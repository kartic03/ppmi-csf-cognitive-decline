"""
Phase 2, step 04: calibration + conformal prediction intervals (task B4).

A) Continuous endpoint (EB MoCA-slope, clinical+CSF Ridge model -- the RQ1
   model from B3):
   - Out-of-fold (OOF) predictions via the leak-safe nested CV harness.
   - Calibration regression: regress observed slope on OOF predicted slope.
     calibration_slope ~1 and calibration_intercept ~0 = well-calibrated.
   - LOO split-conformal prediction intervals at a nominal level (default 90%).
     Empirical coverage = fraction of OOF subjects whose true slope falls in
     the interval.
   - Binned calibration curve: mean predicted vs mean observed per decile bin.

   CAVEAT (recorded here and in output JSON):
     The conformal intervals cover the EB-shrunken slope estimate (the
     analysis target), not the latent true rate. The marginal-coverage
     guarantee assumes exchangeability, which fails under a future
     PPMI -> PDBP distribution shift. B5 will use weighted/transductive
     conformal for that external-validation setting.

B) Binary endpoint (PD-MCI/PDD conversion, clinical+CSF logistic model):
   - OOF predicted probabilities via the harness (subject-level folds;
     in-fold preprocessing + conversion label).
   - In-fold Platt and isotonic recalibration: both recalibrators are fit
     on an inner calibration split of the training fold only -- never on
     the test fold.
   - Metrics: AUC, Brier score, ECE (10-bin uniform), and reliability
     diagram data, reported BEFORE and AFTER each recalibration method.

Output: data/processed/phase2/calibration.json

Parallelism: all GridSearchCV and LogisticRegression calls use n_jobs=1.
A nested loky pool (n_jobs=-1) inside heavy outer CV loops causes
intermittent SIGSEGV (same crash fixed in cv.py and 03_increment.py).

Usage:
    pixi run python src/phase2/04_calibration.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Import B1 utilities via cv.py (re-exported for downstream tasks)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv import nested_cv, fit_eb_params, eb_slopes, load_moca  # noqa: E402

# ---------------------------------------------------------------------------
# Import loaders from 03_increment.py via importlib (digit-prefix filename)
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_spec_inc = importlib.util.spec_from_file_location(
    "_increment_mod_b4", os.path.join(_here, "03_increment.py")
)
_increment_mod = importlib.util.module_from_spec(_spec_inc)
_spec_inc.loader.exec_module(_increment_mod)

_load_csf_aug_df = _increment_mod._load_csf_aug_df
_make_clinical_builder = _increment_mod._make_clinical_builder
CLINICAL_COLS = _increment_mod.CLINICAL_COLS
RIDGE_ALPHA_GRID = _increment_mod.RIDGE_ALPHA_GRID

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CURATED = os.path.join(_ROOT, "data", "processed", "phase1", "curated_cut.parquet")
_OUTCOME = os.path.join(_ROOT, "data", "processed", "phase2", "outcome.parquet")
_NULISA = os.path.join(
    _ROOT, "data", "raw", "ALL Proteomic Analysis", "converted",
    "PPMI_Project_282_NULISAseq_CNSDiseasePanel_NPQCounts_20260120.csv",
)
_DATA_OUT_DIR = os.path.join(_ROOT, "data", "processed", "phase2")
_JSON_OUT = os.path.join(_DATA_OUT_DIR, "calibration.json")


# ---------------------------------------------------------------------------
# Internal preprocessor factory (n_jobs=1 safe)
# ---------------------------------------------------------------------------

def _default_preprocessor_b4():
    """Median imputation + z-score standardisation."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# Low-level metric functions (all public for testing)
# ---------------------------------------------------------------------------

def compute_calibration_slope_intercept(y_true, y_pred):
    """
    Calibration regression: regress y_true on y_pred (OLS).

    A well-calibrated predictor satisfies:
        slope  ~1  (prediction scale matches truth scale)
        intercept ~0  (no systematic over/under-prediction bias)

    Parameters
    ----------
    y_true : array-like  (observed values, e.g. EB slopes)
    y_pred : array-like  (OOF predicted values)

    Returns
    -------
    (slope, intercept) as floats.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    slope, intercept, _, _, _ = stats.linregress(y_pred[mask], y_true[mask])
    return float(slope), float(intercept)


def conformal_intervals_loo(y_true, y_pred, alpha=0.10):
    """
    Leave-one-out split conformal prediction intervals for regression.

    For each OOF point i, the conformal quantile is computed from all other
    OOF residuals (n-1 calibration residuals). The empirical coverage is the
    fraction of OOF points whose true value falls within the interval.

    The nominal coverage guarantee (1 - alpha) holds approximately under
    exchangeability of residuals. This function applies the LOO quantile
    construction to K-fold OOF residuals rather than n separate model refits,
    so coverage is approximate under exchangeability of fold residuals -- not
    an exact finite-sample LOO guarantee. The LOO level uses the exact
    finite-sample correction ceil(n*(1-alpha))/(n-1) (tighter than the
    continuous approximation (1-alpha)*(1+1/(n-1))).

    CAVEAT: the conformal intervals cover the EB-shrunken slope estimate (the
    analysis target), not the latent true rate. The marginal-coverage guarantee
    assumes exchangeability, which fails under a PPMI -> PDBP distribution
    shift. B5 will use weighted/transductive conformal for the external-
    validation setting.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    alpha  : float  (1 - nominal coverage; default 0.10 for 90% intervals)

    Returns
    -------
    (empirical_coverage, mean_interval_width) as floats.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    if n < 3:
        return float("nan"), float("nan")

    residuals = np.abs(y_true - y_pred)
    covered = 0
    widths = []

    for i in range(n):
        loo_res = np.concatenate([residuals[:i], residuals[i + 1:]])
        # LOO conformal quantile: exact finite-sample level ceil(n*(1-alpha))/(n-1)
        # is slightly tighter than the continuous approximation and is the
        # correct discrete quantile for marginal coverage on exchangeable data.
        level = min(1.0, np.ceil(n * (1 - alpha)) / (n - 1))
        q = float(np.quantile(loo_res, level))
        if residuals[i] <= q:
            covered += 1
        widths.append(2.0 * q)

    coverage = float(covered / n)
    mean_width = float(np.mean(widths))
    return coverage, mean_width


def _calibration_curve_binned(y_true, y_pred, n_bins=10):
    """
    Binned calibration curve for continuous predictions.

    Bins are defined by equal-quantile ranges of y_pred (deciles by default).
    Returns a list of dicts: [{mean_pred, mean_obs, n}, ...].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y_pred, percentiles)
    curve = []
    for j, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if j == n_bins - 1:
            mask_b = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask_b = (y_pred >= lo) & (y_pred < hi)
        if mask_b.sum() == 0:
            continue
        curve.append({
            "mean_pred": round(float(y_pred[mask_b].mean()), 4),
            "mean_obs": round(float(y_true[mask_b].mean()), 4),
            "n": int(mask_b.sum()),
        })
    return curve


def _ece(y_true, y_prob, n_bins=10):
    """
    Expected calibration error (uniform bin width on [0, 1]).

    Returns (ece_float, bin_data_list).
    bin_data_list entries: {bin_lo, bin_hi, n, mean_pred, mean_true}.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    bin_data = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        mean_true = float(y_true[mask].mean())
        ece += n_bin / n * abs(mean_pred - mean_true)
        bin_data.append({
            "bin_lo": round(float(lo), 3),
            "bin_hi": round(float(hi), 3),
            "n": n_bin,
            "mean_pred": round(mean_pred, 4),
            "mean_true": round(mean_true, 4),
        })
    return float(ece), bin_data


# ---------------------------------------------------------------------------
# Continuous endpoint: OOF predictions + calibration
# ---------------------------------------------------------------------------

def run_continuous_calibration(moca_long, curated_path, nulisa_path, cv_config):
    """
    Calibration analysis for the continuous EB MoCA-slope endpoint.

    Uses the clinical+CSF Ridge model (same feature set as B3 RQ1).
    OOF predictions are obtained from a single-seed nested CV run (canonical
    seed only -- calibration metrics are slope/intercept and conformal
    coverage, not R2; additional seeds add computation without benefit).

    Returns a dict for the 'continuous' key of calibration.json.
    """
    outcome_df = pd.read_parquet(_OUTCOME)
    slope_ids = set(outcome_df["subject_id"].astype(int))

    clin_builder = _make_clinical_builder(curated_path, slope_ids)
    csf_aug = _load_csf_aug_df(curated_path, nulisa_path, slope_ids)

    # Analytic IDs: clinical + CSF complete (same as B3 RQ1)
    clin_probe = clin_builder(list(csf_aug.index))
    analytic_ids = clin_probe.index.intersection(csf_aug.index).tolist()
    print(f"Continuous calibration analytic N: {len(analytic_ids)}")

    def csf_clin_builder(subject_ids):
        """Clinical + CSF feature builder (clinical joined with CSF)."""
        base = clin_builder(subject_ids)
        csf_sub = csf_aug.loc[csf_aug.index.intersection(base.index)]
        return base.join(csf_sub, how="inner")

    # Single-seed CV for calibration (canonical seed 0)
    calib_cv_config = {
        "n_outer": int(cv_config.get("n_outer", 5)),
        "n_inner": int(cv_config.get("n_inner", 5)),
        "seeds": [int(list(cv_config.get("seeds", [0]))[0])],
    }

    print("Running clinical+CSF nested CV for calibration (canonical seed)...")
    result = nested_cv(
        feature_builder=csf_clin_builder,
        moca_long=moca_long,
        model_factory=lambda: Ridge(),
        param_grid={"alpha": RIDGE_ALPHA_GRID},
        cv_config=calib_cv_config,
        analytic_ids=analytic_ids,
    )

    oof_pred = result["oof_predictions"]
    oof_truth = result["oof_truth"]
    common = oof_pred.index.intersection(oof_truth.index)
    y_pred = oof_pred.loc[common].to_numpy(float)
    y_true = oof_truth.loc[common].to_numpy(float)

    # Calibration slope and intercept
    slope, intercept = compute_calibration_slope_intercept(y_true, y_pred)
    print(f"Calibration slope={slope:.4f}, intercept={intercept:.4f}")

    # LOO conformal intervals at 90% (nominal)
    alpha = 0.10
    coverage, mean_width = conformal_intervals_loo(y_true, y_pred, alpha=alpha)
    print(f"Conformal coverage (nominal 90%): {coverage:.4f}, "
          f"mean width: {mean_width:.4f}")

    # Binned calibration curve (decile bins)
    curve = _calibration_curve_binned(y_true, y_pred, n_bins=10)

    return {
        "n": int(len(common)),
        "calib_slope": round(slope, 4),
        "calib_intercept": round(intercept, 4),
        "conformal_level": float(1 - alpha),
        "conformal_coverage": round(coverage, 4),
        "mean_interval_width": round(mean_width, 4),
        "calibration_curve": curve,
        "caveat": (
            "Conformal intervals cover the EB-shrunken slope estimate (the "
            "analysis target), not the latent true rate. The marginal-coverage "
            "guarantee assumes exchangeability, which fails under the future "
            "PPMI->PDBP distribution shift; B5 will use weighted/transductive "
            "conformal for that external-validation setting."
        ),
    }


# ---------------------------------------------------------------------------
# Binary endpoint: OOF predictions with in-fold recalibration
# ---------------------------------------------------------------------------

def _load_conversion_labels(curated_path):
    """
    Load the binary PD-MCI/PDD conversion label from curated_cut.parquet.

    Definition:
      - Include: PD subjects (COHORT==1) with cogstate==1 at the BL visit.
      - converted = 1 if the subject ever had cogstate >= 2 at a non-BL visit.

    Returns pd.Series indexed by PATNO (baseline-normal PD only), values 0/1.
    """
    curated = pd.read_parquet(curated_path)
    pd_df = curated[curated["COHORT"] == 1].copy()

    # Baseline-normal: cogstate == 1 at BL
    bl = pd_df[pd_df["EVENT_ID"] == "BL"].set_index("PATNO")
    baseline_normal_ids = bl[bl["cogstate"] == 1].index

    # Ever converted: cogstate >= 2 at any non-BL visit
    converted_ids = set(
        pd_df[
            (pd_df["PATNO"].isin(baseline_normal_ids)) &
            (pd_df["EVENT_ID"] != "BL") &
            (pd_df["cogstate"] >= 2)
        ]["PATNO"].unique()
    )

    y = pd.Series(
        [1.0 if int(pid) in converted_ids else 0.0 for pid in baseline_normal_ids],
        index=baseline_normal_ids,
        name="converted",
        dtype=float,
    )
    return y


def _binary_oof_cv(
    feature_builder,
    y_series,
    all_ids,
    cv_config=None,
    calib_frac=0.25,
    _calib_spy=None,
):
    """
    OOF binary probability predictions with in-fold Platt and isotonic
    recalibration.

    In each outer fold the training subjects are split further:
      inner_train (~75%) : base logistic model fitting
      inner_calib (~25%) : recalibrator fitting (Platt + isotonic)

    The recalibrator NEVER sees test-fold subjects or their labels. If
    _calib_spy is provided it is called with
    (inner_calib_ids, test_ids, inner_train_ids) for each outer fold; callers
    use this to verify leak-safety (see test_recalibration_in_fold_only in
    test_calibration.py).

    n_jobs=1 throughout: avoid nested loky pool SIGSEGV risk.

    Parameters
    ----------
    feature_builder : callable  (subject_ids) -> pd.DataFrame
    y_series        : pd.Series  (PATNO -> binary label 0/1)
    all_ids         : array-like  (analytic subject IDs)
    cv_config       : dict  (n_outer, seeds)
    calib_frac      : float  (fraction of train fold used for recalibration)
    _calib_spy      : callable(inner_calib_ids, test_ids, inner_train_ids) or None

    Returns
    -------
    dict with pd.Series:
        oof_raw      : raw model probability
        oof_platt    : Platt-recalibrated probability
        oof_isotonic : isotonic-recalibrated probability
        oof_truth    : binary label
    """
    cfg = cv_config or {}
    n_outer = int(cfg.get("n_outer", 5))
    seed = int(list(cfg.get("seeds", [0]))[0])

    all_ids = np.array(sorted(set(int(i) for i in all_ids)))
    rng = np.random.default_rng(seed)

    oof_raw = {}
    oof_platt = {}
    oof_isotonic = {}
    oof_truth = {}

    kf = KFold(n_splits=n_outer, shuffle=True, random_state=seed)

    for tr_idx, te_idx in kf.split(all_ids):
        train_ids_fold = all_ids[tr_idx]
        test_ids_fold = all_ids[te_idx]

        # Labels for this fold
        y_train_full = y_series.reindex(train_ids_fold).dropna()
        y_test_full = y_series.reindex(test_ids_fold).dropna()

        # Features (built inside fold)
        X_train_raw = feature_builder(y_train_full.index.to_numpy())
        X_test_raw = feature_builder(y_test_full.index.to_numpy())

        train_common = X_train_raw.index.intersection(y_train_full.index)
        test_common = X_test_raw.index.intersection(y_test_full.index)

        if len(train_common) < 20 or len(test_common) < 1:
            continue
        if len(np.unique(y_train_full.loc[train_common].to_numpy())) < 2:
            continue

        # Split training fold into inner_train and inner_calib
        train_common_arr = train_common.to_numpy()
        n_train = len(train_common_arr)
        n_calib = max(5, int(n_train * calib_frac))
        calib_idx = rng.choice(n_train, n_calib, replace=False)
        calib_mask = np.zeros(n_train, dtype=bool)
        calib_mask[calib_idx] = True

        inner_train_ids = train_common_arr[~calib_mask]
        inner_calib_ids = train_common_arr[calib_mask]

        # Leak-safety spy: called BEFORE any model fitting
        if _calib_spy is not None:
            _calib_spy(inner_calib_ids, test_common.to_numpy(), inner_train_ids)

        X_inner_train = X_train_raw.loc[inner_train_ids]
        X_inner_calib = X_train_raw.loc[inner_calib_ids]
        X_test = X_test_raw.loc[test_common]

        y_inner_train = y_train_full.loc[inner_train_ids].to_numpy()
        y_inner_calib = y_train_full.loc[inner_calib_ids].to_numpy()
        y_te = y_test_full.loc[test_common].to_numpy()

        # Preprocessing: fit on INNER TRAIN only (leak-safe)
        pre = _default_preprocessor_b4()
        X_it_pp = pre.fit_transform(X_inner_train)
        X_ic_pp = pre.transform(X_inner_calib)
        X_te_pp = pre.transform(X_test)

        # Skip folds with single class in inner_train
        if len(np.unique(y_inner_train)) < 2:
            continue

        # Base logistic model (no inner tuning -- keeps loop serial and fast)
        base = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, n_jobs=1)
        base.fit(X_it_pp, y_inner_train)
        p_calib_raw = base.predict_proba(X_ic_pp)[:, 1]
        p_test_raw = base.predict_proba(X_te_pp)[:, 1]

        # Default: no recalibration (raw predictions used if calib is degenerate)
        p_test_platt = p_test_raw.copy()
        p_test_iso = p_test_raw.copy()

        # In-fold recalibration (fit on inner_calib ONLY)
        if len(np.unique(y_inner_calib)) >= 2:
            # Platt scaling: univariate logistic regression on raw probabilities
            platt = LogisticRegression(
                C=1e6, max_iter=1000, random_state=seed, n_jobs=1
            )
            platt.fit(p_calib_raw.reshape(-1, 1), y_inner_calib)
            p_test_platt = platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

            # Isotonic regression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_calib_raw, y_inner_calib)
            p_test_iso = iso.predict(p_test_raw)

        for sid, pr, pp, pi, yt in zip(
            test_common, p_test_raw, p_test_platt, p_test_iso, y_te
        ):
            oof_raw[int(sid)] = float(pr)
            oof_platt[int(sid)] = float(pp)
            oof_isotonic[int(sid)] = float(pi)
            oof_truth[int(sid)] = float(yt)

    return {
        "oof_raw": pd.Series(oof_raw, name="p_raw"),
        "oof_platt": pd.Series(oof_platt, name="p_platt"),
        "oof_isotonic": pd.Series(oof_isotonic, name="p_isotonic"),
        "oof_truth": pd.Series(oof_truth, name="converted"),
    }


def run_binary_calibration(curated_path, nulisa_path, cv_config):
    """
    Calibration analysis for the binary PD-MCI/PDD conversion endpoint.

    Analytic set: baseline-normal PD (cogstate==1 at BL) with complete
    clinical + CSF features. Uses a logistic regression model with in-fold
    Platt and isotonic recalibration.

    Returns a dict for the 'binary' key of calibration.json.
    """
    y_all = _load_conversion_labels(curated_path)
    print(f"Binary endpoint: {len(y_all)} baseline-normal subjects, "
          f"{int(y_all.sum())} converters")

    # Build clinical feature builder for ALL baseline-normal PD subjects
    all_baseline_ids = set(y_all.index.astype(int))
    clin_builder = _make_clinical_builder(curated_path, all_baseline_ids)

    # CSF augmentation for the binary analytic set
    csf_aug = _load_csf_aug_df(curated_path, nulisa_path, all_baseline_ids)

    # Combined feature builder
    def csf_clin_builder(subject_ids):
        base = clin_builder(subject_ids)
        csf_sub = csf_aug.loc[csf_aug.index.intersection(base.index)]
        return base.join(csf_sub, how="inner")

    # Analytic IDs: clinical + CSF complete + has conversion label
    clin_probe = clin_builder(list(csf_aug.index))
    csf_clin_ids = clin_probe.index.intersection(csf_aug.index)
    analytic_ids = csf_clin_ids.intersection(y_all.index).tolist()
    y_analytic = y_all.loc[analytic_ids]

    n_subjects = len(analytic_ids)
    n_events = int(y_analytic.sum())
    print(f"Binary CSF-complete analytic N: {n_subjects}, events: {n_events}")

    if n_events < 20 or (n_subjects - n_events) < 20:
        print("Too few events or non-events; returning empty binary result.")
        return {
            "n": n_subjects, "events": n_events,
            "auc": None, "brier_raw": None, "brier_platt": None,
            "brier_isotonic": None, "ece_raw": None, "ece_platt": None,
            "ece_isotonic": None, "reliability": [],
        }

    calib_cv_config = {
        "n_outer": int(cv_config.get("n_outer", 5)),
        "seeds": [int(list(cv_config.get("seeds", [0]))[0])],
    }

    print("Running binary OOF CV with in-fold Platt + isotonic recalibration...")
    oof = _binary_oof_cv(
        feature_builder=csf_clin_builder,
        y_series=y_analytic,
        all_ids=analytic_ids,
        cv_config=calib_cv_config,
    )

    # Align OOF series by index
    common = oof["oof_truth"].index
    y_true = oof["oof_truth"].loc[common].to_numpy(float)
    p_raw = oof["oof_raw"].reindex(common).to_numpy(float)
    p_platt = oof["oof_platt"].reindex(common).to_numpy(float)
    p_iso = oof["oof_isotonic"].reindex(common).to_numpy(float)

    # Metrics
    auc = float(roc_auc_score(y_true, p_raw))
    brier_raw = float(brier_score_loss(y_true, p_raw))
    brier_platt = float(brier_score_loss(y_true, p_platt))
    brier_iso = float(brier_score_loss(y_true, p_iso))

    ece_raw, _ = _ece(y_true, p_raw)
    ece_platt, _ = _ece(y_true, p_platt)
    ece_iso, rel_data = _ece(y_true, p_iso)

    print(f"Binary AUC={auc:.4f}")
    print(f"Brier: raw={brier_raw:.4f}, Platt={brier_platt:.4f}, "
          f"isotonic={brier_iso:.4f}")
    print(f"ECE:   raw={ece_raw:.4f}, Platt={ece_platt:.4f}, "
          f"isotonic={ece_iso:.4f}")

    return {
        "n": n_subjects,
        "events": n_events,
        "auc": round(auc, 4),
        "brier_raw": round(brier_raw, 4),
        "brier_platt": round(brier_platt, 4),
        "brier_isotonic": round(brier_iso, 4),
        "ece_raw": round(ece_raw, 4),
        "ece_platt": round(ece_platt, 4),
        "ece_isotonic": round(ece_iso, 4),
        "reliability": rel_data,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_calibration(
    curated_path=None,
    nulisa_path=None,
    cv_config=None,
    moca_long=None,
):
    """
    Run the full B4 calibration analysis (continuous + binary endpoints).

    Parameters
    ----------
    curated_path : str or None
    nulisa_path  : str or None
    cv_config    : dict or None  (n_outer, n_inner, seeds)
    moca_long    : pd.DataFrame or None

    Returns
    -------
    dict  (content for calibration.json)
    """
    curated_path = curated_path or _CURATED
    nulisa_path = nulisa_path or _NULISA

    if cv_config is None:
        cv_config = {"n_outer": 5, "n_inner": 5, "seeds": list(range(10))}

    if moca_long is None:
        moca_long = load_moca(curated_path)

    print("=== B4: Continuous endpoint (EB MoCA-slope, clinical+CSF Ridge) ===")
    cont = run_continuous_calibration(moca_long, curated_path, nulisa_path, cv_config)

    print("\n=== B4: Binary endpoint (PD-MCI/PDD conversion, clinical+CSF logistic) ===")
    binary = run_binary_calibration(curated_path, nulisa_path, cv_config)

    return {"continuous": cont, "binary": binary}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    result = run_calibration()

    os.makedirs(_DATA_OUT_DIR, exist_ok=True)
    with open(_JSON_OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {_JSON_OUT}")
    return result


if __name__ == "__main__":
    main()
