"""
Phase 2, step 03: biomarker increment test (task B3).

RQ1 — CSF increment over clinical (the headline positive):
  On CSF-complete subjects (clinical + NULISA NEFL + pTau181 + Abeta42/40 +
  pSNCA-129, N~484-607), compute paired dR2 of clinical+CSF over
  clinical-only. Both models use the SAME outer CV folds, so dR2 is paired,
  not a difference of two independent runs.

  Permutation negative control (200 shuffles): shuffle the CSF block across
  subjects (breaking the link to outcome), recompute dR2, report null mean
  (should be ~0) and permutation p-value (fraction of shuffles >= observed).

  Sensitivity: recompute CSF dR2 using ols_slope and slope_excl_baseline
  as the outcome (not the EB slope). This confirms the increment is not an
  artifact of shrinkage or baseline coupling.

RQ2 — BBB predictive null (pre-registered as null):
  (a) NULISA BBB block (N~617): dR2 of (NfL+clinical+PDGFRB/ICAM1/VCAM1/VEGFA)
      over (NfL+clinical).
  (b) Q-albumin subgroup (N~319): dR2 of (NfL+clinical+Q-albumin) over
      (NfL+clinical) on the Q-albumin subgroup.
  Report honestly whether dR2 crosses zero or not. Do not spin a null as
  positive.

Output: data/processed/phase2/increment.json

Leak-safety:
  - preprocessor.fit is called on TRAIN subjects only per outer fold.
  - fit_eb_params is called on TRAIN subjects only per outer fold.
  - augmentation columns are NOT shuffled before CV (only in permutation runs).
  - analytic_ids is the pre-computed intersection of subjects with all required
    features; both base and augmented models are restricted to this set.

Usage:
    pixi run python src/phase2/03_increment.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Import B1/B2 utilities via cv.py (re-exported for B3)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv import fit_eb_params, eb_slopes, load_moca  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CURATED = os.path.join(_ROOT, "data", "processed", "phase1", "curated_cut.parquet")
_OUTCOME = os.path.join(_ROOT, "data", "processed", "phase2", "outcome.parquet")
_QALB = os.path.join(_ROOT, "data", "processed", "phase1", "qalbumin.csv")
_NULISA = os.path.join(
    _ROOT, "data", "raw", "ALL Proteomic Analysis", "converted",
    "PPMI_Project_282_NULISAseq_CNSDiseasePanel_NPQCounts_20260120.csv",
)
_DATA_OUT_DIR = os.path.join(_ROOT, "data", "processed", "phase2")
_JSON_OUT = os.path.join(_DATA_OUT_DIR, "increment.json")

# ---------------------------------------------------------------------------
# Ridge hyperparameter grids
# ---------------------------------------------------------------------------
RIDGE_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
PERM_RIDGE_ALPHA = 1.0   # fixed alpha for permutation runs (no inner CV)

# Full-cohort (N=816) clinical R2 from B2 (commit dcaeb85).
# NOTE: do NOT add this to the N=607 paired dR2. The additive clinical+CSF
# figure is the paired rq1.r2_aug over rq1.r2_base (both on N=607 subjects).
# Report clinical+CSF as: R2 0.0692 -> 0.1273 (dR2 +0.058, paired, N=607).
BASELINE_CLINICAL_R2_FULL_COHORT_N816 = 0.0764

# Clinical predictor columns (from B2)
CLINICAL_COLS = [
    "age", "SEX", "EDUCYRS", "moca", "updrs3_score",
    "duration_yrs", "LEDD", "APOE_e4", "gba_status", "MIA_PUTAMEN_BILAT",
]


# ---------------------------------------------------------------------------
# Default factories
# ---------------------------------------------------------------------------

def _default_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def _default_model():
    return Ridge()


# ---------------------------------------------------------------------------
# Core: paired increment CV
# ---------------------------------------------------------------------------

def paired_increment_cv(
    base_builder,
    aug_df,
    moca_long,
    model_factory=None,
    preprocessor_factory=None,
    param_grid=None,
    cv_config=None,
    analytic_ids=None,
    outcome_col="eb_slope",
):
    """
    Paired increment: base and base+augmented model on IDENTICAL outer folds.

    For each outer fold, both models are trained on the same training subjects
    and evaluated on the same test subjects. The dR2 is therefore truly paired
    (not a difference of two independent runs).

    Harness note — why this function exists instead of calling nested_cv twice:
        cv.py exports nested_cv, which wraps a SINGLE model through nested CV.
        A single nested_cv call cannot simultaneously maintain predictions from
        TWO models (base and augmented) on the SAME outer-fold test subjects.
        Calling nested_cv twice independently would produce predictions from
        different random fold splits (even with the same seed, subject ordering
        in the two calls can diverge once the analytic set differs by one
        subject). To guarantee the paired property, this function reimplements
        the outer CV loop explicitly, running BOTH models on each fold's
        train/test split before moving to the next fold. The inner CV for
        hyperparameter tuning is structurally identical to nested_cv's inner CV
        (same KFold seed and n_splits).

    Parameters
    ----------
    base_builder : callable
        (subject_ids) -> pd.DataFrame indexed by subject_id (base features).
    aug_df : pd.DataFrame
        Indexed by subject_id (augmentation columns only). Joined with base
        features to form the augmented input.
    moca_long : pd.DataFrame
        Longitudinal MoCA table (PATNO, EVENT_ID, age_at_visit, moca).
    model_factory : callable or None
        () -> sklearn estimator. Default: Ridge().
    preprocessor_factory : callable or None
        () -> sklearn Pipeline. Default: SimpleImputer + StandardScaler.
    param_grid : dict or None
        Hyperparameter grid for inner GridSearchCV. None = no inner tuning.
    cv_config : dict or None
        n_outer (default 5), n_inner (default 5), seeds (default [0..9]).
    analytic_ids : array-like or None
        Subjects with BOTH base and augmentation features. If None, inferred
        from aug_df.index intersected with base_builder output.
    outcome_col : str
        Column from eb_slopes() to use as outcome. One of:
        'eb_slope' (default, primary), 'ols_slope', 'slope_excl_baseline'.

    Returns
    -------
    dict with keys:
        oof_base        : pd.Series (subject_id -> base-model OOF prediction)
        oof_aug         : pd.Series (subject_id -> augmented-model OOF prediction)
        oof_truth       : pd.Series (subject_id -> true outcome)
        r2_base         : float  (pooled OOF R2, base model, canonical seed)
        r2_aug          : float  (pooled OOF R2, augmented model, canonical seed)
        dr2             : float  (r2_aug - r2_base, canonical seed)
        per_seed_dr2    : list[float]  (dr2 per seed)
        dr2_repeat_lo   : float  (min dr2 across seeds)
        dr2_repeat_hi   : float  (max dr2 across seeds)
        n_subjects      : int
    """
    if model_factory is None:
        model_factory = _default_model
    if preprocessor_factory is None:
        preprocessor_factory = _default_preprocessor

    cfg = cv_config or {}
    n_outer = int(cfg.get("n_outer", 5))
    n_inner = int(cfg.get("n_inner", 5))
    seeds = [int(s) for s in cfg.get("seeds", list(range(10)))]
    canonical_seed = seeds[0]

    # Determine analytic subject set
    if analytic_ids is not None:
        valid_ids = np.array(sorted(set(int(i) for i in analytic_ids)))
    else:
        # Infer from aug_df index (all subjects with augmentation features)
        valid_ids = np.array(sorted(int(i) for i in aug_df.index))
    n_subjects = len(valid_ids)
    if n_subjects == 0:
        raise ValueError("paired_increment_cv: analytic subject set is empty.")

    def _aug_builder(subject_ids):
        """Base features joined with aug_df (inner join on subjects)."""
        base = base_builder(subject_ids)
        aug_subset = aug_df.loc[aug_df.index.intersection(base.index)]
        return base.join(aug_subset, how="inner")

    canonical_oof = None
    per_seed_dr2 = []

    for seed in seeds:
        kf = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
        oof_base_pred = {}
        oof_aug_pred = {}
        oof_truth_dict = {}

        for tr_idx, te_idx in kf.split(valid_ids):
            train_ids = valid_ids[tr_idx]
            test_ids = valid_ids[te_idx]

            # ---- OUTCOME: refit EB params on train only (leak-safe) --------
            p = fit_eb_params(moca_long, train_ids)
            df_tr_y = eb_slopes(moca_long, train_ids, p)
            df_te_y = eb_slopes(moca_long, test_ids, p)

            if df_tr_y.empty or df_te_y.empty:
                continue
            if outcome_col not in df_tr_y.columns:
                raise ValueError(
                    f"outcome_col='{outcome_col}' not in eb_slopes output. "
                    f"Available: {df_tr_y.columns.tolist()}"
                )

            y_tr_full = df_tr_y[outcome_col].dropna()
            y_te_full = df_te_y[outcome_col].dropna()

            # ---- BASE FEATURES (train and test) ----------------------------
            Xb_tr_raw = base_builder(y_tr_full.index.to_numpy())
            Xb_te_raw = base_builder(y_te_full.index.to_numpy())

            # ---- AUGMENTED FEATURES (train and test) -----------------------
            Xa_tr_raw = _aug_builder(y_tr_full.index.to_numpy())
            Xa_te_raw = _aug_builder(y_te_full.index.to_numpy())

            # ---- COMMON SUBJECTS: identical for both models ----------------
            train_common = (
                Xb_tr_raw.index
                .intersection(y_tr_full.index)
                .intersection(Xa_tr_raw.index)
            )
            test_common = (
                Xb_te_raw.index
                .intersection(y_te_full.index)
                .intersection(Xa_te_raw.index)
            )

            if len(train_common) < n_inner + 1 or len(test_common) < 1:
                continue

            Xb_tr = Xb_tr_raw.loc[train_common]
            Xa_tr = Xa_tr_raw.loc[train_common]
            y_tr = y_tr_full.loc[train_common].to_numpy()

            Xb_te = Xb_te_raw.loc[test_common]
            Xa_te = Xa_te_raw.loc[test_common]
            y_te = y_te_full.loc[test_common].to_numpy()

            # ---- PREPROCESSING: fit on train only (leak-safe) -------------
            pre_base = preprocessor_factory()
            Xb_tr_pp = pre_base.fit_transform(Xb_tr)
            Xb_te_pp = pre_base.transform(Xb_te)

            pre_aug = preprocessor_factory()
            Xa_tr_pp = pre_aug.fit_transform(Xa_tr)
            Xa_te_pp = pre_aug.transform(Xa_te)

            # ---- INNER CV (shared structure; stateless KFold) --------------
            inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=seed)

            # Fit base model. n_jobs=1 (not -1): the inner grid is tiny, and
            # n_jobs=-1's loky process pool nests with BLAS threads + the heavy
            # pandas groupby in the permutation loop -> intermittent SIGSEGV.
            # Serial is deterministic and gives identical estimates.
            base_est = model_factory()
            if param_grid:
                base_est = GridSearchCV(
                    base_est, param_grid, cv=inner_cv,
                    scoring="r2", refit=True, n_jobs=1,
                )
            base_est.fit(Xb_tr_pp, y_tr)

            # Fit augmented model. n_jobs=1 (not -1): same reasoning as the
            # base-model fit above -- serial avoids the loky/BLAS/SIGSEGV risk.
            aug_est = model_factory()
            if param_grid:
                aug_est = GridSearchCV(
                    aug_est, param_grid, cv=inner_cv,
                    scoring="r2", refit=True, n_jobs=1,
                )
            aug_est.fit(Xa_tr_pp, y_tr)

            # ---- PREDICT on SAME test subjects (paired property) ----------
            y_base_pred = base_est.predict(Xb_te_pp)
            y_aug_pred = aug_est.predict(Xa_te_pp)

            for sid, bp, ap, yt in zip(test_common, y_base_pred, y_aug_pred, y_te):
                oof_base_pred[int(sid)] = float(bp)
                oof_aug_pred[int(sid)] = float(ap)
                oof_truth_dict[int(sid)] = float(yt)

        if not oof_truth_dict:
            per_seed_dr2.append(float("nan"))
            continue

        keys = sorted(oof_truth_dict)
        truths = [oof_truth_dict[k] for k in keys]
        base_preds = [oof_base_pred[k] for k in keys]
        aug_preds = [oof_aug_pred[k] for k in keys]

        seed_r2_base = float(r2_score(truths, base_preds))
        seed_r2_aug = float(r2_score(truths, aug_preds))
        seed_dr2 = seed_r2_aug - seed_r2_base
        per_seed_dr2.append(seed_dr2)

        if seed == canonical_seed:
            canonical_oof = {
                "r2_base": seed_r2_base,
                "r2_aug": seed_r2_aug,
                "dr2": seed_dr2,
                "oof_base": pd.Series(oof_base_pred, name="predicted_base"),
                "oof_aug": pd.Series(oof_aug_pred, name="predicted_aug"),
                "oof_truth": pd.Series(oof_truth_dict, name=outcome_col),
            }

    valid_dr2s = [v for v in per_seed_dr2 if np.isfinite(v)]
    if len(valid_dr2s) >= 2:
        dr2_repeat_lo = float(np.min(valid_dr2s))
        dr2_repeat_hi = float(np.max(valid_dr2s))
    elif len(valid_dr2s) == 1:
        dr2_repeat_lo = dr2_repeat_hi = valid_dr2s[0]
    else:
        dr2_repeat_lo = dr2_repeat_hi = float("nan")

    if canonical_oof is None:
        canonical_oof = {
            "r2_base": float("nan"), "r2_aug": float("nan"), "dr2": float("nan"),
            "oof_base": pd.Series(dtype=float),
            "oof_aug": pd.Series(dtype=float),
            "oof_truth": pd.Series(dtype=float),
        }

    return {
        **canonical_oof,
        "per_seed_dr2": per_seed_dr2,
        "dr2_repeat_lo": dr2_repeat_lo,
        "dr2_repeat_hi": dr2_repeat_hi,
        "n_subjects": n_subjects,
    }


# ---------------------------------------------------------------------------
# Permutation null
# ---------------------------------------------------------------------------

def compute_permutation_null(
    base_builder,
    aug_df,
    moca_long,
    observed_dr2,
    base_r2=None,
    cv_config=None,
    analytic_ids=None,
    outcome_col="eb_slope",
    n_perm=200,
    rng_seed=42,
):
    """
    Permutation null distribution of dR2.

    For each of n_perm shuffles:
      1. Permute the rows of aug_df (all columns move together, preserving
         within-block correlation structure while breaking the link to outcome).
      2. Run the augmented model with the shuffled aug_df using a fixed Ridge
         alpha (no inner tuning, canonical seed only — for speed).
      3. Record perm_dR2 = R2(shuffled_aug_fixed) - R2(base_fixed).

    Both base and augmented use the SAME fixed Ridge alpha so the null
    distribution is centered near zero (consistent comparison). perm_p is
    computed against observed_dr2_fixed (the fixed-alpha reference dR2 computed
    internally), NOT against the tuned headline dR2 passed in observed_dr2.
    This is intentional: Ridge(alpha=1.0) bias-invariance means the permutation
    comparison is internally consistent. The passed observed_dr2 (from the tuned
    main run with inner CV) is stored as observed_dr2_tuned for reporting only.
    The returned perm_null_mean reflects this consistent null.

    Seed note: the permutation loop uses the canonical seed (seeds[0]) only,
    for speed. The main dR2 reported in paired_increment_cv averages over all
    seeds (default 10). This is a known minor asymmetry — it does not affect
    the validity of the p-value.

    Parameters
    ----------
    base_builder     : callable  (same as in paired_increment_cv)
    aug_df           : pd.DataFrame  (augmentation columns, indexed by subject_id)
    moca_long        : pd.DataFrame
    observed_dr2     : float  (dR2 from the tuned main run; stored for context
                       but the p-value is computed against the fixed-alpha
                       observed increment for consistency)
    base_r2          : float or None  (unused; kept for backward compatibility)
    cv_config        : dict or None  (only n_outer and seeds[0] used)
    analytic_ids     : array-like or None
    outcome_col      : str  (same as in paired_increment_cv)
    n_perm           : int  (number of shuffles, default 200)
    rng_seed         : int  (master seed for reproducible shuffles)

    Returns
    -------
    dict with keys:
        perm_dr2s           : list[float]  (one dR2 per permutation, fixed alpha)
        perm_null_mean      : float  (mean of perm_dr2s; should be near 0)
        perm_p              : float  (fraction of perm_dr2s >= observed_dr2_fixed)
        observed_dr2_fixed  : float  (dR2 with fixed alpha; reference for perm_p)
        observed_dr2_tuned  : float  (dR2 from tuned main run; stored for report)
    """
    rng = np.random.default_rng(rng_seed)

    cfg = cv_config or {}
    n_outer = int(cfg.get("n_outer", 5))
    canonical_seed = int(cfg.get("seeds", [0])[0])

    if analytic_ids is not None:
        valid_ids = np.array(sorted(set(int(i) for i in analytic_ids)))
    else:
        valid_ids = np.array(sorted(int(i) for i in aug_df.index))

    # Fixed Ridge (no inner CV) — ensures consistent comparison between
    # permutation runs and the fixed-alpha reference (null centered near 0)
    perm_model_factory = lambda: Ridge(alpha=PERM_RIDGE_ALPHA)
    perm_preprocessor_factory = _default_preprocessor
    n_aug_subjects = len(aug_df)
    aug_index = aug_df.index.to_numpy()

    def _run_model_fixed(feature_builder_fn):
        """
        Pooled OOF R2 for a model using feature_builder_fn, fixed alpha,
        canonical seed. Used for both base and augmented runs.
        """
        kf = KFold(n_splits=n_outer, shuffle=True, random_state=canonical_seed)
        oof_pred = {}
        oof_truth = {}

        for tr_idx, te_idx in kf.split(valid_ids):
            train_ids = valid_ids[tr_idx]
            test_ids = valid_ids[te_idx]

            p = fit_eb_params(moca_long, train_ids)
            df_tr_y = eb_slopes(moca_long, train_ids, p)
            df_te_y = eb_slopes(moca_long, test_ids, p)

            if df_tr_y.empty or df_te_y.empty:
                continue
            y_tr_full = df_tr_y[outcome_col].dropna()
            y_te_full = df_te_y[outcome_col].dropna()

            X_tr_raw = feature_builder_fn(y_tr_full.index.to_numpy())
            X_te_raw = feature_builder_fn(y_te_full.index.to_numpy())

            train_common = X_tr_raw.index.intersection(y_tr_full.index)
            test_common = X_te_raw.index.intersection(y_te_full.index)

            if len(train_common) < 2 or len(test_common) < 1:
                continue

            X_tr = X_tr_raw.loc[train_common]
            X_te = X_te_raw.loc[test_common]
            y_tr = y_tr_full.loc[train_common].to_numpy()
            y_te = y_te_full.loc[test_common].to_numpy()

            pre = perm_preprocessor_factory()
            X_tr_pp = pre.fit_transform(X_tr)
            X_te_pp = pre.transform(X_te)

            m = perm_model_factory()
            m.fit(X_tr_pp, y_tr)
            y_pred = m.predict(X_te_pp)

            for sid, pred, yt in zip(test_common, y_pred, y_te):
                oof_pred[int(sid)] = float(pred)
                oof_truth[int(sid)] = float(yt)

        if not oof_truth:
            return float("nan")
        keys = sorted(oof_truth)
        return float(r2_score(
            [oof_truth[k] for k in keys],
            [oof_pred[k] for k in keys],
        ))

    def _aug_builder_fn(the_aug_df):
        def _builder(subject_ids):
            base = base_builder(subject_ids)
            aug_subset = the_aug_df.loc[the_aug_df.index.intersection(base.index)]
            return base.join(aug_subset, how="inner")
        return _builder

    # One-time fixed-alpha reference runs (defines the consistent null baseline)
    r2_base_fixed = _run_model_fixed(base_builder)
    r2_aug_real_fixed = _run_model_fixed(_aug_builder_fn(aug_df))
    observed_dr2_fixed = r2_aug_real_fixed - r2_base_fixed

    # Permutation loop: shuffle aug columns, compute dR2 against fixed base
    perm_dr2s = []
    for _ in range(n_perm):
        perm_idx = rng.permutation(n_aug_subjects)
        shuffled = aug_df.iloc[perm_idx].copy()
        shuffled.index = aug_index  # same subject IDs, shuffled values
        r2_aug_perm = _run_model_fixed(_aug_builder_fn(shuffled))
        perm_dr2 = r2_aug_perm - r2_base_fixed
        perm_dr2s.append(float(perm_dr2))

    perm_dr2s_arr = np.array(perm_dr2s)
    perm_null_mean = float(np.nanmean(perm_dr2s_arr))
    # p-value: fraction of null draws >= fixed-alpha observed increment.
    # REPORTING NOTE: a perm_p of 0.0 (no null shuffle exceeded the observed
    # increment) must be reported in the manuscript as p < 1/n_perm (e.g.,
    # p < 0.005 for n_perm=200, p < 0.002 for n_perm=500). Never report
    # "p = 0" because this estimator lacks the (b+1)/(m+1) continuity
    # correction; it can only bound the true p, not pin it at zero.
    valid = perm_dr2s_arr[np.isfinite(perm_dr2s_arr)]
    perm_p = float(
        np.mean(valid >= observed_dr2_fixed)
    ) if len(valid) > 0 else float("nan")

    return {
        "perm_dr2s": perm_dr2s,
        "perm_null_mean": perm_null_mean,
        "perm_p": perm_p,
        "observed_dr2_fixed": observed_dr2_fixed,
        "observed_dr2_tuned": observed_dr2,
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_nulisa_wide(nulisa_path, slope_ids, targets):
    """
    Load NULISA CSV, filter to Sample/passed, pivot to wide (subject x target).

    Takes earliest event per PATNO per Target. Returns pd.DataFrame indexed
    by PATNO with one column per target in `targets`. Only subjects in
    slope_ids are returned. Values are log1p-transformed NPQ counts.
    """
    nulisa = pd.read_csv(nulisa_path)
    samples = nulisa[
        (nulisa["SampleType"] == "Sample") & (nulisa["SampleQC"] == "passed")
    ].dropna(subset=["PATNO"]).copy()
    samples["PATNO"] = samples["PATNO"].astype(int)
    samples = samples[samples["Target"].isin(targets)]
    samples = samples[samples["PATNO"].isin(slope_ids)]
    # Earliest event per PATNO per Target
    samples_sorted = samples.sort_values("CLINICAL_EVENT")
    earliest = samples_sorted.groupby(["PATNO", "Target"], as_index=False).first()
    # Log1p-transform NPQ (protein counts, always >= 0)
    earliest["NPQ"] = np.log1p(earliest["NPQ"].clip(lower=0.0))
    wide = earliest.pivot(index="PATNO", columns="Target", values="NPQ")
    wide.columns.name = None
    return wide


def _load_csf_aug_df(curated_path, nulisa_path, slope_ids):
    """
    Build the CSF augmentation DataFrame for RQ1.

    Columns:
      NEFL        : log1p(NULISA NPQ) — primary NfL source
      pSNCA_129   : log1p(NULISA NPQ for pSNCA-129) — primary alpha-syn source
      pTau181     : log1p(IU_pTau181_CSF from curated BL)
      ab_ratio    : IU_ABeta42_CSF / IU_ABeta40_CSF (Abeta42/40 ratio, raw)
      ABeta42     : log1p(IU_ABeta42_CSF)

    Returns pd.DataFrame indexed by PATNO (CSF-complete subjects only).
    """
    # NULISA targets for CSF
    nulisa_targets = ["NEFL", "pSNCA-129"]
    nulisa_wide = _load_nulisa_wide(nulisa_path, slope_ids, nulisa_targets)
    nulisa_wide = nulisa_wide.rename(columns={"pSNCA-129": "pSNCA_129"})

    # Curated BL CSF markers
    curated = pd.read_parquet(curated_path)
    bl = curated[
        (curated["COHORT"] == 1) & (curated["EVENT_ID"] == "BL")
    ].set_index("PATNO")
    bl = bl[bl.index.isin(slope_ids)]

    csf_cols = ["IU_pTau181_CSF", "IU_ABeta42_CSF", "IU_ABeta40_CSF"]
    bl_csf = bl[csf_cols].copy()

    # Compute Abeta42/40 ratio (only where both are positive)
    mask = (bl_csf["IU_ABeta42_CSF"] > 0) & (bl_csf["IU_ABeta40_CSF"] > 0)
    bl_csf["ab_ratio"] = np.where(
        mask,
        bl_csf["IU_ABeta42_CSF"] / bl_csf["IU_ABeta40_CSF"],
        np.nan,
    )
    bl_csf["pTau181"] = np.log1p(bl_csf["IU_pTau181_CSF"].clip(lower=0.0))
    bl_csf["ABeta42"] = np.log1p(bl_csf["IU_ABeta42_CSF"].clip(lower=0.0))
    bl_csf = bl_csf.drop(columns=csf_cols)

    # Join NULISA + curated CSF (inner: require both)
    csf_aug = nulisa_wide.join(bl_csf, how="inner")

    # CSF-complete: all columns non-missing
    csf_core = ["NEFL", "pSNCA_129", "pTau181", "ab_ratio", "ABeta42"]
    csf_aug = csf_aug[csf_core].dropna()

    return csf_aug


def _load_bbb_nulisa_aug_df(curated_path, nulisa_path, slope_ids):
    """
    Build base (clinical + NfL) and BBB augmentation DataFrames for RQ2a.

    Returns (nfl_aug_df, bbb_aug_df) both indexed by PATNO.
      nfl_aug_df : NEFL only (to augment clinical for the RQ2 base model)
      bbb_aug_df : PDGFRB, ICAM1, VCAM1, VEGFA (the BBB augmentation)
    """
    bbb_targets = ["NEFL", "PDGFRB", "ICAM1", "VCAM1", "VEGFA"]
    wide = _load_nulisa_wide(nulisa_path, slope_ids, bbb_targets)

    # Subjects with all targets non-missing
    wide = wide.dropna()

    nfl_df = wide[["NEFL"]].copy()
    bbb_df = wide[["PDGFRB", "ICAM1", "VCAM1", "VEGFA"]].copy()

    return nfl_df, bbb_df


def _load_qalb_aug_df(qalb_path, nulisa_path, slope_ids):
    """
    Build Q-albumin augmentation DataFrame for RQ2b.

    Returns (nfl_qalb_subgroup_ids, qalb_aug_df, nfl_df) for the Q-albumin
    subgroup (subjects with BOTH NULISA and Q-albumin).
      nfl_df         : NEFL indexed by PATNO (Q-albumin subgroup)
      qalb_aug_df    : qalb column indexed by PATNO (Q-albumin subgroup)
    """
    # Q-albumin
    qalb = pd.read_csv(qalb_path)
    qalb["PATNO"] = qalb["PATNO"].astype(int)
    qalb = qalb[qalb["PATNO"].isin(slope_ids)].copy()
    qalb = qalb.dropna(subset=["qalb"]).set_index("PATNO")
    # Log1p-transform Q-albumin (right-skewed ratio)
    qalb["qalb"] = np.log1p(qalb["qalb"].clip(lower=0.0))
    qalb_aug = qalb[["qalb"]].copy()

    # NULISA NEFL for Q-albumin subgroup
    nulisa_wide = _load_nulisa_wide(nulisa_path, slope_ids, ["NEFL"])
    nulisa_wide = nulisa_wide.dropna()

    # Intersection: subjects with BOTH Q-albumin and NULISA NEFL
    common_ids = qalb_aug.index.intersection(nulisa_wide.index)
    qalb_aug = qalb_aug.loc[common_ids]
    nfl_df = nulisa_wide.loc[common_ids]

    return nfl_df, qalb_aug


# ---------------------------------------------------------------------------
# Feature builder for clinical block (replicates B2 logic)
# ---------------------------------------------------------------------------

def _make_clinical_builder(curated_path, slope_ids):
    """
    Returns a feature builder for the clinical block (Block 1 from B0/B2).
    """
    curated = pd.read_parquet(curated_path)
    bl = curated[
        (curated["COHORT"] == 1) & (curated["EVENT_ID"] == "BL")
    ].copy()
    bl["gba_status"] = bl["subgroup"].str.contains("GBA", na=False).astype(float)
    bl = bl.set_index("PATNO")
    bl = bl[bl.index.isin(slope_ids)]

    def builder(subject_ids):
        ids = set(int(i) for i in np.asarray(list(subject_ids)))
        valid = ids & set(bl.index)
        return bl.loc[sorted(valid), CLINICAL_COLS].copy()

    return builder


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_increment(cv_config=None, n_perm=200, curated_path=None,
                  nulisa_path=None, qalb_path=None, outcome_path=None,
                  moca_long=None):
    """
    Run the full B3 analysis: RQ1 (CSF increment) + RQ2 (BBB null).

    Parameters
    ----------
    cv_config : dict or None
        CV configuration. Default: n_outer=5, n_inner=5, seeds=0..9.
    n_perm : int
        Number of permutations for the null control. Default 200.
    curated_path, nulisa_path, qalb_path, outcome_path : str or None
        Paths to data files. Defaults to project standard paths.
    moca_long : pd.DataFrame or None
        Pre-loaded longitudinal MoCA table. If None, loaded via load_moca().

    Returns
    -------
    dict : full results for JSON output.
    """
    curated_path = curated_path or _CURATED
    nulisa_path = nulisa_path or _NULISA
    qalb_path = qalb_path or _QALB
    outcome_path = outcome_path or _OUTCOME

    if cv_config is None:
        cv_config = {
            "n_outer": 5,
            "n_inner": 5,
            "seeds": list(range(10)),
        }

    # Load longitudinal MoCA
    if moca_long is None:
        moca_long = load_moca(curated_path)

    # Slope-eligible subject IDs
    outcome_df = pd.read_parquet(outcome_path)
    slope_ids = set(outcome_df["subject_id"].astype(int))
    print(f"Slope-eligible subjects: {len(slope_ids)}")

    # Clinical feature builder
    clin_builder = _make_clinical_builder(curated_path, slope_ids)

    # ========================================================================
    # RQ1: CSF increment
    # ========================================================================
    print("\n=== RQ1: CSF INCREMENT ===")

    csf_aug = _load_csf_aug_df(curated_path, nulisa_path, slope_ids)
    # Analytic IDs for RQ1: subjects with clinical + CSF features
    clin_probe = clin_builder(list(csf_aug.index))
    csf_analytic_ids = clin_probe.index.intersection(csf_aug.index).tolist()
    print(f"CSF-complete analytic N: {len(csf_analytic_ids)}")

    # Primary outcome: EB slope
    print("Running RQ1 (EB slope, paired increment)...")
    rq1_eb = paired_increment_cv(
        base_builder=clin_builder,
        aug_df=csf_aug,
        moca_long=moca_long,
        model_factory=_default_model,
        param_grid={"alpha": RIDGE_ALPHA_GRID},
        cv_config=cv_config,
        analytic_ids=csf_analytic_ids,
        outcome_col="eb_slope",
    )
    print(
        f"  R2_base={rq1_eb['r2_base']:.4f}, R2_aug={rq1_eb['r2_aug']:.4f}, "
        f"dR2={rq1_eb['dr2']:.4f} "
        f"range [{rq1_eb['dr2_repeat_lo']:.4f}, {rq1_eb['dr2_repeat_hi']:.4f}]"
    )

    # Permutation null
    print(f"Running RQ1 permutation null ({n_perm} shuffles)...")
    perm_cv_config = {
        "n_outer": cv_config.get("n_outer", 5),
        "n_inner": cv_config.get("n_inner", 5),
        "seeds": [int(cv_config.get("seeds", [0])[0])],
    }
    rq1_perm = compute_permutation_null(
        base_builder=clin_builder,
        aug_df=csf_aug,
        moca_long=moca_long,
        observed_dr2=rq1_eb["dr2"],
        base_r2=rq1_eb["r2_base"],
        cv_config=perm_cv_config,
        analytic_ids=csf_analytic_ids,
        outcome_col="eb_slope",
        n_perm=n_perm,
        rng_seed=42,
    )
    print(
        f"  Perm null mean={rq1_perm['perm_null_mean']:.4f}, "
        f"perm_p={rq1_perm['perm_p']:.3f}"
    )

    # Sensitivity: ols_slope
    print("Running RQ1 sensitivity (ols_slope)...")
    rq1_ols = paired_increment_cv(
        base_builder=clin_builder,
        aug_df=csf_aug,
        moca_long=moca_long,
        model_factory=_default_model,
        param_grid={"alpha": RIDGE_ALPHA_GRID},
        cv_config=cv_config,
        analytic_ids=csf_analytic_ids,
        outcome_col="ols_slope",
    )
    print(f"  dR2 (ols_slope)={rq1_ols['dr2']:.4f} "
          f"range [{rq1_ols['dr2_repeat_lo']:.4f}, {rq1_ols['dr2_repeat_hi']:.4f}]")

    # Sensitivity: slope_excl_baseline
    print("Running RQ1 sensitivity (slope_excl_baseline)...")
    rq1_excl = paired_increment_cv(
        base_builder=clin_builder,
        aug_df=csf_aug,
        moca_long=moca_long,
        model_factory=_default_model,
        param_grid={"alpha": RIDGE_ALPHA_GRID},
        cv_config=cv_config,
        analytic_ids=csf_analytic_ids,
        outcome_col="slope_excl_baseline",
    )
    print(f"  dR2 (slope_excl_baseline)={rq1_excl['dr2']:.4f} "
          f"range [{rq1_excl['dr2_repeat_lo']:.4f}, {rq1_excl['dr2_repeat_hi']:.4f}]")

    rq1_result = {
        "n": len(csf_analytic_ids),
        "r2_base": rq1_eb["r2_base"],
        "r2_aug": rq1_eb["r2_aug"],
        "dr2": rq1_eb["dr2"],
        "repeat_lo": rq1_eb["dr2_repeat_lo"],
        "repeat_hi": rq1_eb["dr2_repeat_hi"],
        "per_seed_dr2": rq1_eb["per_seed_dr2"],
        "perm_p": rq1_perm["perm_p"],
        "perm_null_mean": rq1_perm["perm_null_mean"],
        "perm_observed_dr2_fixed": rq1_perm["observed_dr2_fixed"],
        "dr2_ols_slope": rq1_ols["dr2"],
        "dr2_slope_excl_baseline": rq1_excl["dr2"],
        "dr2_ols_slope_repeat_lo": rq1_ols["dr2_repeat_lo"],
        "dr2_ols_slope_repeat_hi": rq1_ols["dr2_repeat_hi"],
        "dr2_excl_repeat_lo": rq1_excl["dr2_repeat_lo"],
        "dr2_excl_repeat_hi": rq1_excl["dr2_repeat_hi"],
        # Base/augmented R2 for the sensitivity outcomes (added 2026-08-04).
        # Previously only dr2 + its repeat range were stored, so a reader could
        # not tell WHY the baseline-excluded arm attenuates: whether both models
        # degrade together (a signal-to-noise effect, the outcome gets noisier
        # once the baseline visit is dropped) or the augmented model degrades
        # selectively (which would mean the CSF increment is partly carried by
        # coupling to baseline MoCA). Storing both makes that answerable.
        "r2_base_ols_slope": rq1_ols["r2_base"],
        "r2_aug_ols_slope": rq1_ols["r2_aug"],
        "r2_base_slope_excl_baseline": rq1_excl["r2_base"],
        "r2_aug_slope_excl_baseline": rq1_excl["r2_aug"],
    }

    # ========================================================================
    # RQ2: BBB predictive null
    # ========================================================================
    print("\n=== RQ2: BBB PREDICTIVE NULL ===")

    # RQ2a: NULISA BBB block
    print("Loading BBB NULISA block...")
    nfl_df, bbb_nulisa_aug = _load_bbb_nulisa_aug_df(curated_path, nulisa_path, slope_ids)
    # Base for RQ2: clinical + NfL
    # Analytic IDs: subjects with clinical + NULISA (all 5 targets)
    nulisa_analytic_ids_probe = clin_builder(list(nfl_df.index))
    nulisa_analytic_ids = (
        nulisa_analytic_ids_probe.index
        .intersection(nfl_df.index)
        .intersection(bbb_nulisa_aug.index)
        .tolist()
    )
    print(f"BBB NULISA analytic N: {len(nulisa_analytic_ids)}")

    # Define a builder for (clinical + NfL) combined, to serve as the
    # base for the BBB increment test. The r2_base of rq2a (below) gives
    # the clinical+NfL R2 directly — no separate rq2_base run needed.
    def clin_nfl_builder(subject_ids):
        base = clin_builder(subject_ids)
        nfl_sub = nfl_df.loc[nfl_df.index.intersection(base.index)]
        return base.join(nfl_sub, how="inner")

    print("Running RQ2a BBB increment (clinical+NfL vs clinical+NfL+BBB)...")
    rq2a = paired_increment_cv(
        base_builder=clin_nfl_builder,
        aug_df=bbb_nulisa_aug,
        moca_long=moca_long,
        model_factory=_default_model,
        param_grid={"alpha": RIDGE_ALPHA_GRID},
        cv_config=cv_config,
        analytic_ids=nulisa_analytic_ids,
        outcome_col="eb_slope",
    )
    print(
        f"  dR2={rq2a['dr2']:.4f} "
        f"range [{rq2a['dr2_repeat_lo']:.4f}, {rq2a['dr2_repeat_hi']:.4f}]"
    )

    # Permutation null for RQ2a — 500 shuffles to tighten the p-value bound
    # for this contested finding (pre-registered null, unexpectedly non-null).
    # RQ2b stays at n_perm (default 200) since it is cleanly null.
    rq2a_n_perm = 500
    print(f"Running RQ2a permutation null ({rq2a_n_perm} shuffles)...")
    rq2a_perm = compute_permutation_null(
        base_builder=clin_nfl_builder,
        aug_df=bbb_nulisa_aug,
        moca_long=moca_long,
        observed_dr2=rq2a["dr2"],
        base_r2=rq2a["r2_base"],
        cv_config=perm_cv_config,
        analytic_ids=nulisa_analytic_ids,
        outcome_col="eb_slope",
        n_perm=rq2a_n_perm,
        rng_seed=43,
    )
    print(
        f"  Perm null mean={rq2a_perm['perm_null_mean']:.4f}, "
        f"perm_p={rq2a_perm['perm_p']:.3f}"
    )

    # RQ2b: Q-albumin subgroup
    print("\nLoading Q-albumin subgroup...")
    nfl_qalb_df, qalb_aug = _load_qalb_aug_df(qalb_path, nulisa_path, slope_ids)
    qalb_analytic_ids_probe = clin_builder(list(nfl_qalb_df.index))
    qalb_analytic_ids = (
        qalb_analytic_ids_probe.index
        .intersection(nfl_qalb_df.index)
        .intersection(qalb_aug.index)
        .tolist()
    )
    print(f"Q-albumin subgroup analytic N: {len(qalb_analytic_ids)}")

    def clin_nfl_qalb_builder(subject_ids):
        base = clin_builder(subject_ids)
        nfl_sub = nfl_qalb_df.loc[nfl_qalb_df.index.intersection(base.index)]
        return base.join(nfl_sub, how="inner")

    print("Running RQ2b Q-albumin increment (clinical+NfL vs clinical+NfL+Qalb)...")
    rq2b = paired_increment_cv(
        base_builder=clin_nfl_qalb_builder,
        aug_df=qalb_aug,
        moca_long=moca_long,
        model_factory=_default_model,
        param_grid={"alpha": RIDGE_ALPHA_GRID},
        cv_config=cv_config,
        analytic_ids=qalb_analytic_ids,
        outcome_col="eb_slope",
    )
    print(
        f"  dR2={rq2b['dr2']:.4f} "
        f"range [{rq2b['dr2_repeat_lo']:.4f}, {rq2b['dr2_repeat_hi']:.4f}]"
    )

    # Permutation null for RQ2b
    print(f"Running RQ2b permutation null ({n_perm} shuffles)...")
    rq2b_perm = compute_permutation_null(
        base_builder=clin_nfl_qalb_builder,
        aug_df=qalb_aug,
        moca_long=moca_long,
        observed_dr2=rq2b["dr2"],
        base_r2=rq2b["r2_base"],
        cv_config=perm_cv_config,
        analytic_ids=qalb_analytic_ids,
        outcome_col="eb_slope",
        n_perm=n_perm,
        rng_seed=44,
    )
    print(
        f"  Perm null mean={rq2b_perm['perm_null_mean']:.4f}, "
        f"perm_p={rq2b_perm['perm_p']:.3f}"
    )

    rq2a_result = {
        "n": len(nulisa_analytic_ids),
        "r2_base_clin_nfl": rq2a["r2_base"],   # R2 of clinical+NfL model (base of rq2a)
        "r2_aug": rq2a["r2_aug"],
        "dr2": rq2a["dr2"],
        "repeat_lo": rq2a["dr2_repeat_lo"],
        "repeat_hi": rq2a["dr2_repeat_hi"],
        "per_seed_dr2": rq2a["per_seed_dr2"],
        "perm_p": rq2a_perm["perm_p"],
        "perm_null_mean": rq2a_perm["perm_null_mean"],
        "perm_observed_dr2_fixed": rq2a_perm["observed_dr2_fixed"],
        # Pre-registered as null; data unexpectedly shows non-null signal
        # for the NULISA vascular panel (PDGFRB/ICAM1/VCAM1/VEGFA).
        # Report honestly — do not spin as a BBB-permeability finding.
        "verdict": (
            "null" if rq2a_perm["perm_p"] > 0.05
            else "unexpected-non-null (pre-registered as null)"
        ),
    }

    rq2b_result = {
        "n": len(qalb_analytic_ids),
        "r2_base_clin_nfl": rq2b["r2_base"],
        "r2_aug": rq2b["r2_aug"],
        "dr2": rq2b["dr2"],
        "repeat_lo": rq2b["dr2_repeat_lo"],
        "repeat_hi": rq2b["dr2_repeat_hi"],
        "per_seed_dr2": rq2b["per_seed_dr2"],
        "perm_p": rq2b_perm["perm_p"],
        "perm_null_mean": rq2b_perm["perm_null_mean"],
        "perm_observed_dr2_fixed": rq2b_perm["observed_dr2_fixed"],
        "verdict": "null" if rq2b_perm["perm_p"] > 0.05 else "non-null",
    }

    result = {
        "baseline_clinical_r2_full_cohort_N816": BASELINE_CLINICAL_R2_FULL_COHORT_N816,
        "rq1_csf": rq1_result,
        "rq2_bbb": {
            "nulisa_block": rq2a_result,
            "qalb": rq2b_result,
        },
    }

    # Print verdicts
    print("\n=== VERDICTS ===")
    rq1_verdict = (
        "POSITIVE (dR2 > 0, perm_p < 0.05)"
        if rq1_result["dr2"] > 0 and rq1_result["perm_p"] < 0.05
        else "NULL (dR2 not significantly above permutation null)"
        if rq1_result["perm_p"] >= 0.05
        else f"MARGINAL (dR2={rq1_result['dr2']:.4f}, perm_p={rq1_result['perm_p']:.3f})"
    )
    print(f"RQ1 CSF increment: {rq1_verdict}")
    print(
        f"  dR2={rq1_result['dr2']:.4f} [{rq1_result['repeat_lo']:.4f}, "
        f"{rq1_result['repeat_hi']:.4f}] | perm_p={rq1_result['perm_p']:.3f}"
    )
    print(
        f"  Sensitivity: ols_slope dR2={rq1_result['dr2_ols_slope']:.4f}, "
        f"slope_excl_baseline dR2={rq1_result['dr2_slope_excl_baseline']:.4f}"
    )
    print(f"RQ2 NULISA BBB: {rq2a_result['verdict']} "
          f"(dR2={rq2a_result['dr2']:.4f}, perm_p={rq2a_result['perm_p']:.3f})")
    print(f"RQ2 Q-albumin:  {rq2b_result['verdict']} "
          f"(dR2={rq2b_result['dr2']:.4f}, perm_p={rq2b_result['perm_p']:.3f})")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    result = run_increment()

    os.makedirs(_DATA_OUT_DIR, exist_ok=True)
    with open(_JSON_OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {_JSON_OUT}")
    return result


if __name__ == "__main__":
    main()
