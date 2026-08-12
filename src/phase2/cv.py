"""
Phase 2 nested-CV harness (task B2) — leak-safe by design.

Public entry point: `nested_cv`.

Callers (B2, B3, B4) pass:
  feature_builder      : (subject_ids) -> pd.DataFrame indexed by subject_id.
  moca_long            : longitudinal MoCA table (PATNO, EVENT_ID, age_at_visit, moca).
  model_factory        : () -> sklearn estimator (model only, no preprocessing).
  preprocessor_factory : () -> sklearn-compatible transformer (default: imputer+scaler).
  param_grid           : dict for inner GridSearchCV (e.g., {'alpha': [0.01, 0.1, 1, 10]}).
  cv_config            : dict with n_outer, n_inner, seeds, n_repeats.

Leak-safety contract (verified by test_cv.py):
  1. preprocessor.fit (or fit_transform) is called exactly once per outer fold,
     on train subjects only — never on the test fold or the full dataset.
  2. fit_eb_params() is called once per outer fold with train subject IDs only.
     Test subjects never inform mu or tau2.
  3. eb_slopes() uses the train-fold EB params to compute test-fold outcomes.
     Test subjects' own visit data drives their OLS slope; the shrinkage target
     (mu, tau2) comes from the training fold.
  4. The outer split is on SUBJECTS (not rows). Each subject appears in the
     test fold exactly once across the n_outer folds (standard K-fold).

Repeated nested CV:
  The whole outer loop is repeated across `len(seeds)` different random seeds.

  Canonical R2 (reported as `r2_mean`): the POOLED out-of-fold R2 for the
  canonical seed, computed as r2_score over the concatenated OOF predictions
  and truths from that seed's full K-fold run. This is consistent with and
  reproducible from the stored `oof_predictions`/`oof_truth` series.
  The arithmetic mean of per-fold R2 values is also kept as `r2_fold_mean`
  for reference, but is NOT the canonical metric.

  Repeat range (`r2_repeat_lo`/`r2_repeat_hi`): the min and max of the
  per-seed pooled-OOF R2 across all N CV repeats (seeds). This is the
  OBSERVED RANGE across repeated runs, NOT a 95% confidence interval — with
  N=10 seeds, a percentile-based CI would be unreliable.

  OOF predictions returned are from the canonical seed (seeds[0]).

Clean import for downstream tasks (B3, B4):
  `fit_eb_params`, `eb_slopes`, and `load_moca` are re-exported at module
  level so downstream modules can do:
      import sys, os
      sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
      from cv import nested_cv, fit_eb_params, eb_slopes, load_moca
  without their own importlib dance for the digit-prefixed 01_outcome.py.
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Load B1 outcome module (filename starts with digit, can't use regular import)
# Module-level names so tests can monkeypatch them.
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_outcome_cv", os.path.join(_here, "01_outcome.py")
)
_outcome_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_outcome_mod)

fit_eb_params = _outcome_mod.fit_eb_params   # patchable in tests
eb_slopes = _outcome_mod.eb_slopes           # patchable in tests
load_moca = _outcome_mod.load_moca           # re-exported for B3/B4


# ---------------------------------------------------------------------------
# Default preprocessor factory
# ---------------------------------------------------------------------------

def _default_preprocessor_factory():
    """Median imputation followed by z-score standardisation."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def nested_cv(
    feature_builder,
    moca_long,
    model_factory,
    preprocessor_factory=None,
    param_grid=None,
    scoring="r2",
    cv_config=None,
    analytic_ids=None,
):
    """
    Leak-safe nested cross-validation harness.

    Parameters
    ----------
    feature_builder : callable
        (subject_ids) -> pd.DataFrame indexed by subject_id.
        May contain NaN; imputation happens inside each fold on train only.
        The returned index determines the analytic subject set if analytic_ids
        is not provided.
    moca_long : pd.DataFrame
        Longitudinal MoCA table with columns:
        PATNO (int), EVENT_ID (str), age_at_visit (float), moca (float).
    model_factory : callable
        () -> sklearn estimator (model only, without preprocessing).
        A fresh estimator is created per outer fold.
    preprocessor_factory : callable or None
        () -> sklearn-compatible transformer with fit, transform, fit_transform.
        Default: Pipeline([SimpleImputer(median), StandardScaler()]).
        Inject a RecordingTransformer here for leak-safety tests.
    param_grid : dict or None
        Hyperparameter grid for GridSearchCV (inner tuning).
        Keys match the estimator's parameter names.
        Example: {'alpha': [0.01, 0.1, 1, 10, 100]}.
        None = no inner tuning; model used with its default parameters.
    scoring : str
        Sklearn scoring string for inner CV. Default 'r2'.
    cv_config : dict or None
        n_outer  : int       (default 5)
        n_inner  : int       (default 5)
        seeds    : list[int] (default [0, 1, 2, 3, 4])
    analytic_ids : array-like or None
        Explicit list of subject IDs forming the analytic set. If None, the
        set is inferred by calling feature_builder(moca_long['PATNO'].unique())
        and taking the returned index.

    Returns
    -------
    dict with keys:
        oof_predictions : pd.Series  (subject_id -> predicted eb_slope, canonical seed)
        oof_truth       : pd.Series  (subject_id -> true eb_slope, canonical seed)
        per_fold_r2     : list[float] (per-fold R2, canonical seed)
        r2_fold_mean    : float (arithmetic mean of per_fold_r2, canonical seed; kept
                          for reference — NOT the canonical reported metric)
        r2_mean         : float POOLED OOF R2 for the canonical seed:
                          r2_score(all_oof_truth, all_oof_pred). This is the canonical
                          metric, consistent with and reproducible from oof_predictions/
                          oof_truth. Differs from r2_fold_mean when fold sizes are unequal.
        per_seed_r2     : list[float] (pooled OOF R2 per seed, length = n_repeats)
        r2_repeat_lo    : float (min of per_seed_r2 — observed range across N repeats,
                          NOT a 95% CI)
        r2_repeat_hi    : float (max of per_seed_r2)
        n_subjects      : int
        n_folds         : int
        n_repeats       : int
    """
    if preprocessor_factory is None:
        preprocessor_factory = _default_preprocessor_factory

    # Parse config
    cfg = cv_config or {}
    n_outer = int(cfg.get("n_outer", 5))
    n_inner = int(cfg.get("n_inner", 5))
    seeds = [int(s) for s in cfg.get("seeds", list(range(5)))]
    canonical_seed = seeds[0]

    # Determine analytic subject set
    if analytic_ids is not None:
        valid_ids = np.array(sorted(set(int(i) for i in analytic_ids)))
    else:
        # Infer from feature_builder: call with all moca PATNOs and take the index
        all_moca_ids = moca_long["PATNO"].unique()
        X_probe = feature_builder(all_moca_ids)
        valid_ids = np.array(sorted(int(i) for i in X_probe.index.unique()))

    n_subjects = len(valid_ids)
    if n_subjects == 0:
        raise ValueError("Analytic subject set is empty.")

    canonical_result = None
    per_seed_r2 = []

    for seed in seeds:
        kf = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
        oof_pred = {}
        oof_truth = {}
        fold_r2s = []

        for tr_idx, te_idx in kf.split(valid_ids):
            train_ids = valid_ids[tr_idx]
            test_ids = valid_ids[te_idx]

            # --- OUTCOME: refit EB params on train only (leak-safety point 2) ---
            p = fit_eb_params(moca_long, train_ids)
            df_train_y = eb_slopes(moca_long, train_ids, p)
            # Test outcomes use TRAIN shrinkage params (leak-safety point 3)
            df_test_y = eb_slopes(moca_long, test_ids, p)

            if df_train_y.empty or df_test_y.empty:
                continue

            y_train = df_train_y["eb_slope"]
            y_test = df_test_y["eb_slope"]

            # --- FEATURES: build inside fold, align with valid outcomes ---
            X_train_raw = feature_builder(y_train.index.to_numpy())
            X_test_raw = feature_builder(y_test.index.to_numpy())

            # Align y to X (both may return fewer subjects due to internal filters)
            train_common = X_train_raw.index.intersection(y_train.index)
            test_common = X_test_raw.index.intersection(y_test.index)

            if len(train_common) < n_inner or len(test_common) < 1:
                continue

            X_train = X_train_raw.loc[train_common]
            y_train_aligned = y_train.loc[train_common]
            X_test = X_test_raw.loc[test_common]
            y_test_aligned = y_test.loc[test_common]

            # --- PREPROCESSING: fit on TRAIN only (leak-safety point 1) ---
            preprocessor = preprocessor_factory()
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # --- MODEL: tune by inner CV on train only ---
            model = model_factory()
            if param_grid:
                inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=seed)
                model = GridSearchCV(
                    model,
                    param_grid,
                    cv=inner_cv,
                    scoring=scoring,
                    refit=True,
                    # serial inner grid: n_jobs=-1's loky process pool nests with
                    # BLAS threads and pandas state in heavy outer CV / permutation
                    # loops, causing intermittent SIGSEGV. The inner grid is tiny,
                    # so serial costs nothing and gives identical estimates.
                    n_jobs=1,
                )
            model.fit(X_train_arr, y_train_aligned.to_numpy())

            # --- PREDICT TEST fold ---
            y_pred = model.predict(X_test_arr)
            fold_r2 = float(r2_score(y_test_aligned.to_numpy(), y_pred))
            fold_r2s.append(fold_r2)

            for sid, pred_val in zip(test_common, y_pred):
                oof_pred[int(sid)] = float(pred_val)
                oof_truth[int(sid)] = float(y_test_aligned.loc[sid])

        if not fold_r2s:
            per_seed_r2.append(float("nan"))
            continue

        # Pooled OOF R2: r2_score over the concatenated OOF series (not fold-mean).
        # This is the canonical metric — consistent with the stored oof series.
        seed_pooled_r2 = float(
            r2_score(list(oof_truth.values()), list(oof_pred.values()))
        )
        per_seed_r2.append(seed_pooled_r2)

        if seed == canonical_seed:
            canonical_result = {
                "oof_predictions": pd.Series(oof_pred, name="predicted"),
                "oof_truth": pd.Series(oof_truth, name="eb_slope"),
                "per_fold_r2": fold_r2s,
                "r2_fold_mean": float(np.mean(fold_r2s)),  # kept for reference
                "r2_mean": seed_pooled_r2,  # canonical: pooled OOF R2
            }

    # Repeat range: min/max of per-seed pooled-OOF R2 across all N repeats.
    # With N=10 seeds, percentile-based CIs are unreliable; min/max is honest.
    # This is the observed range of R2 across repeated CV runs, NOT a 95% CI.
    valid_r2s = [r for r in per_seed_r2 if not np.isnan(r)]
    if len(valid_r2s) >= 2:
        r2_repeat_lo = float(np.min(valid_r2s))
        r2_repeat_hi = float(np.max(valid_r2s))
    elif len(valid_r2s) == 1:
        r2_repeat_lo = r2_repeat_hi = valid_r2s[0]
    else:
        r2_repeat_lo = r2_repeat_hi = float("nan")

    if canonical_result is None:
        canonical_result = {
            "oof_predictions": pd.Series(dtype=float),
            "oof_truth": pd.Series(dtype=float),
            "per_fold_r2": [],
            "r2_fold_mean": float("nan"),
            "r2_mean": float("nan"),
        }

    return {
        **canonical_result,
        "per_seed_r2": per_seed_r2,
        "r2_repeat_lo": r2_repeat_lo,
        "r2_repeat_hi": r2_repeat_hi,
        "n_subjects": n_subjects,
        "n_folds": n_outer,
        "n_repeats": len(seeds),
    }
