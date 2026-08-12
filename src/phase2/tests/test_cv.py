"""
Leak-safety tests for the nested-CV harness (cv.py), task B2.

Four mandatory leak-safety assertions:
  1. Preprocessing transformer is fit ONLY on train-fold subjects (never on the
     full dataset or on a set that includes test subjects).
  2. fit_eb_params() is called with train subject IDs only, never the full set.
  3. End-to-end OOF predictions have the expected shape and correct index.
  4. The canonical R2 (r2_mean) equals r2_score computed directly from the
     stored oof_predictions/oof_truth series (M4 consistency test).

Run:
    pixi run python -m pytest src/phase2/tests/test_cv.py -v
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CV_PATH = os.path.join(_THIS_DIR, "..", "cv.py")


def _load_cv():
    spec = importlib.util.spec_from_file_location("cv_b2", _CV_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cv = _load_cv()


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_moca(n_subjects=24, n_visits=5, seed=42):
    """
    Synthetic longitudinal MoCA table.
    All subjects have n_visits >= 4 with span = n_visits - 1 >= 3 yr.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for pat in range(1, n_subjects + 1):
        start_age = 60.0 + rng.randn() * 5
        baseline = float(np.clip(25.0 + rng.randn() * 2, 20, 30))
        true_slope = rng.normal(-0.3, 0.5)
        for v in range(n_visits):
            age = start_age + v * 1.0
            score = float(np.clip(
                baseline + true_slope * v + rng.randn() * 0.5, 0, 30
            ))
            rows.append({
                "PATNO": pat,
                "EVENT_ID": f"V{v:02d}",
                "age_at_visit": age,
                "moca": score,
            })
    return pd.DataFrame(rows)


def _make_feature_builder(n_subjects=24, n_features=5, nan_frac=0.05, seed=42):
    """Returns a feature builder closure for subjects 1..n_subjects."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_subjects, n_features),
        index=np.arange(1, n_subjects + 1),
        columns=[f"f{i}" for i in range(n_features)],
    )
    mask = rng.rand(*X.shape) < nan_frac
    for r, c in zip(*np.where(mask)):
        X.iloc[r, c] = np.nan

    def builder(subject_ids):
        ids = np.asarray(list(subject_ids), dtype=int)
        return X.loc[X.index.isin(ids)].copy()

    return builder


# ---------------------------------------------------------------------------
# Recording transformer (leak-safety probe)
# ---------------------------------------------------------------------------

class RecordingTransformer:
    """
    Wraps a sklearn-compatible transformer and records which subject IDs
    (DataFrame index values) were passed to fit / fit_transform.

    Usage: inject via preprocessor_factory=lambda: recorder.
    After nested_cv runs, recorder.fit_subject_ids is a list of frozensets,
    one entry per outer fold, each containing the train-fold subject IDs.
    """

    def __init__(self, inner=None):
        if inner is None:
            inner = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
        self.inner = inner
        self.fit_subject_ids: list = []  # list[frozenset]

    def _record(self, X):
        if hasattr(X, "index"):
            self.fit_subject_ids.append(frozenset(X.index))
        else:
            self.fit_subject_ids.append(frozenset(range(len(X))))

    def fit(self, X, y=None):
        self._record(X)
        self.inner.fit(X, y)
        return self

    def transform(self, X):
        return self.inner.transform(X)

    def fit_transform(self, X, y=None):
        self._record(X)
        return self.inner.fit_transform(X, y)


# ===========================================================================
# Test 1: Leak-safety — preprocessor fit on train subjects ONLY
# ===========================================================================

def test_preprocessor_fit_on_train_only():
    """
    The preprocessing transformer's fit (or fit_transform) must be called
    exactly n_outer times (one per fold) and each call must receive a STRICT
    SUBSET of the full analytic subject IDs.

    A fit call that receives the full set would indicate the preprocessor was
    fit on all subjects (including the test fold) — a data-leakage bug.
    """
    n_subjects = 24
    n_outer = 4
    moca = _make_moca(n_subjects=n_subjects, n_visits=5)
    all_ids = frozenset(range(1, n_subjects + 1))
    builder = _make_feature_builder(n_subjects=n_subjects)

    recorder = RecordingTransformer()

    cv_config = {"n_outer": n_outer, "n_inner": 3, "seeds": [0]}
    _ = _cv.nested_cv(
        feature_builder=builder,
        moca_long=moca,
        model_factory=lambda: Ridge(alpha=1.0),
        preprocessor_factory=lambda: recorder,
        param_grid=None,
        cv_config=cv_config,
    )

    # Exactly n_outer fit calls
    assert len(recorder.fit_subject_ids) == n_outer, (
        f"Expected {n_outer} fit calls (one per outer fold), "
        f"got {len(recorder.fit_subject_ids)}"
    )

    for i, fitted_ids in enumerate(recorder.fit_subject_ids):
        # Each fit call must be a STRICT subset (not equal to all_ids)
        assert fitted_ids < all_ids, (
            f"Outer fold {i}: preprocessor.fit received {len(fitted_ids)} subjects "
            f"but the full analytic set has {len(all_ids)} subjects. "
            f"The fit set must be a strict subset (train only). "
            f"Full-dataset fit = leakage."
        )
        # Sanity: train set should be (n_outer-1)/n_outer of total
        expected_approx = n_subjects * (n_outer - 1) / n_outer
        assert len(fitted_ids) >= expected_approx * 0.8, (
            f"Fold {i}: fitted only {len(fitted_ids)} subjects, expected ~{expected_approx:.0f}"
        )


# ===========================================================================
# Test 2: Leak-safety — fit_eb_params receives train IDs only
# ===========================================================================

def test_outcome_refit_uses_train_ids_only():
    """
    fit_eb_params() must be called exactly n_outer times (one per outer fold)
    and each call must receive only the training-fold subject IDs — never the
    full set, never the test subjects.

    This is verified by spying on cv.fit_eb_params at the module level.
    Test subjects must never appear in any fit_eb_params call.
    """
    n_subjects = 24
    n_outer = 4
    moca = _make_moca(n_subjects=n_subjects, n_visits=5)
    all_ids = frozenset(range(1, n_subjects + 1))
    builder = _make_feature_builder(n_subjects=n_subjects)

    calls = []
    original_fn = _cv.fit_eb_params

    def spy_fit_eb_params(moca_long, subject_ids):
        calls.append(frozenset(int(s) for s in np.asarray(list(subject_ids))))
        return original_fn(moca_long, subject_ids)

    # Monkeypatch at module level so the harness uses the spy
    _cv.fit_eb_params = spy_fit_eb_params
    try:
        cv_config = {"n_outer": n_outer, "n_inner": 3, "seeds": [0]}
        _cv.nested_cv(
            feature_builder=builder,
            moca_long=moca,
            model_factory=lambda: Ridge(alpha=1.0),
            cv_config=cv_config,
        )
    finally:
        _cv.fit_eb_params = original_fn

    # Called once per outer fold
    assert len(calls) == n_outer, (
        f"Expected {n_outer} fit_eb_params calls, got {len(calls)}"
    )

    for i, call_ids in enumerate(calls):
        # Must be a strict subset (test subjects never included)
        assert call_ids < all_ids, (
            f"Fold {i}: fit_eb_params received {len(call_ids)} subjects "
            f"but the analytic set has {len(all_ids)}. "
            f"Must be a strict subset (train fold only). "
            f"Test subjects must never inform mu/tau2."
        )

    # The union of all train sets should cover all subjects
    # (each subject is in the test set exactly once in K-fold)
    assert set.union(*[set(c) for c in calls]) == set(all_ids), (
        "The union of all fit_eb_params call sets should cover all subjects "
        "(each subject is in the test set exactly once)."
    )


# ===========================================================================
# Test 3: End-to-end OOF shape and index correctness
# ===========================================================================

def test_end_to_end_oof_shape():
    """
    On a tiny synthetic dataset, nested_cv must return:
      - oof_predictions: Series of length n_subjects, index = subject IDs
      - oof_truth: same length and index
      - per_fold_r2: list of length n_outer
      - per_seed_r2: list of length n_repeats
    """
    n_subjects = 20
    n_outer = 4
    seeds = [0, 1]
    moca = _make_moca(n_subjects=n_subjects, n_visits=5)
    builder = _make_feature_builder(n_subjects=n_subjects)

    cv_config = {"n_outer": n_outer, "n_inner": 3, "seeds": seeds}
    result = _cv.nested_cv(
        feature_builder=builder,
        moca_long=moca,
        model_factory=lambda: Ridge(alpha=1.0),
        cv_config=cv_config,
    )

    # Required keys
    for key in ("oof_predictions", "oof_truth", "per_fold_r2", "per_seed_r2",
                "r2_mean", "r2_fold_mean", "r2_repeat_lo", "r2_repeat_hi",
                "n_subjects", "n_folds", "n_repeats"):
        assert key in result, f"Missing key: {key}"

    oof_pred = result["oof_predictions"]
    oof_truth = result["oof_truth"]

    # Every subject predicted exactly once (from canonical seed)
    assert len(oof_pred) == n_subjects, (
        f"Expected {n_subjects} OOF predictions, got {len(oof_pred)}"
    )
    assert set(oof_pred.index) == set(range(1, n_subjects + 1)), (
        "OOF prediction index does not match expected subject IDs"
    )
    assert oof_pred.index.equals(oof_truth.index), (
        "oof_predictions and oof_truth indices are not aligned"
    )

    # per_fold_r2 length
    assert len(result["per_fold_r2"]) == n_outer

    # per_seed_r2 length
    assert len(result["per_seed_r2"]) == len(seeds)

    # n_* fields
    assert result["n_subjects"] == n_subjects
    assert result["n_folds"] == n_outer
    assert result["n_repeats"] == len(seeds)


# ===========================================================================
# Test 4: repeated-CV interval has correct length
# ===========================================================================

def test_per_seed_r2_length():
    """per_seed_r2 must have length == len(seeds) and n_repeats must match."""
    n_subjects = 20
    n_repeats = 3
    moca = _make_moca(n_subjects=n_subjects, n_visits=5)
    builder = _make_feature_builder(n_subjects=n_subjects)

    cv_config = {"n_outer": 4, "n_inner": 3, "seeds": list(range(n_repeats))}
    result = _cv.nested_cv(
        feature_builder=builder,
        moca_long=moca,
        model_factory=lambda: Ridge(alpha=1.0),
        cv_config=cv_config,
    )

    assert result["n_repeats"] == n_repeats
    assert len(result["per_seed_r2"]) == n_repeats


# ===========================================================================
# Test 5: OOF predictions vs truth — no identical values (sanity)
# ===========================================================================

def test_oof_predictions_not_trivially_constant():
    """
    With meaningful synthetic data (true slopes vary across subjects), the
    OOF predictions should not all be identical (no null-predictor collapse).
    """
    n_subjects = 24
    moca = _make_moca(n_subjects=n_subjects, n_visits=5, seed=7)
    builder = _make_feature_builder(n_subjects=n_subjects, seed=7)

    cv_config = {"n_outer": 4, "n_inner": 3, "seeds": [0]}
    result = _cv.nested_cv(
        feature_builder=builder,
        moca_long=moca,
        model_factory=lambda: Ridge(alpha=1.0),
        cv_config=cv_config,
    )

    pred_std = result["oof_predictions"].std()
    assert pred_std > 1e-6, (
        f"OOF predictions are nearly constant (std={pred_std:.2e}); "
        "the model may not be running correctly."
    )


# ===========================================================================
# Test 6: param_grid tuning does not bypass fold structure
# ===========================================================================

def test_param_grid_inner_cv():
    """
    When param_grid is provided, the harness tunes the model via inner CV
    on the train fold only. The result should still be leak-safe and the
    OOF predictions should have the correct shape.
    """
    n_subjects = 24
    moca = _make_moca(n_subjects=n_subjects, n_visits=5)
    builder = _make_feature_builder(n_subjects=n_subjects)

    cv_config = {"n_outer": 4, "n_inner": 3, "seeds": [0]}
    result = _cv.nested_cv(
        feature_builder=builder,
        moca_long=moca,
        model_factory=lambda: Ridge(),
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        cv_config=cv_config,
    )

    assert len(result["oof_predictions"]) == n_subjects
    # With inner tuning the model should still converge
    assert not any(np.isnan(result["oof_predictions"]))


# ===========================================================================
# Test 7 (M4): Canonical R2 equals r2_score computed from stored OOF series
# ===========================================================================

def test_canonical_r2_equals_pooled_oof_r2():
    """
    The reported r2_mean must equal r2_score(oof_truth, oof_pred) computed
    directly from the stored oof_truth/oof_predictions Series (canonical seed).

    This closes the gap that existed when r2_mean was the arithmetic mean of
    per-fold R2 values: that number was inconsistent with the stored OOF series
    and could not be reproduced from it when fold sizes differed. After the B2
    fix, r2_mean IS the pooled OOF R2, so this equality must hold exactly.
    """
    n_subjects = 24
    moca = _make_moca(n_subjects=n_subjects, n_visits=5, seed=13)
    builder = _make_feature_builder(n_subjects=n_subjects, seed=13)

    cv_config = {"n_outer": 4, "n_inner": 3, "seeds": [0]}
    result = _cv.nested_cv(
        feature_builder=builder,
        moca_long=moca,
        model_factory=lambda: Ridge(alpha=1.0),
        cv_config=cv_config,
    )

    oof_pred = result["oof_predictions"]
    oof_truth = result["oof_truth"]

    # Recompute pooled OOF R2 from the stored series
    # (pandas Series.values is a property returning ndarray, not a method)
    recomputed_r2 = r2_score(oof_truth.values, oof_pred.values)

    assert abs(result["r2_mean"] - recomputed_r2) < 1e-12, (
        f"r2_mean={result['r2_mean']:.10f} does not match pooled OOF R2="
        f"{recomputed_r2:.10f} computed from oof_truth/oof_predictions. "
        "The canonical metric must be reproducible from the stored OOF series."
    )
