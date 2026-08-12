"""
Tests for B4: calibration + conformal prediction intervals.

Three required assertions (task-B4 spec):
  1. Conformal empirical coverage is within tolerance of the nominal level on
     a synthetic dataset (nominal 0.90 -> empirical in [0.85, 0.95]).
     This proves the conformal machinery is correct, not just that it runs.
  2. Recalibration is fit in-fold only -- the inner calibration set never
     overlaps with the test fold subjects (no test/full-data leak).
  3. A perfectly-calibrated synthetic predictor yields calibration slope ~1
     and intercept ~0 (sanity check for the calibration regression).

All tests use synthetic data; they do not touch real PPMI data.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CALIB_PATH = os.path.join(_HERE, "..", "04_calibration.py")

_spec = importlib.util.spec_from_file_location("_calib_mod", _CALIB_PATH)
_calib_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_calib_mod)

compute_calibration_slope_intercept = _calib_mod.compute_calibration_slope_intercept
conformal_intervals_loo = _calib_mod.conformal_intervals_loo
_binary_oof_cv = _calib_mod._binary_oof_cv


# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------
FAST_CV = {"n_outer": 4, "seeds": [0]}
N_CONFORMAL = 300
N_BINARY = 120


# ---------------------------------------------------------------------------
# Test 1: conformal coverage near nominal
# ---------------------------------------------------------------------------

def test_conformal_coverage_near_nominal():
    """
    LOO split conformal intervals at nominal 0.90 must yield empirical
    coverage in [0.85, 0.95] on a synthetic regression dataset.

    This proves the conformal machinery is correct (not just that it runs):
    an incorrect quantile formula or residual computation would push coverage
    well outside the tolerance band.
    """
    rng = np.random.default_rng(0)
    n = N_CONFORMAL
    X = rng.normal(0, 1, (n, 3))
    coef = np.array([0.5, -0.3, 0.2])
    y = X @ coef + rng.normal(0, 0.3, n)

    # 5-fold OOF predictions with Ridge
    oof_pred = np.zeros(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in kf.split(X):
        m = Ridge(alpha=0.01)
        m.fit(X[tr], y[tr])
        oof_pred[te] = m.predict(X[te])

    nominal = 0.90
    coverage, mean_width = conformal_intervals_loo(y, oof_pred, alpha=1 - nominal)

    assert 0.85 <= coverage <= 0.95, (
        f"Conformal empirical coverage {coverage:.4f} is outside [0.85, 0.95] "
        f"for nominal level {nominal}. "
        "This indicates the conformal machinery is incorrect."
    )
    assert mean_width > 0.0, "Mean interval width must be positive."


# ---------------------------------------------------------------------------
# Test 2: recalibration fit in-fold only (no test/full-data leak)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_binary_data():
    """Synthetic binary dataset for recalibration leak-safety tests."""
    rng = np.random.default_rng(123)
    n = N_BINARY
    subject_ids = np.arange(1, n + 1)

    y = pd.Series(
        (rng.random(n) < 0.30).astype(float),
        index=subject_ids,
        name="converted",
    )
    features = rng.standard_normal((n, 5))
    # Slight signal so the model is not trivially constant
    features[y.to_numpy() == 1, 0] += 0.5
    feature_df = pd.DataFrame(
        features, index=subject_ids, columns=[f"f{i}" for i in range(5)]
    )

    def feature_builder(ids):
        parsed = [int(s) for s in ids]
        valid = feature_df.index.intersection(parsed)
        return feature_df.loc[valid].copy()

    return {
        "y": y,
        "feature_builder": feature_builder,
        "all_ids": subject_ids.tolist(),
    }


def test_recalibration_in_fold_only(synthetic_binary_data):
    """
    The Platt/isotonic recalibrator must be fit on an inner calibration
    split of the training fold only -- never on the test fold subjects or
    the inner training subjects.

    The _calib_spy callback receives (inner_calib_ids, test_ids,
    inner_train_ids) for each outer fold. We verify:
      1. calib_ids and test_ids are disjoint (no test leak).
      2. calib_ids is a strict subset of all subject IDs (not the full set).
      3. calib_ids and inner_train_ids are disjoint (witnesses the
         train/calib partition -- proving the split is genuine).
    """
    d = synthetic_binary_data
    calib_records = []

    def spy(calib_ids, test_ids, inner_train_ids):
        calib_records.append(
            (frozenset(int(s) for s in calib_ids),
             frozenset(int(s) for s in test_ids),
             frozenset(int(s) for s in inner_train_ids))
        )

    _binary_oof_cv(
        feature_builder=d["feature_builder"],
        y_series=d["y"],
        all_ids=d["all_ids"],
        cv_config=FAST_CV,
        _calib_spy=spy,
    )

    assert len(calib_records) > 0, (
        "Spy was never called -- _calib_spy is not wired up in _binary_oof_cv."
    )

    all_ids_set = frozenset(int(s) for s in d["all_ids"])
    for fold_i, (calib_ids, test_ids, inner_train_ids) in enumerate(calib_records):
        # No overlap: recalibrator never sees test-fold subjects
        overlap_test = calib_ids & test_ids
        assert len(overlap_test) == 0, (
            f"Fold {fold_i}: recalibrator calibration set overlaps with the "
            f"test fold by {len(overlap_test)} subjects: {overlap_test}. "
            "This is a data leak -- the recalibrator must not see test labels."
        )
        # Strict subset: calib set is smaller than the full subject pool
        assert calib_ids < all_ids_set, (
            f"Fold {fold_i}: calibration set equals all_ids -- the recalibrator "
            "likely saw the full training fold or the entire dataset."
        )
        # Train/calib partition: calib_ids and inner_train_ids are disjoint
        overlap_train = calib_ids & inner_train_ids
        assert len(overlap_train) == 0, (
            f"Fold {fold_i}: recalibrator calibration set overlaps with the "
            f"inner training set by {len(overlap_train)} subjects: {overlap_train}. "
            "The train/calib split is not genuine -- these must be disjoint."
        )


# ---------------------------------------------------------------------------
# Test 3: perfectly-calibrated predictor yields slope ~1, intercept ~0
# ---------------------------------------------------------------------------

def test_perfect_calibration_slope_intercept():
    """
    Calibration regression direction test.

    Case A -- perfect predictor: slope ~1, intercept ~0. Sanity check that
    the function runs and produces the right sign.

    Case B -- attenuated predictor (y_pred = 0.5*y_true + noise): this is
    the direction-sensitive test. With the correct regression direction
    (observed ~ predicted, predicted on x-axis), the true slope should be
    ~2 because y_true ~ 2*y_pred. A swapped-direction implementation would
    return slope ~0.5. We assert slope in [1.5, 2.5] so a direction bug
    causes this test to fail.
    """
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.normal(-0.3, 0.5, n)

    # Case A: nearly perfect predictor
    y_pred_perfect = y_true + rng.normal(0, 0.01, n)
    slope_a, intercept_a = compute_calibration_slope_intercept(y_true, y_pred_perfect)

    assert abs(slope_a - 1.0) < 0.10, (
        f"Calibration slope for a perfect predictor: expected ~1.0, got {slope_a:.4f}. "
        "This suggests the regression direction in "
        "compute_calibration_slope_intercept is incorrect."
    )
    assert abs(intercept_a) < 0.05, (
        f"Calibration intercept for a perfect predictor: expected ~0.0, "
        f"got {intercept_a:.4f}. "
        "This suggests a systematic bias in the calibration regression."
    )

    # Case B: attenuated predictor -- direction-sensitive test
    # y_pred = 0.5 * y_true + noise, so y_true ~ 2 * y_pred
    # linregress(y_pred, y_true) should give slope ~2 (scale compression detected)
    # A swapped-direction implementation would give slope ~0.5 and fail here.
    y_pred_attenuated = 0.5 * y_true + rng.normal(0, 0.02, n)
    slope_b, _ = compute_calibration_slope_intercept(y_true, y_pred_attenuated)

    assert 1.5 <= slope_b <= 2.5, (
        f"Calibration slope for an attenuated predictor (y_pred=0.5*y_true): "
        f"expected ~2.0 (in [1.5, 2.5]), got {slope_b:.4f}. "
        "With the correct direction (observed ~ predicted), an attenuated "
        "predictor yields slope>1. A slope near 0.5 indicates the regression "
        "axes are swapped in compute_calibration_slope_intercept."
    )
