"""
Integration tests for the clinical baseline model (02_baseline.py), task B2.

These tests use real PPMI data (curated_cut.parquet + outcome.parquet) and are
automatically skipped when those files are absent. They are designed to be
lightweight (small n_repeats) so they complete within the standard pytest run.

Run:
    pixi run python -m pytest src/phase2/tests/test_baseline.py -v
"""
import importlib.util
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_BASELINE_PATH = os.path.join(_THIS_DIR, "..", "02_baseline.py")
_PARQUET = os.path.join(_ROOT, "data", "processed", "phase1", "curated_cut.parquet")
_OUTCOME = os.path.join(_ROOT, "data", "processed", "phase2", "outcome.parquet")

# Skip all tests when data files are absent (CI without PPMI data)
pytestmark = pytest.mark.skipif(
    not os.path.exists(_PARQUET) or not os.path.exists(_OUTCOME),
    reason="PPMI data files not present; skipping integration tests",
)


def _load_baseline():
    spec = importlib.util.spec_from_file_location("baseline_b2", _BASELINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test 1: Feature builder returns correct columns and index
# ---------------------------------------------------------------------------

def test_feature_builder_columns_and_index():
    """
    make_clinical_feature_builder() returns a builder whose output has:
      - exactly CLINICAL_COLS columns
      - index values that are a subset of the requested subject IDs
      - all index values are in the slope-eligible set
    """
    mod = _load_baseline()
    builder = mod.make_clinical_feature_builder()

    import pandas as pd
    outcome = pd.read_parquet(_OUTCOME)
    # Request a 50-subject subsample of slope-eligible subjects
    slope_ids = outcome["subject_id"].tolist()[:50]
    X = builder(slope_ids)

    assert list(X.columns) == mod.CLINICAL_COLS, (
        f"Columns mismatch. Expected: {mod.CLINICAL_COLS}, Got: {list(X.columns)}"
    )
    assert set(X.index).issubset(set(slope_ids)), (
        "Feature builder returned subjects not in the requested subset"
    )
    # All returned subjects must be slope-eligible
    all_slope_ids = set(outcome["subject_id"])
    assert set(X.index).issubset(all_slope_ids), (
        "Feature builder returned non-slope-eligible subjects"
    )


# ---------------------------------------------------------------------------
# Test 2: Analytic N is in the expected range
# ---------------------------------------------------------------------------

def test_analytic_n_in_expected_range():
    """
    The clinical-block analytic N (slope-eligible subjects with BL features)
    must be in the range [700, 820] to match the B0 report (~771-816).
    """
    mod = _load_baseline()
    builder = mod.make_clinical_feature_builder()

    import pandas as pd
    outcome = pd.read_parquet(_OUTCOME)
    all_slope_ids = outcome["subject_id"].tolist()
    X = builder(all_slope_ids)

    assert 700 <= len(X) <= 820, (
        f"Analytic N={len(X)} is outside the expected range [700, 820]. "
        "Check the feature builder's subject filter."
    )


# ---------------------------------------------------------------------------
# Test 3: run_baseline produces required output structure
# ---------------------------------------------------------------------------

def test_run_baseline_output_structure():
    """
    run_baseline() must return a dict with keys 'ridge', 'hgbm', 'n_subjects',
    'n_folds', 'n_repeats', and each model dict must contain the CV metric keys.
    Uses a tiny CV config to keep runtime < 60 s.
    """
    mod = _load_baseline()
    result = mod.run_baseline(
        cv_config={"n_outer": 3, "n_inner": 3, "seeds": [0, 1]}
    )

    required_top = {"ridge", "hgbm", "n_subjects", "n_folds", "n_repeats"}
    assert required_top.issubset(result.keys()), (
        f"Missing top-level keys: {required_top - set(result.keys())}"
    )

    for model_key in ("ridge", "hgbm"):
        sub = result[model_key]
        required_sub = {"r2_mean", "r2_repeat_lo", "r2_repeat_hi", "per_fold_r2", "per_seed_r2"}
        assert required_sub.issubset(sub.keys()), (
            f"{model_key}: Missing keys: {required_sub - set(sub.keys())}"
        )
        assert isinstance(sub["r2_mean"], float), (
            f"{model_key}.r2_mean must be a float"
        )
        assert len(sub["per_fold_r2"]) == result["n_folds"], (
            f"{model_key}.per_fold_r2 length mismatch"
        )
        assert len(sub["per_seed_r2"]) == result["n_repeats"], (
            f"{model_key}.per_seed_r2 length mismatch"
        )


# ---------------------------------------------------------------------------
# Test 4: ridge R2 is non-trivially positive
# ---------------------------------------------------------------------------

def test_ridge_r2_positive():
    """
    The clinical block (age, sex, educ, baseline MoCA, UPDRS-III, etc.) is
    known to predict MoCA decline. The nested-CV R2 must be > 0, otherwise
    the model is worse than predicting the mean — a signal that something has
    gone wrong (e.g., outcome is shuffled or features are misaligned).
    """
    mod = _load_baseline()
    result = mod.run_baseline(
        cv_config={"n_outer": 3, "n_inner": 3, "seeds": [0, 1]}
    )

    r2 = result["ridge"]["r2_mean"]
    assert r2 > 0.0, (
        f"Ridge nested-CV R2={r2:.4f} is not positive. "
        "Clinical predictors should explain some variance in MoCA slope. "
        "Check for subject-ID misalignment or outcome/feature mismatch."
    )


# ---------------------------------------------------------------------------
# Test 5: n_subjects matches the feature builder's output
# ---------------------------------------------------------------------------

def test_n_subjects_consistent():
    """
    The n_subjects reported in run_baseline() must equal the number of subjects
    returned by the feature builder for the full slope-eligible set.
    """
    mod = _load_baseline()

    import pandas as pd
    outcome = pd.read_parquet(_OUTCOME)
    all_slope_ids = outcome["subject_id"].tolist()
    builder = mod.make_clinical_feature_builder()
    X_full = builder(all_slope_ids)

    result = mod.run_baseline(
        cv_config={"n_outer": 3, "n_inner": 3, "seeds": [0]}
    )

    assert result["n_subjects"] == len(X_full), (
        f"n_subjects={result['n_subjects']} in result does not match "
        f"feature builder output n={len(X_full)}"
    )
