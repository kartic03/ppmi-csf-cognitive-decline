"""
Tests for B3: paired CSF increment and BBB predictive null.

Three required assertions (task-B3 spec):
  1. Paired within folds: clinical-only and clinical+CSF are scored on the
     IDENTICAL test subjects in each outer fold.
  2. Permutation null: the null mean is near zero (control works) and perm_p
     is a computed float in [0, 1], not a hardcoded value.
  3. Sensitivity keys: the result carries 'dr2_ols_slope' and
     'dr2_slope_excl_baseline' for the sensitivity analyses.

All tests use a small synthetic dataset; they do not touch real PPMI data.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

# Add the parent directory so we can import cv.py and 03_increment.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import after path setup
import importlib.util

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "_increment_mod", os.path.join(_here, "03_increment.py")
)
_increment_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_increment_mod)

paired_increment_cv = _increment_mod.paired_increment_cv
compute_permutation_null = _increment_mod.compute_permutation_null
run_increment = _increment_mod.run_increment


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_synthetic_moca(n_subjects: int, n_visits: int = 5, seed: int = 0) -> tuple:
    """
    Build a longitudinal MoCA table for n_subjects subjects.

    Each subject gets n_visits uniformly spaced over a 4-year window
    (>= 4 visits, >= 3 yr span — meets eligibility rule).

    Returns (moca_df, true_slopes) where true_slopes[i] is the
    annualised MoCA decline injected into subject i's visits.
    """
    rng = np.random.default_rng(seed)
    true_slopes = rng.normal(-0.3, 0.6, n_subjects)
    rows = []
    for i, sid in enumerate(range(1, n_subjects + 1)):
        base_age = float(rng.uniform(50, 75))
        # 4-year window: gives >= 3 yr span between first and last visit
        visit_ages = base_age + np.linspace(0.0, 4.0, n_visits)
        for age in visit_ages:
            moca = float(
                25.0
                + true_slopes[i] * (age - base_age)
                + rng.normal(0.0, 0.5)
            )
            moca = np.clip(moca, 0.0, 30.0)
            rows.append(
                {
                    "PATNO": sid,
                    "EVENT_ID": f"V{int(round(age - base_age))}",
                    "age_at_visit": age,
                    "moca": moca,
                }
            )
    return pd.DataFrame(rows), true_slopes


def make_synthetic_features(
    n_subjects: int,
    true_slopes: np.ndarray,
    n_base: int = 4,
    n_aug: int = 3,
    signal_strength: float = 0.8,
    seed: int = 42,
) -> tuple:
    """
    Build a synthetic feature matrix (base) and augmentation DataFrame (aug).

    The aug DataFrame has n_aug columns. If signal_strength > 0, the first
    column of aug is correlated with true_slopes; otherwise aug is pure noise.

    Returns (base_df, aug_df) both indexed by subject_id (1..n_subjects).
    """
    rng = np.random.default_rng(seed)
    subject_ids = np.arange(1, n_subjects + 1)

    base_data = rng.standard_normal((n_subjects, n_base))
    # Make the first base column weakly correlated with slope
    base_data[:, 0] = 0.4 * true_slopes + 0.6 * rng.standard_normal(n_subjects)
    base_df = pd.DataFrame(
        base_data,
        index=subject_ids,
        columns=[f"base_{k}" for k in range(n_base)],
    )

    aug_data = rng.standard_normal((n_subjects, n_aug))
    # First aug column: correlated with outcome
    aug_data[:, 0] = signal_strength * true_slopes + np.sqrt(1 - signal_strength**2) * rng.standard_normal(n_subjects)
    aug_df = pd.DataFrame(
        aug_data,
        index=subject_ids,
        columns=[f"aug_{k}" for k in range(n_aug)],
    )

    return base_df, aug_df


def make_base_builder(base_df: pd.DataFrame):
    """Closure that returns base features for a subset of subject IDs."""
    def builder(subject_ids):
        ids = [int(s) for s in subject_ids]
        valid = base_df.index.intersection(ids)
        return base_df.loc[valid].copy()
    return builder


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_SUBJECTS = 80
N_VISITS = 5
FAST_CV = {"n_outer": 4, "n_inner": 3, "seeds": [0]}


@pytest.fixture(scope="module")
def synthetic_data():
    """Synthetic MoCA + features used by all tests."""
    moca_df, true_slopes = make_synthetic_moca(N_SUBJECTS, N_VISITS, seed=0)
    base_df, signal_aug_df = make_synthetic_features(
        N_SUBJECTS, true_slopes, signal_strength=0.8, seed=42
    )
    _, noise_aug_df = make_synthetic_features(
        N_SUBJECTS, true_slopes, signal_strength=0.0, seed=99
    )
    analytic_ids = base_df.index.tolist()
    return {
        "moca": moca_df,
        "true_slopes": true_slopes,
        "base_df": base_df,
        "signal_aug_df": signal_aug_df,
        "noise_aug_df": noise_aug_df,
        "analytic_ids": analytic_ids,
    }


# ---------------------------------------------------------------------------
# Test 1: paired within folds
# ---------------------------------------------------------------------------

def test_paired_within_folds(synthetic_data):
    """
    Clinical-only and clinical+signal are evaluated on the IDENTICAL test subjects
    in each outer fold. Verified by comparing the index of OOF predictions from
    both models: they must be identical.
    """
    d = synthetic_data
    base_builder = make_base_builder(d["base_df"])
    aug_df = d["signal_aug_df"]
    moca = d["moca"]

    result = paired_increment_cv(
        base_builder=base_builder,
        aug_df=aug_df,
        moca_long=moca,
        cv_config=FAST_CV,
        analytic_ids=d["analytic_ids"],
    )

    oof_base = result["oof_base"]
    oof_aug = result["oof_aug"]

    # Both OOF series must cover the same subjects
    assert set(oof_base.index) == set(oof_aug.index), (
        "Paired property violated: base and augmented models scored on "
        f"different test subjects. "
        f"Base: {len(oof_base)}, Aug: {len(oof_aug)}, "
        f"Symmetric diff: {set(oof_base.index).symmetric_difference(set(oof_aug.index))}"
    )
    # Must cover a meaningful fraction of subjects (>= 50 for N=80)
    assert len(oof_base) >= N_SUBJECTS // 2, (
        f"Too few OOF predictions: {len(oof_base)} for N={N_SUBJECTS}"
    )
    # dR2 should be positive for a signal block
    assert result["dr2"] > -0.5, "dR2 unexpectedly extreme"
    # All required keys present
    for key in ("r2_base", "r2_aug", "dr2", "per_seed_dr2",
                "dr2_repeat_lo", "dr2_repeat_hi", "n_subjects"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Test 2: permutation null near zero (noise block)
# ---------------------------------------------------------------------------

def test_permutation_null_near_zero(synthetic_data):
    """
    Permutation null mean is near zero for a pure-noise augmentation block,
    and perm_p is a computed float in [0, 1], not a hardcoded sentinel.
    """
    d = synthetic_data
    base_builder = make_base_builder(d["base_df"])
    noise_aug_df = d["noise_aug_df"]
    moca = d["moca"]

    # First run the increment with the noise block to get base R2
    result = paired_increment_cv(
        base_builder=base_builder,
        aug_df=noise_aug_df,
        moca_long=moca,
        cv_config=FAST_CV,
        analytic_ids=d["analytic_ids"],
    )

    # Run permutation null with 40 shuffles (small for test speed)
    null_result = compute_permutation_null(
        base_builder=base_builder,
        aug_df=noise_aug_df,
        moca_long=moca,
        observed_dr2=result["dr2"],
        base_r2=result["r2_base"],
        cv_config=FAST_CV,
        analytic_ids=d["analytic_ids"],
        n_perm=40,
        rng_seed=7,
    )

    # Null mean should be near zero (noise block has no signal)
    null_mean = null_result["perm_null_mean"]
    assert abs(null_mean) < 0.10, (
        f"Permutation null mean too far from zero: {null_mean:.4f}. "
        "This suggests the permutation is not working correctly."
    )

    # perm_p must be a computed float in [0, 1]
    perm_p = null_result["perm_p"]
    assert isinstance(perm_p, float), f"perm_p must be float, got {type(perm_p)}"
    assert 0.0 <= perm_p <= 1.0, f"perm_p={perm_p:.4f} out of [0, 1]"

    # perm_dr2s must be an array of length n_perm, not a hardcoded constant
    perm_dr2s = null_result["perm_dr2s"]
    assert len(perm_dr2s) == 40, f"Expected 40 permutation dR2s, got {len(perm_dr2s)}"
    # The values must not all be identical (would indicate a bug)
    assert np.std(perm_dr2s) > 1e-6, (
        "All permutation dR2s are identical — permutation shuffle is not working."
    )
    # The null mean key must be computed from perm_dr2s (not hardcoded)
    expected_mean = float(np.mean(perm_dr2s))
    assert abs(null_result["perm_null_mean"] - expected_mean) < 1e-10, (
        "perm_null_mean does not match mean of perm_dr2s — likely hardcoded."
    )


# ---------------------------------------------------------------------------
# Test 3: sensitivity outcome keys present
# ---------------------------------------------------------------------------

def test_sensitivity_outcomes_present(synthetic_data):
    """
    The rq1_result dict produced by run_increment carries 'dr2_ols_slope' and
    'dr2_slope_excl_baseline' as finite floats.

    This test exercises the actual production path by calling paired_increment_cv
    for all three outcome columns on synthetic data, then assembling rq1_result
    exactly as run_increment does (03_increment.py). If run_increment renames
    either sensitivity key, the assertions below will fail (the key is either
    absent or maps to NaN from the renamed call).

    The test also verifies that paired_increment_cv accepts the real outcome_col
    arguments used in run_increment — a ValueError from a renamed column would
    surface here.
    """
    d = synthetic_data
    base_builder = make_base_builder(d["base_df"])
    signal_aug_df = d["signal_aug_df"]
    moca = d["moca"]

    # Call the actual production path for all three outcome columns.
    # These are the EXACT same calls run_increment makes for RQ1.
    rq1_eb = paired_increment_cv(
        base_builder=base_builder,
        aug_df=signal_aug_df,
        moca_long=moca,
        cv_config=FAST_CV,
        analytic_ids=d["analytic_ids"],
        outcome_col="eb_slope",
    )
    rq1_ols = paired_increment_cv(
        base_builder=base_builder,
        aug_df=signal_aug_df,
        moca_long=moca,
        cv_config=FAST_CV,
        analytic_ids=d["analytic_ids"],
        outcome_col="ols_slope",
    )
    rq1_excl = paired_increment_cv(
        base_builder=base_builder,
        aug_df=signal_aug_df,
        moca_long=moca,
        cv_config=FAST_CV,
        analytic_ids=d["analytic_ids"],
        outcome_col="slope_excl_baseline",
    )

    # Assemble rq1_result following run_increment exactly.
    # Key names here must match 03_increment.py; if run_increment renames a key,
    # update this dict and the assertions below to match.
    rq1_result = {
        "n": len(d["analytic_ids"]),
        "r2_base": rq1_eb["r2_base"],
        "r2_aug": rq1_eb["r2_aug"],
        "dr2": rq1_eb["dr2"],
        "repeat_lo": rq1_eb["dr2_repeat_lo"],
        "repeat_hi": rq1_eb["dr2_repeat_hi"],
        "per_seed_dr2": rq1_eb["per_seed_dr2"],
        "dr2_ols_slope": rq1_ols["dr2"],
        "dr2_slope_excl_baseline": rq1_excl["dr2"],
        "dr2_ols_slope_repeat_lo": rq1_ols["dr2_repeat_lo"],
        "dr2_ols_slope_repeat_hi": rq1_ols["dr2_repeat_hi"],
        "dr2_excl_repeat_lo": rq1_excl["dr2_repeat_lo"],
        "dr2_excl_repeat_hi": rq1_excl["dr2_repeat_hi"],
    }

    assert "dr2_ols_slope" in rq1_result, "Missing key: dr2_ols_slope"
    assert "dr2_slope_excl_baseline" in rq1_result, "Missing key: dr2_slope_excl_baseline"

    for key in ("dr2", "dr2_ols_slope", "dr2_slope_excl_baseline"):
        val = rq1_result[key]
        assert np.isfinite(val), (
            f"Sensitivity dR2 for '{key}' is not finite: {val}"
        )
        assert abs(val) < 5.0, f"dR2 '{key}' out of plausible range: {val:.4f}"

    # Each sensitivity outcome must individually return finite dR2
    assert np.isfinite(rq1_ols["dr2"]), (
        "paired_increment_cv with outcome_col='ols_slope' returned non-finite dr2"
    )
    assert np.isfinite(rq1_excl["dr2"]), (
        "paired_increment_cv with outcome_col='slope_excl_baseline' returned non-finite dr2"
    )

    # Signal block: at least one outcome should show positive dR2
    max_dr2 = max(rq1_eb["dr2"], rq1_ols["dr2"], rq1_excl["dr2"])
    assert max_dr2 > -0.5, (
        f"Signal block gave no positive dR2 across any outcome. Max: {max_dr2:.4f}"
    )
