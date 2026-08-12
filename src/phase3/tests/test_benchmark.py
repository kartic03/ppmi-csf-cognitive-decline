"""
Tests for src/phase3/01_benchmark.py (task C1).

TDD: these tests FAIL before the implementation exists and PASS after.

Test assertions (per task specification):
  1. EQUIVALENCE MARGIN and MINIMUM-DETECTABLE-DIFFERENCE are present in the
     output and the equivalence verdict is computed from them (not hard-coded).
  2. All arms use the SAME outer folds/outcome: identical fold subject-ID
     partitions across all seven arms.
  3. On a small synthetic dataset where the high-dim block is pure noise, the
     benchmark correctly returns equivalent=True. Where the high-dim block
     carries strong extra signal, it returns equivalent=False.

TabPFN notes: the synthetic machinery tests replace TabPFN with a fast Ridge
stub via model_overrides so the test suite runs in under ~15 s. The real
TabPFN is exercised only by the main 01_benchmark.py script on real PPMI data.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Module loader (01_benchmark.py has a digit prefix, cannot be imported
# directly; use the same importlib pattern as the rest of this codebase)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCHMARK_PATH = os.path.join(os.path.dirname(_HERE), "01_benchmark.py")


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("_benchmark_mod", _BENCHMARK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_moca(n_subjects=30, n_visits=5, seed=0, true_slopes=None):
    """
    Build a synthetic longitudinal MoCA table.

    Each subject receives n_visits spaced ~0.9 yr apart (total span ~3.6 yr,
    satisfying the >=4 visits and >=3 yr eligibility rule).
    """
    rng = np.random.default_rng(seed)
    if true_slopes is None:
        true_slopes = rng.normal(-0.2, 0.4, n_subjects)

    rows = []
    for i in range(n_subjects):
        base_age = float(rng.uniform(55.0, 72.0))
        base_moca = float(rng.uniform(22.0, 29.0))
        for v in range(n_visits):
            age = base_age + v * 0.9
            moca = base_moca + true_slopes[i] * (age - base_age) + float(rng.normal(0, 0.4))
            rows.append({
                "PATNO": i + 1,
                "EVENT_ID": f"V{v:02d}",
                "age_at_visit": age,
                "moca": float(np.clip(moca, 0.0, 30.0)),
            })
    return pd.DataFrame(rows), true_slopes


def _make_feature_frames(n_subjects, n_est, n_hd, rng, true_slopes,
                          signal_in="established", signal_strength=2.0):
    """
    Build synthetic feature DataFrames indexed by 1..n_subjects.

    signal_in:
        "established" - one established feature correlates with the slope;
                        high-dim columns are pure noise.
        "highdim"     - one high-dim feature strongly predicts the slope;
                        established columns are pure noise.
        "none"        - all pure noise.
    """
    ids = np.arange(1, n_subjects + 1)
    est_data = rng.standard_normal((n_subjects, n_est))
    hd_data = rng.standard_normal((n_subjects, n_hd))

    if signal_in == "established":
        est_data[:, 0] += true_slopes * signal_strength
    elif signal_in == "highdim":
        hd_data[:, 0] += true_slopes * signal_strength

    est_df = pd.DataFrame(
        est_data, index=ids,
        columns=[f"est_{j}" for j in range(n_est)],
    )
    hd_df = pd.DataFrame(
        hd_data, index=ids,
        columns=[f"hd_{j}" for j in range(n_hd)],
    )
    both_df = pd.concat([est_df, hd_df], axis=1)
    return est_df, hd_df, both_df


def _make_builders_from_dfs(est_df, hd_df, both_df):
    def est_builder(subject_ids):
        ids = [int(i) for i in subject_ids]
        return est_df.loc[[i for i in ids if i in est_df.index]]

    def hd_builder(subject_ids):
        ids = [int(i) for i in subject_ids]
        return hd_df.loc[[i for i in ids if i in hd_df.index]]

    def both_builder(subject_ids):
        ids = [int(i) for i in subject_ids]
        return both_df.loc[[i for i in ids if i in both_df.index]]

    return est_builder, hd_builder, both_builder


# ---------------------------------------------------------------------------
# Test 1: equivalence MARGIN and MDD present; verdict derived from inputs
# ---------------------------------------------------------------------------

def test_equivalence_verdict_derived_not_hardcoded():
    """
    EQUIVALENCE_MARGIN and min_detectable_diff must appear in the output, and
    the equivalent flag must flip when margin or the difference changes. This
    verifies that the verdict is computed from the inputs, not hard-coded.
    """
    bm = _load_benchmark()
    margin = bm.EQUIVALENCE_MARGIN
    assert margin > 0, "EQUIVALENCE_MARGIN must be a positive float"

    # Difference well within margin => equivalent
    v1 = bm.build_equivalence_verdict(
        best_diff=0.005, range_lo=-0.01, range_hi=0.012, margin=margin, mdd=0.025
    )
    assert "margin" in v1
    assert "min_detectable_diff" in v1
    assert v1["equivalent"] is True

    # Difference's upper range exceeds margin => not equivalent
    v2 = bm.build_equivalence_verdict(
        best_diff=0.05, range_lo=0.03, range_hi=0.07, margin=margin, mdd=0.04
    )
    assert v2["equivalent"] is False

    # Narrowing margin from 0.05 to 0.01 must flip verdict for diff~0.025
    v3 = bm.build_equivalence_verdict(
        best_diff=0.025, range_lo=0.02, range_hi=0.03, margin=0.05, mdd=0.01
    )
    v4 = bm.build_equivalence_verdict(
        best_diff=0.025, range_lo=0.02, range_hi=0.03, margin=0.01, mdd=0.01
    )
    assert v3["equivalent"] is True   # range_hi=0.03 < margin=0.05
    assert v4["equivalent"] is False  # range_hi=0.03 >= margin=0.01


def test_verdict_straddling_margin_is_inconclusive_not_superior():
    """
    When the per-seed difference range STRADDLES the margin (range_lo < margin
    <= range_hi) - e.g. one seed negative, the canonical diff below the margin,
    but the best seed above it - the verdict must NOT claim high-dim is superior.
    A 'genuinely beats' claim requires the LOWER bound to clear the margin, not
    just the upper bound. Guards against declaring non-equivalence off a single
    favorable seed.
    """
    bm = _load_benchmark()
    margin = 0.02
    # diffs like [-0.01, +0.005, +0.04]: mean +0.011 (< margin), one seed < 0,
    # but range_hi=0.04 >= margin. Old upper-bound-only logic called this
    # "may genuinely beat"; correct logic calls it inconclusive.
    v = bm.build_equivalence_verdict(
        best_diff=0.011, range_lo=-0.01, range_hi=0.04, margin=margin, mdd=0.05
    )
    assert v["equivalent"] is False      # range_hi >= margin
    assert v["superior"] is False        # range_lo < margin -> NOT superior
    assert v["status"] == "inconclusive"
    assert "INCONCLUSIVE" in v["verdict"]

    # And a range whose LOWER bound clears the margin IS superior.
    v2 = bm.build_equivalence_verdict(
        best_diff=0.045, range_lo=0.0335, range_hi=0.0606, margin=margin, mdd=0.027
    )
    assert v2["superior"] is True
    assert v2["status"] == "superior"


# ---------------------------------------------------------------------------
# Test 2: all seven arms use the same outer folds/outcome
# ---------------------------------------------------------------------------

def test_same_outer_folds_across_arms():
    """
    When all arms receive the same analytic_ids and cv_config, the canonical-
    seed OOF subject IDs must be identical across all eight arms.
    """
    from sklearn.linear_model import Ridge

    bm = _load_benchmark()

    n_subj = 32
    rng = np.random.default_rng(42)
    moca, slopes = _make_moca(n_subjects=n_subj, n_visits=5, seed=42)
    est_df, hd_df, both_df = _make_feature_frames(n_subj, 5, 10, rng, slopes)

    analytic_ids = np.arange(1, n_subj + 1)
    est_b, hd_b, both_b = _make_builders_from_dfs(est_df, hd_df, both_df)

    stub = lambda: Ridge()
    cv_config = {"n_outer": 3, "n_inner": 3, "seeds": [0, 1]}

    output = bm.run_benchmark_on_data(
        established_builder=est_b,
        highdim_builder=hd_b,
        both_builder=both_b,
        moca_long=moca,
        analytic_ids=analytic_ids,
        cv_config=cv_config,
        model_overrides={"tabpfn": stub},
    )

    oof_ids = output["_oof_ids"]
    all_arms = list(oof_ids.keys())
    assert len(all_arms) == 8, f"Expected 8 arms, got {all_arms}"

    reference = oof_ids[all_arms[0]]
    for arm in all_arms[1:]:
        assert oof_ids[arm] == reference, (
            f"Arm '{arm}' has different OOF subject IDs than '{all_arms[0]}':\n"
            f"  {all_arms[0]}: {reference[:5]}...\n"
            f"  {arm}: {oof_ids[arm][:5]}..."
        )


# ---------------------------------------------------------------------------
# Test 3a: pure noise in high-dim => equivalent=True
# ---------------------------------------------------------------------------

def test_noise_highdim_returns_equivalent():
    """
    When the high-dim block is pure noise and established features carry the
    signal, the benchmark must return equivalent=True (high-dim adds nothing
    meaningful). Uses a Ridge stub in place of TabPFN.
    """
    from sklearn.linear_model import Ridge

    bm = _load_benchmark()

    n_subj = 40
    rng = np.random.default_rng(7)
    true_slopes = rng.normal(-0.3, 0.5, n_subj)
    moca, _ = _make_moca(n_subjects=n_subj, n_visits=5, seed=7, true_slopes=true_slopes)

    # Strong signal in established; high-dim is pure noise
    est_df, hd_df, both_df = _make_feature_frames(
        n_subj, 5, 20, rng, true_slopes,
        signal_in="established", signal_strength=3.0,
    )

    analytic_ids = np.arange(1, n_subj + 1)
    est_b, hd_b, both_b = _make_builders_from_dfs(est_df, hd_df, both_df)

    stub = lambda: Ridge()
    cv_config = {"n_outer": 3, "n_inner": 3, "seeds": [0, 1, 2]}

    output = bm.run_benchmark_on_data(
        established_builder=est_b,
        highdim_builder=hd_b,
        both_builder=both_b,
        moca_long=moca,
        analytic_ids=analytic_ids,
        cv_config=cv_config,
        model_overrides={"tabpfn": stub},
    )

    contrast = output["result"]["key_contrast"]
    assert contrast["equivalent"] is True, (
        f"Expected equivalent=True for noise high-dim. "
        f"Got: best_diff={contrast['best_highdim_minus_established']:.4f}, "
        f"range_hi={contrast['range_hi']:.4f}, margin={contrast['margin']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3b: strong signal only in high-dim => equivalent=False
# ---------------------------------------------------------------------------

def test_signal_highdim_returns_not_equivalent():
    """
    When the high-dim block contains a near-perfect predictor and the
    established block is pure noise, the benchmark must return equivalent=False
    (high-dim genuinely beats established). Uses a Ridge stub for TabPFN.
    """
    from sklearn.linear_model import Ridge

    bm = _load_benchmark()

    n_subj = 50
    rng = np.random.default_rng(99)
    true_slopes = rng.normal(-0.5, 0.8, n_subj)

    # Build very clean MoCA series (low noise) so EB captures the signal
    moca_rows = []
    for i, slope in enumerate(true_slopes):
        base_age = float(rng.uniform(55.0, 70.0))
        base_moca = float(rng.uniform(20.0, 27.0))
        for v in range(5):
            age = base_age + v * 0.9
            moca_rows.append({
                "PATNO": i + 1,
                "EVENT_ID": f"V{v:02d}",
                "age_at_visit": age,
                "moca": float(np.clip(
                    base_moca + slope * (age - base_age) + rng.normal(0, 0.08),
                    0.0, 30.0,
                )),
            })
    moca = pd.DataFrame(moca_rows)

    ids = np.arange(1, n_subj + 1)

    # Established: pure noise
    est_data = rng.standard_normal((n_subj, 5))
    est_df = pd.DataFrame(est_data, index=ids,
                          columns=[f"est_{j}" for j in range(5)])

    # High-dim: first column is a near-perfect predictor
    hd_col0 = true_slopes * 10.0 + rng.standard_normal(n_subj) * 0.2
    hd_data = np.column_stack([hd_col0, rng.standard_normal((n_subj, 29))])
    hd_df = pd.DataFrame(hd_data, index=ids,
                         columns=[f"hd_{j}" for j in range(30)])
    both_df = pd.concat([est_df, hd_df], axis=1)

    est_b, hd_b, both_b = _make_builders_from_dfs(est_df, hd_df, both_df)

    stub = lambda: Ridge()
    cv_config = {"n_outer": 3, "n_inner": 3, "seeds": [0, 1, 2]}

    output = bm.run_benchmark_on_data(
        established_builder=est_b,
        highdim_builder=hd_b,
        both_builder=both_b,
        moca_long=moca,
        analytic_ids=ids,
        cv_config=cv_config,
        model_overrides={"tabpfn": stub},
    )

    contrast = output["result"]["key_contrast"]
    assert contrast["equivalent"] is False, (
        f"Expected equivalent=False for signal-rich high-dim. "
        f"Got: best_diff={contrast['best_highdim_minus_established']:.4f}, "
        f"range_hi={contrast['range_hi']:.4f}, margin={contrast['margin']:.4f}"
    )
