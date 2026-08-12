"""
Tests for src/phase2/01_outcome.py — leakage-safe EB MoCA-slope outcome (B1).

The primary assertion is LEAKAGE SAFETY: the EB shrinkage parameters
(mu, tau2) estimated from a training subset must not change when non-training
subjects are present or absent in the input DataFrame.

Run:
    pixi run python -m pytest src/phase2/tests/test_outcome.py -v
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Module loader (handles the digit-prefixed filename "01_outcome.py")
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTCOME_PATH = os.path.join(_THIS_DIR, "..", "01_outcome.py")


def _load_outcome():
    spec = importlib.util.spec_from_file_location("outcome_b1", _OUTCOME_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_outcome()
fit_eb_params = _mod.fit_eb_params
eb_slopes = _mod.eb_slopes


# ---------------------------------------------------------------------------
# Synthetic MoCA table factory
# ---------------------------------------------------------------------------

def make_moca(n_subjects=30, seed=42):
    """
    Generate a synthetic longitudinal MoCA table.
    Each subject gets 6 visits across 4 years so all pass eligibility
    (>=4 visits, >=3 yr span).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pat in range(1, n_subjects + 1):
        ages = np.linspace(60.0, 64.0, 6) + rng.uniform(-0.1, 0.1, 6)
        ages = np.sort(ages)
        true_slope = rng.normal(-0.3, 0.5)
        baseline = rng.uniform(22.0, 28.0)
        scores = baseline + true_slope * (ages - ages[0]) + rng.normal(0, 0.5, 6)
        scores = np.clip(scores, 0.0, 30.0)
        for i, (age, score) in enumerate(zip(ages, scores)):
            rows.append({
                "PATNO": pat,
                "EVENT_ID": f"V{i:02d}",
                "age_at_visit": float(age),
                "moca": float(score),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# Test 1: LEAKAGE SAFETY - fit_eb_params ignores non-training subjects
# ===========================================================================

def test_fit_eb_params_ignores_non_training_rows():
    """
    fit_eb_params(moca, A) must return identical mu and tau2 regardless
    of whether extra subjects (B) are present in the moca DataFrame.
    Only rows for A may influence the returned parameters.

    This is the core CV leakage-safety check: if a test subject's rows
    are present in the DataFrame but not in subject_ids, the parameters
    must be unaffected.
    """
    moca = make_moca(n_subjects=30)
    all_ids = list(moca["PATNO"].unique())
    A = set(all_ids[:15])   # training subjects
    B = set(all_ids[15:])   # held-out (must be invisible)

    # Case 1: full table (contains B rows)
    params_with_B = fit_eb_params(moca, A)

    # Case 2: B rows removed from the DataFrame
    moca_A_only = moca[moca["PATNO"].isin(A)].copy()
    params_without_B = fit_eb_params(moca_A_only, A)

    assert np.isclose(params_with_B["mu"], params_without_B["mu"], atol=1e-12), (
        f"mu changed with non-training rows present: "
        f"{params_with_B['mu']:.10f} vs {params_without_B['mu']:.10f}"
    )
    assert np.isclose(params_with_B["tau2"], params_without_B["tau2"], atol=1e-12), (
        f"tau2 changed with non-training rows present: "
        f"{params_with_B['tau2']:.10f} vs {params_without_B['tau2']:.10f}"
    )
    assert params_with_B["n_used"] == params_without_B["n_used"], (
        "n_used changed when non-training rows present"
    )


def test_fit_eb_params_perturbed_non_training_rows_unchanged():
    """
    Even if B subjects' MoCA scores are replaced with extreme values,
    fit_eb_params(moca_perturbed, A) must still return the same mu/tau2.
    """
    moca = make_moca(n_subjects=30)
    all_ids = list(moca["PATNO"].unique())
    A = set(all_ids[:15])
    B = set(all_ids[15:])

    params_original = fit_eb_params(moca, A)

    # Perturb B subjects' scores to extreme values
    moca_perturbed = moca.copy()
    moca_perturbed.loc[moca_perturbed["PATNO"].isin(B), "moca"] = 999.0

    params_perturbed = fit_eb_params(moca_perturbed, A)

    assert np.isclose(params_original["mu"], params_perturbed["mu"], atol=1e-12), (
        "mu changed when non-training subjects' data was perturbed"
    )
    assert np.isclose(params_original["tau2"], params_perturbed["tau2"], atol=1e-12), (
        "tau2 changed when non-training subjects' data was perturbed"
    )


# ===========================================================================
# Test 2: LEAKAGE SAFETY - eb_slopes is independent per subject
# ===========================================================================

def test_eb_slopes_per_subject_independence():
    """
    The EB slope for any single subject must be identical whether computed
    alone or alongside other test subjects. No cross-contamination between
    test subjects is allowed (the passed eb_params are the only coupling).
    """
    moca = make_moca(n_subjects=20)
    all_ids = list(moca["PATNO"].unique())
    train_ids = set(all_ids[:10])
    test_ids = set(all_ids[10:])

    params = fit_eb_params(moca, train_ids)

    # Batch computation
    slopes_batch = eb_slopes(moca, test_ids, params)

    # Individual computation for each test subject
    for sid in test_ids:
        slopes_single = eb_slopes(moca, {sid}, params)
        if sid not in slopes_batch.index:
            # Subject may be ineligible; verify it is also absent individually
            assert sid not in slopes_single.index, (
                f"Subject {sid} absent in batch but present in individual call"
            )
            continue
        if sid not in slopes_single.index:
            pytest.fail(f"Subject {sid} present in batch but absent in individual call")

        for col in ("eb_slope", "ols_slope", "slope_excl_baseline"):
            batch_val = slopes_batch.loc[sid, col]
            single_val = slopes_single.loc[sid, col]
            both_nan = np.isnan(batch_val) and np.isnan(single_val)
            if not both_nan:
                assert np.isclose(batch_val, single_val, atol=1e-12), (
                    f"Subject {sid} col={col}: batch={batch_val:.10f} "
                    f"vs single={single_val:.10f}"
                )


# ===========================================================================
# Test 3: Sensitivity columns exist and shrinkage is non-trivial
# ===========================================================================

def test_sensitivity_columns_present_and_shrinkage_active():
    """
    eb_slopes must return ols_slope and slope_excl_baseline columns.
    Because EB shrinkage is active when tau2 > 0, eb_slope must differ
    from ols_slope for subjects whose sampling variance (v_i) is non-zero.
    Also, slope_excl_baseline must differ from ols_slope for some subjects.
    """
    moca = make_moca(n_subjects=30)
    all_ids = moca["PATNO"].unique()
    params = fit_eb_params(moca, all_ids)
    out = eb_slopes(moca, all_ids, params)

    # Required columns
    for col in ("eb_slope", "ols_slope", "slope_excl_baseline", "n_visits", "span_years"):
        assert col in out.columns, f"Missing required column: {col}"

    # Shrinkage must be non-trivial
    assert params["tau2"] > 0, f"tau2={params['tau2']}: shrinkage is zero (degenerate data)"
    eb_ols_diff = (out["eb_slope"] - out["ols_slope"]).abs()
    assert eb_ols_diff.max() > 1e-6, (
        f"EB slope equals OLS slope for all subjects; shrinkage not applied. "
        f"tau2={params['tau2']:.6f}"
    )

    # slope_excl_baseline must differ from ols_slope for some subjects
    excl_diffs = (out["slope_excl_baseline"] - out["ols_slope"]).abs().dropna()
    assert len(excl_diffs) > 0, "slope_excl_baseline is NaN for all subjects"
    assert excl_diffs.max() > 1e-6, (
        "slope_excl_baseline is identical to ols_slope for all subjects"
    )


# ===========================================================================
# Test 4: Eligibility rule enforced in both fit and transform
# ===========================================================================

def test_eligibility_rule_enforced():
    """
    Subjects with < 4 visits OR < 3 yr span must not appear in eb_slopes output
    and must not contribute to fit_eb_params.
    """
    rows = []

    # Subject 999: only 3 visits (ineligible — too few visits)
    for age, score in zip([60.0, 61.5, 63.0], [27.0, 26.5, 26.0]):
        rows.append({"PATNO": 999, "EVENT_ID": f"V{int(age)}", "age_at_visit": age, "moca": score})

    # Subject 998: 4 visits but only 2 yr span (ineligible — too short)
    for i, (age, score) in enumerate(zip([60.0, 60.5, 61.0, 62.0], [26.0, 25.5, 25.0, 24.5])):
        rows.append({"PATNO": 998, "EVENT_ID": f"V{i}", "age_at_visit": age, "moca": score})

    # Subject 1000: 5 visits, 4 yr span (eligible)
    for age, score in zip([60.0, 61.0, 62.0, 63.0, 64.0], [27.0, 26.5, 26.0, 25.5, 25.0]):
        rows.append({"PATNO": 1000, "EVENT_ID": f"V{int(age)}", "age_at_visit": age, "moca": score})

    moca = pd.DataFrame(rows)
    all_ids = {999, 998, 1000}

    params = fit_eb_params(moca, all_ids)
    # Only subject 1000 contributes to params
    assert params["n_used"] == 1, f"Expected 1 eligible subject, got {params['n_used']}"

    out = eb_slopes(moca, all_ids, params)
    assert 999 not in out.index, "Subject 999 (3 visits) must be excluded"
    assert 998 not in out.index, "Subject 998 (<3 yr span) must be excluded"
    assert 1000 in out.index, "Subject 1000 (eligible) must be included"


# ===========================================================================
# Test 5: n_visits and span_years are correct
# ===========================================================================

def test_metadata_columns_correct():
    """n_visits and span_years in output must match the raw data."""
    rows = []
    ages = [60.0, 61.0, 62.0, 63.5, 64.5]
    scores = [27.0, 26.5, 26.0, 25.5, 25.0]
    for i, (age, score) in enumerate(zip(ages, scores)):
        rows.append({"PATNO": 1, "EVENT_ID": f"V{i}", "age_at_visit": age, "moca": score})
    moca = pd.DataFrame(rows)

    params = fit_eb_params(moca, {1})
    out = eb_slopes(moca, {1}, params)

    assert 1 in out.index
    assert out.loc[1, "n_visits"] == 5
    assert np.isclose(out.loc[1, "span_years"], 64.5 - 60.0, atol=1e-6)


# ===========================================================================
# Test 6: CV caller pattern — train params not updated by test call
# ===========================================================================

def test_cv_caller_pattern_train_params_unchanged():
    """
    The standard CV usage pattern:
        p = fit_eb_params(moca, train_ids)
        train_y = eb_slopes(moca, train_ids, p)
        test_y  = eb_slopes(moca, test_ids, p)   # test never informs mu/tau2

    Calling eb_slopes on test_ids must not change the params dict.
    """
    moca = make_moca(n_subjects=20)
    all_ids = list(moca["PATNO"].unique())
    train_ids = set(all_ids[:10])
    test_ids = set(all_ids[10:])

    params = fit_eb_params(moca, train_ids)
    mu_before = params["mu"]
    tau2_before = params["tau2"]

    _ = eb_slopes(moca, train_ids, params)
    _ = eb_slopes(moca, test_ids, params)

    assert params["mu"] == mu_before, "eb_slopes mutated params['mu']"
    assert params["tau2"] == tau2_before, "eb_slopes mutated params['tau2']"


# ===========================================================================
# Test 7 (M3): tau2=0 collapses to population mean mu
# ===========================================================================

def test_tau2_zero_collapses_to_mu():
    """
    When fit_eb_params receives a single eligible subject, tau2=0 (method of
    moments needs >=2 subjects for Var(ddof=1)). eb_slopes for that subject
    must then return eb_slope == params['mu'], i.e. full collapse to the
    population mean with no subject-specific adjustment.
    """
    rows = []
    for i, (age, score) in enumerate(
        zip([60.0, 61.0, 62.0, 63.0, 64.0], [27.0, 26.5, 26.0, 25.5, 25.0])
    ):
        rows.append(
            {"PATNO": 1, "EVENT_ID": f"V{i}", "age_at_visit": age, "moca": score}
        )
    moca = pd.DataFrame(rows)

    params = fit_eb_params(moca, {1})
    assert params["tau2"] == 0.0, (
        f"Expected tau2=0 for single eligible subject, got {params['tau2']}"
    )
    assert params["n_used"] == 1

    out = eb_slopes(moca, {1}, params)
    assert 1 in out.index, "Single eligible subject must appear in eb_slopes output"
    assert np.isclose(out.loc[1, "eb_slope"], params["mu"], atol=1e-12), (
        f"tau2=0: expected eb_slope={params['mu']:.10f} (mu), "
        f"got {out.loc[1, 'eb_slope']:.10f}"
    )


# ===========================================================================
# Test 8 (M4): eligibility span threshold boundary
# ===========================================================================

def test_eligibility_span_boundary():
    """
    Subject with span just under 3.0 yr (2.999) is excluded regardless of
    visit count. Subject with span just over 3.0 yr (3.001) and >=4 visits
    is included.
    """
    rows = []

    # Subject 1: 4 visits, span = 2.999 yr — excluded (span < MIN_SPAN=3.0)
    for i, (age, score) in enumerate(
        zip([60.0, 60.5, 61.0, 62.999], [27.0, 26.8, 26.5, 26.0])
    ):
        rows.append({"PATNO": 1, "EVENT_ID": f"V{i}", "age_at_visit": age, "moca": score})

    # Subject 2: 4 visits, span = 3.001 yr — included
    for i, (age, score) in enumerate(
        zip([60.0, 60.5, 61.0, 63.001], [27.0, 26.8, 26.5, 26.0])
    ):
        rows.append({"PATNO": 2, "EVENT_ID": f"V{i}", "age_at_visit": age, "moca": score})

    moca = pd.DataFrame(rows)

    params = fit_eb_params(moca, {1, 2})
    assert params["n_used"] == 1, (
        f"Only subject 2 (span>3.0) should be eligible; got n_used={params['n_used']}"
    )

    out = eb_slopes(moca, {1, 2}, params)
    assert 1 not in out.index, "Subject 1 (span=2.999 yr) must be excluded"
    assert 2 in out.index, "Subject 2 (span=3.001 yr, 4 visits) must be included"
