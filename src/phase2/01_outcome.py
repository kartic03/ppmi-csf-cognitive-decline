"""
Phase 2, step 01: leakage-safe EB MoCA-slope outcome (task B1).

Primary outcome for the npj PD prediction pillar: per-subject annualized MoCA
slope estimated via Empirical Bayes (EB) shrinkage.

Unlike the Phase 1 implementation, the shrinkage parameters (population mean
slope mu and between-subject variance tau2) are estimated from a TRAINING
subset only. A CV caller refit them inside each outer fold:

    p          = fit_eb_params(moca, train_ids)
    train_y    = eb_slopes(moca, train_ids, p)
    test_y     = eb_slopes(moca, test_ids, p)   # test never informs mu/tau2

OLS slopes b_i are computed from each subject's own visits only (fold-safe by
construction). Only mu and tau2 need fold-localization.

Eligibility: >=4 MoCA visits with non-missing age_at_visit AND >=3.0 yr span
between first and last visit (matches B0 report and existing Phase 1 scripts).

Sensitivity outcomes (to verify EB shrinkage is not manufacturing signal):
    ols_slope           : unshrunken per-subject OLS slope
    slope_excl_baseline : slope refit excluding the earliest visit

Intercept-slope correlation is reported as a diagnostic for baseline-level
coupling. Time-centering within each subject before OLS breaks this coupling.
"""
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root; overridden by callers passing parquet_path)
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_IN = os.path.join(_ROOT, "data", "processed", "phase1", "curated_cut.parquet")
_DATA_OUT = os.path.join(_ROOT, "data", "processed", "phase2", "outcome.parquet")

# ---------------------------------------------------------------------------
# Eligibility thresholds (must match B0 report)
# ---------------------------------------------------------------------------
MIN_VISITS: int = 4
MIN_SPAN: float = 3.0  # years between first and last visit


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ols_slope_centered(ages, scores):
    """
    OLS annualized slope and its sampling variance using time-centered ages.

    Time-centering (subtracting the subject-specific mean age) decouples the
    intercept from the slope so that a high baseline level does not spuriously
    pull the shrinkage target toward zero.

    Parameters
    ----------
    ages   : array-like of floats  (age at each visit, in years)
    scores : array-like of floats  (MoCA score at each visit)

    Returns
    -------
    (slope, var_slope) or (nan, nan) if data are insufficient or degenerate.
    """
    t = np.asarray(ages, dtype=float)
    y = np.asarray(scores, dtype=float)
    n = len(t)
    if n < 3:
        return np.nan, np.nan
    tc = t - t.mean()          # time-center within subject
    Sxx = (tc ** 2).sum()
    if Sxx <= 0.0:
        return np.nan, np.nan
    b = (tc * (y - y.mean())).sum() / Sxx
    resid = y - (y.mean() + b * tc)
    s2 = (resid ** 2).sum() / (n - 2)
    return float(b), float(s2 / Sxx)


def _apply_eligibility(moca_long, subject_ids):
    """
    Filter moca_long to rows for subject_ids, then apply the eligibility rule.

    Returns
    -------
    sub      : DataFrame of eligible rows only
    elig_set : set of eligible PATNOs
    """
    ids = set(subject_ids)
    sub = moca_long[moca_long["PATNO"].isin(ids)].copy()
    sub = sub.dropna(subset=["PATNO", "age_at_visit", "moca"])
    if sub.empty:
        return sub, set()
    g = sub.groupby("PATNO")
    nvis = g["EVENT_ID"].nunique()
    span = g["age_at_visit"].agg(lambda s: s.max() - s.min())
    keep = nvis.index[(nvis >= MIN_VISITS) & (span >= MIN_SPAN)]
    elig = sub[sub["PATNO"].isin(keep)]
    return elig, set(keep)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_eb_params(moca_long, subject_ids):
    """
    Estimate EB shrinkage parameters from ONLY the given subjects.

    This is the fold-safe estimator. Pass training-fold subject IDs; the
    function touches only those rows in moca_long (rows for other subjects
    are ignored even if present in the DataFrame).

    Parameters
    ----------
    moca_long   : pd.DataFrame
        Longitudinal MoCA table with columns:
        PATNO (int), EVENT_ID (str), age_at_visit (float), moca (float).
    subject_ids : array-like
        PATNOs whose data may be used (e.g. training-fold subjects).

    Returns
    -------
    dict with keys:
        mu           : grand mean of OLS slopes across eligible subjects
        tau2         : between-subject slope variance (method of moments, >=0)
        residual_var : mean within-subject residual variance (diagnostic)
        n_used       : number of subjects contributing to the estimate
    """
    sub, _ = _apply_eligibility(moca_long, subject_ids)

    bs, vs = [], []
    for _, s in sub.groupby("PATNO"):
        b, v = _ols_slope_centered(s["age_at_visit"], s["moca"])
        # v=0 with tau2>0: w = tau2/(tau2+0) = 1, no shrinkage (OLS slope exact).
        # tau2=0 regardless of v: w=0 (collapse to population mean mu).
        if np.isfinite(b) and np.isfinite(v):
            bs.append(b)
            vs.append(v)

    if len(bs) == 0:
        return {"mu": 0.0, "tau2": 0.0, "residual_var": np.nan, "n_used": 0}

    bs = np.asarray(bs)
    vs = np.asarray(vs)
    mu = float(bs.mean())
    # Method-of-moments: tau2 = Var(b_i) - mean(v_i), clamped at 0
    # Need at least 2 subjects for Var(ddof=1); single-subject => tau2 = 0
    if len(bs) >= 2:
        tau2 = float(max(0.0, float(bs.var(ddof=1)) - float(vs.mean())))
    else:
        tau2 = 0.0
    residual_var = float(vs.mean())

    return {"mu": mu, "tau2": tau2, "residual_var": residual_var, "n_used": len(bs)}


def eb_slopes(moca_long, subject_ids, eb_params):
    """
    Compute per-subject EB-shrunken annualized MoCA slopes.

    Uses the shrinkage parameters in eb_params (NOT re-estimated from subject_ids).
    This is safe for test-fold subjects: their own visit data drives b_i and v_i,
    but mu and tau2 come from the training fold only.

    Parameters
    ----------
    moca_long   : pd.DataFrame
        Longitudinal MoCA table (same schema as fit_eb_params).
    subject_ids : array-like
        PATNOs for whom slopes are computed.
    eb_params   : dict
        Output of fit_eb_params. Must contain 'mu' and 'tau2'.

    Returns
    -------
    pd.DataFrame indexed by subject_id with columns:
        eb_slope            : EB-shrunken annualized slope (primary outcome)
        ols_slope           : unshrunken per-subject OLS slope (sensitivity)
        slope_excl_baseline : slope refit without the earliest visit (sensitivity)
        n_visits            : number of visits used
        span_years          : age range between first and last visit
    Only subjects passing the eligibility rule are included.
    """
    mu = eb_params["mu"]
    tau2 = eb_params["tau2"]

    sub, _ = _apply_eligibility(moca_long, subject_ids)

    rows = []
    for pat, s in sub.groupby("PATNO"):
        s = s.sort_values("age_at_visit").reset_index(drop=True)
        ages = s["age_at_visit"].to_numpy(float)
        scores = s["moca"].to_numpy(float)
        n = len(ages)
        span = ages[-1] - ages[0]

        # Full-record OLS slope (time-centered)
        b, v = _ols_slope_centered(ages, scores)
        # v=0 and tau2>0: w=1, no shrinkage (OLS slope exact). tau2=0: w=0 (collapse to mu).
        if not (np.isfinite(b) and np.isfinite(v)):
            continue

        # EB shrinkage using PASSED params only
        w = tau2 / (tau2 + v) if (tau2 + v) > 0.0 else 0.0
        eb = mu + w * (b - mu)

        # Sensitivity: slope excluding the earliest visit
        # After removing one visit, n-1 >= 3 (guaranteed by eligibility n >= 4)
        b_excl, _ = _ols_slope_centered(ages[1:], scores[1:])

        rows.append({
            "subject_id": pat,
            "eb_slope": eb,
            "ols_slope": b,
            "slope_excl_baseline": b_excl,
            "n_visits": n,
            "span_years": span,
        })

    if not rows:
        return pd.DataFrame(
            columns=["subject_id", "eb_slope", "ols_slope",
                     "slope_excl_baseline", "n_visits", "span_years"]
        ).set_index("subject_id")

    out = pd.DataFrame(rows).set_index("subject_id")
    return out


def intercept_slope_correlation(moca_long, subject_ids=None):
    """
    Pearson correlation between per-subject mean MoCA (intercept at centered
    time) and OLS slope, across all eligible subjects.

    A negative correlation indicates that subjects with higher baseline MoCA
    have steeper declines (regression to the mean). Time-centering the ages
    before fitting isolates the slope from the intercept, mitigating this
    coupling in the EB shrinkage step.

    Parameters
    ----------
    moca_long   : pd.DataFrame
    subject_ids : array-like or None
        If None, uses all PATNOs in moca_long.

    Returns
    -------
    (pearson_r, n_subjects)
    """
    if subject_ids is None:
        subject_ids = moca_long["PATNO"].unique()

    sub, _ = _apply_eligibility(moca_long, subject_ids)

    intercepts, slopes = [], []
    for _, s in sub.groupby("PATNO"):
        ages = s["age_at_visit"].to_numpy(float)
        scores = s["moca"].to_numpy(float)
        b, v = _ols_slope_centered(ages, scores)
        if np.isfinite(b) and np.isfinite(v):
            intercepts.append(float(scores.mean()))  # mean level at centered time
            slopes.append(b)

    n = len(slopes)
    if n < 3:
        return np.nan, n

    r = float(np.corrcoef(np.asarray(intercepts), np.asarray(slopes))[0, 1])
    return r, n


def load_moca(parquet_path=None):
    """
    Load and filter the PD MoCA longitudinal table from curated_cut.parquet.

    Returns the columns needed for outcome computation:
    PATNO, EVENT_ID, age_at_visit, moca.
    """
    path = parquet_path if parquet_path is not None else _DATA_IN
    cut = pd.read_parquet(path)
    pd_cut = (
        cut[cut["COHORT"] == 1]
        .dropna(subset=["moca", "age_at_visit"])[
            ["PATNO", "EVENT_ID", "age_at_visit", "moca"]
        ]
        .copy()
    )
    return pd_cut


def main():
    """
    Build the full-cohort outcome table (for inspection) and write
    data/processed/phase2/outcome.parquet.

    In production modeling (B2-B4), callers refit EB params inside each
    outer CV fold using fit_eb_params(moca, train_ids). This main() uses
    the full cohort only for inspection and the written parquet.
    """
    moca = load_moca()
    all_pd_ids = moca["PATNO"].unique()

    # Full-cohort EB params (diagnostic; CV callers refit on train-fold only)
    params = fit_eb_params(moca, all_pd_ids)
    print(
        f"Full-cohort EB params: mu={params['mu']:.4f} MoCA/yr, "
        f"tau2={params['tau2']:.4f}, residual_var={params['residual_var']:.4f}, "
        f"n_eligible={params['n_used']}"
    )

    # Intercept-slope correlation (diagnostic for baseline coupling)
    r, n_corr = intercept_slope_correlation(moca, all_pd_ids)
    print(
        f"\nIntercept-slope correlation (n={n_corr}): r={r:.4f}"
    )
    if r < 0:
        print(
            "  Negative: higher baseline MoCA associates with steeper decline "
            "(regression-to-mean). Time-centering isolates slope from intercept."
        )
    else:
        print("  Near-zero or positive: minimal baseline coupling in this cohort.")

    # Compute EB slopes for all eligible subjects
    out = eb_slopes(moca, all_pd_ids, params)
    n_analytic = len(out)
    print(f"\nAnalytic N (slope-eligible, valid EB): {n_analytic}")
    print(
        f"EB slope : mean={out['eb_slope'].mean():.4f}, "
        f"SD={out['eb_slope'].std():.4f} MoCA/yr"
    )
    print(
        f"OLS slope: mean={out['ols_slope'].mean():.4f}, "
        f"SD={out['ols_slope'].std():.4f} MoCA/yr"
    )
    shrinkage_check = (out["eb_slope"] - out["ols_slope"]).abs().describe()
    print(f"  |EB - OLS| max={shrinkage_check['max']:.4f}, "
          f"mean={shrinkage_check['mean']:.4f}  (shrinkage is non-trivial)")

    # Write output
    os.makedirs(os.path.dirname(_DATA_OUT), exist_ok=True)
    out.reset_index().to_parquet(_DATA_OUT, index=False)
    print(f"\nWrote {_DATA_OUT}  ({n_analytic} subjects)")

    return out, params, r


if __name__ == "__main__":
    main()
