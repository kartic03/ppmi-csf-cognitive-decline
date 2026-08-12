"""
14_mr_estimate.py
-----------------
MR estimation per (protein, platform) with null-vs-underpowered decision gate.

Consumes : data/processed/mr/harmonized.parquet
Produces : data/processed/mr/mr_results.json

Algorithm per group
-------------------
1. n_instruments = row count; F = mean (beta_exp/se_exp)**2; method = wald | ivw
2. Wald (n=1): theta = beta_out/beta_exp; se_theta = |se_out/beta_exp|
   IVW fixed-effect (n>=2):
       theta = sum(beta_exp*beta_out/se_out**2) / sum(beta_exp**2/se_out**2)
       se_theta = sqrt(1 / sum(beta_exp**2/se_out**2))
3. OR = exp(theta); 95-pct CI = exp(theta +/- 1.96*se_theta)
4. min_detectable_or_80 = exp((1.96+0.842)*se_theta)   [80% power, alpha=0.05 two-sided]
5. verdict:
   - "positive"     if ci_low > 1.0 OR ci_high < 1.0
   - "null"         elif ci_low > 0.91 AND ci_high < 1.10   (CI tight around 1)
   - "inconclusive" otherwise
6. has_strand_ambiguous = any instrument in group has strand_resolved == False
7. Sensitivity estimators (non-gating, only when n_instruments >= 3):
   - MR-Egger (WLS with intercept, instruments oriented to beta_exp > 0)
   - Weighted-median (bootstrap SE, seed=42)

TIMP1 has zero cis-pQTLs (absent from Le Guen X-GWAS). A sentinel record with
verdict="not_testable" is appended so downstream stages have an explicit entry.
Note: ci_high is set to 0.0 (not JSON null) for the TIMP1 sentinel so that the
verdict-loop test (`ci_high >= 1.10 and ci_low <= 1.0`) short-circuits safely.
"""

import json
import math
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm


INPUT = pathlib.Path("data/processed/mr/harmonized.parquet")
OUTPUT = pathlib.Path("data/processed/mr/mr_results.json")


def _wald(grp: pd.DataFrame):
    row = grp.iloc[0]
    theta = row.beta_out / row.beta_exp
    se_theta = abs(row.se_out / row.beta_exp)
    return theta, se_theta


def _ivw(grp: pd.DataFrame):
    w = grp.beta_exp ** 2 / grp.se_out ** 2
    theta = float(np.sum(grp.beta_exp * grp.beta_out / grp.se_out ** 2) / np.sum(w))
    se_theta = float(math.sqrt(1.0 / float(np.sum(w))))
    return theta, se_theta


def leave_one_out(grp: pd.DataFrame) -> dict:
    """Leave-one-out IVW + per-instrument weight share (requires n >= 2).

    Added 2026-08-04. A cis-MR null is only as trustworthy as its instrument
    set: if one SNP carries almost all the inverse-variance weight, the pooled
    estimate is effectively that SNP restated, and MR-Egger/weighted-median
    "sensitivity" checks inherit the same dominance rather than testing it.
    PDGFRB/deCODE is exactly this case (one instrument at ~93.5% weight), so
    the weight distribution and the LOO estimates are now emitted as first-class
    fields instead of being recoverable only by hand from harmonized.parquet.

    Reports, per instrument: its weight share, its Wald ratio, and the IVW
    estimate with that instrument removed.  Also reports whether every LOO
    estimate keeps the same null/non-null verdict as the full fit, which is the
    property a null claim actually needs.
    """
    n = len(grp)
    if n < 2:
        return {
            "loo_max_weight_frac": None,
            "loo_max_weight_rsid": None,
            "loo_verdict_stable": None,
            "loo": None,
        }

    w = (grp.beta_exp ** 2 / grp.se_out ** 2).to_numpy(float)
    wfrac = w / w.sum()
    rsids = grp["rsid"].astype(str).to_numpy()
    ratios = (grp.beta_out / grp.beta_exp).to_numpy(float)

    theta_all, se_all = _ivw(grp)
    full_excludes_null = not (
        math.exp(theta_all - 1.96 * se_all) < 1.0 < math.exp(theta_all + 1.96 * se_all)
    )

    rows = []
    stable = True
    for i in range(n):
        sub = grp.drop(grp.index[i])
        t, s = _ivw(sub)
        lo, hi = math.exp(t - 1.96 * s), math.exp(t + 1.96 * s)
        excl = not (lo < 1.0 < hi)
        if excl != full_excludes_null:
            stable = False
        rows.append({
            "dropped_rsid": rsids[i],
            "weight_frac": round(float(wfrac[i]), 6),
            "wald_ratio": round(float(ratios[i]), 6),
            "wald_or": round(float(math.exp(ratios[i])), 6),
            "loo_or": round(float(math.exp(t)), 6),
            "loo_ci_low": round(float(lo), 6),
            "loo_ci_high": round(float(hi), 6),
            "loo_excludes_null": bool(excl),
        })

    imax = int(np.argmax(wfrac))
    return {
        "loo_max_weight_frac": round(float(wfrac[imax]), 6),
        "loo_max_weight_rsid": rsids[imax],
        # True = every LOO fit reaches the same null/non-null call as the full
        # fit. This is what licenses a null; a dominant instrument does not by
        # itself invalidate one, but it must be disclosed either way.
        "loo_verdict_stable": bool(stable),
        "loo": rows,
    }


def egger(grp: pd.DataFrame) -> dict:
    """MR-Egger sensitivity estimator (requires n >= 3).

    Orients all instruments so beta_exp > 0 before the regression (standard
    Egger convention).  Returns a dict with slope, slope SE, intercept, and
    intercept p-value.  A small intercept p-value flags directional pleiotropy.
    """
    be = grp.beta_exp.values.copy()
    bo = grp.beta_out.values.copy()
    se = grp.se_out.values.copy()

    # Orient so all exposure betas are positive
    flip = be < 0
    be[flip] = -be[flip]
    bo[flip] = -bo[flip]

    weights = 1.0 / se ** 2
    X = sm.add_constant(be)
    result = sm.WLS(bo, X, weights=weights).fit()

    intercept_idx = 0  # add_constant puts intercept first
    slope_idx = 1

    return {
        "egger_slope": float(result.params[slope_idx]),
        "egger_slope_se": float(result.bse[slope_idx]),
        "egger_intercept": float(result.params[intercept_idx]),
        "egger_intercept_p": float(result.pvalues[intercept_idx]),
    }


def weighted_median(grp: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    """Weighted-median MR estimator (requires n >= 3).

    Per-instrument ratio r_i = beta_out_i / beta_exp_i with weight
    w_i = (beta_exp_i / se_out_i)^2.  Bootstrap SE uses 1000 resamples
    (numpy Generator seeded at 42 for reproducibility).
    """
    be = grp.beta_exp.values.astype(float)
    bo = grp.beta_out.values.astype(float)
    se = grp.se_out.values.astype(float)

    def _wmedian(be_, bo_, se_):
        ratios = bo_ / be_
        w = (be_ / se_) ** 2
        w_norm = w / w.sum()
        order = np.argsort(ratios)
        cumw = np.cumsum(w_norm[order])
        # Find index where cumulative weight first reaches 0.5
        idx = np.searchsorted(cumw, 0.5)
        idx = min(idx, len(ratios) - 1)
        return ratios[order[idx]]

    theta = _wmedian(be, bo, se)

    rng = np.random.default_rng(seed)
    n = len(be)
    boot_thetas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_thetas.append(_wmedian(be[idx], bo[idx], se[idx]))
    wm_se = float(np.std(boot_thetas, ddof=1))

    return {
        "wmedian_theta": float(theta),
        "wmedian_or": float(math.exp(theta)),
        "wmedian_se": wm_se,
    }


def _verdict(ci_low: float, ci_high: float) -> str:
    if ci_low > 1.0 or ci_high < 1.0:
        return "positive"
    if ci_low > 0.91 and ci_high < 1.10:
        return "null"
    return "inconclusive"


def estimate_group(protein: str, platform: str, grp: pd.DataFrame) -> dict:
    n = len(grp)
    assert (grp.beta_exp != 0).all(), (
        f"Zero exposure beta detected in group ({protein}, {platform}) — "
        "cannot compute MR ratio."
    )
    F = float(np.mean((grp.beta_exp / grp.se_exp) ** 2))
    method = "wald" if n == 1 else "ivw"

    theta, se_theta = _wald(grp) if n == 1 else _ivw(grp)

    or_ = math.exp(theta)
    ci_low = math.exp(theta - 1.96 * se_theta)
    ci_high = math.exp(theta + 1.96 * se_theta)
    mdo80 = math.exp((1.96 + 0.842) * se_theta)

    has_strand_ambiguous = bool((~grp.strand_resolved).any())

    rec = {
        "protein": protein,
        "platform": platform,
        "n_instruments": n,
        "F": round(F, 2),
        "method": method,
        "or": round(or_, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "min_detectable_or_80": round(mdo80, 6),
        "verdict": _verdict(ci_low, ci_high),
        "has_strand_ambiguous": has_strand_ambiguous,
    }

    # Instrument-influence diagnostics (n >= 2). Emitted for every group, not
    # just multi-instrument ones, so a dominant-instrument estimate is always
    # visible in the artifact rather than only in harmonized.parquet.
    rec.update(leave_one_out(grp))

    # Sensitivity estimators (non-gating; only meaningful with >= 3 instruments)
    if n >= 3:
        rec.update(egger(grp))
        rec.update(weighted_median(grp))
    else:
        rec.update({
            "egger_slope": None,
            "egger_slope_se": None,
            "egger_intercept": None,
            "egger_intercept_p": None,
            "wmedian_theta": None,
            "wmedian_or": None,
            "wmedian_se": None,
        })

    return rec


def timp1_sentinel() -> dict:
    """
    TIMP1 has no testable cis-pQTLs (absent from Le Guen X-GWAS).
    ci_high is set to 0.0 (a numeric sentinel < 1.10) so that the verdict-loop
    comparison `ci_high >= 1.10` evaluates to False without TypeError.
    All other numeric fields are null.
    """
    return {
        "protein": "TIMP1",
        "platform": "both",
        "n_instruments": 0,
        "F": None,
        "method": None,
        "or": None,
        "ci_low": None,
        "ci_high": 0.0,
        "min_detectable_or_80": None,
        "verdict": "not_testable",
        "has_strand_ambiguous": False,
        "egger_slope": None,
        "egger_slope_se": None,
        "egger_intercept": None,
        "egger_intercept_p": None,
        "wmedian_theta": None,
        "wmedian_or": None,
        "wmedian_se": None,
    }


def main():
    df = pd.read_parquet(INPUT)

    results = []
    for (protein, platform), grp in df.groupby(["protein", "platform"]):
        rec = estimate_group(protein, platform, grp.reset_index(drop=True))
        results.append(rec)

    results.append(timp1_sentinel())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} records to {OUTPUT}")

    # Print results table for inspection
    print()
    print(f"{'Protein':<8} {'Platform':<8} {'N':>3} {'F':>8} {'OR':>6} {'CI_low':>7} {'CI_high':>8} {'MDO80':>7} {'Verdict':<15} {'StrandAmb'}")
    print("-" * 90)
    for r in results:
        if r["verdict"] == "not_testable":
            print(f"{r['protein']:<8} {r['platform']:<8} {'0':>3} {'—':>8} {'—':>6} {'—':>7} {'—':>8} {'—':>7} not_testable")
            continue
        print(
            f"{r['protein']:<8} {r['platform']:<8} {r['n_instruments']:>3} {r['F']:>8.1f} "
            f"{r['or']:>6.3f} {r['ci_low']:>7.3f} {r['ci_high']:>8.3f} {r['min_detectable_or_80']:>7.3f} "
            f"{r['verdict']:<15} {r['has_strand_ambiguous']}"
        )


if __name__ == "__main__":
    main()
