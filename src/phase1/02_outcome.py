"""
Phase 1, step 02: build the leak-free progression OUTCOME (DESIGN_SPEC_v2 Section 3).

Primary outcome = per-subject annualized progression rate, estimated as an
EMPIRICAL-BAYES (shrunken) slope. Two-stage, pure-numpy (statsmodels MixedLM segfaults
intermittently in this env; the manual EB estimator is transparent and reproducible):

  Stage 1: per-subject OLS slope b_i on (age_at_visit, scale), with its sampling
           variance v_i = residual_var / Sxx. Time centred within subject so the
           intercept is the mean level, not the baseline visit (anti-coupling).
  Stage 2: tau^2 = max(0, Var(b_i) - mean(v_i))  (between-subject slope variance, MoM).
           EB slope  = mu + w_i (b_i - mu),  w_i = tau^2 / (tau^2 + v_i)  (shrinkage).
           Reliability = tau^2 / (tau^2 + mean(v_i)).

Shrinkage de-couples a noisy high baseline from a spuriously steep slope (regression to
the mean), the exact issue the review flagged. Baseline severity enters models as a
SEPARATE covariate from the BL visit (never the slope anchor).

Scales: MDS-UPDRS Part II (primary; treatment-robust ADL), Part III + total (sensitivity).
Inclusion: >=4 visits with the scale AND >=3 years follow-up span.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

OUT = "data/processed/phase1"
cut = pd.read_parquet(os.path.join(OUT, "curated_cut.parquet"))
pd_pat = cut.loc[cut["COHORT"] == 1, "PATNO"].unique()
SCALES = {"updrs2_score": "PartII", "updrs3_score": "PartIII", "updrs_totscore": "Total"}
MIN_VISITS, MIN_SPAN = 4, 3.0


def ols_slope(t, y):
    t = np.asarray(t, float); y = np.asarray(y, float)
    tc = t - t.mean(); Sxx = (tc ** 2).sum()
    if Sxx <= 0 or len(t) < 3:
        return np.nan, np.nan
    b = (tc * (y - y.mean())).sum() / Sxx
    resid = y - (y.mean() + b * tc)
    s2 = (resid ** 2).sum() / (len(t) - 2)
    return b, s2 / Sxx


def split_half_r(d, col):
    a, b = [], []
    for _, s in d.sort_values("age_at_visit").groupby("PATNO"):
        t = s["age_at_visit"].to_numpy(float); y = s[col].to_numpy(float)
        be, _ = ols_slope(t[0::2], y[0::2]); bo, _ = ols_slope(t[1::2], y[1::2])
        if np.isfinite(be) and np.isfinite(bo):
            a.append(be); b.append(bo)
    a, b = np.asarray(a), np.asarray(b)
    r = pearsonr(a, b)[0]
    return r, spearmanr(a, b)[0], 2 * r / (1 + r), len(a)


out = pd.DataFrame({"PATNO": pd_pat}).set_index("PATNO")
summary = []
for col, name in SCALES.items():
    d = (cut[cut["PATNO"].isin(pd_pat)]
         .dropna(subset=[col, "age_at_visit"])[["PATNO", "EVENT_ID", "age_at_visit", col]].copy())
    g = d.groupby("PATNO")
    nvis = g["EVENT_ID"].nunique()
    span = g["age_at_visit"].agg(lambda s: s.max() - s.min())
    keep = nvis.index[(nvis >= MIN_VISITS) & (span >= MIN_SPAN)]
    d = d[d["PATNO"].isin(keep)].copy()

    bs, vs, pats = [], [], []
    for pat, s in d.groupby("PATNO"):
        b, v = ols_slope(s["age_at_visit"], s[col])
        if np.isfinite(b) and np.isfinite(v) and v > 0:
            bs.append(b); vs.append(v); pats.append(pat)
    bs, vs = np.asarray(bs), np.asarray(vs)
    mu = bs.mean()
    tau2 = max(0.0, bs.var(ddof=1) - vs.mean())
    w = tau2 / (tau2 + vs)
    eb = mu + w * (bs - mu)
    reliability = tau2 / (tau2 + vs.mean())
    out = out.join(pd.Series(dict(zip(pats, eb)), name=f"ebslope_{name}"))

    rp, rs, sb, nrel = split_half_r(d, col)
    summary.append({"scale": name, "n": len(pats), "mu_per_yr": round(mu, 3),
                    "eb_median": round(float(np.median(eb)), 3),
                    "eb_IQR": f"{np.percentile(eb,25):.2f},{np.percentile(eb,75):.2f}",
                    "reliability_EB": round(reliability, 3),
                    "splithalf_SB": round(sb, 3), "n_rel": nrel})

bl = cut[(cut["PATNO"].isin(pd_pat)) & (cut["EVENT_ID"] == "BL")].set_index("PATNO")
for col, name in SCALES.items():
    out[f"bl_{name}"] = bl[col]
out["bl_age"] = bl["age"]; out["sex"] = bl["SEX"]; out["duration_yrs"] = bl["duration_yrs"]
out.to_csv(os.path.join(OUT, "outcome.csv"))

print("=== Outcome: EB-shrunken annualized progression slopes ===")
print(pd.DataFrame(summary).to_string(index=False))

setA = set(pd.read_csv(os.path.join(OUT, "setA_patnos.csv"))["PATNO"])
elig = set(out["ebslope_PartII"].dropna().index)
datscan = set(bl.index[bl["MIA_PUTAMEN_BILAT"].notna()])
csf = set(bl.index[bl["ptau"].notna() | bl["IU_pTau181_CSF"].notna()])
setB = set(pd_pat) & datscan & csf
print(f"\nPart II outcome-eligible PD: {len(elig)} | Set A: {len(setA & elig)} | Set B: {len(setB & elig)}")
print(f"Wrote outcome.csv ({out['ebslope_PartII'].notna().sum()} Part II slopes).")
