"""
Phase 1, step 06: USE the PPMI NULISA CSF proteomics (the data the user downloaded).
Direct observational test of MEASURED CSF soluble PDGFRβ (the brain-relevant BBB-injury
marker; the plasma-genetic MR was null) in PD:
  (1) does CSF sPDGFRβ predict cognitive decline (MoCA EB-slope) OVER age+educ+baseline-MoCA+NfL?
  (2) how does sPDGFRβ co-track NfL, synuclein species, and age?
NULISA NPQ values; baseline visit; Ridge nested CV; pure-numpy metrics.
"""
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold

rng = np.random.default_rng(42)
OUT = "data/processed/phase1"
NUL = ("data/raw/ALL Proteomic Analysis/converted/"
       "PPMI_Project_282_NULISAseq_CNSDiseasePanel_NPQCounts_20260120.csv")
TARGETS = ["PDGFRB", "NEFL", "GFAP", "ICAM1", "VCAM1", "Oligo-SNCA", "pSNCA-129", "S100B", "VEGFA"]

def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - p) ** 2).sum() / ss if ss > 0 else np.nan

# ---- NULISA baseline CSF matrix ----
n = pd.read_csv(NUL, low_memory=False)
n = n[(n["SampleType"] == "Sample") & (n["SampleQC"] == "passed") & (n["Target"].isin(TARGETS))]
n["PATNO"] = pd.to_numeric(n["PATNO"], errors="coerce")
n = n.dropna(subset=["PATNO"]); n["PATNO"] = n["PATNO"].astype(int)
print("CLINICAL_EVENT values:", n["CLINICAL_EVENT"].value_counts().head(8).to_dict())
# baseline = BL if present else earliest event
ev_order = {"BL": 0, "SC": -1, "V02": 2, "V04": 4, "V06": 6, "V08": 8}
n["evr"] = n["CLINICAL_EVENT"].map(ev_order).fillna(99)
n = n.sort_values(["PATNO", "Target", "evr"])
base = n.groupby(["PATNO", "Target"]).first().reset_index()
mat = base.pivot(index="PATNO", columns="Target", values="NPQ")
print(f"NULISA baseline CSF matrix: {mat.shape[0]} subjects x {mat.shape[1]} targets")

# ---- cognitive outcome (MoCA EB-slope) among PD ----
cut = pd.read_parquet(os.path.join(OUT, "curated_cut.parquet"))
pdp = cut[cut["COHORT"] == 1]
bl = pdp[pdp["EVENT_ID"] == "BL"].set_index("PATNO")
def ols_slope(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    tc = t - t.mean(); sxx = (tc ** 2).sum()
    if sxx <= 0 or len(t) < 3: return np.nan, np.nan
    b = (tc * (y - y.mean())).sum() / sxx
    s2 = ((y - (y.mean() + b * tc)) ** 2).sum() / (len(t) - 2)
    return b, s2 / sxx
d = pdp.dropna(subset=["moca", "age_at_visit"])
keep = [p for p, g in d.groupby("PATNO") if g["EVENT_ID"].nunique() >= 4
        and g["age_at_visit"].max() - g["age_at_visit"].min() >= 3]
bs, vs, pats = [], [], []
for p in keep:
    g = d[d.PATNO == p]; b, v = ols_slope(g["age_at_visit"], g["moca"])
    if np.isfinite(v) and v > 0: bs.append(b); vs.append(v); pats.append(p)
bs, vs = np.array(bs), np.array(vs)
tau2 = max(0.0, bs.var(ddof=1) - vs.mean()); w = tau2 / (tau2 + vs)
moca = pd.Series(bs.mean() + w * (bs - bs.mean()), index=pats, name="moca_slope")

# ---- assemble PD analysis frame ----
F = mat.join(moca, how="inner")
F["age"] = bl["age"]; F["educ"] = bl["EDUCYRS"]; F["bl_moca"] = bl["moca"]
F = F.dropna(subset=["moca_slope"])
pd_n = F.shape[0]
print(f"\nPD subjects with CSF NULISA + MoCA-slope outcome: {pd_n}")

# ---- co-tracking correlations (honest context) ----
print("\n--- sPDGFRβ co-tracking (Pearson r) ---")
for t in ["NEFL", "age", "Oligo-SNCA", "pSNCA-129", "GFAP", "ICAM1"]:
    sub = F[["PDGFRB", t]].dropna()
    r = np.corrcoef(sub["PDGFRB"], sub[t])[0, 1] if len(sub) > 5 else np.nan
    print(f"  PDGFRB vs {t:10s}: r={r:+.2f}  (n={len(sub)})")

# ---- incremental test: does CSF sPDGFRβ add over age+educ+baseline-MoCA+NfL? ----
def oof(X, y):
    return cross_val_predict(Pipeline([("i", SimpleImputer(strategy="median")),
        ("s", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-3, 3, 25)))]),
        X, y, cv=KFold(10, shuffle=True, random_state=42))
def boot(y, pa, pb, n=2000):
    y = np.asarray(y); idx = np.arange(len(y)); dl = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        dl.append(r2(y[b], pb[b]) - r2(y[b], pa[b]))
    return np.median(dl), np.percentile(dl, 2.5), np.percentile(dl, 97.5)

BASE = ["age", "educ", "bl_moca", "NEFL"]
dd = F[["moca_slope"] + BASE + ["PDGFRB"]].dropna()
y = dd["moca_slope"].to_numpy()
print(f"\n--- MoCA-slope prediction (n={len(dd)}) ---")
p0 = oof(dd[BASE].to_numpy(), y); p1 = oof(dd[BASE + ["PDGFRB"]].to_numpy(), y)
print(f"  base (age+educ+blMoCA+NfL):  R2={r2(y,p0):+.3f}")
print(f"  + CSF sPDGFRβ:               R2={r2(y,p1):+.3f}")
md, lo, hi = boot(y, p0, p1)
flag = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
print(f"  dR2 [sPDGFRβ increment]: {md:+.3f} [95%CI {lo:+.3f},{hi:+.3f}]{flag}")
print("\nBar: sPDGFRβ must add OVER NfL+age+baseline cognition. CI excludes 0 = BBB headline alive.")
