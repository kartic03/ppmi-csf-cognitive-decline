"""Why did the APOE e4 positive control fail?

The rank-11 run gave APOE e4 a dR2 of -0.0003 on the EB MoCA slope. APOE e4 is
among the most replicated predictors of cognitive decline in PD, so a flat
result is either:

  (A) the OUTCOME is insensitive -- the EB MoCA slope over this window cannot
      register a known effect, which would limit how much the negative controls
      in the same run are worth; or
  (B) the EFFECT IS ENDPOINT-SPECIFIC -- APOE acts on conversion to dementia
      rather than on within-range MoCA slope in early PD, and its effect grows
      with follow-up length.

These have opposite implications for the manuscript, so distinguish them rather
than pick one. Tests:
  1. APOE e4 on the BINARY conversion endpoint (where the literature's effect
     actually lives). Fires -> (B). Flat -> (A).
  2. APOE e4 dose (0/1/2) rather than carrier status, in case dichotomising
     threw the signal away.
  3. Restricted to long-follow-up subjects, where APOE has room to act.
  4. Unadjusted association with both endpoints -- if APOE shows NO marginal
     association at all in this cohort, neither (A) nor (B) applies and the
     cohort simply does not reproduce the literature effect.
"""
import os, sys, importlib.util, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = os.path.expanduser("~/pd_repro")
spec = importlib.util.spec_from_file_location(
    "inc", os.path.join(ROOT, "src/phase2/03_increment.py"))
inc = importlib.util.module_from_spec(spec); sys.modules["inc"] = inc
spec.loader.exec_module(inc)
SEEDS = range(10)


def cv_auc(X, y):
    a = []
    for s in SEEDS:
        oof = np.full(len(y), np.nan)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            p = Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler()),
                          ("m", LogisticRegression(max_iter=5000))])
            p.fit(X.iloc[tr], y.iloc[tr]); oof[te] = p.predict_proba(X.iloc[te])[:, 1]
        a.append(roc_auc_score(y, oof))
    return float(np.mean(a))


cur = pd.read_parquet(inc._CURATED)
pdf = cur[cur["COHORT"] == 1]
bl = pdf[pdf["EVENT_ID"] == "BL"].copy()
bl["gba_status"] = bl["subgroup"].str.contains("GBA", na=False).astype(float)
bl = bl.set_index("PATNO"); bl.index = bl.index.astype(int)

# binary conversion endpoint (same definition as 04_calibration / the refit)
norm = bl[bl["cogstate"] == 1].index
conv = set(pdf[(pdf["PATNO"].isin(norm)) & (pdf["EVENT_ID"] != "BL")
               & (pdf["cogstate"] >= 2)]["PATNO"].astype(int).unique())
y_bin = pd.Series([1.0 if i in conv else 0.0 for i in norm], index=norm, dtype=float)

# continuous outcome
outc = pd.read_parquet(inc._OUTCOME).set_index("subject_id")
slope = outc.iloc[:, 0] if "eb_slope" not in outc.columns else outc["eb_slope"]
slope.index = slope.index.astype(int)

nvis = pdf.groupby("PATNO")["EVENT_ID"].nunique()
nvis.index = nvis.index.astype(int)

print("=" * 84)
print("4. MARGINAL ASSOCIATION — does this cohort reproduce the effect at all?")
print("=" * 84)
for name, var in [("APOE e4 carrier", (bl["APOE_e4"] > 0).astype(float).where(bl["APOE_e4"].notna())),
                  ("APOE e4 dose", bl["APOE_e4"]),
                  ("GBA carrier", bl["gba_status"])]:
    v = var.dropna()
    # continuous
    i = v.index.intersection(slope.index)
    g1, g0 = slope.loc[i][v.loc[i] > 0], slope.loc[i][v.loc[i] == 0]
    if len(g1) > 5 and len(g0) > 5:
        t, p = stats.mannwhitneyu(g1, g0)
        print(f"  {name:16s} vs EB slope    : carriers n={len(g1):4d} "
              f"median {g1.median():+.4f} | non n={len(g0):4d} median "
              f"{g0.median():+.4f}   p={p:.4f}")
    # binary
    j = v.index.intersection(y_bin.index)
    if len(j) > 20:
        vb = (v.loc[j] > 0).astype(int); yb = y_bin.loc[j]
        r1, r0 = yb[vb == 1].mean(), yb[vb == 0].mean()
        tab = pd.crosstab(vb, yb)
        p = stats.fisher_exact(tab)[1] if tab.shape == (2, 2) else np.nan
        print(f"  {name:16s} vs conversion  : carriers {r1:.1%} "
              f"(n={int((vb==1).sum())}) | non {r0:.1%} "
              f"(n={int((vb==0).sum())})   Fisher p={p:.4f}")
    print()

print("=" * 84)
print("1+2. APOE ON THE BINARY ENDPOINT (incremental AUC over clinical)")
print("=" * 84)
CL = [c for c in inc.CLINICAL_COLS if c != "APOE_e4"]
for lbl, col in [("APOE e4 carrier", (bl["APOE_e4"] > 0).astype(float).where(bl["APOE_e4"].notna())),
                 ("APOE e4 dose", bl["APOE_e4"])]:
    X0 = bl[CL]
    idx = y_bin.index.intersection(X0[X0.notna().sum(1) >= len(CL) - 1].index)
    idx = idx.intersection(col.dropna().index)
    yb = y_bin.loc[idx]
    a0 = cv_auc(X0.loc[idx], yb)
    a1 = cv_auc(X0.loc[idx].assign(_v=col.loc[idx]), yb)
    print(f"  {lbl:16s} n={len(idx):4d}  AUC {a0:.4f} -> {a1:.4f}  "
          f"dAUC {a1-a0:+.4f}")

print("\n" + "=" * 84)
print("3. LONG-FOLLOW-UP SUBJECTS ONLY (APOE needs time to act)")
print("=" * 84)
med = nvis.median()
for lbl, keep in [(f"visits > {med:.0f}", nvis[nvis > med].index),
                  (f"visits <= {med:.0f}", nvis[nvis <= med].index)]:
    col = (bl["APOE_e4"] > 0).astype(float).where(bl["APOE_e4"].notna())
    X0 = bl[CL]
    idx = (y_bin.index.intersection(X0[X0.notna().sum(1) >= len(CL) - 1].index)
           .intersection(col.dropna().index).intersection(pd.Index(keep)))
    yb = y_bin.loc[idx]
    if yb.nunique() < 2 or len(yb) < 80:
        print(f"  {lbl:16s} n={len(yb)} — too small"); continue
    a0 = cv_auc(X0.loc[idx], yb)
    a1 = cv_auc(X0.loc[idx].assign(_v=col.loc[idx]), yb)
    print(f"  {lbl:16s} n={len(idx):4d} events={int(yb.sum()):3d}  "
          f"AUC {a0:.4f} -> {a1:.4f}  dAUC {a1-a0:+.4f}")
