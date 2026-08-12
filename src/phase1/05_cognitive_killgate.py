"""
Phase 1, step 05: COGNITIVE kill-gate (the BBB pivot).

Q-albumin biology is strong at the cognitive/dementia end of the Lewy-body spectrum
(PDD > PD, SMD 0.482) and null at the motor end (our step-04 finding). Test whether
Q-albumin predicts PD COGNITIVE decline, on existing PPMI data.

Two outcomes:
  A. continuous MoCA EB-shrunken slope (decline rate; lower MoCA = worse).
  B. incident conversion to cognitive impairment (cogstate>=2 = PD-MCI/PDD) among
     baseline cognitively-normal PD.
Honest test: does Q-albumin add OVER age + education + baseline cognition + NfL
(Qalb co-tracks age and NfL, so that is the bar). Penalized models, nested CV,
bootstrap CIs. Pure-numpy metrics (env's sklearn metrics crash).
"""
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold

rng = np.random.default_rng(42)
OUT = "data/processed/phase1"
cut = pd.read_parquet(os.path.join(OUT, "curated_cut.parquet"))
qa = pd.read_csv(os.path.join(OUT, "qalbumin.csv")).set_index("PATNO")
pdp = cut[cut["COHORT"] == 1].copy()
bl = pdp[pdp["EVENT_ID"] == "BL"].set_index("PATNO")

def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - p) ** 2).sum() / ss if ss > 0 else np.nan
def auc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return np.nan
    r = np.empty(len(s)); r[s.argsort()] = np.arange(1, len(s) + 1)
    return (r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
def ols_slope(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    tc = t - t.mean(); sxx = (tc ** 2).sum()
    if sxx <= 0 or len(t) < 3: return np.nan, np.nan
    b = (tc * (y - y.mean())).sum() / sxx
    s2 = ((y - (y.mean() + b * tc)) ** 2).sum() / (len(t) - 2)
    return b, s2 / sxx

# ---- Outcome A: MoCA EB slope ----
d = pdp.dropna(subset=["moca", "age_at_visit"])
g = d.groupby("PATNO")
keep = g["EVENT_ID"].nunique()[lambda s: s >= 4].index
keep = [p for p in keep if d.loc[d.PATNO == p, "age_at_visit"].agg(lambda s: s.max()-s.min()) >= 3]
bs, vs, pats = [], [], []
for p in keep:
    s = d[d.PATNO == p]
    b, v = ols_slope(s["age_at_visit"], s["moca"])
    if np.isfinite(v) and v > 0: bs.append(b); vs.append(v); pats.append(p)
bs, vs = np.array(bs), np.array(vs)
tau2 = max(0.0, bs.var(ddof=1) - vs.mean()); w = tau2 / (tau2 + vs)
moca_slope = pd.Series(bs.mean() + w * (bs - bs.mean()), index=pats, name="moca_slope")

# ---- Outcome B: incident cognitive impairment (cogstate>=2) among baseline-normal ----
base_state = bl["cogstate"]
ever_impaired = pdp.groupby("PATNO")["cogstate"].max()
conv = pd.Series(index=base_state.index, dtype=float)
normal_bl = base_state[base_state == 1].index
conv.loc[normal_bl] = (ever_impaired.reindex(normal_bl) >= 2).astype(float)

# ---- features ----
def coalesce(*c):
    s = bl[c[0]].copy()
    for x in c[1:]: s = s.fillna(bl[x])
    return s
F = pd.DataFrame(index=bl.index)
F["age"] = bl["age"]; F["sex"] = pd.factorize(bl["SEX"])[0]; F["educ"] = bl["EDUCYRS"]
F["bl_moca"] = bl["moca"]; F["duration"] = bl["duration_yrs"]
F["saa_pos"] = (bl["CSFSAA"] == 1).astype(float)
F["nfl"] = coalesce("NFL_CSF", "nfl_serum"); F["ptau"] = coalesce("ptau", "IU_pTau181_CSF")
F["abeta"] = coalesce("abeta", "IU_ABeta42_CSF"); F["asyn"] = bl["asyn"]
F["qalb"] = qa["qalb"].reindex(F.index)
m = F[["age", "qalb"]].dropna()
F.loc[m.index, "qalb_resid"] = m["qalb"] - LinearRegression().fit(m[["age"]], m["qalb"]).predict(m[["age"]])

CLIN = ["age", "sex", "educ", "bl_moca", "duration", "saa_pos"]
CSFB = ["nfl", "ptau", "abeta", "asyn"]
QALB = ["qalb_resid"]
qalb_pd = set(qa.index) & set(bl.index)

def ridge(): return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                              ("m", RidgeCV(alphas=np.logspace(-3, 3, 25)))])
def logit(): return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                              ("m", LogisticRegression(C=0.3, max_iter=2000))])

def boot(y, pa, pb, fn, n=2000):
    y = np.asarray(y); idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        try: d.append(fn(y[b], pb[b]) - fn(y[b], pa[b]))
        except Exception: pass
    return np.median(d), np.percentile(d, 2.5), np.percentile(d, 97.5)

def run_cont(name, ids, blocks):
    sub = [p for p in ids if p in moca_slope.index]
    df = F.loc[sub].join(moca_slope).dropna(subset=["moca_slope"] + sum(blocks.values(), []))
    y = df["moca_slope"].to_numpy()
    print(f"\n== {name}  MoCA-slope (n={len(df)}) ==")
    preds = {}
    for lab, feats in blocks.items():
        p = cross_val_predict(ridge(), df[feats].to_numpy(), y, cv=KFold(10, shuffle=True, random_state=42))
        preds[lab] = p; print(f"   {lab:14s} R2={r2(y,p):+.3f}")
    labs = list(blocks)
    for i in range(1, len(labs)):
        md, lo, hi = boot(y, preds[labs[i-1]], preds[labs[i]], r2)
        s = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        print(f"   dR2[{labs[i]} - {labs[i-1]}]: {md:+.3f} [{lo:+.3f},{hi:+.3f}]{s}")

def run_conv(name, ids, blocks):
    sub = [p for p in ids if p in conv.index and np.isfinite(conv.get(p, np.nan))]
    df = F.loc[sub].join(conv.rename("y")).dropna(subset=["y"] + sum(blocks.values(), []))
    y = df["y"].to_numpy().astype(int)
    if y.sum() < 10 or (1 - y).sum() < 10:
        print(f"\n== {name}  conversion (n={len(df)}, converters={int(y.sum())}) -- too few, skip ==="); return
    print(f"\n== {name}  conversion to PD-MCI/PDD (n={len(df)}, converters={int(y.sum())}) ==")
    preds = {}
    for lab, feats in blocks.items():
        p = cross_val_predict(logit(), df[feats].to_numpy(), y, cv=StratifiedKFold(10, shuffle=True, random_state=42),
                              method="predict_proba")[:, 1]
        preds[lab] = p; print(f"   {lab:14s} AUC={auc(y,p):.3f}")
    labs = list(blocks)
    for i in range(1, len(labs)):
        md, lo, hi = boot(y, preds[labs[i-1]], preds[labs[i]], auc)
        s = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        print(f"   dAUC[{labs[i]} - {labs[i-1]}]: {md:+.3f} [{lo:+.3f},{hi:+.3f}]{s}")

setB = sorted(set(bl.index))
setA = sorted(qalb_pd)
print("COGNITIVE KILL-GATE. Honest bar: Q-albumin must add OVER age+educ+baseline-MoCA+NfL.")
run_cont("Set B GENERAL", setB, {"clinical": CLIN, "+CSF/NfL": CLIN+CSFB})
run_cont("Set A BBB", setA, {"clinical": CLIN, "+CSF/NfL": CLIN+CSFB, "+Q-albumin": CLIN+CSFB+QALB})
run_conv("Set B GENERAL", setB, {"clinical": CLIN, "+CSF/NfL": CLIN+CSFB})
run_conv("Set A BBB", setA, {"clinical": CLIN, "+CSF/NfL": CLIN+CSFB, "+Q-albumin": CLIN+CSFB+QALB})
print("\nGREEN if Q-albumin dR2/dAUC CI excludes 0; AMBER if positive but CI includes 0; RED if null.")
