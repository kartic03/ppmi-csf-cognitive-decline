"""
Phase 1, step 04: THE KILL GATE (DESIGN_SPEC_v2 Section 7).

Decisive cheap experiment: does any biomarker fusion beat the realistic clinical+SAA
baseline at predicting the Part II EB-slope, and does Q-albumin add anything?

Models (penalized Ridge, few pre-specified predictors per the power gate; nested CV):
  M0  clinical+SAA   : age, sex, baseline Part II severity, duration, SAA status
  M1  + biomarkers   : + CSF (p-tau, Abeta, asyn, NfL) + DaTSCAN SBR (putamen, caudate)
  M2  + Q-albumin    : + age-residualized Q-albumin   (Set A only)
General question (M0 vs M1) on Set B (n~641); BBB question (M1 vs M2, M0 vs M2) on Set A (n~287).
Metric: out-of-fold R2 and MAE; incremental dR2 with PAIRED bootstrap 95% CI.
Permutation negative control: shuffle the biomarker block, confirm dR2 ~ 0.
Decision: GREEN / AMBER / RED.
"""
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import cross_val_predict, KFold

# pure-numpy metrics (sklearn's r2_score crashes on this env's numpy/array-API path)
def r2_score(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss_res = ((y - p) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
def mean_absolute_error(y, p):
    return float(np.abs(np.asarray(y, float) - np.asarray(p, float)).mean())

rng = np.random.default_rng(42)
OUT = "data/processed/phase1"
cut = pd.read_parquet(os.path.join(OUT, "curated_cut.parquet"))
oc = pd.read_csv(os.path.join(OUT, "outcome.csv")).set_index("PATNO")
qa = pd.read_csv(os.path.join(OUT, "qalbumin.csv")).set_index("PATNO")

# baseline (BL) biomarker frame
bl = cut[cut["EVENT_ID"] == "BL"].set_index("PATNO")
def coalesce(*cols):
    s = bl[cols[0]].copy()
    for c in cols[1:]:
        s = s.fillna(bl[c])
    return s

df = pd.DataFrame(index=oc.index)
df["y"] = oc["ebslope_PartII"]
# clinical+SAA
df["age"] = oc["bl_age"]
df["sex"] = pd.factorize(oc["sex"])[0]
df["bl_sev"] = oc["bl_PartII"]
df["duration"] = oc["duration_yrs"]
df["saa_pos"] = (bl["CSFSAA"] == 1).astype(float).reindex(df.index)
# biomarkers
df["ptau"] = coalesce("ptau", "IU_pTau181_CSF").reindex(df.index)
df["abeta"] = coalesce("abeta", "IU_ABeta42_CSF").reindex(df.index)
df["asyn"] = bl["asyn"].reindex(df.index)
df["nfl"] = coalesce("NFL_CSF", "nfl_serum").reindex(df.index)
df["putamen"] = bl["MIA_PUTAMEN_BILAT"].reindex(df.index)
df["caudate"] = bl["MIA_CAUDATE_BILAT"].reindex(df.index)
# age-residualized Q-albumin
df["qalb"] = qa["qalb"].reindex(df.index)
m = df[["age", "qalb"]].dropna()
lr = LinearRegression().fit(m[["age"]], m["qalb"])
df.loc[m.index, "qalb_resid"] = m["qalb"] - lr.predict(m[["age"]])

CLIN = ["age", "sex", "bl_sev", "duration", "saa_pos"]
BIO = ["ptau", "abeta", "asyn", "nfl", "putamen", "caudate"]
QALB = ["qalb_resid"]

def pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("m", RidgeCV(alphas=np.logspace(-3, 3, 25)))])

def oof(X, y):
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    return cross_val_predict(pipe(), X, y, cv=cv)

def boot_delta(y, pa, pb, n=2000):
    y = np.asarray(y); idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        d.append(r2_score(y[b], pb[b]) - r2_score(y[b], pa[b]))
    d = np.array(d)
    return np.median(d), np.percentile(d, 2.5), np.percentile(d, 97.5)

def run(sub, feats):
    d = df.loc[sub, ["y"] + feats].dropna(subset=["y"])
    y = d["y"].to_numpy()
    p = oof(d[feats], y)
    return y, p, d.index

def report(setname, sub, blocks):
    print(f"\n===== {setname} (n={len(sub)}) =====")
    preds = {}
    common = None
    for label, feats in blocks.items():
        d = df.loc[sub, ["y"] + feats].dropna(subset=["y"] + feats)
        common = d.index if common is None else common.intersection(d.index)
    for label, feats in blocks.items():
        y = df.loc[common, "y"].to_numpy()
        p = oof(df.loc[common, feats], y)
        preds[label] = p
        print(f"  {label:18s} R2={r2_score(y,p):+.3f}  MAE={mean_absolute_error(y,p):.3f}")
    y = df.loc[common, "y"].to_numpy()
    labels = list(blocks)
    for i in range(1, len(labels)):
        a, b = labels[i-1], labels[i]
        md, lo, hi = boot_delta(y, preds[a], preds[b])
        star = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        print(f"  dR2 [{b} - {a}]: {md:+.3f}  95%CI [{lo:+.3f},{hi:+.3f}]{star}")
    return preds, common, y

# Set A / Set B membership (modality availability; y-independent)
setA0 = set(pd.read_csv(os.path.join(OUT, "setA_patnos.csv"))["PATNO"])
datscan = set(bl.index[bl["MIA_PUTAMEN_BILAT"].notna()])
csf = set(bl.index[bl["ptau"].notna() | bl["IU_pTau181_CSF"].notna()])
setB0 = set(cut.loc[cut["COHORT"] == 1, "PATNO"]) & datscan & csf

SCALES = {"ebslope_PartII": "Part II (PRIMARY)", "ebslope_PartIII": "Part III (sens.)",
          "ebslope_Total": "Total (sens.)"}
print("Models = penalized Ridge, nested 10-fold CV. dR2 with paired bootstrap 95% CI.")
for ycol, yname in SCALES.items():
    df["y"] = oc[ycol]
    has_y = set(df.dropna(subset=["y"]).index)
    print(f"\n############ OUTCOME: {yname} ############")
    report(f"Set B GENERAL", sorted(setB0 & has_y),
           {"clinical+SAA": CLIN, "+biomarkers": CLIN + BIO})
    report(f"Set A BBB", sorted(setA0 & has_y),
           {"clinical+SAA": CLIN, "+biomarkers": CLIN + BIO, "+Q-albumin": CLIN + BIO + QALB})

# ---- numpy-array helpers (avoid pandas-indexing fragility in cross_val_predict) ----
def oof_np(X, y, est=None):
    pl = est or pipe()
    return cross_val_predict(pl, X, y, cv=KFold(5, shuffle=True, random_state=1))

def auc_np(y_true, score):  # Mann-Whitney AUC, pure numpy
    y_true = np.asarray(y_true).astype(int); score = np.asarray(score, float)
    pos, neg = score[y_true == 1], score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = score.argsort(); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

df["y"] = oc["ebslope_PartII"]
sub = sorted(setB0 & set(df.dropna(subset=["y"]).index))
d = df.loc[sub, ["y"] + CLIN + BIO].dropna()
yv = d["y"].to_numpy(); Xc = d[CLIN].to_numpy(); Xb = d[CLIN + BIO].to_numpy()
biocols = list(range(len(CLIN), len(CLIN) + len(BIO)))

print("\n--- permutation negative control (Part II, Set B, biomarker block shuffled, 100x) ---")
base_r2 = r2_score(yv, oof_np(Xc, yv))
obs = r2_score(yv, oof_np(Xb, yv)) - base_r2
perm = []
for k in range(100):
    Xp = Xb.copy(); pidx = np.random.default_rng(k).permutation(len(yv))
    Xp[:, biocols] = Xp[pidx][:, biocols]
    perm.append(r2_score(yv, oof_np(Xp, yv)) - base_r2)
perm = np.array(perm)
print(f"  observed dR2={obs:+.3f}; perm null mean={perm.mean():+.3f} sd={perm.std():.3f}; "
      f"p={(perm >= obs).mean():.3f}")

print("\n--- robustness: nonlinear model + rapid/slow AUC (Part II, Set B) ---")
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
hgb = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", HGBR(max_depth=3, max_iter=200,
               learning_rate=0.05, l2_regularization=1.0, random_state=42))])
r2c = r2_score(yv, oof_np(Xc, yv, hgb)); r2b = r2_score(yv, oof_np(Xb, yv, hgb))
print(f"  HistGBM R2: clinical+SAA={r2c:+.3f}  +biomarkers={r2b:+.3f}  dR2={r2b-r2c:+.3f}")
ybin = (yv > np.median(yv)).astype(int)
pc = oof_np(Xc, yv); pb = oof_np(Xb, yv)  # ridge scores as risk
print(f"  rapid/slow AUC (Ridge score): clinical+SAA={auc_np(ybin, pc):.3f}  "
      f"+biomarkers={auc_np(ybin, pb):.3f}")

print("\nGREEN if a biomarker dR2 CI excludes 0 (and perm p<0.05); AMBER if positive but CI"
      " includes 0; RED if no incremental signal.")
