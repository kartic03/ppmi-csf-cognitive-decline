"""
Cognition project, Phase 1 KILL GATE: does the NULISA CSF proteomic panel (132 targets)
add incremental discrimination OVER a clinical + established-CSF baseline for predicting
conversion to PD-MCI/PDD among baseline cognitively-normal PD?
Penalized logistic, nested CV, AUC, paired-bootstrap dAUC + permutation control. Pure-numpy metrics.
"""
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

rng = np.random.default_rng(42)
OUT = "data/processed/phase1"
NUL = ("data/raw/ALL Proteomic Analysis/converted/"
       "PPMI_Project_282_NULISAseq_CNSDiseasePanel_NPQCounts_20260120.csv")

def auc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return np.nan
    r = np.empty(len(s)); r[s.argsort()] = np.arange(1, len(s) + 1)
    return (r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)

# ---- NULISA baseline CSF matrix (all 132 targets) ----
n = pd.read_csv(NUL, usecols=["SampleType", "SampleQC", "PATNO", "CLINICAL_EVENT", "Target", "NPQ"],
                low_memory=False)
n = n[(n["SampleType"] == "Sample") & (n["SampleQC"] == "passed")].copy()
n["PATNO"] = pd.to_numeric(n["PATNO"], errors="coerce")
n = n.dropna(subset=["PATNO"]); n["PATNO"] = n["PATNO"].astype(int)
order = {"BL": 0, "SC": -1, "V02": 2, "V04": 4, "V06": 6, "V08": 8, "V10": 10, "V12": 12}
n["evr"] = n["CLINICAL_EVENT"].map(order).fillna(99)
n = n.sort_values(["PATNO", "Target", "evr"])
nb = n.groupby(["PATNO", "Target"]).first().reset_index()
prot = nb.pivot(index="PATNO", columns="Target", values="NPQ")
PROT_COLS = list(prot.columns)
print(f"NULISA baseline matrix: {prot.shape[0]} subjects x {prot.shape[1]} proteins")

# ---- outcome + clinical + established CSF ----
cut = pd.read_parquet(os.path.join(OUT, "curated_cut.parquet"))
pdp = cut[cut["COHORT"] == 1]; bl = pdp[pdp["EVENT_ID"] == "BL"].set_index("PATNO")
base = bl["cogstate"]; ever = pdp.groupby("PATNO")["cogstate"].max()
normal = base[base == 1].index
conv = pd.Series((ever.reindex(normal) >= 2).astype(float), index=normal, name="y")

def coalesce(*c):
    s = bl[c[0]].copy()
    for x in c[1:]: s = s.fillna(bl[x])
    return s
F = pd.DataFrame(index=bl.index)
F["age"] = bl["age"]; F["sex"] = pd.factorize(bl["SEX"])[0]; F["educ"] = bl["EDUCYRS"]
F["bl_moca"] = bl["moca"]; F["duration"] = bl["duration_yrs"]; F["bl_updrs3"] = bl["updrs3_score"]
F["nfl"] = coalesce("NFL_CSF", "nfl_serum"); F["abeta"] = coalesce("abeta", "IU_ABeta42_CSF")
F["ptau"] = coalesce("ptau", "IU_pTau181_CSF"); F["asyn"] = bl["asyn"]
F["saa_pos"] = (bl["CSFSAA"] == 1).astype(float)
F = F.join(prot).join(conv)
F = F.loc[F.index.isin(normal)].dropna(subset=["y"])
F = F[F.index.isin(prot.index)]  # require proteomics
y = F["y"].to_numpy().astype(int)
print(f"Analytic cohort (baseline-normal PD with NULISA): n={len(F)}, converters={int(y.sum())}")

CLIN = ["age", "sex", "educ", "bl_moca", "duration", "bl_updrs3"]
CSF = ["nfl", "abeta", "ptau", "asyn", "saa_pos"]

def logit(): return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
    ("m", LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc", max_iter=3000))])
def oof(cols, est=None):
    return cross_val_predict(est or logit(), F[cols].to_numpy(), y,
        cv=StratifiedKFold(10, shuffle=True, random_state=42), method="predict_proba")[:, 1]
def bootdiff(pa, pb, n=2000):
    idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        d.append(auc(y[b], pb[b]) - auc(y[b], pa[b]))
    return np.median(d), np.percentile(d, 2.5), np.percentile(d, 97.5)

blocks = {"clinical": CLIN, "+established CSF": CLIN + CSF, "+NULISA proteomics": CLIN + CSF + PROT_COLS}
preds = {}
print("\n=== Cognitive-conversion kill-gate (penalized logistic, nested 10-fold CV) ===")
for lab, cols in blocks.items():
    p = oof(cols); preds[lab] = p
    print(f"  {lab:22s} AUC={auc(y,p):.3f}")
labs = list(blocks)
for i in range(1, len(labs)):
    md, lo, hi = bootdiff(preds[labs[i-1]], preds[labs[i]])
    flag = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
    print(f"  dAUC [{labs[i]} - {labs[i-1]}]: {md:+.3f} [95%CI {lo:+.3f},{hi:+.3f}]{flag}")

# gradient-boosting check on full set (nonlinearity)
hgb = Pipeline([("i", SimpleImputer(strategy="median")),
    ("m", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
        l2_regularization=1.0, random_state=42))])
p_hgb = oof(CLIN + CSF + PROT_COLS, hgb)
print(f"\n  HistGBM (clinical+CSF+proteomics) AUC={auc(y,p_hgb):.3f}")

# permutation negative control for the proteomic increment
print("\n--- permutation control (proteomic block shuffled, 100x) ---")
base_auc = auc(y, preds["+established CSF"])
obs = auc(y, preds["+NULISA proteomics"]) - base_auc
Xfull = F[CLIN + CSF + PROT_COLS].to_numpy(); pcol = list(range(len(CLIN + CSF), Xfull.shape[1]))
perm = []
for k in range(100):
    Xp = Xfull.copy(); Xp[:, pcol] = Xp[rng.permutation(len(y))][:, pcol]
    pp = cross_val_predict(logit(), Xp, y, cv=StratifiedKFold(10, shuffle=True, random_state=42),
                           method="predict_proba")[:, 1]
    perm.append(auc(y, pp) - base_auc)
perm = np.array(perm)
print(f"  observed dAUC={obs:+.3f}; perm null mean={perm.mean():+.3f} sd={perm.std():.3f}; p={(perm>=obs).mean():.3f}")
print("\nGREEN if proteomic dAUC CI excludes 0 (& perm p<0.05); AMBER if positive/CI crosses; RED if null.")
