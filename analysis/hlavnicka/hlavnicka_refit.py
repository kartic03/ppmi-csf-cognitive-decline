"""RANK 8 — HLAVNICKA FOUR-QUESTION REFIT.

Hlavnicka et al. 2025, npj Parkinsons Dis (PMID 40274837,
doi:10.1038/s41531-025-00958-5): four questions predict cognitive decline in
de novo PD -- age of disease onset, history of stroke, history of fainting,
vocalization during dreams. Cross-validated PPMI AUC 0.70 +/- 0.10 (N=186);
0.79 overall PPMI; 0.78 in BIO-PD (N=48).

THE EDITORIAL QUESTION THIS ANSWERS: is a lumbar puncture worth it over four
free questions?

THIS IS NOT A REPLICATION. Their cohort was de novo PD (N=186) with binary
decline at 2- and 4-year horizons. Ours is the cogstate conversion endpoint on
a broader cohort over a longer window. So this is THEIR PREDICTORS, OUR COHORT,
OUR ENDPOINT -- and their 0.70 must not be quoted as the number being beaten.

PREDICTOR SOURCING (all structured; free text deliberately avoided after it
proved only 43.6% sensitive vs the structured stroke field):
  1. onset age          -> `ageonset` (curated cut)
  2. stroke             -> `STROKE`   (RBDSQ comorbidity block)
  4. dream vocalization -> `DRMVERBL` (RBDSQ item)
  3. fainting           -> AMBIGUOUS. The PPMI data dictionary labels SCOPA-AUT
     items only as "SCOPA Item 13/14/15" with no item text, so the specific
     fainting item cannot be verified. Run ALL candidate operationalisations
     and report whether the conclusion is stable.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = os.path.expanduser("~/pd_repro")
DL = r"/mnt/c/Users/Kartic Mishra/Downloads"
CURATED = os.path.join(ROOT, "data/processed/phase1/curated_cut.parquet")
RBDSQ = os.path.join(DL, "REM_Sleep_Behavior_Disorder_Screening_Questionnaire_04Aug2026.csv")
SCOPA = os.path.join(DL, "SCOPA-AUT_04Aug2026.csv")
SEEDS = list(range(10))
NFOLD = 5


def binary_endpoint(cur):
    pdf = cur[cur["COHORT"] == 1]
    bl = pdf[pdf["EVENT_ID"] == "BL"].set_index("PATNO")
    normal = bl[bl["cogstate"] == 1].index
    conv = set(pdf[(pdf["PATNO"].isin(normal)) & (pdf["EVENT_ID"] != "BL")
                   & (pdf["cogstate"] >= 2)]["PATNO"].unique())
    return pd.Series([1.0 if int(p) in conv else 0.0 for p in normal],
                     index=normal, name="converted", dtype=float)


def cv_auc(X, y):
    """Repeated stratified 5-fold, pooled OOF AUC per seed; returns mean and range."""
    aucs = []
    for s in SEEDS:
        oof = np.full(len(y), np.nan)
        skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=s)
        for tr, te in skf.split(X, y):
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler()),
                             ("m", LogisticRegression(max_iter=2000))])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.min(aucs)), float(np.max(aucs))


def main():
    cur = pd.read_parquet(CURATED)
    y = binary_endpoint(cur)
    print(f"Binary conversion endpoint: N={len(y)}, events={int(y.sum())} "
          f"({y.mean():.1%})\n")

    bl = cur[(cur["COHORT"] == 1) & (cur["EVENT_ID"] == "BL")].copy()
    bl["gba_status"] = bl["subgroup"].str.contains("GBA", na=False).astype(float)
    bl = bl.set_index("PATNO")

    rb = pd.read_csv(RBDSQ, low_memory=False)
    stroke = rb.groupby("PATNO")["STROKE"].max()
    verbal = rb.groupby("PATNO")["DRMVERBL"].max()

    sc = pd.read_csv(SCOPA, low_memory=False)
    cands = {}
    for it in ["SCAU13", "SCAU14", "SCAU15", "SCAU16"]:
        if it in sc.columns:
            v = sc.copy()
            v[it] = v[it].replace(9.0, np.nan)      # 9 = not applicable
            cands[it] = v.groupby("PATNO")[it].max()
    cands["scopa_cv"] = bl["scopa_cv"]
    cands["orthostasis"] = bl["orthostasis"]

    print("=" * 84)
    print("HLAVNICKA-4 REFIT — sensitivity across fainting operationalisations")
    print("=" * 84)
    rows = []
    for fname, fser in cands.items():
        X = pd.DataFrame({
            "onset_age": bl["ageonset"],
            "stroke": stroke,
            "fainting": fser,
            "dream_vocal": verbal,
        })
        idx = [i for i in y.index if i in X.index]
        Xi, yi = X.loc[idx], y.loc[idx]
        keep = Xi.notna().sum(axis=1) >= 3          # need most predictors present
        Xi, yi = Xi[keep], yi[keep]
        if yi.nunique() < 2 or len(yi) < 100:
            print(f"  {fname:14s} n={len(yi)} — insufficient, skipped"); continue
        m, lo, hi = cv_auc(Xi, yi)
        rows.append({"fainting_var": fname, "n": int(len(yi)),
                     "events": int(yi.sum()), "auc": m, "lo": lo, "hi": hi})
        print(f"  {fname:14s} n={len(yi):4d} events={int(yi.sum()):3d}  "
              f"AUC={m:.4f}  [seed range {lo:.4f}, {hi:.4f}]")

    print("\n" + "=" * 84)
    print("COMPARATORS ON THE IDENTICAL ENDPOINT AND FOLDS")
    print("=" * 84)
    CLIN = ["age", "SEX", "EDUCYRS", "moca", "updrs3_score", "duration_yrs",
            "LEDD", "APOE_e4", "gba_status", "MIA_PUTAMEN_BILAT"]
    ENR = [c for c in CLIN if c != "MIA_PUTAMEN_BILAT"] + [
        "MIA_CAUDATE_BILAT", "COG_COMPOSITE_INT", "gds", "stai", "rem"]
    for label, cols in [("our clinical (10)", CLIN), ("our enriched (14)", ENR)]:
        X = bl[cols]
        idx = [i for i in y.index if i in X.index]
        Xi, yi = X.loc[idx], y.loc[idx]
        keep = Xi.notna().sum(axis=1) >= len(cols) - 2
        Xi, yi = Xi[keep], yi[keep]
        m, lo, hi = cv_auc(Xi, yi)
        rows.append({"fainting_var": label, "n": int(len(yi)),
                     "events": int(yi.sum()), "auc": m, "lo": lo, "hi": hi})
        print(f"  {label:18s} n={len(yi):4d} events={int(yi.sum()):3d}  "
              f"AUC={m:.4f}  [seed range {lo:.4f}, {hi:.4f}]")

    print("\nNOTE: Hlavnicka reported cross-validated PPMI AUC 0.70 +/- 0.10 on")
    print("N=186 de novo PD with a 2-4 year binary horizon. Different cohort,")
    print("different endpoint, different window -- NOT a like-for-like number.")

    out = os.path.join(ROOT, "data/processed/phase2/hlavnicka_refit.json")
    json.dump({"endpoint": "cogstate conversion (baseline-normal PD)",
               "n_total": int(len(y)), "events_total": int(y.sum()),
               "seeds": SEEDS, "results": rows,
               "caveat": "their predictors, our cohort, our endpoint; "
                         "fainting item not verifiable from the PPMI dictionary"},
              open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
