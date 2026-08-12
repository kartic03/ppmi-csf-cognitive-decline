"""HLAVNICKA HEAD-TO-HEAD on the paper's own CSF-complete set.

The editorial question npj PD will actually ask:

    on the SAME subjects and the SAME folds, does a lumbar puncture buy
    anything over four free questions?

An earlier pass hand-rolled a CSF block from curated columns and collapsed the
sample to n=147 with a 42% event rate -- a selected subset whose numbers were
not trustworthy (CSF alone scored 0.39, i.e. worse than chance, which is the
signature of selection, not signal). This version imports the project's own
`_load_csf_aug_df` and `CLINICAL_COLS` from src/phase2/03_increment.py so the
CSF block is EXACTLY the one behind the published +0.058 dR2.

THIS IS NOT A REPLICATION of Hlavnicka. Their cohort was de novo PD (N=186)
with a 2-4 year binary horizon; ours is the cogstate conversion endpoint on a
broader cohort over a longer window. Their predictors, our cohort, our
endpoint -- their 0.70 +/- 0.10 is NOT a number being beaten.
"""
import os, sys, json, importlib.util
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

ROOT = os.path.expanduser("~/pd_repro")
DL = r"/mnt/c/Users/Kartic Mishra/Downloads"
SEEDS = list(range(10))

spec = importlib.util.spec_from_file_location(
    "inc", os.path.join(ROOT, "src/phase2/03_increment.py"))
inc = importlib.util.module_from_spec(spec)
sys.modules["inc"] = inc
spec.loader.exec_module(inc)


C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def cv_auc(X, y, seeds=SEEDS):
    """Nested 5x5 CV, L2 penalty tuned in the inner loop.

    The penalty is NOT optional here. Arms range from 1 to 22 features against
    127 events (EPV as low as 5.8), so an unpenalised fit would overfit the
    wide arms specifically -- which are exactly the CSF-augmented ones. Tuning
    inside the training folds mirrors the published ridge design and keeps the
    comparison from being decided by feature count.
    """
    aucs = []
    for s in seeds:
        oof = np.full(len(y), np.nan)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            p = GridSearchCV(
                Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler()),
                          ("m", LogisticRegression(max_iter=5000))]),
                {"m__C": C_GRID}, scoring="roc_auc",
                cv=StratifiedKFold(5, shuffle=True, random_state=s), n_jobs=-1)
            p.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = p.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.min(aucs)), float(np.max(aucs))


def main():
    cur = pd.read_parquet(inc._CURATED)
    pdf = cur[cur["COHORT"] == 1]
    bl0 = pdf[pdf["EVENT_ID"] == "BL"].set_index("PATNO")
    normal = bl0[bl0["cogstate"] == 1].index
    conv = set(pdf[(pdf["PATNO"].isin(normal)) & (pdf["EVENT_ID"] != "BL")
                   & (pdf["cogstate"] >= 2)]["PATNO"].unique())
    y_all = pd.Series([1.0 if int(p) in conv else 0.0 for p in normal],
                      index=normal, dtype=float)
    y_all.index = y_all.index.astype(int)

    # --- the paper's own CSF block, unmodified -----------------------------
    slope_ids = set(pd.read_parquet(inc._OUTCOME)["subject_id"].astype(int))
    csf = inc._load_csf_aug_df(inc._CURATED, inc._NULISA, slope_ids)
    csf.index = csf.index.astype(int)
    print(f"paper CSF-complete block: n={len(csf)}, cols={list(csf.columns)}")

    bl = bl0.copy()
    bl.index = bl.index.astype(int)
    bl["gba_status"] = bl["subgroup"].str.contains("GBA", na=False).astype(float)

    rb = pd.read_csv(os.path.join(DL, "REM_Sleep_Behavior_Disorder_Screening_Questionnaire_04Aug2026.csv"),
                     low_memory=False)
    stroke = rb.groupby("PATNO")["STROKE"].max()
    verbal = rb.groupby("PATNO")["DRMVERBL"].max()
    sc = pd.read_csv(os.path.join(DL, "SCOPA-AUT_04Aug2026.csv"), low_memory=False)

    CLIN = bl[inc.CLINICAL_COLS]
    ENR = bl[[c for c in inc.CLINICAL_COLS if c != "MIA_PUTAMEN_BILAT"] +
             ["MIA_CAUDATE_BILAT", "COG_COMPOSITE_INT", "gds", "stai", "rem"]]

    def h4(item):
        f = sc.copy(); f[item] = f[item].replace(9.0, np.nan)
        return pd.DataFrame({"onset_age": bl["ageonset"], "stroke": stroke,
                             "fainting": f.groupby("PATNO")[item].max(),
                             "dream_vocal": verbal})

    # ONE index shared by every arm: endpoint + paper-CSF + clinical + H4
    H4 = h4("SCAU15")
    idx = (y_all.index.intersection(csf.index)
           .intersection(CLIN[CLIN.notna().sum(1) >= CLIN.shape[1] - 1].index)
           .intersection(H4[H4.notna().sum(1) >= 3].index))
    y = y_all.loc[idx]
    print(f"\nCOMMON INDEX (endpoint + paper CSF + clinical + H4): "
          f"n={len(y)}, events={int(y.sum())} ({y.mean():.1%})\n")

    res = {}
    print("=" * 80)
    print("A. FAINTING-ITEM SENSITIVITY (the item is not verifiable from the "
          "PPMI dictionary)")
    print("=" * 80)
    faint_arms = {}
    for it in ["SCAU13", "SCAU14", "SCAU15", "SCAU16"]:
        m, lo, hi = cv_auc(h4(it).loc[idx], y)
        faint_arms[it] = {"auc": m, "lo": lo, "hi": hi}
        print(f"  Hlavnicka-4 [{it}]        AUC={m:.4f}  [{lo:.4f}, {hi:.4f}]")
    spread = max(a["auc"] for a in faint_arms.values()) - \
             min(a["auc"] for a in faint_arms.values())
    best = max(faint_arms, key=lambda k: faint_arms[k]["auc"])
    print(f"  -> spread across items = {spread:.4f}; most favourable = {best}")

    arms = {
        "Hlavnicka-4 (best item)":  H4.loc[idx] if best == "SCAU15" else h4(best).loc[idx],
        "  onset age alone":        H4.loc[idx, ["onset_age"]],
        "  H4 minus onset age":     H4.loc[idx, ["stroke", "fainting", "dream_vocal"]],
        "CSF alone (paper block)":  csf.loc[idx],
        "clinical (10)":            CLIN.loc[idx],
        "clinical + CSF":           pd.concat([CLIN, csf], axis=1).loc[idx],
        "enriched (14)":            ENR.loc[idx],
        "enriched + CSF":           pd.concat([ENR, csf], axis=1).loc[idx],
        "enriched + H4":            pd.concat([ENR, H4[["stroke", "fainting", "dream_vocal"]]], axis=1).loc[idx],
        "enriched + H4 + CSF":      pd.concat([ENR, H4[["stroke", "fainting", "dream_vocal"]], csf], axis=1).loc[idx],
    }

    print("\n" + "=" * 80)
    print(f"B. ALL ARMS, IDENTICAL SUBJECTS AND FOLDS (n={len(y)})")
    print("=" * 80)
    print(f"{'arm':<28s} {'p':>3s}  {'AUC':>7s}   seed range")
    for name, X in arms.items():
        m, lo, hi = cv_auc(X, y)
        res[name] = {"p": int(X.shape[1]), "auc": m, "lo": lo, "hi": hi}
        print(f"{name:<28s} {X.shape[1]:>3d}  {m:.4f}   [{lo:.4f}, {hi:.4f}]")

    print("\n" + "=" * 80)
    print("C. THE DELTAS THAT DECIDE THE FRAMING")
    print("=" * 80)
    for a, b in [("clinical (10)", "clinical + CSF"),
                 ("enriched (14)", "enriched + CSF"),
                 ("enriched + H4", "enriched + H4 + CSF")]:
        print(f"  CSF adds on top of [{a:<16s}] = {res[b]['auc'] - res[a]['auc']:+.4f}")
    print(f"  four free questions vs our enriched block = "
          f"{res['Hlavnicka-4 (best item)']['auc'] - res['enriched (14)']['auc']:+.4f}")
    print(f"  H4 adds on top of [enriched (14)]        = "
          f"{res['enriched + H4']['auc'] - res['enriched (14)']['auc']:+.4f}")

    out = os.path.join(ROOT, "data/processed/phase2/hlavnicka_head2head.json")
    json.dump({"endpoint": "cogstate conversion, baseline-normal PD",
               "n": int(len(y)), "events": int(y.sum()), "seeds": SEEDS,
               "csf_block": list(csf.columns),
               "fainting_sensitivity": faint_arms,
               "fainting_spread": spread, "fainting_best_item": best,
               "arms": res,
               "caveats": [
                   "their predictors, our cohort, our endpoint -- not a replication",
                   "the fainting item is NOT identifiable from the PPMI data "
                   "dictionary; the reported H4 arm is the most favourable of four",
                   "brackets are cross-seed ranges, not confidence intervals",
               ]}, open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
