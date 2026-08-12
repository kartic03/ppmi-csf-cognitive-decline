"""THE PHASE-1-ONLY VISIT SPLIT — is the increment's concentration genuine
early exit, or administrative censoring?

THE OPEN QUESTION. The CSF increment is concentrated in short-follow-up
subjects: few-visit dR2 +0.0770 vs many-visit +0.0067. The standing explanation
is largely administrative -- every many-visit subject is enrolment phase 1 and
every phase-2 subject is few-visit, so the long-followed group is a survivor
cohort with compressed decline variance. But 219 few-visit subjects ARE phase 1,
which means the two explanations are separable: repeat the split INSIDE phase 1,
where enrolment date is no longer confounded with follow-up length.

  - concentration PERSISTS within phase 1  -> genuine early exit / dropout
  - concentration VANISHES within phase 1  -> administrative censoring, and the
    original finding is an artifact of enrolment wave

DESIGN NOTE -- THE TRAP THIS AVOIDS. An earlier attempt at a stratified version
of this analysis passed `analytic_ids=<stratum>` to paired_increment_cv, which
RETRAINS the model inside each stratum. At n=263 that produced negative R2 for
both models and was meaningless. Stratification must be applied AFTER the fact,
by partitioning the full-sample out-of-fold predictions -- the models are fit
once on everyone, and the strata only decide how predictions are scored. That is
what this script does.

DOUBLE ANCHOR. The full-sample run must reproduce dR2 = +0.0581, and the
few/many partition must reproduce +0.0770 / +0.0067 from the committed
c1_visitcount_strata.json. If both land, the visit-count definition matches the
original analysis and the new phase-restricted cells can be trusted.
"""
import os, sys, json, importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = os.path.expanduser("~/pd_repro")
spec = importlib.util.spec_from_file_location(
    "inc", os.path.join(ROOT, "src/phase2/03_increment.py"))
inc = importlib.util.module_from_spec(spec); sys.modules["inc"] = inc
spec.loader.exec_module(inc)

CV = {"n_outer": 5, "n_inner": 5, "seeds": list(range(10))}
GRID = {"alpha": inc.RIDGE_ALPHA_GRID}


def cell(oof_b, oof_a, truth, idx, label):
    idx = pd.Index(idx).intersection(truth.index)
    if len(idx) < 20:
        return {"cell": label, "n": int(len(idx)), "note": "too small to score"}
    t = truth.loc[idx]
    b, a = oof_b.loc[idx], oof_a.loc[idx]
    rb, ra = r2_score(t, b), r2_score(t, a)
    # RMSE is on the outcome's own scale, so it does NOT reward a stratum merely
    # for having more variance to explain. R2 does. Cells here differ in outcome
    # SD by ~1.5x, so the RMSE gain is the check that decides whether a large
    # dR2 is real signal or just a wider target.
    rmse_b = float(np.sqrt(((t - b) ** 2).mean()))
    rmse_a = float(np.sqrt(((t - a) ** 2).mean()))
    return {"cell": label, "n": int(len(idx)), "r2_base": float(rb),
            "r2_aug": float(ra), "dr2": float(ra - rb),
            "rmse_base": rmse_b, "rmse_aug": rmse_a,
            "rmse_gain": rmse_b - rmse_a,
            "rmse_gain_pct_of_sd": 100 * (rmse_b - rmse_a) / float(t.std()),
            "outcome_sd": float(t.std())}


def run(outcome_col):
    moca = inc.load_moca(inc._CURATED)
    slope_ids = set(pd.read_parquet(inc._OUTCOME)["subject_id"].astype(int))
    clin = inc._make_clinical_builder(inc._CURATED, slope_ids)
    csf = inc._load_csf_aug_df(inc._CURATED, inc._NULISA, slope_ids)
    csf.index = csf.index.astype(int)
    ids = clin(list(csf.index)).index.intersection(csf.index).tolist()

    r = inc.paired_increment_cv(
        base_builder=clin, aug_df=csf, moca_long=moca,
        model_factory=inc._default_model, param_grid=GRID, cv_config=CV,
        analytic_ids=ids, outcome_col=outcome_col)
    ob, oa, tr = r["oof_base"], r["oof_aug"], r["oof_truth"]
    ob.index = ob.index.astype(int); oa.index = oa.index.astype(int)
    tr.index = tr.index.astype(int)
    return r, ob, oa, tr


def main():
    cur = pd.read_parquet(inc._CURATED)
    bl = cur[(cur["COHORT"] == 1) & (cur["EVENT_ID"] == "BL")].set_index("PATNO")
    bl.index = bl.index.astype(int)
    phase = bl["enroll_phase"]

    moca = inc.load_moca(inc._CURATED)
    pid = "PATNO" if "PATNO" in moca.columns else moca.columns[0]
    nvis = moca.groupby(pid).size()
    nvis.index = nvis.index.astype(int)

    out = {}
    for outcome in ("eb_slope", "ols_slope"):
        print("=" * 86)
        print(f"OUTCOME = {outcome}")
        print("=" * 86)
        r, ob, oa, tr = run(outcome)
        print(f"  FULL SAMPLE  n={r['n_subjects']}  R2 {r['r2_base']:.4f} -> "
              f"{r['r2_aug']:.4f}   dR2 {r['dr2']:+.4f}")

        if outcome == "eb_slope":
            if abs(r["dr2"] - 0.0581) > 0.002:
                print(f"  !! ANCHOR FAILED (expected +0.0581). STOPPING.")
                sys.exit(1)
            print("  -> full-sample anchor OK")

        med = nvis.loc[nvis.index.intersection(tr.index)].median()
        few = nvis[nvis <= med].index
        many = nvis[nvis > med].index
        print(f"  visit-count median = {med:.0f}")

        rows = []
        print("\n  A. ORIGINAL SPLIT (anchor: +0.0770 / +0.0067 on eb_slope)")
        rows.append(cell(ob, oa, tr, few, "few-visit (all phases)"))
        rows.append(cell(ob, oa, tr, many, "many-visit (all phases)"))

        p1 = phase[phase == 1].index
        p2 = phase[phase == 2].index
        print("\n  B. THE QUESTION — SPLIT WITHIN ENROLMENT PHASE 1 ONLY")
        rows.append(cell(ob, oa, tr, p1.intersection(few), "phase 1, few-visit"))
        rows.append(cell(ob, oa, tr, p1.intersection(many), "phase 1, many-visit"))
        print("\n  C. PHASE 2 FOR CONTRAST")
        rows.append(cell(ob, oa, tr, p2.intersection(few), "phase 2, few-visit"))
        rows.append(cell(ob, oa, tr, p2.intersection(many), "phase 2, many-visit"))

        for c in rows:
            if "dr2" in c:
                print(f"    {c['cell']:<26s} n={c['n']:>4d}  dR2 {c['dr2']:>+7.4f}"
                      f"   RMSE {c['rmse_base']:.4f}->{c['rmse_aug']:.4f} "
                      f"(gain {c['rmse_gain']:>+.4f} = "
                      f"{c['rmse_gain_pct_of_sd']:>+5.1f}% of sd)   "
                      f"sd {c['outcome_sd']:.3f}")
            else:
                print(f"    {c['cell']:<26s} n={c['n']:>4d}  {c['note']}")

        # the decisive contrast
        d = {c["cell"]: c for c in rows}
        if "dr2" in d.get("phase 1, few-visit", {}) and \
           "dr2" in d.get("phase 1, many-visit", {}):
            gap_all = d["few-visit (all phases)"]["dr2"] - d["many-visit (all phases)"]["dr2"]
            gap_p1 = d["phase 1, few-visit"]["dr2"] - d["phase 1, many-visit"]["dr2"]
            print(f"\n    few-minus-many gap, ALL phases : {gap_all:+.4f}")
            print(f"    few-minus-many gap, PHASE 1    : {gap_p1:+.4f}")
            frac = gap_p1 / gap_all if gap_all != 0 else float("nan")
            print(f"    fraction of the gap surviving inside phase 1: {frac:.0%}")
            out[f"{outcome}_gap_all"] = float(gap_all)
            out[f"{outcome}_gap_phase1"] = float(gap_p1)
            out[f"{outcome}_gap_fraction_surviving"] = float(frac)
        out[outcome] = {"full_sample": {k: float(r[k]) for k in
                                        ("r2_base", "r2_aug", "dr2")},
                        "n_full": int(r["n_subjects"]),
                        "visit_median": float(med), "cells": rows}
        print()

    p = os.path.join(ROOT, "data/processed/phase2/phase1_visit_split.json")
    json.dump({
        "design": "full-sample paired increment; strata applied AFTER the fact "
                  "by partitioning out-of-fold predictions (models fit once on "
                  "everyone). Retraining within strata was tried previously and "
                  "produced negative R2 at n=263.",
        "cv_config": CV, "results": out,
        "caveat": "cells are scored on shared out-of-fold predictions, so their "
                  "R2 values are not independent estimates and have no CIs here",
    }, open(p, "w"), indent=2)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
