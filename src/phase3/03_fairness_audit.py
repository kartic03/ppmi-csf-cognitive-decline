"""
Fairness / equity subgroup audit of the primary prediction model (Pillar A).

Takes the leak-safe out-of-fold predictions from the PRIMARY parsimonious model
(established markers = clinical + 5 CSF, penalized Ridge — the model Framing 1
recommends, NOT the non-replicating TabPFN proteome arm) and breaks performance
+ calibration down by equity-relevant subgroups:
  sex, age tertile, education (<=12 vs >12 yrs), APOE-e4 carrier, GBA carrier.

For each subgroup: N, R2, calibration slope (OLS observed~predicted, ideal 1.0),
calibration intercept, RMSE. Plus a per-axis disparity summary (R2 spread and
worst calibration-slope deviation). A Collection theme ("equity / real-world
impact") and a standard reviewer expectation for a clinical multimodal model.

Reuses the merged nested_cv harness + benchmark loaders; canonical-seed OOF (the
same predictions behind the reported established-model R2=0.127). CPU is fine
(Ridge only). Output: data/processed/phase3/fairness_audit.json
"""
import importlib.util
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "phase2"))
from cv import nested_cv, load_moca  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "bench_mod", os.path.join(_HERE, "01_benchmark.py"))
bm = importlib.util.module_from_spec(_spec)
sys.modules["bench_mod"] = bm
_spec.loader.exec_module(bm)


def _metrics(truth, pred):
    truth = np.asarray(truth, float)
    pred = np.asarray(pred, float)
    m = np.isfinite(truth) & np.isfinite(pred)
    truth, pred = truth[m], pred[m]
    out = {"n": int(len(truth))}
    if len(truth) < 10:
        out.update({"r2": None, "cal_slope": None, "cal_intercept": None, "rmse": None,
                    "note": "n<10, not reported"})
        return out
    out["r2"] = float(r2_score(truth, pred))
    out["rmse"] = float(np.sqrt(np.mean((truth - pred) ** 2)))
    if float(np.std(pred)) < 1e-9:
        out["cal_slope"] = out["cal_intercept"] = None
    else:
        slope, intercept = np.polyfit(pred, truth, 1)
        out["cal_slope"] = float(slope)
        out["cal_intercept"] = float(intercept)
    return out


def _subgroup_axes(cov):
    """Return {axis_name: {level_label: boolean mask}} over the cov DataFrame."""
    axes = {}
    # Sex (PPMI: 0 / 1)
    axes["sex"] = {"sex=0": cov["SEX"] == 0, "sex=1": cov["SEX"] == 1}
    # Age tertiles
    qs = cov["age"].quantile([1 / 3, 2 / 3]).values
    axes["age_tertile"] = {
        f"age<={qs[0]:.0f}": cov["age"] <= qs[0],
        f"age_{qs[0]:.0f}-{qs[1]:.0f}": (cov["age"] > qs[0]) & (cov["age"] <= qs[1]),
        f"age>{qs[1]:.0f}": cov["age"] > qs[1],
    }
    # Education: <=12 (high school or less) vs >12 (the standard equity dichotomy)
    axes["education"] = {"educ<=12": cov["EDUCYRS"] <= 12, "educ>12": cov["EDUCYRS"] > 12}
    # APOE-e4 carrier (>=1 allele) vs non
    axes["apoe_e4"] = {"apoe4_noncarrier": cov["APOE_e4"] == 0, "apoe4_carrier": cov["APOE_e4"] >= 1}
    # GBA carrier
    axes["gba"] = {"gba_noncarrier": cov["gba_status"] == 0, "gba_carrier": cov["gba_status"] == 1}
    return axes


def main():
    moca = load_moca(bm._CURATED)
    slope_ids = set(pd.read_parquet(bm._OUTCOME)["subject_id"].astype(int))
    bl_clinical = bm._load_curated_bl(bm._CURATED, slope_ids)
    csf_df = bm._load_established_csf(bm._CURATED, bm._NULISA, slope_ids)
    established_builder = bm._make_established_builder(bl_clinical, csf_df)
    analytic_ids = np.array(sorted(
        set(int(i) for i in slope_ids) & set(bl_clinical.index) & set(csf_df.index)))
    print(f"Analytic N = {len(analytic_ids)}", flush=True)

    # Primary model OOF (canonical seed) = the reported established-markers model.
    res = nested_cv(
        feature_builder=established_builder, moca_long=moca,
        model_factory=lambda: Ridge(), param_grid=bm.RIDGE_ALPHA_GRID,
        cv_config={"n_outer": 5, "n_inner": 5, "seeds": list(range(10))},
        analytic_ids=analytic_ids,
    )
    truth = res["oof_truth"]
    pred = res["oof_predictions"].reindex(truth.index)
    overall = _metrics(truth.values, pred.values)
    print(f"OVERALL: n={overall['n']} R2={overall['r2']:+.4f} "
          f"cal_slope={overall['cal_slope']:.3f}", flush=True)

    cov = bl_clinical.reindex(truth.index)[["SEX", "age", "EDUCYRS", "APOE_e4", "gba_status"]]

    result = {"n": overall["n"], "overall": overall, "axes": {}}
    for axis, levels in _subgroup_axes(cov).items():
        result["axes"][axis] = {}
        r2s, slopes = [], []
        print(f"\n[{axis}]", flush=True)
        for label, mask in levels.items():
            idx = cov.index[mask.fillna(False).values]
            mt = _metrics(truth.reindex(idx).values, pred.reindex(idx).values)
            result["axes"][axis][label] = mt
            if mt.get("r2") is not None:
                r2s.append(mt["r2"])
                if mt.get("cal_slope") is not None:
                    slopes.append(mt["cal_slope"])
                print(f"  {label:<22} n={mt['n']:<4} R2={mt['r2']:+.4f} "
                      f"cal_slope={mt['cal_slope']:.3f} rmse={mt['rmse']:.4f}", flush=True)
            else:
                print(f"  {label:<22} n={mt['n']:<4} (n<10, skipped)", flush=True)
        result["axes"][axis]["_disparity"] = {
            "r2_spread": float(max(r2s) - min(r2s)) if len(r2s) >= 2 else None,
            "worst_cal_slope_dev": float(max(abs(s - 1.0) for s in slopes)) if slopes else None,
        }

    # Age-outcome gradient diagnostic: explains the age-tertile R2 finding.
    # If the youngest tertile is near-flat with low variance, its negative R2 is
    # a near-absent base rate (decline is age-gated), NOT a model failure.
    y = truth  # EB-shrunken MoCA slope (the target), canonical-seed OOF subjects
    age_all = cov["age"]
    qa = age_all.quantile([1 / 3, 2 / 3]).values
    grad = {}
    overall_var = float(np.nanvar(y.values))
    for lbl, mask in {
        f"age<={qa[0]:.0f}": age_all <= qa[0],
        f"age_{qa[0]:.0f}-{qa[1]:.0f}": (age_all > qa[0]) & (age_all <= qa[1]),
        f"age>{qa[1]:.0f}": age_all > qa[1],
    }.items():
        yy = y[mask.fillna(False).values].dropna().values
        grad[lbl] = {
            "n": int(len(yy)), "mean_slope": float(np.mean(yy)),
            "sd_slope": float(np.std(yy)),
            "pct_decliners_lt_-0.3": float(100 * np.mean(yy < -0.3)),
            "variance_ratio_vs_overall": float(np.var(yy) / overall_var),
        }
    result["age_outcome_gradient"] = grad
    print("\n=== AGE-OUTCOME GRADIENT (why young tertile R2 is negative) ===")
    for lbl, g in grad.items():
        print(f"  {lbl:<14} n={g['n']:<4} mean={g['mean_slope']:+.4f} sd={g['sd_slope']:.4f} "
              f"decliners={g['pct_decliners_lt_-0.3']:.1f}% var_ratio={g['variance_ratio_vs_overall']:.2f}")

    os.makedirs(bm._DATA_OUT_DIR, exist_ok=True)
    out_path = os.path.join(bm._DATA_OUT_DIR, "fairness_audit.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== DISPARITY SUMMARY (R2 spread / worst calibration-slope deviation) ===")
    for axis in result["axes"]:
        d = result["axes"][axis]["_disparity"]
        sp = "n/a" if d["r2_spread"] is None else f"{d['r2_spread']:.4f}"
        cd = "n/a" if d["worst_cal_slope_dev"] is None else f"{d['worst_cal_slope_dev']:.3f}"
        print(f"  {axis:<14} R2 spread {sp:<8} worst |cal_slope-1| {cd}")
    print(f"\nWrote {out_path}")
    return result


if __name__ == "__main__":
    main()
