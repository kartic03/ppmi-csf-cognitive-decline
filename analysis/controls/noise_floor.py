"""The 'noise floor' is a single draw, and single draws vary.

`batch1_checks.json` records dR2 = -0.0093 for clinical + 5 random Gaussians.
The rank-11 re-run, with a different RNG seed, got +0.0066 on the same
subjects. Both are draws from the same null distribution, so neither is "the"
floor -- and quoting one as a floor understates how large a null increment can
look by chance.

Estimate the distribution over 20 independent draws and report the upper tail,
which is the number that actually matters: how big can dR2 get from pure noise?
"""
import os, sys, json, importlib.util
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/pd_repro")
spec = importlib.util.spec_from_file_location(
    "inc", os.path.join(ROOT, "src/phase2/03_increment.py"))
inc = importlib.util.module_from_spec(spec); sys.modules["inc"] = inc
spec.loader.exec_module(inc)

CV = {"n_outer": 5, "n_inner": 5, "seeds": list(range(10))}
NDRAW = 20

moca = inc.load_moca(inc._CURATED)
slope_ids = set(pd.read_parquet(inc._OUTCOME)["subject_id"].astype(int))
clin = inc._make_clinical_builder(inc._CURATED, slope_ids)
csf = inc._load_csf_aug_df(inc._CURATED, inc._NULISA, slope_ids)
csf.index = csf.index.astype(int)
ids = clin(list(csf.index)).index.intersection(csf.index).tolist()

cur = pd.read_parquet(inc._CURATED)
bl = cur[(cur["COHORT"] == 1) & (cur["EVENT_ID"] == "BL")].set_index("PATNO")
bl.index = bl.index.astype(int)

vals = []
for d in range(NDRAW):
    rng = np.random.default_rng(1000 + d)
    noise = pd.DataFrame(rng.standard_normal((len(bl), 5)), index=bl.index,
                         columns=[f"n{i}" for i in range(5)])
    r = inc.paired_increment_cv(
        base_builder=clin, aug_df=noise, moca_long=moca,
        model_factory=inc._default_model,
        param_grid={"alpha": inc.RIDGE_ALPHA_GRID},
        cv_config=CV, analytic_ids=ids, outcome_col="eb_slope")
    vals.append(r["dr2"])
    print(f"  draw {d:2d}: dR2 {r['dr2']:+.4f}", flush=True)

v = np.array(vals)
print("\n" + "=" * 72)
print(f"  draws            : {NDRAW} x 5 random Gaussians, n={len(ids)}")
print(f"  mean             : {v.mean():+.4f}")
print(f"  sd               : {v.std(ddof=1):.4f}")
print(f"  min / max        : {v.min():+.4f} / {v.max():+.4f}")
print(f"  95th percentile  : {np.percentile(v, 95):+.4f}   <- the number to quote")
print(f"  CSF increment    : +0.0581  ({(0.0581 - v.mean()) / v.std(ddof=1):.1f} "
      f"SD above the noise mean)")
print("=" * 72)
print("\n  The single-draw '-0.0093 noise floor' should be replaced by this")
print("  distribution. A null increment can reach the 95th percentile above by")
print("  chance alone, so that -- not a lucky negative draw -- is the bar.")

json.dump({"n_draws": NDRAW, "n_subjects": len(ids), "dr2_draws": vals,
           "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
           "min": float(v.min()), "max": float(v.max()),
           "p95": float(np.percentile(v, 95)),
           "csf_increment": 0.0581,
           "note": "replaces the single-draw -0.0093 figure in batch1_checks.json"},
          open(os.path.join(ROOT, "data/processed/phase2/noise_floor_distribution.json"), "w"),
          indent=2)
print("\nWrote data/processed/phase2/noise_floor_distribution.json")
