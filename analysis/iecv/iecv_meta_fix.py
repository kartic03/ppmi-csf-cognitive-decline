"""Recompute the IECV calibration meta-analysis EXCLUDING the pooled small-site
cluster.

Cluster "-1" is not a site -- it is 38 small sites glued together so that no
subject is dropped from the pooled estimate. At n=289 it carries the smallest
standard error and therefore dominates a random-effects pooling, which would
make a "between-site" meta-analysis mostly an artifact of that glued cluster.
The first run included it, contradicting its own stated caveat. This recomputes
from the saved per-cluster data (no refitting needed) and stores BOTH, with the
real-sites-only version as primary.
"""
import json, os
import numpy as np
from scipy import stats

P = os.path.expanduser("~/pd_repro/data/processed/phase2/iecv_site.json")
d = json.load(open(P))


def dl(y, se):
    y, v = np.asarray(y, float), np.asarray(se, float) ** 2
    k = len(y)
    w = 1 / v
    fixed = (w * y).sum() / w.sum()
    Q = (w * (y - fixed) ** 2).sum()
    df = k - 1
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    ws = 1 / (v + tau2)
    est = (ws * y).sum() / ws.sum()
    se_p = np.sqrt(1 / ws.sum())
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    pi = stats.t.ppf(0.975, k - 2) * np.sqrt(tau2 + se_p ** 2) if k > 2 else np.nan
    return {"estimate": float(est), "se": float(se_p),
            "ci_lo": float(est - 1.96 * se_p), "ci_hi": float(est + 1.96 * se_p),
            "tau2": float(tau2), "I2_percent": float(I2), "k": int(k),
            "pi_lo": float(est - pi) if np.isfinite(pi) else None,
            "pi_hi": float(est + pi) if np.isfinite(pi) else None}


pc = d["site"]["per_cluster"]
real = [c for c in pc if c["cluster"] != "-1.0"]
meta_real = dl([c["calib_slope"] for c in real], [c["calib_slope_se"] for c in real])

print("=" * 78)
print("CALIBRATION SLOPE — random-effects meta-analysis")
print("=" * 78)
print(f"  including pooled small-site cluster (k={d['site']['calibration_meta']['k']}): "
      f"{d['site']['calibration_meta']['estimate']:.3f} "
      f"[{d['site']['calibration_meta']['ci_lo']:.3f}, "
      f"{d['site']['calibration_meta']['ci_hi']:.3f}]  "
      f"I2={d['site']['calibration_meta']['I2_percent']:.1f}%")
print(f"  REAL SITES ONLY (k={meta_real['k']}) : {meta_real['estimate']:.3f} "
      f"[{meta_real['ci_lo']:.3f}, {meta_real['ci_hi']:.3f}]  "
      f"I2={meta_real['I2_percent']:.1f}%, tau2={meta_real['tau2']:.4f}")
print(f"  95% prediction interval for a NEW site: "
      f"[{meta_real['pi_lo']:.3f}, {meta_real['pi_hi']:.3f}]")
print()
sl = [c["calib_slope"] for c in real]
print(f"  observed per-site slopes: min {min(sl):.3f}, max {max(sl):.3f}, "
      f"n sites {len(sl)}")
print(f"  slopes below 1.0: {sum(1 for s in sl if s < 1)} of {len(sl)}")
print()
print("  NOTE ON I^2 = 0%: with k=10 clusters of 24-75 subjects, each slope has")
print("  a large standard error, so I^2 has little power. Read it as")
print("  'between-site heterogeneity is NOT DETECTABLE at this cluster size',")
print("  NOT as 'sites are homogeneous'.")

d["site"]["calibration_meta_real_sites_only"] = meta_real
d["site"]["calibration_meta_note"] = (
    "PRIMARY = calibration_meta_real_sites_only (k=10). The pooled small-site "
    "cluster '-1.0' is 38 sites glued together, not a site; at n=289 it has the "
    "smallest SE and would dominate the pooling. I^2 is underpowered at these "
    "cluster sizes and must not be read as evidence of homogeneity.")
d["caveats"] = [c for c in d["caveats"] if "excluded from the meta" not in c] + [
    "the pooled small-site cluster is retained in the POOLED R2 (so no subject "
    "is dropped) but excluded from the primary calibration meta-analysis"]
json.dump(d, open(P, "w"), indent=2)
print(f"\nUpdated {P}")
