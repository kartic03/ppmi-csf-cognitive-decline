"""GBA as a SCALE ANCHOR for what dR2 means.

The rank-11 run produced an uncomfortable juxtaposition worth reporting rather
than hiding:

  GBA carrier status has a HUGE marginal effect -- 57.6% vs 21.7% conversion
  (Fisher p < 1e-4), EB slope median -0.193 vs -0.047 -- and yet contributes
  only dR2 = +0.0116 when added to the clinical block.

  The CSF block contributes +0.0581, five times more.

That is a genuinely useful anchor for readers who cannot intuit dR2: the CSF
increment is several times the increment of a variable with a 36-percentage-
point difference in conversion rate.

BUT IT IS NOT A FAIR HEAD-TO-HEAD and must not be written as one. GBA carriers
are ~10% of the sample, so the variable is near-constant for 90% of subjects
and its achievable R2 contribution is bounded by prevalence. The CSF markers
vary continuously across everyone. This script quantifies that bound so the
comparison can be stated with its caveat attached instead of as a bare ratio.
"""
import os, sys, importlib.util, json
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/pd_repro")
spec = importlib.util.spec_from_file_location(
    "inc", os.path.join(ROOT, "src/phase2/03_increment.py"))
inc = importlib.util.module_from_spec(spec); sys.modules["inc"] = inc
spec.loader.exec_module(inc)

cur = pd.read_parquet(inc._CURATED)
bl = cur[(cur["COHORT"] == 1) & (cur["EVENT_ID"] == "BL")].copy()
bl["gba_status"] = bl["subgroup"].str.contains("GBA", na=False).astype(float)
bl = bl.set_index("PATNO"); bl.index = bl.index.astype(int)

outc = pd.read_parquet(inc._OUTCOME).set_index("subject_id")
y = outc["eb_slope"] if "eb_slope" in outc.columns else outc.iloc[:, 0]
y.index = y.index.astype(int)

g = bl["gba_status"].dropna()
i = g.index.intersection(y.index)
g, yy = g.loc[i], y.loc[i]

prev = g.mean()
d1, d0 = yy[g == 1], yy[g == 0]
diff = d1.mean() - d0.mean()
# variance a binary predictor can explain, at best, given its prevalence
r2_max = (prev * (1 - prev) * diff ** 2) / yy.var()

print("=" * 76)
print("GBA AS A SCALE ANCHOR")
print("=" * 76)
print(f"  n                              : {len(yy)}")
print(f"  carrier prevalence             : {prev:.1%}  ({int(g.sum())} carriers)")
print(f"  EB slope, carriers vs non      : {d1.mean():+.4f} vs {d0.mean():+.4f}"
      f"   (difference {diff:+.4f})")
print(f"  outcome SD                     : {yy.std():.4f}")
print(f"  standardised effect (Cohen d)  : {diff / yy.std():.3f}")
print()
print(f"  MAXIMUM R2 a binary predictor at this prevalence and effect")
print(f"  size can explain, marginally   : {r2_max:.4f}")
print(f"  observed incremental dR2       : +0.0116")
print(f"  CSF block incremental dR2      : +0.0581")
print()
print("READING")
print("-" * 76)
print(f"  GBA's ceiling is {r2_max:.4f}, so its observed +0.0116 is already a")
print(f"  large fraction of what a variable affecting {prev:.0%} of the cohort could")
print("  possibly contribute. The CSF block faces no such prevalence ceiling.")
print("  So write the comparison as a SCALE ANCHOR -- 'the CSF increment is")
print("  several times that of GBA carrier status, a variable with a 36-point")
print("  difference in conversion rate' -- and state the prevalence caveat in")
print("  the same sentence. Do NOT write 'CSF outperforms GBA'.")

json.dump({"n": int(len(yy)), "prevalence": float(prev),
           "mean_carrier": float(d1.mean()), "mean_noncarrier": float(d0.mean()),
           "difference": float(diff), "cohens_d": float(diff / yy.std()),
           "r2_ceiling_at_this_prevalence": float(r2_max),
           "observed_dr2": 0.0116, "csf_dr2": 0.0581,
           "caveat": "prevalence-bounded; not a fair head-to-head with a "
                     "continuous block"},
          open(os.path.join(ROOT, "data/processed/phase2/gba_scale_anchor.json"), "w"),
          indent=2)
print("\nWrote data/processed/phase2/gba_scale_anchor.json")
