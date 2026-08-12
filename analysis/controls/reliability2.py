"""Compute the EB reliability of the MoCA-slope outcome actually used.

WHY THIS IS BEING CHECKED. The simulated review's Devil's Advocate asserted that
"outcome reliability was measured at 0.68, which caps achievable R2". That 0.68
comes from the progress log of 2026-06-23 and refers to the *Part II UPDRS motor*
outcome of the ABANDONED motor-progression design -- not the MoCA slope this
paper predicts. Before that claim is written into the manuscript it has to be
recomputed for the right outcome, or dropped.

Reliability of an EB-shrunken slope, following the phase1 estimator
(src/phase1/02_outcome.py:78):  reliability = tau2 / (tau2 + mean(v_i))
where v_i is the sampling variance of subject i's OLS slope.
"""
import importlib.util, sys, os
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/pd_repro")
sys.path.insert(0, os.path.join(ROOT, "src/phase2"))
spec = importlib.util.spec_from_file_location(
    "inc", os.path.join(ROOT, "src/phase2/03_increment.py"))
inc = importlib.util.module_from_spec(spec); sys.modules["inc"] = inc
spec.loader.exec_module(inc)
from cv import fit_eb_params, load_moca  # noqa: E402

moca = load_moca(inc._CURATED)
slope_ids = set(pd.read_parquet(inc._OUTCOME)["subject_id"].astype(int))
p = fit_eb_params(moca, list(slope_ids))
tau2, rvar = p["tau2"], p["residual_var"]

pid = "PATNO" if "PATNO" in moca.columns else moca.columns[0]
tcol = [c for c in moca.columns if "age" in c.lower() or "time" in c.lower()][0]
ycol = "moca" if "moca" in moca.columns else moca.columns[-1]
print(f"columns -> subject={pid}  time={tcol}  value={ycol}")

vs, ns = [], []
for sid, g in moca[moca[pid].astype(int).isin(slope_ids)].groupby(pid):
    g = g[[tcol, ycol]].dropna()
    if len(g) < 3:
        continue
    t = g[tcol].to_numpy(float)
    sxx = ((t - t.mean()) ** 2).sum()
    if sxx <= 0:
        continue
    vs.append(rvar / sxx)
    ns.append(len(g))

vs = np.array(vs); ns = np.array(ns)
rel = tau2 / (tau2 + vs.mean())

print()
print("=" * 70)
print("EB RELIABILITY OF THE MoCA SLOPE (the outcome this paper predicts)")
print("=" * 70)
print(f"  subjects with >=3 MoCA visits : {len(vs)}")
print(f"  tau2 (between-subject slope var): {tau2:.5f}")
print(f"  residual variance               : {rvar:.5f}")
print(f"  mean within-subject slope var   : {vs.mean():.5f}")
print(f"  RELIABILITY                     : {rel:.3f}")
print()
print(f"  ceiling on achievable R2 (= reliability): {rel:.3f}")
print(f"  observed clinical+CSF R2               : 0.1273")
print(f"  fraction of the attainable ceiling used: {0.1273 / rel:.1%}")
print()
print("  Reliability by visit count (few vs many, split at median 8):")
for lab, m in (("<=8 visits", ns <= 8), (" >8 visits", ns > 8)):
    if m.sum():
        r = tau2 / (tau2 + vs[m].mean())
        print(f"    {lab}: n={int(m.sum()):4d}  mean v={vs[m].mean():.5f}  "
              f"reliability={r:.3f}")
print()
print("  NOTE: the 0.68 figure in the 2026-06-23 progress log is the Part II")
print("  UPDRS MOTOR outcome of the abandoned design. It is NOT this outcome")
print("  and must not be quoted for the MoCA slope.")
