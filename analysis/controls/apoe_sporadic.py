"""Does the APOE null dissolve when the cohort is restricted to SPORADIC PD?

Botta et al. 2026, npj Parkinsons Dis (PMID 41730900, doi:10.1038/s41531-026-01290-2)
-- same journal, five months before our target submission -- report that APOE e4
predicts faster cognitive decline in SPORADIC PD but has NO effect in GBA1-PD or
LRRK2-PD, using pooled PPMI + CPP data.

Our cohort pools subtypes (1080 sporadic, 184 LRRK2, 114 GBA, 38 SNCA, 29 PRKN...).
If Botta is right, pooling should DILUTE the APOE effect, and the flat APOE result
we found is the predicted consequence rather than an anomaly.

That converts a bare null into a literature-consistent, mechanistically explained
finding -- but only if it actually holds. If APOE stays flat in sporadic-only PD,
we FAIL to replicate Botta in the subgroup where they found the effect, which is a
different and more awkward result that must be reported as such.

Tested on both endpoints, since Botta used longitudinal cognitive decline.
"""
import os, sys, json, importlib.util
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.expanduser("~/pd_repro")
spec = importlib.util.spec_from_file_location(
    "inc", os.path.join(ROOT, "src/phase2/03_increment.py"))
inc = importlib.util.module_from_spec(spec); sys.modules["inc"] = inc
spec.loader.exec_module(inc)

cur = pd.read_parquet(inc._CURATED)
pdf = cur[cur["COHORT"] == 1]
bl = pdf[pdf["EVENT_ID"] == "BL"].set_index("PATNO")
bl.index = bl.index.astype(int)

outc = pd.read_parquet(inc._OUTCOME).set_index("subject_id")
y = outc["eb_slope"] if "eb_slope" in outc.columns else outc.iloc[:, 0]
y.index = y.index.astype(int)

norm = bl[bl["cogstate"] == 1].index
conv = set(pdf[(pdf["PATNO"].isin(norm)) & (pdf["EVENT_ID"] != "BL")
               & (pdf["cogstate"] >= 2)]["PATNO"].astype(int).unique())
ybin = pd.Series([1.0 if i in conv else 0.0 for i in norm], index=norm, dtype=float)

sub = bl["subgroup"].fillna("")
strata = {
    "ALL PD (pooled, as analysed)": bl.index,
    "SPORADIC only (Botta's group)": bl[sub.str.strip() == "Sporadic PD"].index,
    "GBA carriers": bl[sub.str.contains("GBA", na=False)].index,
    "LRRK2 carriers": bl[sub.str.contains("LRRK2", na=False)].index,
}

apoe = (bl["APOE_e4"] > 0).astype(float).where(bl["APOE_e4"].notna())

print("=" * 88)
print("APOE e4 BY GENETIC SUBTYPE — testing Botta 2026's subtype-specific claim")
print("=" * 88)
rows = []
for name, idx in strata.items():
    v = apoe.loc[apoe.index.intersection(idx)].dropna()
    r = {"stratum": name}

    i = v.index.intersection(y.index)
    g1, g0 = y.loc[i][v.loc[i] > 0], y.loc[i][v.loc[i] == 0]
    if len(g1) >= 10 and len(g0) >= 10:
        u, p = stats.mannwhitneyu(g1, g0)
        d = g1.mean() - g0.mean()
        r.update({"slope_n_carrier": len(g1), "slope_n_non": len(g0),
                  "slope_mean_carrier": float(g1.mean()),
                  "slope_mean_non": float(g0.mean()),
                  "slope_diff": float(d), "slope_p": float(p)})
        print(f"\n  {name}")
        print(f"    EB slope   : carriers n={len(g1):4d} mean {g1.mean():+.4f} | "
              f"non n={len(g0):4d} mean {g0.mean():+.4f}")
        print(f"                 difference {d:+.4f}  (negative = carriers decline "
              f"FASTER)   p={p:.4f}")
    else:
        print(f"\n  {name}\n    EB slope   : too few subjects")

    j = v.index.intersection(ybin.index)
    if len(j) >= 40:
        vb = (v.loc[j] > 0).astype(int); yb = ybin.loc[j]
        if vb.nunique() == 2 and yb.nunique() == 2:
            r1, r0 = yb[vb == 1].mean(), yb[vb == 0].mean()
            pf = stats.fisher_exact(pd.crosstab(vb, yb))[1]
            r.update({"conv_carrier_rate": float(r1), "conv_non_rate": float(r0),
                      "conv_n_carrier": int((vb == 1).sum()),
                      "conv_n_non": int((vb == 0).sum()), "conv_p": float(pf)})
            print(f"    conversion : carriers {r1:.1%} (n={int((vb==1).sum())}) | "
                  f"non {r0:.1%} (n={int((vb==0).sum())})   Fisher p={pf:.4f}")
    rows.append(r)

print("\n" + "=" * 88)
print("READING")
print("=" * 88)
spor = next(r for r in rows if r["stratum"].startswith("SPORADIC"))
if spor.get("slope_p", 1) < 0.05 and spor.get("slope_diff", 0) < 0:
    print("  APOE e4 DOES predict faster decline in sporadic PD here. Our pooled")
    print("  null is therefore a DILUTION effect and replicates Botta 2026's")
    print("  subtype-specific finding. Report the subtype breakdown, cite Botta,")
    print("  and note that pooling subtypes masks a real effect.")
else:
    print("  APOE e4 remains flat EVEN IN SPORADIC PD (p="
          f"{spor.get('slope_p', float('nan')):.4f}, difference "
          f"{spor.get('slope_diff', float('nan')):+.4f}).")
    print("  So we do NOT replicate Botta 2026 in the subgroup where they found")
    print("  the effect. This is a genuine non-replication and must be reported")
    print("  as such -- NOT explained away as subtype dilution. Candidate reasons")
    print("  to state: different outcome (EB MoCA slope vs their mixed-model")
    print("  decline), different follow-up window, and their pooled PPMI+CPP")
    print("  sample differs from our CSF-complete analytic subset.")

json.dump({"source_claim": "Botta 2026 PMID 41730900 doi:10.1038/s41531-026-01290-2 "
                           "— APOE e4 effect in sporadic PD, absent in GBA1/LRRK2",
           "strata": rows},
          open(os.path.join(ROOT, "data/processed/phase2/apoe_by_subtype.json"), "w"),
          indent=2)
print("\nWrote data/processed/phase2/apoe_by_subtype.json")
