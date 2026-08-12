"""
Phase 1, step 01: compute the real Q-albumin (CSF/serum albumin) BBB-integrity index
from the PPMI biospecimen file, and report its overlap with the PD progression cohort.

Q-albumin = (CSF albumin / plasma albumin) * 1000.  Both compartments are in ng/ml in PPMI,
so the ratio is dimensionless and the *1000 gives the conventional index (normal ~5-8).

This is modality 2 in DESIGN_SPEC_v2 and the BINDING constraint on Set A size.
No fabrication: Q-albumin is only computed where the SAME subject has BOTH CSF and plasma
albumin measured at the SAME clinical event.
"""
import os
import pandas as pd
import numpy as np

BASE = "data/raw/PPMI new downloads"
OUT = "data/processed/phase1"
os.makedirs(OUT, exist_ok=True)

# ---- 1. Q-albumin from biospecimen (long format) ----
bio = pd.read_csv(os.path.join(BASE, "Current_Biospecimen_Analysis_Results_22Jun2026.csv"),
                  low_memory=False)
alb = bio[bio["TESTNAME"].isin(["CSF Albumin", "Plasma Albumin"])].copy()
alb["val"] = pd.to_numeric(alb["TESTVALUE"], errors="coerce")
alb = alb.dropna(subset=["val"])

# pivot to one row per (subject, clinical event) with both compartments
wide = (alb.pivot_table(index=["PATNO", "CLINICAL_EVENT"], columns="TESTNAME",
                        values="val", aggfunc="mean").reset_index())
wide = wide.dropna(subset=["CSF Albumin", "Plasma Albumin"])
wide["qalb"] = wide["CSF Albumin"] / wide["Plasma Albumin"] * 1000.0

# one Q-albumin per subject: prefer baseline (BL), else earliest available event
order = {"BL": 0, "SC": -1}  # screening before baseline if present
wide["ev_rank"] = wide["CLINICAL_EVENT"].map(order).fillna(
    wide["CLINICAL_EVENT"].str.extract(r"V(\d+)").astype(float)[0])
wide = wide.sort_values(["PATNO", "ev_rank"])
qalb = (wide.groupby("PATNO").first().reset_index()
        .rename(columns={"CLINICAL_EVENT": "qalb_event",
                         "CSF Albumin": "csf_albumin", "Plasma Albumin": "plasma_albumin"}))
qalb = qalb[["PATNO", "qalb_event", "csf_albumin", "plasma_albumin", "qalb"]]
qalb.to_csv(os.path.join(OUT, "qalbumin.csv"), index=False)

print(f"Q-albumin computed for {len(qalb)} subjects (paired CSF+plasma albumin).")
print(f"  event used: {qalb['qalb_event'].value_counts().to_dict()}")
print(f"  qalb distribution: min={qalb['qalb'].min():.2f} median={qalb['qalb'].median():.2f} "
      f"p95={qalb['qalb'].quantile(.95):.2f} max={qalb['qalb'].max():.2f}")

# ---- 2. overlap with the PD progression cohort (curated cut) ----
cut = pd.read_parquet(os.path.join(OUT, "curated_cut.parquet"))
pd_pat = set(cut.loc[cut["COHORT"] == 1, "PATNO"].unique())  # COHORT 1 = Parkinson's Disease
print(f"\nPD subjects (curated cut, COHORT==1): {len(pd_pat)}")

# progression-eligible: >=4 visits with the primary scale (use UPDRS-III and UPDRS-II)
def n_visits(scale):
    s = cut[cut["PATNO"].isin(pd_pat)].dropna(subset=[scale])
    return s.groupby("PATNO")["EVENT_ID"].nunique()

v3 = n_visits("updrs3_score"); v2 = n_visits("updrs2_score")
ge4_v3 = set(v3[v3 >= 4].index); ge4_v2 = set(v2[v2 >= 4].index)

qalb_pd = pd_pat & set(qalb["PATNO"])
print(f"PD with Q-albumin: {len(qalb_pd)}")

# baseline-modality availability among PD (curated cut BL row)
bl = cut[(cut["PATNO"].isin(pd_pat)) & (cut["EVENT_ID"] == "BL")].set_index("PATNO")
def have(col):
    return set(bl.index[bl[col].notna()])
csf_core = have("ptau") | have("IU_pTau181_CSF")
datscan = have("MIA_PUTAMEN_BILAT")
saa = have("CSFSAA")

print("\n--- Binding-constraint (Set A) funnel, PD subjects ---")
print(f"PD with >=4 UPDRS-III visits:                 {len(pd_pat & ge4_v3)}")
print(f"  + Q-albumin:                                {len(pd_pat & ge4_v3 & qalb_pd)}")
print(f"  + Q-albumin + baseline DaTSCAN:             {len(pd_pat & ge4_v3 & qalb_pd & datscan)}")
print(f"  + Q-albumin + DaTSCAN + baseline CSF:       {len(pd_pat & ge4_v3 & qalb_pd & datscan & csf_core)}")
print(f"  + Q-alb + DaTSCAN + CSF + SAA (full Set A):  {len(pd_pat & ge4_v3 & qalb_pd & datscan & csf_core & saa)}")
print(f"\nSet B (PD, >=4 visits, DaTSCAN + CSF, Q-alb optional): "
      f"{len(pd_pat & ge4_v3 & datscan & csf_core)}")

setA = sorted(pd_pat & ge4_v3 & qalb_pd & datscan & csf_core)
pd.Series(setA, name="PATNO").to_csv(os.path.join(OUT, "setA_patnos.csv"), index=False)
print(f"\nWrote setA_patnos.csv ({len(setA)} subjects) and qalbumin.csv")
