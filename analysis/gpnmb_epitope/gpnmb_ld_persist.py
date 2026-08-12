"""Persist the GPNMB cross-platform LD analysis into the epitope artifact.

The epitope check came back all-non-coding, so the binding-artifact explanation
for the GPNMB cross-platform discordance is not supported at lead-variant level
and the discordance was left unexplained. This adds the LD evidence, which does
offer a concrete explanation:

  - the two platforms' instrument sets share NO rsID
  - exactly ONE cross-platform pair is in strong LD (r2 = 0.705)
  - but each platform's STRONGEST instrument is essentially independent of
    everything on the other platform

i.e. the deCODE and UKB-PPP MR estimates are largely NOT instrumented by the
same genetic variation, which is a better-grounded explanation for the
discordance than an epitope difference.
"""
import json, os, subprocess, tempfile
import pandas as pd

ROOT = os.path.expanduser("~/pd_repro")
INST = os.path.join(ROOT, "data/processed/mr/poscontrol_instruments.csv")
ART = os.path.join(ROOT, "data/processed/mr/epitope_check_gpnmb.json")
EUR = os.path.join(ROOT, "data/raw/mr/ld_1000g_eur/EUR")

inst = pd.read_csv(INST)
g = inst[inst.protein == "GPNMB"].copy()
tmp = tempfile.mkdtemp()
snp = os.path.join(tmp, "snps.txt")
open(snp, "w").write("\n".join(g.rsid.dropna().unique()) + "\n")

subprocess.run(["plink", "--bfile", EUR, "--extract", snp, "--r2",
                "--ld-window", "99999", "--ld-window-kb", "10000",
                "--ld-window-r2", "0", "--out", os.path.join(tmp, "ld")],
               capture_output=True)
ld = pd.read_csv(os.path.join(tmp, "ld.ld"), sep=r"\s+")
plat = dict(zip(g.rsid, g.platform))
F = dict(zip(g.rsid, g.F))
ld["p_a"] = ld.SNP_A.map(plat)
ld["p_b"] = ld.SNP_B.map(plat)
cross = ld[ld.p_a != ld.p_b].sort_values("R2", ascending=False)

# strongest instrument per platform, and its best cross-platform LD partner
strongest = {}
for p in ("decode", "ukbppp"):
    sub = g[g.platform == p]
    top = sub.loc[sub.F.idxmax()]
    c = cross[(cross.SNP_A == top.rsid) | (cross.SNP_B == top.rsid)]
    strongest[p] = {
        "rsid": top.rsid, "F": float(top.F),
        "max_cross_platform_r2": float(c.R2.max()) if len(c) else None}

pairs = [{"snp_a": r.SNP_A, "platform_a": r.p_a, "snp_b": r.SNP_B,
          "platform_b": r.p_b, "r2": float(r.R2)} for r in cross.itertuples()]

block = {
    "ld_reference": "1000 Genomes EUR (N=503) — the same weak panel already "
                    "disclosed as the likely cause of the SuSiE credible-set "
                    "failures; these r2 values are correspondingly imprecise",
    "shared_rsids_between_platforms": 0,
    "n_cross_platform_pairs": len(pairs),
    "max_cross_platform_r2": float(cross.R2.max()),
    "n_pairs_r2_above_0.2": int((cross.R2 > 0.2).sum()),
    "strongest_instrument_per_platform": strongest,
    "cross_platform_pairs": pairs,
    "reading": (
        "The instrument sets share no rsID. Exactly one cross-platform pair is "
        "in strong LD (rs191297708 UKB / rs79914289 deCODE, r2=0.705); all "
        "other 14 pairs are below r2=0.08. Critically, the STRONGEST instrument "
        "on each platform is nearly independent of everything on the other "
        "(deCODE rs10250602 F=715, max cross r2=0.074; UKB rs78840640 F=1405, "
        "max cross r2=0.011). So the two platforms' MR estimates are largely "
        "NOT instrumented by the same genetic variation. This is a concrete "
        "explanation for the discordance and does not require an epitope "
        "difference — but it is an observation about instrument selection, not "
        "proof, and it does NOT restore the pre-registered both-platform gate."),
}

d = json.load(open(ART))
d["cross_platform_ld"] = block
d["verdict_combined"] = (
    "Epitope/binding artifact NOT supported (0/8 instruments coding on either "
    "platform). The likelier explanation is that the platforms instrument "
    "different variation: no shared rsID, and each platform's strongest "
    "instrument is nearly independent of the other's. The UKB-PPP arm of the "
    "PDGFRB null REMAINS UNGATED and must stay disclosed.")
json.dump(d, open(ART, "w"), indent=2)

print("cross-platform pairs:", len(pairs))
print("max r2:", round(float(cross.R2.max()), 4))
print("pairs with r2 > 0.2:", int((cross.R2 > 0.2).sum()))
for p, v in strongest.items():
    print(f"  strongest {p:8s} {v['rsid']:14s} F={v['F']:8.1f}  "
          f"max cross-platform r2 = {v['max_cross_platform_r2']:.4f}")
print(f"\nUpdated {ART}")
