"""
MR step 02: two-sample MR of BBB proteins (with strong cis instruments) -> PD risk
(Nalls 2019, ieu-b-7). Single-cis-SNP = Wald ratio; >=2 = IVW. Pure-numpy.
Causal estimate = OR for PD per 1-SD increase in (genetically predicted) plasma protein.
Harmonize exposure/outcome alleles before MR.
"""
import os, json, urllib.request
import numpy as np

TOKEN = open(os.path.expanduser("~/.opengwas_token")).read().strip()
API = "https://api.opengwas.io/api"
OUT = "data/raw/mr"
PD = "ieu-b-7"

def api_post(path, body):
    req = urllib.request.Request(f"{API}/{path}", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
                 "X-Api-Source": "ars-mr"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())

instruments = json.load(open(f"{OUT}/instruments.json"))
rsids = sorted({s["rsid"] for v in instruments.values() for s in v["snps"]})
# PD associations for all instrument SNPs
pd_assoc = api_post("associations", {"id": [PD], "variant": rsids})
pdmap = {a["rsid"]: a for a in pd_assoc}

print("=== Two-sample MR: BBB protein -> Parkinson's disease (Nalls 2019) ===")
print(f"{'protein':8s} {'nSNP':4s} {'OR/SD':>7s} {'95% CI':>16s} {'p':>9s}  method")
results = {}
for prot, v in instruments.items():
    bx, by, bxse, byse = [], [], [], []
    used = []
    for s in v["snps"]:
        rs = s["rsid"]
        if rs not in pdmap:
            continue
        o = pdmap[rs]
        ea_x, nea_x, b_x = s["ea"], s["nea"], float(s["beta"])
        ea_y, nea_y, b_y = o["ea"], o["nea"], float(o["beta"])
        # harmonize to the protein effect allele
        if ea_x == ea_y and nea_x == nea_y:
            pass
        elif ea_x == nea_y and nea_x == ea_y:
            b_y = -b_y
        else:
            continue  # allele mismatch / ambiguous -> drop
        bx.append(b_x); bxse.append(float(s["se"]))
        by.append(b_y); byse.append(float(o["se"])); used.append(rs)
    bx, by, bxse, byse = map(np.array, (bx, by, bxse, byse))
    if len(bx) == 0:
        print(f"{prot:8s}  no harmonized SNPs"); continue
    if len(bx) == 1:
        beta = by[0] / bx[0]
        se = abs(byse[0] / bx[0])  # leading-order Wald SE
        method = "Wald"
    else:
        w = 1.0 / (byse ** 2)  # IVW (fixed effect), weight by outcome variance
        beta = np.sum(w * bx * by) / np.sum(w * bx ** 2)
        se = np.sqrt(1.0 / np.sum(w * bx ** 2))
        method = f"IVW({len(bx)})"
    z = beta / se
    from math import erf, sqrt, exp
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    lo, hi = beta - 1.96 * se, beta + 1.96 * se
    results[prot] = {"snps": used, "beta": beta, "se": se, "p": p,
                     "or": exp(beta), "or_lo": exp(lo), "or_hi": exp(hi), "method": method}
    print(f"{prot:8s} {len(bx):4d} {exp(beta):7.2f} [{exp(lo):5.2f},{exp(hi):5.2f}]  {p:9.2e}  {method}  "
          f"({','.join(used)})")

json.dump(results, open(f"{OUT}/mr_results.json", "w"), indent=2)
print("\nOR/SD = PD odds ratio per 1-SD higher genetically-predicted plasma protein.")
print("NOTE: single-SNP Wald has no pleiotropy sensitivity; colocalization (next step) is the")
print("key robustness check to separate causality from LD confounding.")
