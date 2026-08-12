"""
MR step 03: BBB proteins (PDGFRB, ICAM1) -> PD PROGRESSION (Tan 2020).
Outcomes: cognitive (GCST011037), motor (GCST011038), composite (GCST011039).
Continuous outcomes -> causal estimate = change in progression (SD units) per 1-SD protein.
"""
import os, json, urllib.request
import numpy as np
from math import erf, sqrt

TOKEN = open(os.path.expanduser("~/.opengwas_token")).read().strip()
API = "https://api.opengwas.io/api"; OUT = "data/raw/mr"
OUTCOMES = {"cognitive": "ebi-a-GCST011037", "motor": "ebi-a-GCST011038",
            "composite": "ebi-a-GCST011039"}

def api_post(path, body):
    req = urllib.request.Request(f"{API}/{path}", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
                 "X-Api-Source": "ars-mr"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())

instr = json.load(open(f"{OUT}/instruments.json"))
rsids = sorted({s["rsid"] for v in instr.values() for s in v["snps"]})
assoc = api_post("associations", {"id": list(OUTCOMES.values()), "variant": rsids})
# index by (dataset_id, rsid)
amap = {}
for a in assoc:
    amap[(a["id"], a["rsid"])] = a

def pnorm(z): return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

print("=== MR: BBB protein -> PD progression (Tan 2020) ===")
print(f"{'protein':8s} {'outcome':10s} {'nSNP':4s} {'beta/SD':>8s} {'95% CI':>16s} {'p':>9s}")
rows = []
for prot, v in instr.items():
    for oname, oid in OUTCOMES.items():
        bx, by, byse = [], [], []
        for s in v["snps"]:
            o = amap.get((oid, s["rsid"]))
            if not o:
                continue
            ea_x, nea_x, b_x = s["ea"], s["nea"], float(s["beta"])
            ea_y, nea_y, b_y = o["ea"], o["nea"], float(o["beta"])
            if ea_x == ea_y and nea_x == nea_y:
                pass
            elif ea_x == nea_y and nea_x == ea_y:
                b_y = -b_y
            else:
                continue
            bx.append(b_x); by.append(b_y); byse.append(float(o["se"]))
        if not bx:
            print(f"{prot:8s} {oname:10s}  no SNP in outcome"); continue
        bx, by, byse = map(np.array, (bx, by, byse))
        if len(bx) == 1:
            beta = by[0] / bx[0]; se = abs(byse[0] / bx[0])
        else:
            w = 1 / byse ** 2
            beta = np.sum(w * bx * by) / np.sum(w * bx ** 2)
            se = np.sqrt(1 / np.sum(w * bx ** 2))
        p = pnorm(beta / se)
        rows.append({"protein": prot, "outcome": oname, "nsnp": len(bx),
                     "beta": beta, "se": se, "p": p})
        print(f"{prot:8s} {oname:10s} {len(bx):4d} {beta:+8.3f} "
              f"[{beta-1.96*se:+.3f},{beta+1.96*se:+.3f}] {p:9.2e}")

json.dump(rows, open(f"{OUT}/mr_progression.json", "w"), indent=2)
print("\nbeta/SD = change in progression (SD units) per 1-SD higher genetically-predicted protein.")
print("Tan progression GWAS are small (n~2800); CIs are wide. Bonferroni: 2 proteins x 3 outcomes.")
