"""
MR step 04: disease-agnostic scan. Test the strong BBB-protein instruments
(PDGFRB rs2304058, ICAM1 rs5498) for CAUSAL effects across BBB-relevant CNS diseases.
Find where BBB is causal (the headline), not assume PD.
Exploratory MR-pheWAS; hits get followed up with coloc + replication.
"""
import os, json, re, urllib.request
import numpy as np
from math import erf, sqrt, exp

TOKEN = open(os.path.expanduser("~/.opengwas_token")).read().strip()
API = "https://api.opengwas.io/api"; OUT = "data/raw/mr"

def api_post(path, body):
    req = urllib.request.Request(f"{API}/{path}", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
                 "X-Api-Source": "ars-mr"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())

recs = list(json.load(open(f"{OUT}/gwasinfo_all.json")).values())

# curated BBB-relevant CNS diseases; pick the largest European case/control GWAS each
DISEASES = {
    "Alzheimer's disease": r"alzheimer",
    "Multiple sclerosis": r"multiple sclerosis",
    "Ischaemic stroke": r"ischa?emic stroke|ischemic stroke",
    "Small-vessel stroke": r"small vessel",
    "ALS": r"amyotrophic lateral",
    "Epilepsy": r"^epilepsy|epilepsy$|focal epilepsy|generalized epilepsy",
    "Migraine": r"migraine",
    "Intracerebral haemorrhage": r"intracerebral h",
    "Frontotemporal dementia": r"frontotemporal",
    "Lewy body dementia": r"lewy body",
    "Bipolar disorder": r"bipolar disorder",   # non-vascular CNS contrast
}
EXCL = re.compile(r"illness|father|mother|sibling|family history", re.I)

chosen = {}
for name, pat in DISEASES.items():
    cands = [r for r in recs if re.search(pat, str(r.get("trait", "")), re.I)
             and not EXCL.search(str(r.get("trait", "")))
             and (r.get("ncase") or 0) >= 500
             and str(r.get("population", "")).lower().startswith("euro")]
    if not cands:
        cands = [r for r in recs if re.search(pat, str(r.get("trait", "")), re.I)
                 and not EXCL.search(str(r.get("trait", ""))) and (r.get("sample_size") or 0) > 2000]
    if cands:
        best = max(cands, key=lambda r: (r.get("ncase") or r.get("sample_size") or 0))
        chosen[name] = best

print("=== chosen disease GWAS ===")
for n, r in chosen.items():
    print(f"  {n:26s} {r['id']:22s} ncase={r.get('ncase')} N={r.get('sample_size')} {r.get('author')} {r.get('year')}")

instr = json.load(open(f"{OUT}/instruments.json"))
ids = [r["id"] for r in chosen.values()]
rsids = sorted({s["rsid"] for v in instr.values() for s in v["snps"]})
assoc = api_post("associations", {"id": ids, "variant": rsids})
amap = {(a["id"], a["rsid"]): a for a in assoc}

def pnorm(z): return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

print("\n=== MR: BBB protein -> disease (OR per SD protein; Wald, single cis-SNP) ===")
print(f"{'protein':7s} {'disease':26s} {'OR':>6s} {'95% CI':>14s} {'p':>9s}")
rows = []
for prot, v in instr.items():
    s = v["snps"][0]  # lead cis-SNP
    for name, r in chosen.items():
        o = amap.get((r["id"], s["rsid"]))
        if not o:
            continue
        b_x = float(s["beta"])
        ea_x, nea_x = s["ea"], s["nea"]
        ea_y, nea_y, b_y, se_y = o["ea"], o["nea"], float(o["beta"]), float(o["se"])
        if ea_x == ea_y and nea_x == nea_y:
            pass
        elif ea_x == nea_y and nea_x == ea_y:
            b_y = -b_y
        else:
            continue
        beta = b_y / b_x; se = abs(se_y / b_x); p = pnorm(beta / se)
        rows.append({"protein": prot, "disease": name, "id": r["id"], "beta": beta, "se": se, "p": p, "or": exp(beta)})
        flag = "  ***" if p < 0.05 / (len(chosen) * 2) else ("  *" if p < 0.05 else "")
        print(f"{prot:7s} {name:26s} {exp(beta):6.2f} [{exp(beta-1.96*se):4.2f},{exp(beta+1.96*se):4.2f}] {p:9.2e}{flag}")

json.dump(rows, open(f"{OUT}/mr_phewas.json", "w"), indent=2)
nsig = sum(1 for r in rows if r["p"] < 0.05 / (len(chosen) * 2))
print(f"\n*** = passes Bonferroni ({len(chosen)*2} tests). {nsig} causal hit(s). "
      "Hits need colocalization + replication before any claim.")
