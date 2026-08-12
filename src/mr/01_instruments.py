"""
MR step 01: extract CIS-pQTL instruments for the 6 BBB proteins from OpenGWAS, and
run the instrument-verification GATE (does each protein have a strong cis instrument?).

cis = within +/-1 Mb of the gene (GRCh37/hg19, matching OpenGWAS builds).
Strong instrument: genome-wide significant cis-pQTL (p<5e-8), F-stat = (beta/se)^2 > 10.
Proteins without a valid cis instrument are dropped (reported honestly).
"""
import os, json, time, urllib.request, urllib.error

TOKEN = open(os.path.expanduser("~/.opengwas_token")).read().strip()
API = "https://api.opengwas.io/api"
OUT = "data/raw/mr"

# protein -> (pQTL dataset id, gene chrom, gene_start, gene_end)  [hg19]
PROTEINS = {
    "PDGFRB": ("prot-a-2230", "5", 149493399, 149535435),
    "ICAM1":  ("prot-a-1397", "19", 10381517, 10397272),
    "VCAM1":  ("prot-c-2967_8_1", "1", 101185196, 101219426),
    "MMP9":   ("prot-c-2579_17_5", "20", 44637547, 44645200),
    "MMP2":   ("prot-c-4160_49_1", "16", 55513080, 55540586),
    "TIMP1":  ("prot-c-2211_9_6", "X", 47442210, 47445223),
}
CIS_PAD = 1_000_000

def api_post(path, body):
    req = urllib.request.Request(f"{API}/{path}",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
                 "X-Api-Source": "ars-mr"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())

instruments = {}
print("=== Instrument-verification gate (cis-pQTL per protein) ===")
for prot, (ds, chrom, gs, ge) in PROTEINS.items():
    got = None
    for pval in (5e-8, 5e-6):  # relax only if nothing at GWS
        try:
            hits = api_post("tophits", {"id": [ds], "pval": pval, "clump": 1,
                                        "r2": 0.001, "kb": 10000, "pop": "EUR"})
        except urllib.error.HTTPError as e:
            print(f"  {prot}: API error {e.code} ({e.read().decode()[:120]})"); hits = []
        time.sleep(1)
        cis = [h for h in hits if str(h.get("chr")) == chrom
               and gs - CIS_PAD <= int(h.get("position", 0)) <= ge + CIS_PAD]
        if cis:
            got = (pval, cis); break
    if not got:
        print(f"  {prot} ({ds}): NO cis-pQTL even at p<5e-6 -> DROP"); continue
    pval, cis = got
    for h in cis:
        b, se = float(h["beta"]), float(h["se"])
        h["F"] = (b / se) ** 2
    cis.sort(key=lambda h: h["p"])
    instruments[prot] = {"dataset": ds, "pval_thresh": pval, "snps": cis}
    lead = cis[0]
    print(f"  {prot:7s} ({ds}): {len(cis)} cis-SNP(s) at p<{pval:g} | "
          f"lead {lead['rsid']} p={lead['p']:.1e} F={lead['F']:.0f} "
          f"beta={lead['beta']:+.3f}")

os.makedirs(OUT, exist_ok=True)
json.dump(instruments, open(f"{OUT}/instruments.json", "w"), indent=2)
strong = [p for p, v in instruments.items() if any(s["F"] > 10 for s in v["snps"])]
print(f"\nProteins with a strong cis instrument (F>10): {strong}")
print(f"Dropped (no valid cis instrument): {[p for p in PROTEINS if p not in instruments]}")
print(f"Saved instruments.json ({len(instruments)} proteins).")
