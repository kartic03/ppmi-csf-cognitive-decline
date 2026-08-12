"""GPNMB EPITOPE CHECK — the arm that leaves the UKB-PPP null ungated.

THE PROBLEM THIS ADDRESSES. The pre-registered positive-control gate requires
GPNMB to recover on BOTH pQTL platforms. It recovers strongly on deCODE
(OR 1.492 [1.289, 1.727], SomaScan aptamer) and flatly fails on UKB-PPP
(OR 0.994 [0.896, 1.102], Olink antibody) -- and NOT for want of instruments,
since UKB-PPP yields MORE of them (5 vs 3). The consequence is that the UKB-PPP
arm of the PDGFRB null is ungated. The project has recorded a SomaScan-vs-Olink
epitope difference as a candidate explanation but never tested it.

THE HYPOTHESIS. Affinity reagents bind an epitope. A variant that changes the
protein sequence can change binding without changing abundance, so a cis-pQTL
can be a binding artifact rather than a real expression signal -- and aptamers
and antibodies bind different epitopes, so the same variant can behave
differently across platforms. If the platform whose instruments are CODING is
also the platform driving the discordance, the artifact explanation gains
support. If every instrument on both platforms is non-coding, it loses support
and the discordance needs a different explanation.

WHAT THIS CAN AND CANNOT SETTLE. Only the lead clumped variants are annotated,
not their high-LD proxies -- the same limitation already disclosed for the
PDGFRB/ICAM1 check. A non-coding lead can still tag a coding variant. So a null
result here BOUNDS the artifact hypothesis, it does not refute it.
"""
import json, os, sys, time
import urllib.request
import pandas as pd

ROOT = os.path.expanduser("~/pd_repro")
INST = os.path.join(ROOT, "data/processed/mr/poscontrol_instruments.csv")
OUT = os.path.join(ROOT, "data/processed/mr/epitope_check_gpnmb.json")
VEP = "https://rest.ensembl.org/vep/human/id/{}?content-type=application/json"
CODING = {"missense_variant", "stop_gained", "stop_lost", "start_lost",
          "frameshift_variant", "inframe_insertion", "inframe_deletion",
          "protein_altering_variant", "coding_sequence_variant",
          "splice_acceptor_variant", "splice_donor_variant"}


def vep(rsid, tries=3):
    for a in range(tries):
        try:
            with urllib.request.urlopen(VEP.format(rsid), timeout=45) as r:
                return json.load(r)
        except Exception as e:
            if a == tries - 1:
                print(f"    VEP failed for {rsid}: {e}")
                return None
            time.sleep(3 * (a + 1))
    return None


def main():
    df = pd.read_csv(INST)
    df = df[df["protein"] == "GPNMB"].copy()
    print(f"GPNMB instruments: {len(df)} "
          f"({(df.platform == 'decode').sum()} deCODE, "
          f"{(df.platform == 'ukbppp').sum()} UKB-PPP)\n")

    rows = []
    for _, r in df.iterrows():
        rsid = r["rsid"]
        rec = {"protein": "GPNMB", "platform": r["platform"], "rsid": rsid,
               "pos": int(r["pos"]) if pd.notna(r["pos"]) else None,
               "F": float(r["F"]) if pd.notna(r["F"]) else None,
               "palindromic": bool(r["palindromic"]),
               "most_severe": None, "coding": None,
               "protein_change": None, "consequences": [], "vep_ok": False}
        if not isinstance(rsid, str) or not rsid.startswith("rs"):
            print(f"  {r['platform']:8s} {str(rsid):14s} — no usable rsID, skipped")
            rows.append(rec); continue

        j = vep(rsid)
        if not j:
            rows.append(rec); continue
        rec["vep_ok"] = True
        e = j[0]
        rec["most_severe"] = e.get("most_severe_consequence")
        cons, prot = set(), None
        for tc in e.get("transcript_consequences", []):
            cons.update(tc.get("consequence_terms", []))
            if tc.get("amino_acids") and tc.get("protein_start"):
                prot = f"{tc['amino_acids']}@{tc['protein_start']}"
        rec["consequences"] = sorted(cons)
        rec["coding"] = bool(cons & CODING)
        rec["protein_change"] = prot
        flag = "  <== CODING" if rec["coding"] else ""
        print(f"  {r['platform']:8s} {rsid:14s} F={rec['F']:>8.1f}  "
              f"{rec['most_severe']}{flag}")
        rows.append(rec)
        time.sleep(0.4)

    ok = [r for r in rows if r["vep_ok"]]
    dec = [r for r in ok if r["platform"] == "decode"]
    ukb = [r for r in ok if r["platform"] == "ukbppp"]
    n_cod_dec = sum(1 for r in dec if r["coding"])
    n_cod_ukb = sum(1 for r in ukb if r["coding"])

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  deCODE (SomaScan, aptamer) : {n_cod_dec}/{len(dec)} coding")
    print(f"  UKB-PPP (Olink, antibody)  : {n_cod_ukb}/{len(ukb)} coding")
    print()

    if n_cod_dec == 0 and n_cod_ukb == 0:
        verdict = ("no_coding_variants_on_either_platform")
        print("  EVERY annotated instrument on BOTH platforms is NON-CODING.")
        print("  The epitope/binding-artifact explanation for the cross-platform")
        print("  discordance is NOT SUPPORTED at the level of the lead variants.")
        print("  The GPNMB discordance therefore remains UNEXPLAINED, and the")
        print("  UKB-PPP arm of the PDGFRB null stays ungated and disclosed.")
        print("  Do NOT upgrade this to 'the discordance is explained'.")
    elif n_cod_dec > n_cod_ukb:
        verdict = "coding_variants_concentrated_on_decode"
        print("  Coding variants sit on the platform that RECOVERS the control.")
        print("  Consistent with a SomaScan binding artifact inflating the deCODE")
        print("  signal -- which would weaken the positive control itself, not")
        print("  just explain the UKB failure. Follow up before relying on it.")
    else:
        verdict = "coding_variants_present_see_detail"
        print("  Coding variants are present; inspect the per-variant detail")
        print("  before drawing a platform-level conclusion.")

    print("\n  LIMITATION (unchanged from the PDGFRB/ICAM1 check): only lead")
    print("  clumped variants were annotated, not high-LD proxies. A non-coding")
    print("  lead can still tag a coding variant, so this BOUNDS the artifact")
    print("  hypothesis rather than refuting it.")

    json.dump({"source": "Ensembl VEP REST",
               "exposure": "GPNMB (positive control)",
               "n_instruments": len(rows), "n_annotated": len(ok),
               "n_coding_decode": n_cod_dec, "n_coding_ukbppp": n_cod_ukb,
               "verdict": verdict,
               "context": {
                   "decode_or": 1.491785, "decode_ci": [1.288809, 1.726729],
                   "ukbppp_or": 0.994015, "ukbppp_ci": [0.896358, 1.102312],
                   "note": "deCODE recovers, UKB-PPP does not; UKB-PPP is not "
                           "instrument-poor (5 vs 3)"},
               "limitation": "lead clumped variants only, not high-LD proxies; "
                             "a null bounds the artifact hypothesis, it does not "
                             "refute it",
               "variants": rows}, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
