#!/usr/bin/env python3
"""
17_synthesize.py
----------------
Cross-platform MR synthesis for the BBB-protein panel.

Integrates MR estimates, colocalization, positive-control gate, and
pleiotropy scan into one per-protein causal-call table.

Synthesis rules
---------------
1. Positive-control gate: if GPNMB does NOT recover the expected direction,
   no protein may receive causal_call="null" (pipeline unproven).
2. "null" only when BOTH platforms give verdict "null" AND both PP4 < 0.8
   AND the positive control passed.
3. "discordant" when the two platforms disagree (one null, one positive)
   regardless of coloc (pp4=0 on the positive side means the positive signal
   does not share a causal variant with PD risk -- likely an artifact).
4. "inconclusive" for single-platform results, underpowered estimates, or
   wide-CI positives with pp4=0.
5. "not_testable" for TIMP1 (no trans-instruments from X-GWAS, no
   cis-pQTLs that are instrument-strength).

Inputs
------
  data/processed/mr/mr_results.json
  data/processed/mr/coloc_results.json
  data/processed/mr/poscontrol_result.json
  data/raw/mr/mr_phewas.json

Outputs
-------
  data/processed/mr/mr_final_table.csv   (one row per study protein)
  results/tables/mr_summary.csv           (publication-ready summary + footnote)
"""

import csv
import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

MR_RESULTS_PATH  = ROOT / "data/processed/mr/mr_results.json"
COLOC_PATH       = ROOT / "data/processed/mr/coloc_results.json"
POSCONTROL_PATH  = ROOT / "data/processed/mr/poscontrol_result.json"
PHEWAS_PATH      = ROOT / "data/raw/mr/mr_phewas.json"

OUT_FINAL   = ROOT / "data/processed/mr/mr_final_table.csv"
OUT_SUMMARY = ROOT / "results/tables/mr_summary.csv"

# Canonical protein order (matches analysis plan)
STUDY_PROTEINS = ["PDGFRB", "ICAM1", "VCAM1", "MMP9", "MMP2", "TIMP1"]

PRIMARY_PROTEIN = "PDGFRB"   # canonical BBB-integrity instrument

# Colocalization threshold (pre-registered)
PP4_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_mr(path: Path) -> dict:
    """
    Return {(protein, platform): record} for study proteins.
    TIMP1's "both" sentinel is stored under (TIMP1, "both").
    """
    records = json.loads(path.read_text())
    out: dict = {}
    for rec in records:
        prot = rec["protein"]
        if prot not in STUDY_PROTEINS:
            continue
        plat = rec["platform"]
        out[(prot, plat)] = rec
    return out


def _load_coloc(path: Path):
    """
    Return ({(protein, platform): pp4}, {(protein, platform): method},
            {(protein, platform): pp4_estimable}, positive_control_record).
    GPNMB (positive_control=True) is separated and not stored in the protein map.

    IMPORTANT — pp4 = 0.0 is not always a posterior.  Where coloc.susie found no
    overlapping credible sets (method 'coloc.susie_noCS'), or a run errored or
    had no data ('*_error', '*_nodata'), PP4 is NOT ESTIMABLE and the stored 0.0
    is a field-compatibility placeholder.  The estimable map carries that
    distinction so the published table can state it.  Older coloc_results.json
    files lack the flag; they are treated as estimable only if the method label
    does not itself mark a non-computed state.
    """
    records = json.loads(path.read_text())
    pp4_map: dict = {}
    method_map: dict = {}
    estimable_map: dict = {}
    pc_rec = None
    for rec in records:
        if rec.get("positive_control"):
            pc_rec = rec
            continue
        prot = rec["protein"]
        if prot not in STUDY_PROTEINS:
            continue
        plat = rec["platform"]
        method = rec.get("method", "")
        pp4_map[(prot, plat)] = rec["pp4"]
        method_map[(prot, plat)] = method
        if "pp4_estimable" in rec:
            estimable_map[(prot, plat)] = bool(rec["pp4_estimable"])
        else:
            # Back-compat inference for artifacts written before the flag.
            estimable_map[(prot, plat)] = not (
                method.endswith("_noCS")
                or method.endswith("_error")
                or method.endswith("_nodata")
            )
    return pp4_map, method_map, estimable_map, pc_rec


def _load_pleiotropy(path: Path) -> dict:
    """
    Return {protein: {"flag": bool|None, "notes": str}}.

    flag=True  : one or more phewas hits at p < 0.05 (potential pleiotropy)
    flag=False : in scan, no significant hits
    flag=None  : protein not in the phewas scan
    """
    records = json.loads(path.read_text())
    # Collect significant associations per protein
    sig_hits: dict = {}
    all_proteins: set = set()
    for rec in records:
        prot = rec["protein"]
        if prot not in STUDY_PROTEINS:
            continue
        all_proteins.add(prot)
        if rec["p"] < 0.05:
            sig_hits.setdefault(prot, []).append(
                f"{rec['disease']} (OR={rec['or']:.3f}, p={rec['p']:.4f})"
            )

    result: dict = {}
    for prot in STUDY_PROTEINS:
        if prot not in all_proteins:
            result[prot] = {"flag": None, "notes": "not in phewas scan"}
        elif prot in sig_hits:
            result[prot] = {"flag": True, "notes": "; ".join(sig_hits[prot])}
        else:
            result[prot] = {"flag": False, "notes": "no significant phewas hits"}
    return result


# ---------------------------------------------------------------------------
# Causal-call logic
# ---------------------------------------------------------------------------

def _causal_call(
    protein: str,
    decode_verdict,
    ukbppp_verdict,
    platforms_tested: int,
    decode_pp4,
    ukbppp_pp4,
    poscontrol_passed: bool,
    coloc_pc_passed: bool,
) -> str:
    """
    Determine causal_call for one protein following the pre-registered rules.
    """
    # TIMP1 is always not_testable regardless of other fields
    if protein == "TIMP1":
        return "not_testable"

    # Both platforms tested
    if platforms_tested == 2:
        if decode_verdict == "null" and ukbppp_verdict == "null":
            # Concordant null -- both PP4 must be below threshold
            d_pp4 = decode_pp4 if decode_pp4 is not None else 0.0
            u_pp4 = ukbppp_pp4 if ukbppp_pp4 is not None else 0.0
            if d_pp4 < PP4_THRESHOLD and u_pp4 < PP4_THRESHOLD:
                # Gate on both MR and coloc positive controls
                if poscontrol_passed and coloc_pc_passed:
                    return "null"
                return "inconclusive"
            return "inconclusive"

        if decode_verdict == ukbppp_verdict:
            # Concordant non-null: both inconclusive, or (rarely) both positive
            if decode_verdict == "positive":
                d_pp4 = decode_pp4 if decode_pp4 is not None else 0.0
                u_pp4 = ukbppp_pp4 if ukbppp_pp4 is not None else 0.0
                # Both positive AND at least one colocalizes
                if d_pp4 >= PP4_THRESHOLD or u_pp4 >= PP4_THRESHOLD:
                    return "positive"
            return "inconclusive"

        # One null, one positive (or any other disagreement) -- discordant
        return "discordant"

    # Single platform
    v = decode_verdict if decode_verdict is not None else ukbppp_verdict
    if v == "positive":
        pp4 = decode_pp4 if decode_pp4 is not None else ukbppp_pp4
        pp4 = pp4 if pp4 is not None else 0.0
        # A single-platform positive is credited only with colocalization support
        return "positive" if pp4 >= PP4_THRESHOLD else "inconclusive"
    if v == "null":
        # Single-platform null is not a concordant cross-platform null
        return "inconclusive"
    # "inconclusive" or other
    return v if v else "inconclusive"


# ---------------------------------------------------------------------------
# Main synthesis
# ---------------------------------------------------------------------------

def synthesize():
    mr_map    = _load_mr(MR_RESULTS_PATH)
    pp4_map, coloc_method_map, coloc_estimable_map, coloc_pc = _load_coloc(COLOC_PATH)
    poscontrol = json.loads(POSCONTROL_PATH.read_text())
    pleiotropy = _load_pleiotropy(PHEWAS_PATH)

    poscontrol_passed: bool = poscontrol["recovers_expected_direction"]
    coloc_pc_passed: bool = (coloc_pc is not None) and (coloc_pc["pp4"] >= PP4_THRESHOLD)
    pc_or   = poscontrol["or"]
    pc_low  = poscontrol["ci_low"]
    pc_high = poscontrol["ci_high"]

    rows = []
    for protein in STUDY_PROTEINS:
        dec_rec = mr_map.get((protein, "decode"))
        ukb_rec = mr_map.get((protein, "ukbppp"))
        bot_rec = mr_map.get((protein, "both"))

        # TIMP1: sentinel record with platform="both"
        if bot_rec and bot_rec["verdict"] == "not_testable":
            rows.append({
                "protein":              protein,
                "role":                 "primary" if protein == PRIMARY_PROTEIN else "secondary",
                "platforms_tested":     0,
                "decode_verdict":       None,
                "ukbppp_verdict":       None,
                "decode_or":            None,
                "decode_ci":            None,
                "ukbppp_or":            None,
                "ukbppp_ci":            None,
                "decode_F":             None,
                "ukbppp_F":             None,
                "decode_pp4":           None,
                "ukbppp_pp4":           None,
                "min_detectable_or_80":        None,
                "ukbppp_min_detectable_or_80": None,
                "egger_intercept_p":    None,
                "wmedian_or":           None,
                "has_strand_ambiguous": False,
                "causal_call":          "not_testable",
                "pleiotropy_flag":      pleiotropy[protein]["flag"],
                "pleiotropy_notes":     pleiotropy[protein]["notes"],
            })
            continue

        platforms_tested = int(dec_rec is not None) + int(ukb_rec is not None)

        decode_verdict = dec_rec["verdict"] if dec_rec else None
        ukbppp_verdict = ukb_rec["verdict"] if ukb_rec else None

        # OR and 95% CI strings
        def _ci_str(rec):
            if rec is None or rec.get("ci_low") is None:
                return None
            return f"{rec['ci_low']:.3f}-{rec['ci_high']:.3f}"

        decode_or   = round(dec_rec["or"], 3) if dec_rec and dec_rec.get("or") is not None else None
        ukbppp_or   = round(ukb_rec["or"], 3) if ukb_rec and ukb_rec.get("or") is not None else None
        decode_ci   = _ci_str(dec_rec)
        ukbppp_ci   = _ci_str(ukb_rec)
        decode_F    = dec_rec["F"] if dec_rec else None
        ukbppp_F    = ukb_rec["F"] if ukb_rec else None

        # min-detectable: prefer decode, else ukbppp; also store ukbppp separately
        primary_rec = dec_rec or ukb_rec
        min_det = primary_rec.get("min_detectable_or_80") if primary_rec else None
        ukb_min_det = ukb_rec.get("min_detectable_or_80") if ukb_rec else None

        # Sensitivity fields (MR-Egger intercept, weighted-median OR)
        egger_intercept_p = None
        wmedian_or_val    = None
        for rec in [dec_rec, ukb_rec]:
            if rec and rec.get("egger_intercept_p") is not None and egger_intercept_p is None:
                egger_intercept_p = rec["egger_intercept_p"]
                wmedian_or_val    = rec.get("wmedian_or")

        has_strand_amb = any(
            r.get("has_strand_ambiguous", False)
            for r in [dec_rec, ukb_rec]
            if r is not None
        )

        decode_pp4 = pp4_map.get((protein, "decode"))
        ukbppp_pp4 = pp4_map.get((protein, "ukbppp"))

        call = _causal_call(
            protein, decode_verdict, ukbppp_verdict, platforms_tested,
            decode_pp4, ukbppp_pp4, poscontrol_passed, coloc_pc_passed,
        )

        # Final safety gate: if either positive control (MR or coloc) failed,
        # no null is allowed -- a broken pipeline must not license a null call.
        if (not poscontrol_passed or not coloc_pc_passed) and call == "null":
            call = "inconclusive"

        rows.append({
            "protein":              protein,
            "role":                 "primary" if protein == PRIMARY_PROTEIN else "secondary",
            "platforms_tested":     platforms_tested,
            "decode_verdict":       decode_verdict,
            "ukbppp_verdict":       ukbppp_verdict,
            "decode_or":            decode_or,
            "decode_ci":            decode_ci,
            "ukbppp_or":            ukbppp_or,
            "ukbppp_ci":            ukbppp_ci,
            "decode_F":             decode_F,
            "ukbppp_F":             ukbppp_F,
            "decode_pp4":           decode_pp4,
            "ukbppp_pp4":           ukbppp_pp4,
            # Method label + estimability travel WITH the PP4 so a placeholder
            # zero can never be read as a computed posterior downstream.
            "decode_coloc_method":  coloc_method_map.get((protein, "decode")),
            "ukbppp_coloc_method":  coloc_method_map.get((protein, "ukbppp")),
            "decode_pp4_estimable": coloc_estimable_map.get((protein, "decode")),
            "ukbppp_pp4_estimable": coloc_estimable_map.get((protein, "ukbppp")),
            "min_detectable_or_80":        min_det,
            "ukbppp_min_detectable_or_80": ukb_min_det,
            "egger_intercept_p":    egger_intercept_p,
            "wmedian_or":           wmedian_or_val,
            "has_strand_ambiguous": has_strand_amb,
            "causal_call":          call,
            "pleiotropy_flag":      pleiotropy[protein]["flag"],
            "pleiotropy_notes":     pleiotropy[protein]["notes"],
        })

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Write mr_final_table.csv
    # ------------------------------------------------------------------
    OUT_FINAL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FINAL, index=False, quoting=csv.QUOTE_NONNUMERIC)

    # ------------------------------------------------------------------
    # Write results/tables/mr_summary.csv (publication-ready)
    # ------------------------------------------------------------------
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    def _or_ci(or_val, ci):
        if or_val is None:
            return None
        return f"{or_val:.3f} [{ci}]" if ci else str(or_val)

    def _pp4_cell(val, estimable):
        """Publication cell for PP4.

        A non-estimable PP4 is rendered 'NE (no credible sets)' rather than
        '0.000'.  Printing a placeholder zero here is what let 'no
        colocalization (PP4 = 0)' enter the write-up for loci where no
        posterior was ever computed.
        """
        if val is None:
            return None
        if estimable is False:
            return "NE (no credible sets)"
        return f"{val:.3f}"

    summary_rows = []
    for _, r in df.iterrows():
        summary_rows.append({
            "protein":        r["protein"],
            "role":           r["role"],
            "decode_or_ci":   _or_ci(r["decode_or"], r["decode_ci"]),
            "ukbppp_or_ci":   _or_ci(r["ukbppp_or"], r["ukbppp_ci"]),
            "decode_pp4":     _pp4_cell(r["decode_pp4"], r["decode_pp4_estimable"]),
            "ukbppp_pp4":     _pp4_cell(r["ukbppp_pp4"], r["ukbppp_pp4_estimable"]),
            "causal_call":    r["causal_call"],
        })

    sum_df = pd.DataFrame(summary_rows)

    pc_pp4_str = f"{coloc_pc['pp4']:.3f}" if coloc_pc is not None else "N/A"
    if pc_or is not None and pc_low is not None and pc_high is not None:
        pc_or_str = f"MR OR={pc_or:.2f} [{pc_low:.3f}-{pc_high:.3f}] recovers expected direction"
        pc_status = "PASSED" if poscontrol_passed else "FAILED"
    else:
        pc_or_str = "MR OR=N/A (no GPNMB instruments produced)"
        pc_status = "FAILED"
    footnote = (
        f"# Positive control (GPNMB): {pc_or_str}; "
        f"coloc PP4={pc_pp4_str}. Gate: {pc_status}.\n"
    )

    # Disclose the gate's platform scope.  The pre-registered gate
    # (DESIGN_SPEC_v3_npjPD.md:150) required recovery on BOTH pQTL platforms;
    # as implemented it is deCODE-only.  Where the spec-conformant verdict is
    # not met, the publication-facing table must say so on its face.
    # .get() fallbacks keep this working against poscontrol_result.json files
    # written before these keys existed.
    pc_platforms = poscontrol.get("platforms", {})
    pc_ukb = pc_platforms.get("ukbppp", {})
    both_ok = poscontrol.get("recovers_both_platforms")
    if both_ok is False:
        ukb_or, ukb_lo, ukb_hi = (
            pc_ukb.get("or"), pc_ukb.get("ci_low"), pc_ukb.get("ci_high")
        )
        ukb_str = (
            f"OR={ukb_or:.3f} [{ukb_lo:.3f}-{ukb_hi:.3f}]"
            if None not in (ukb_or, ukb_lo, ukb_hi)
            else "no estimate"
        )
        footnote += (
            "# DEVIATION: the pre-registered gate required the positive control "
            "to recover on BOTH pQTL platforms; it recovers on deCODE only "
            f"(UKB-PPP {ukb_str}, n={pc_ukb.get('n_instruments', 'NA')} "
            "instruments). The UKB-PPP arm of every null call below is therefore "
            "UNGATED, and cross-platform concordance claims must disclose this.\n"
        )

    # PP4 estimability note.  Rendered whenever any cell is a placeholder, so a
    # reader of the table alone cannot mistake 'NE' for a computed zero.
    n_ne = sum(
        1
        for _, r in df.iterrows()
        for est in (r["decode_pp4_estimable"], r["ukbppp_pp4_estimable"])
        if est is False
    )
    if n_ne:
        footnote += (
            f"# PP4: 'NE' = NOT ESTIMABLE ({n_ne} cells). coloc.susie found no "
            "overlapping credible sets between the pQTL and PD signals at that "
            "locus, so no posterior exists. Report as 'no overlapping credible "
            "sets under SuSiE fine-mapping; PP4 not estimable', NOT as "
            "'no colocalization (PP4 = 0)'. CAVEAT ON THE CALLS BELOW: the "
            f"causal_call logic treats a non-estimable PP4 as below the "
            f"PP4>={PP4_THRESHOLD} threshold, i.e. as NOT supporting "
            "colocalization. That is an assumption, not a measurement -- absence "
            "of a computable posterior is not evidence of absent colocalization "
            "-- and it is load-bearing for the 'null' calls.\n"
        )

    with open(OUT_SUMMARY, "w") as fh:
        fh.write(footnote)
        sum_df.to_csv(fh, index=False, quoting=csv.QUOTE_NONNUMERIC)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print(f"[17_synthesize] Positive-control gate: {pc_status}")
    print(f"[17_synthesize] Written: {OUT_FINAL}")
    print(f"[17_synthesize] Written: {OUT_SUMMARY}\n")
    cols = ["protein", "role", "platforms_tested",
            "decode_verdict", "ukbppp_verdict", "causal_call"]
    print(df[cols].to_string(index=False))

    return df


if __name__ == "__main__":
    synthesize()
