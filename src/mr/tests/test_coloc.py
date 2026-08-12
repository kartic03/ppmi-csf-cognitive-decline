"""
Tests for colocalization analysis (task A8).

Method routing:
  - n_signals > 1 -> coloc.susie
  - n_signals == 1 -> coloc.abf

Honest labels:
  - Early exits (no data) produce method + "_nodata" suffix
  - SuSiE no credible-set outcome produces "coloc.susie_noCS"

Positive control:
  - GPNMB (chr7, deCODE) is a known PD-colocalizing locus; PP4 must be > 0.5
    to confirm the pipeline is not broken.
"""

import json
import subprocess


def test_coloc_method_routing():
    """Verify method routing, valid PP4 for all pairs, and GPNMB positive control."""
    subprocess.run(["python", "src/mr/16_coloc_driver.py"], check=True, timeout=1200)

    with open("data/processed/mr/coloc_results.json") as f:
        results = json.load(f)

    res = {(r["protein"], r["platform"]): r for r in results}

    # --- Guard: verify at least one group actually ran R (not all early-exit) ---
    # ICAM1 deCODE has 10 independent signals -> SuSiE-coloc must run R.
    # Acceptable: "coloc.susie" (CS found and overlapping) or "coloc.susie_noCS"
    # (SuSiE ran but found no overlapping CS - biologically valid for BBB proteins).
    # NOT acceptable: "coloc.susie_nodata" (R never ran - data extraction failed).
    icam1_method = res[("ICAM1", "decode")]["method"]
    assert icam1_method in ("coloc.susie", "coloc.susie_noCS"), (
        f"Expected coloc.susie or coloc.susie_noCS for ICAM1/decode (real R run), "
        f"got {icam1_method!r}. "
        "A '_nodata' suffix means R never ran — check data extraction."
    )

    # MMP9 deCODE has 1 signal -> coloc.abf (single-signal ABF, runs R).
    # "coloc.abf" is the expected label when R ran; "coloc.abf_nodata" would mean
    # data extraction failed before R was called.
    mmp9_method = res[("MMP9", "decode")]["method"]
    assert mmp9_method == "coloc.abf", (
        f"Expected coloc.abf for MMP9/decode, got {mmp9_method!r}"
    )

    # --- GPNMB positive control ---
    # GPNMB is a known PD-colocalizing locus. PP4 must be > 0.5.
    # If it comes out near 0, the pipeline has a systematic bug and the
    # all-zero BBB result is an artifact, not a real finding.
    gpnmb_pp4 = res[("GPNMB", "decode")]["pp4"]
    assert gpnmb_pp4 > 0.5, (
        f"GPNMB positive-control PP4={gpnmb_pp4:.4f} is too low (expected >0.5). "
        "This indicates a pipeline bug (build mismatch, LD error, or units issue). "
        "The all-zero BBB result cannot be trusted until GPNMB is fixed."
    )

    # --- All PP4 values must be valid probabilities ---
    for r in results:
        pp4 = r["pp4"]
        assert 0.0 <= pp4 <= 1.0, (
            f"PP4 out of range for {r['protein']}/{r['platform']}: {pp4}"
        )
