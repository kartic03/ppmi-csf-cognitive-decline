import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_harmonize_alleles_and_routing():
    subprocess.run([sys.executable, "src/mr/13_harmonize.py"], check=True)
    df = pd.read_parquet("data/processed/mr/harmonized.parquet")
    # TIMP1 routed to the X-GWAS, never to Nalls
    timp = df[df["protein"] == "TIMP1"]
    assert (timp["outcome_source"] == "leguen_x").all()
    assert (df[df["protein"] != "TIMP1"]["outcome_source"] == "nalls2019").all()
    # palindromes that could not be strand-resolved by EAF are marked
    assert "strand_resolved" in df.columns
    # aligned betas: effect allele matches between exposure and outcome rows kept
    assert df["aligned"].all()


def test_align_alleles_unit():
    """Unit test for align_alleles on synthetic rows: direct, swapped, incompatible."""
    spec = importlib.util.spec_from_file_location(
        "harmonize13",
        Path(__file__).resolve().parent.parent / "13_harmonize.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    align_alleles = mod.align_alleles

    rows = pd.DataFrame({
        "protein":      ["P", "P", "P"],
        "platform":     ["d", "d", "d"],
        "rsid":         ["rs1", "rs2", "rs3"],
        "chrom":        ["1", "1", "1"],
        "pos":          [100, 200, 300],
        "effectAllele": ["A", "A", "A"],
        "otherAllele":  ["G", "G", "G"],
        # rs1: direct match (A/G == A/G)
        # rs2: swapped     (A/G vs G/A)
        # rs3: incompatible (A/G vs C/T)
        "ea_out":       ["A", "G", "C"],
        "oa_out":       ["G", "A", "T"],
        "beta_out":     [0.20, 0.20, 0.20],
        "eaf_out":      [0.30, 0.70, 0.50],
    })

    result = align_alleles(rows)

    # Direct-match row: kept, beta_out unchanged
    rs1 = result[result["rsid"] == "rs1"]
    assert len(rs1) == 1, "direct-match row should be in result"
    assert rs1["aligned"].iloc[0]
    assert abs(rs1["beta_out"].iloc[0] - 0.20) < 1e-9, "direct-match beta_out must be unchanged"

    # Swapped row: kept, beta_out sign-flipped, eaf_out = 1 - original (0.70) = 0.30
    rs2 = result[result["rsid"] == "rs2"]
    assert len(rs2) == 1, "swapped row should be in result"
    assert rs2["aligned"].iloc[0]
    assert abs(rs2["beta_out"].iloc[0] - (-0.20)) < 1e-9, "swapped beta_out must be negated"
    assert abs(rs2["eaf_out"].iloc[0] - 0.30) < 1e-9, "swapped eaf_out must be 1 - original"

    # Incompatible row: dropped from result
    assert len(result[result["rsid"] == "rs3"]) == 0, "incompatible row must be excluded"
