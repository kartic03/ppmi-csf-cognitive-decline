"""
Build GRCh38 cis-window definitions for 6 PD-relevant proteins.

Reads gene spans for PDGFRB, ICAM1, VCAM1, MMP9, TIMP1 from the UKB-PPP
Olink protein map (already GRCh38). MMP2 is hard-coded (not on the Olink
panel). Cis window = gene span +/- 500 kb, start clamped at 0.

Output: data/processed/mr/windows.json
"""

import json
import pathlib

import pandas as pd

OLINK_MAP = pathlib.Path(
    "data/raw/mr/ukbppp/olink_protein_map_3k_v1.tsv"
)
OUT_FILE = pathlib.Path("data/processed/mr/windows.json")
# NOTE: paths above are relative; run this script from the project root.

# Proteins sourced from the Olink panel file
OLINK_PROTEINS = {"PDGFRB", "ICAM1", "VCAM1", "MMP9", "TIMP1"}

# MMP2 is absent from the Olink panel; coordinates from Ensembl GRCh38 (16q12.2)
MMP2_HARDCODED = {
    "chrom": "16",
    "gene_start": 55478191,
    "gene_end": 55506691,
    "strand": 1,  # informational only; not used in cis-window construction
}

CIS_WINDOW_BP = 500_000


def build_windows() -> dict:
    df = pd.read_csv(OLINK_MAP, sep="\t")
    # Keep only the five Olink proteins; deduplicate symbol rows deterministically
    df = df[df["HGNC.symbol"].isin(OLINK_PROTEINS)].copy()
    df = df.drop_duplicates(subset="HGNC.symbol", keep="first")

    windows = {}
    for _, row in df.iterrows():
        symbol = row["HGNC.symbol"]
        chrom = str(row["chr"])
        gene_start = int(row["gene_start"])
        gene_end = int(row["gene_end"])
        strand = int(row["Strand"])

        cis_start = max(0, gene_start - CIS_WINDOW_BP)
        cis_end = gene_end + CIS_WINDOW_BP

        windows[symbol] = {
            "chrom": chrom,
            "gene_start": gene_start,
            "gene_end": gene_end,
            "cis_start": cis_start,
            "cis_end": cis_end,
            "strand": strand,
        }

    missing = set(OLINK_PROTEINS) - set(windows)
    assert not missing, f"Olink map missing proteins: {missing}"

    # Add MMP2 (hard-coded GRCh38 coordinates)
    mmp2 = MMP2_HARDCODED.copy()
    mmp2["cis_start"] = max(0, mmp2["gene_start"] - CIS_WINDOW_BP)
    mmp2["cis_end"] = mmp2["gene_end"] + CIS_WINDOW_BP
    windows["MMP2"] = mmp2

    assert set(windows) == {"PDGFRB", "ICAM1", "VCAM1", "MMP9", "MMP2", "TIMP1"}

    return windows


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    windows = build_windows()
    with open(OUT_FILE, "w") as fh:
        json.dump(windows, fh, indent=2)
    print(f"Wrote {len(windows)} protein windows to {OUT_FILE}")
    for symbol, w in sorted(windows.items()):
        print(
            f"  {symbol}: chr{w['chrom']} "
            f"{w['gene_start']:,}-{w['gene_end']:,} "
            f"(cis {w['cis_start']:,}-{w['cis_end']:,})"
        )


if __name__ == "__main__":
    main()
