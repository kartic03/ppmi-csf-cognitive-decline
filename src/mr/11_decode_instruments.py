"""
MR step 11: Extract cis-pQTL instruments from 6 deCODE SomaScan protein GWAS files.

For each protein:
- Stream-filter sumstats to the cis window (awk, before pandas)
- Keep pval < 5e-8
- Merge EAF from assocvariants.annotated.txt.gz on Name (never use ImpMAF)
- Drop variants in assocvariants.excluded.txt.gz
- Drop multiallelic rows (otherAllele == "!")
- Compute F = (Beta/SE)^2; keep F > 10
- Flag palindromic instruments (A/T or C/G allele pairs)
- LD-clump to independent lead signals with plink vs 1000G EUR
- Write data/processed/mr/instruments_decode.parquet

Run from the project root.
"""

import json
import os
import shutil
import subprocess
import tempfile
from io import StringIO

import pandas as pd

DECODE_DIR = "data/raw/mr/decode_pqtl"
ANN_FILE = f"{DECODE_DIR}/assocvariants.annotated.txt.gz"
EXC_FILE = f"{DECODE_DIR}/assocvariants.excluded.txt.gz"
LD_BFILE = "data/raw/mr/ld_1000g_eur/EUR"
WINDOWS_JSON = "data/processed/mr/windows.json"
OUT_PARQUET = "data/processed/mr/instruments_decode.parquet"

# Map protein name -> sumstats filename
PROTEIN_FILES = {
    "PDGFRB": f"{DECODE_DIR}/3459_49_PDGFRB_PDGF_Rb.txt.gz",
    "ICAM1":  f"{DECODE_DIR}/4342_10_ICAM1_sICAM_1.txt.gz",
    "VCAM1":  f"{DECODE_DIR}/2967_8_VCAM1_VCAM_1.txt.gz",
    "MMP9":   f"{DECODE_DIR}/2579_17_MMP9_MMP_9.txt.gz",
    "MMP2":   f"{DECODE_DIR}/4160_49_MMP2_MMP_2.txt.gz",
    "TIMP1":  f"{DECODE_DIR}/2211_9_TIMP1_TIMP_1.txt.gz",
}

PALINDROME_PAIRS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}


def awk_filter_gz(filepath: str, chrom_decode: str, lo: int, hi: int) -> str:
    """Stream-filter a gzipped tab-delimited file to a genomic window.

    Keeps the header line plus any row where column 1 matches chrom_decode
    and column 2 (integer position) is within [lo, hi].

    Uses a two-stage Popen pipeline (zcat | awk) without shell=True,
    so filepath and chrom_decode are never interpreted as shell code.
    """
    awk_prog = f'NR==1||($1=="{chrom_decode}"&&$2>={lo}&&$2<={hi})'
    with subprocess.Popen(["zcat", filepath], stdout=subprocess.PIPE) as zcat_proc:
        awk_result = subprocess.run(
            ["awk", "-F", "\t", awk_prog],
            stdin=zcat_proc.stdout,
            capture_output=True,
            text=True,
        )
        zcat_proc.stdout.close()  # allow zcat to receive SIGPIPE if awk exits early
    if awk_result.returncode not in (0, 1):
        raise subprocess.CalledProcessError(awk_result.returncode, "awk", awk_result.stderr)
    return awk_result.stdout


def load_excluded_names() -> set:
    """Load the full set of excluded variant Name values from the QC blocklist."""
    print("Loading excluded variants...", flush=True)
    df = pd.read_csv(EXC_FILE, sep="\t", usecols=["Name"])
    return set(df["Name"].tolist())


def load_cis_variants(
    protein: str,
    filepath: str,
    window: dict,
    excluded_names: set,
) -> pd.DataFrame | None:
    """Load and QC cis-window variants for one protein.

    Returns a DataFrame with columns:
        Name, rsids, chrom_raw, pos, effectAllele, otherAllele,
        beta, se, pval, F, eaf, palindromic
    or None if no variants pass all filters.
    """
    chrom = window["chrom"]              # e.g. "5" or "X"
    chrom_decode = f"chr{chrom}"         # e.g. "chr5" or "chrX"
    lo = window["cis_start"]
    hi = window["cis_end"]

    # --- Step 1: stream-filter sumstats to the cis window ---
    raw = awk_filter_gz(filepath, chrom_decode, lo, hi)
    df = pd.read_csv(StringIO(raw), sep="\t")
    if df.empty:
        print(f"  {protein}: no variants in cis window", flush=True)
        return None

    # Normalise column names to lowercase
    df = df.rename(columns={
        "Chrom": "chrom_raw", "Pos": "pos", "rsids": "rsids",
        "effectAllele": "effectAllele", "otherAllele": "otherAllele",
        "Beta": "beta", "Pval": "pval", "SE": "se",
        "N": "n", "ImpMAF": "ImpMAF", "minus_log10_pval": "minus_log10_pval",
    })

    # --- Step 2: genome-wide significance filter ---
    df = df[df["pval"] < 5e-8].copy()
    if df.empty:
        print(f"  {protein}: no genome-wide significant cis variants", flush=True)
        return None

    # --- Step 3: drop multiallelic markers (otherAllele == "!") ---
    df = df[df["otherAllele"] != "!"].copy()
    if df.empty:
        print(f"  {protein}: all significant cis variants are multiallelic", flush=True)
        return None

    # --- Step 4: drop QC-excluded variants ---
    before = len(df)
    df = df[~df["Name"].isin(excluded_names)].copy()
    print(f"  {protein}: {before} -> {len(df)} after exclusion list", flush=True)
    if df.empty:
        return None

    # --- Step 5: F-statistic filter (strong instruments) ---
    df = df[df["se"] > 0].copy()
    df["F"] = (df["beta"] / df["se"]) ** 2
    df = df[df["F"] > 10].copy()
    if df.empty:
        print(f"  {protein}: no variants with F > 10", flush=True)
        return None

    # --- Step 6: merge EAF from annotation file ---
    ann_raw = awk_filter_gz(ANN_FILE, chrom_decode, lo, hi)
    ann = pd.read_csv(StringIO(ann_raw), sep="\t", usecols=["Name", "effectAlleleFreq"])
    ann = ann.rename(columns={"effectAlleleFreq": "eaf"})
    # Deduplicate annotation (keep first occurrence per Name)
    ann = ann.drop_duplicates(subset="Name", keep="first")

    df = df.merge(ann[["Name", "eaf"]], on="Name", how="left")
    df = df[df["eaf"].notna()].copy()
    if df.empty:
        print(f"  {protein}: no variants with EAF in annotation", flush=True)
        return None

    # --- Step 7: flag palindromic instruments ---
    df["palindromic"] = df.apply(
        lambda r: (r["effectAllele"].upper(), r["otherAllele"].upper()) in PALINDROME_PAIRS,
        axis=1,
    )

    print(f"  {protein}: {len(df)} QC-passed cis variants before clumping", flush=True)
    return df


def first_rsid(rsids_val) -> str | None:
    """Extract the first rsid from a (possibly comma-separated) rsids string.

    Returns None for missing values (NA, ., nan).
    """
    if pd.isna(rsids_val):
        return None
    s = str(rsids_val).strip()
    if s in ("NA", ".", "nan", ""):
        return None
    return s.split(",")[0].strip()


def plink_clump(df: pd.DataFrame, protein: str, tmpdir: str) -> pd.DataFrame:
    """Run plink LD-clumping against the EUR 1000G panel.

    Variants without an rsid are excluded from the clump input.
    If no rsid-bearing variants exist (or clumping fails), returns the single
    highest-F variant as a fallback.

    pval=0 rows (floating-point underflow in deCODE data) are assigned a
    rank-based pseudo-p so plink selects the strongest instrument as the index
    SNP rather than breaking ties by genomic position. The real pval column is
    unchanged in the output parquet.
    """
    df = df.copy()
    df["rsid_for_clump"] = df["rsids"].apply(first_rsid)

    # --- C2: rank-based pseudo-p for p=0 underflow rows ---
    # Do NOT recompute via 10**(-minus_log10_pval): in deCODE data, ALL p=0 rows
    # have minus_log10_pval stored as Infinity, so that column cannot distinguish
    # instrument strength. Use F = (Beta/SE)^2 instead — it is always finite and
    # directly measures instrument strength.
    # Primary sort: F descending (highest F = strongest instrument = rank 1 = smallest pseudo-p).
    # Secondary sort: pos descending (deterministic tiebreaker for variants with identical
    # stored Beta/SE due to limited precision in deCODE output; higher position = earlier rank).
    # All pseudo-p values are far below 5e-8 so --clump-p1 threshold is unaffected.
    df["pval_for_clump"] = df["pval"]
    zero = df["pval"] == 0
    if zero.any():
        sort_order = (
            df.loc[zero, ["F", "pos"]]
            .sort_values(["F", "pos"], ascending=[False, False])
            .index
        )
        ranks = pd.Series(
            data=range(1, len(sort_order) + 1),
            index=sort_order,
            dtype=float,
        )
        df.loc[zero, "pval_for_clump"] = ranks * 1e-300

    has_rsid = df[df["rsid_for_clump"].notna()].copy()
    # I2: drop duplicate rsids before indexing to avoid Series-instead-of-scalar issues
    has_rsid = has_rsid.drop_duplicates(subset="rsid_for_clump", keep="first")

    def _warn_rare(row_df: pd.DataFrame) -> None:
        """m1: warn if fallback instrument is a rare variant (eaf < 0.01)."""
        if not row_df.empty and "eaf" in row_df.columns:
            eaf_val = row_df.iloc[0]["eaf"]
            if pd.notna(eaf_val) and float(eaf_val) < 0.01:
                print(
                    f"  WARNING: {protein} fallback instrument eaf={eaf_val:.4f} "
                    f"(rare variant — downstream MR results may be unreliable)",
                    flush=True,
                )

    if has_rsid.empty:
        print(f"  {protein}: no rsid-bearing variants - keeping single highest-F", flush=True)
        fallback = df.nlargest(1, "F").copy()
        _warn_rare(fallback)
        return fallback

    # Write plink association input file. plink sorts internally by pval_for_clump,
    # so the strongest variant wins ties that raw pval=0 would break arbitrarily.
    assoc_file = os.path.join(tmpdir, f"{protein}_assoc.tsv")
    (
        has_rsid[["rsid_for_clump", "pval_for_clump"]]
        .rename(columns={"rsid_for_clump": "rsids"})
        .to_csv(assoc_file, sep="\t", index=False)
    )

    out_prefix = os.path.join(tmpdir, protein)
    cmd = [
        "plink",
        "--bfile", LD_BFILE,
        "--clump", assoc_file,
        "--clump-p1", "5e-8",
        "--clump-r2", "0.001",
        "--clump-kb", "1000",
        "--clump-snp-field", "rsids",
        "--clump-field", "pval_for_clump",
        "--out", out_prefix,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    clump_file = f"{out_prefix}.clumped"

    if not os.path.exists(clump_file):
        stderr_tail = result.stderr[-400:] if result.stderr else ""
        print(
            f"  {protein}: plink produced no .clumped file "
            f"(likely no rsids in LD panel). stderr tail:\n{stderr_tail}",
            flush=True,
        )
        fallback = has_rsid.nlargest(1, "F").copy()
        _warn_rare(fallback)
        return fallback

    clumped = pd.read_csv(clump_file, sep=r"\s+")
    print(
        f"  {protein}: plink selected {len(clumped)} independent index SNP(s)",
        flush=True,
    )

    # Use plink's selected index SNPs directly. The palindromic FLAG column is
    # preserved from load_cis_variants; no preference or substitution is applied
    # here (spec says flag palindromes, not prefer them over non-palindromic leads).
    selected_rsids = list(clumped["SNP"])
    result_df = df[df["rsid_for_clump"].isin(selected_rsids)].copy()
    if result_df.empty:
        print(f"  {protein}: no clumped variants matched - keeping highest-F rsid row", flush=True)
        fallback = has_rsid.nlargest(1, "F").copy()
        _warn_rare(fallback)
        return fallback

    return result_df


def main() -> None:
    """Main entry point: extract and write cis-pQTL instruments for all 6 proteins."""
    with open(WINDOWS_JSON) as fh:
        windows = json.load(fh)
    excluded_names = load_excluded_names()

    tmpdir = tempfile.mkdtemp(prefix="mr_decode_")
    all_instruments = []

    try:
        for protein, filepath in PROTEIN_FILES.items():
            print(f"\n=== {protein} ===", flush=True)
            df = load_cis_variants(protein, filepath, windows[protein], excluded_names)
            if df is None:
                print(f"  {protein}: SKIPPED (no passing variants)", flush=True)
                continue

            clumped = plink_clump(df, protein, tmpdir)

            # Build rsid column (first rsid or null)
            clumped["rsid"] = clumped["rsids"].apply(first_rsid)
            clumped["protein"] = protein

            w = windows[protein]
            # Keep chrom as defined in windows.json (no "chr" prefix)
            clumped["chrom"] = w["chrom"]

            out_cols = [
                "protein", "Name", "rsid", "chrom", "pos",
                "effectAllele", "otherAllele", "eaf",
                "beta", "se", "pval", "F", "palindromic",
            ]
            all_instruments.append(clumped[out_cols].copy())
            print(f"  {protein}: {len(clumped)} final instrument(s)", flush=True)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not all_instruments:
        raise RuntimeError("No instruments found for any protein - aborting.")

    result = pd.concat(all_instruments, ignore_index=True)
    os.makedirs("data/processed/mr", exist_ok=True)
    result.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {len(result)} instruments across {result['protein'].nunique()} proteins to {OUT_PARQUET}", flush=True)

    # --- Sanity checks (printed to stdout, not hard-coded assertions) ---
    pdgfrb = result[result["protein"] == "PDGFRB"]
    if not pdgfrb.empty:
        lead = pdgfrb.nlargest(1, "F").iloc[0]
        lead_rsid = lead["rsid"]
        print(f"\nSANITY: PDGFRB lead instrument = {lead_rsid} (expected rs2304058)", flush=True)
        if lead_rsid == "rs2304058":
            print("  CONFIRMED: PDGFRB lead is rs2304058", flush=True)
        else:
            print(f"  NOTE: PDGFRB lead is {lead_rsid}, not rs2304058", flush=True)
    else:
        print("\nSANITY: PDGFRB has no instruments", flush=True)

    icam1 = result[result["protein"] == "ICAM1"]
    n_icam1 = len(icam1)
    print(f"\nSANITY: ICAM1 has {n_icam1} independent cis instrument(s) (expected >= 2)", flush=True)
    if n_icam1 >= 2:
        print("  CONFIRMED: ICAM1 has >= 2 independent cis signals", flush=True)
    else:
        print(f"  NOTE: ICAM1 has {n_icam1} (expected >= 2)", flush=True)

    print("\nFull instrument table:", flush=True)
    print(result.to_string(), flush=True)


if __name__ == "__main__":
    main()
