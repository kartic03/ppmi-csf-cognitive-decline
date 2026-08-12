"""
MR step 12: Extract cis-pQTL instruments from 5 UKB-PPP Olink protein GWAS tars.

For each protein:
- Extract only the cis-chromosome inner file from the tar (streaming, no full extract)
- Parse REGENIE columns; GENPOS is GRCh38
- Filter to the cis window (cis_start..cis_end from windows.json) and LOG10P > 7.301 (p<5e-8)
- Compute F = (BETA/SE)^2; keep F > 10
- Join rsid from the cis-chromosome rsid map by ID column
- Flag palindromic instruments (A/T or C/G pairs) — flag only, never prefer
- LD-clump to independent signals with plink vs 1000G EUR (r2<0.001, 1000kb window)
- p-underflow handling: for pval==0 rows, assign rank-based pseudo-p (sorted by LOG10P
  descending) so the strongest signal becomes the clump index, not an arbitrary position
- Fallback: if no rsid matches in LD panel, keep single highest-F (highest LOG10P) variant
- Write data/processed/mr/instruments_ukbppp.parquet with A3 schema

Output schema matches 11_decode_instruments.py:
  protein, Name, rsid, chrom, pos, effectAllele, otherAllele, eaf, beta, se, pval, F, palindromic

Run from the project root.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

UKBPPP_DIR = "data/raw/mr/ukbppp"
LD_BFILE = "data/raw/mr/ld_1000g_eur/EUR"
WINDOWS_JSON = "data/processed/mr/windows.json"
OUT_PARQUET = "data/processed/mr/instruments_ukbppp.parquet"

PALINDROME_PAIRS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}

# Map protein -> (tar filename, cis chromosome, inner member name)
# Member names derived from `tar -tf` output; must match exactly.
PROTEIN_CONFIG = {
    "PDGFRB": {
        "tar": f"{UKBPPP_DIR}/PDGFRB_P09619_OID20268_v1_Cardiometabolic.tar",
        "chrom": "5",
        "regenie_chrom": "5",    # REGENIE CHROM column value for this chromosome
        "member": "PDGFRB_P09619_OID20268_v1_Cardiometabolic/discovery_chr5_PDGFRB:P09619:OID20268:v1:Cardiometabolic.gz",
        "rsid_map": f"{UKBPPP_DIR}/olink_rsid_map_mac5_info03_b0_7_chr5_patched_v2.tsv.gz",
    },
    "ICAM1": {
        "tar": f"{UKBPPP_DIR}/ICAM1_P05362_OID20411_v1_Cardiometabolic.tar",
        "chrom": "19",
        "regenie_chrom": "19",
        "member": "ICAM1_P05362_OID20411_v1_Cardiometabolic/discovery_chr19_ICAM1:P05362:OID20411:v1:Cardiometabolic.gz",
        "rsid_map": f"{UKBPPP_DIR}/olink_rsid_map_mac5_info03_b0_7_chr19_patched_v2.tsv.gz",
    },
    "VCAM1": {
        "tar": f"{UKBPPP_DIR}/VCAM1_P19320_OID20396_v1_Cardiometabolic.tar",
        "chrom": "1",
        "regenie_chrom": "1",
        "member": "VCAM1_P19320_OID20396_v1_Cardiometabolic/discovery_chr1_VCAM1:P19320:OID20396:v1:Cardiometabolic.gz",
        "rsid_map": f"{UKBPPP_DIR}/olink_rsid_map_mac5_info03_b0_7_chr1_patched_v2.tsv.gz",
    },
    "MMP9": {
        "tar": f"{UKBPPP_DIR}/MMP9_P14780_OID21103_v1_Neurology.tar",
        "chrom": "20",
        "regenie_chrom": "20",
        "member": "MMP9_P14780_OID21103_v1_Neurology/discovery_chr20_MMP9:P14780:OID21103:v1:Neurology.gz",
        "rsid_map": f"{UKBPPP_DIR}/olink_rsid_map_mac5_info03_b0_7_chr20_patched_v2.tsv.gz",
    },
    "TIMP1": {
        "tar": f"{UKBPPP_DIR}/TIMP1_P01033_OID20418_v1_Cardiometabolic.tar",
        "chrom": "X",
        "regenie_chrom": "23",   # REGENIE encodes chrX as 23
        "member": "TIMP1_P01033_OID20418_v1_Cardiometabolic/discovery_chrX_TIMP1:P01033:OID20418:v1:Cardiometabolic.gz",
        "rsid_map": f"{UKBPPP_DIR}/olink_rsid_map_mac5_info03_b0_7_chrX_patched_v2.tsv.gz",
    },
}

# REGENIE column names (space-separated header in the gz files)
REGENIE_COLS = [
    "CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1",
    "A1FREQ", "INFO", "N", "TEST", "BETA", "SE",
    "CHISQ", "LOG10P", "EXTRA",
]


def extract_cis_chr_data(
    tar_path: str,
    member: str,
    regenie_chrom: str,
    cis_start: int,
    cis_end: int,
) -> pd.DataFrame:
    """Stream-extract the cis-chromosome inner file from the tar and filter to the cis window.

    Uses tar -xOf to stream just one member to stdout, pipes through gunzip,
    then awk-filters to the window before loading into pandas. This avoids
    extracting all 23 chromosomes from the tar.

    regenie_chrom is the value as it appears in the REGENIE CHROM column:
    - autosomes: numeric string ("5", "19", etc.)
    - chrX: "23" (REGENIE encodes X as 23)
    """
    # Filter: keep header (NR==1) or rows where $1 matches regenie_chrom
    # and GENPOS ($2) is within the cis window.
    awk_prog = (
        f'NR==1 || ($1=="{regenie_chrom}" && $2>={cis_start} && $2<={cis_end})'
    )

    with subprocess.Popen(
        ["tar", "-xOf", tar_path, member],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as tar_proc:
        with subprocess.Popen(
            ["gunzip", "-c"],
            stdin=tar_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as gunzip_proc:
            awk_result = subprocess.run(
                ["awk", awk_prog],
                stdin=gunzip_proc.stdout,
                capture_output=True,
                text=True,
            )
            gunzip_proc.stdout.close()
        tar_proc.stdout.close()

    if awk_result.returncode not in (0, 1):
        raise RuntimeError(
            f"awk failed (rc={awk_result.returncode}): {awk_result.stderr[:300]}"
        )

    lines = awk_result.stdout.strip()
    if not lines:
        return pd.DataFrame(columns=REGENIE_COLS)

    # Parse: header is the first line; REGENIE is space-separated
    rows = [line.split() for line in lines.splitlines()]
    header = rows[0]
    # Fix 1: validate that the file header matches our expected REGENIE column spec
    assert header == REGENIE_COLS, (
        f"Unexpected REGENIE header in {tar_path}:\n"
        f"  got:      {header}\n"
        f"  expected: {REGENIE_COLS}"
    )
    data_rows = rows[1:]
    if not data_rows:
        return pd.DataFrame(columns=REGENIE_COLS)

    df = pd.DataFrame(data_rows, columns=header)
    # Cast numeric columns
    for col in ["GENPOS", "BETA", "SE", "A1FREQ", "LOG10P"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_rsid_map(rsid_map_path: str) -> pd.DataFrame:
    """Load the cis-chromosome rsid map; keep only ID and rsid columns."""
    df = pd.read_csv(rsid_map_path, sep="\t", usecols=["ID", "rsid"])
    # Drop rows without an rsid
    df = df[df["rsid"].notna()].drop_duplicates(subset="ID", keep="first")
    return df


def plink_clump(df: pd.DataFrame, protein: str, tmpdir: str) -> pd.DataFrame:
    """Run plink LD-clumping against the EUR 1000G panel.

    Variants without an rsid are excluded from the clump input.
    If no rsid-bearing variants exist or clumping produces no .clumped file,
    returns the single highest-LOG10P variant as a fallback.

    p-underflow handling: pval = 10**(-LOG10P) underflows to 0.0 for strong
    signals. When writing the plink --clump-field input, break p=0 ties by
    LOG10P descending so the strongest variant becomes the clump index.
    All pseudo-p values are far below 5e-8, so --clump-p1 is unaffected.
    The real pval column is unchanged in the output parquet.
    """
    df = df.copy()
    has_rsid = df[df["rsid"].notna()].drop_duplicates(subset="rsid", keep="first").copy()

    def _fallback_strongest(src: pd.DataFrame) -> pd.DataFrame:
        """Return the single row with highest LOG10P (strongest signal)."""
        return src.nlargest(1, "LOG10P").copy()

    def _warn_rare(row_df: pd.DataFrame) -> None:
        """Fix 3: warn if fallback instrument is a rare variant (A1FREQ < 0.01)."""
        if not row_df.empty and "A1FREQ" in row_df.columns:
            row = row_df.iloc[0]
            eaf_val = row["A1FREQ"]
            rsid_val = row["rsid"] if pd.notna(row.get("rsid")) else row["ID"]
            if pd.notna(eaf_val) and float(eaf_val) < 0.01:
                print(
                    f"  WARNING: {protein} fallback instrument "
                    f"rsid={rsid_val} eaf={float(eaf_val):.4f} "
                    f"(rare variant -- downstream MR results may be unreliable)",
                    file=sys.stderr,
                    flush=True,
                )

    if has_rsid.empty:
        print(f"  {protein}: no rsid-bearing variants - keeping single strongest", flush=True)
        fallback = _fallback_strongest(df)
        _warn_rare(fallback)
        return fallback

    # Build pval_for_clump: use real pval, but rank zero-pval rows by LOG10P descending.
    has_rsid["pval_for_clump"] = has_rsid["pval"]
    zero_mask = has_rsid["pval"] == 0.0
    if zero_mask.any():
        # Fix 4: secondary key GENPOS (descending) for deterministic tie-breaking
        sorted_idx = (
            has_rsid.loc[zero_mask, ["LOG10P", "GENPOS"]]
            .sort_values(by=["LOG10P", "GENPOS"], ascending=[False, False])
            .index
        )
        ranks = pd.Series(
            data=range(1, len(sorted_idx) + 1),
            index=sorted_idx,
            dtype=float,
        )
        has_rsid.loc[zero_mask, "pval_for_clump"] = ranks * 1e-300

    assoc_file = os.path.join(tmpdir, f"{protein}_assoc.tsv")
    (
        has_rsid[["rsid", "pval_for_clump"]]
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
        "--clump-snp-field", "rsid",
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
        fallback = _fallback_strongest(has_rsid)
        _warn_rare(fallback)
        return fallback

    clumped = pd.read_csv(clump_file, sep=r"\s+")
    print(
        f"  {protein}: plink selected {len(clumped)} independent index SNP(s)",
        flush=True,
    )

    selected_rsids = set(clumped["SNP"].tolist())
    result_df = df[df["rsid"].isin(selected_rsids)].copy()

    if result_df.empty:
        print(
            f"  {protein}: no clumped rsids matched data rows - fallback to strongest",
            flush=True,
        )
        fallback = _fallback_strongest(has_rsid)
        _warn_rare(fallback)
        return fallback

    return result_df


def process_protein(protein: str, config: dict, window: dict, tmpdir: str) -> pd.DataFrame | None:
    """Extract and QC cis instruments for one protein.

    Returns a DataFrame with the A3 output schema or None if no passing variants.
    """
    print(f"\n=== {protein} ===", flush=True)

    chrom = window["chrom"]          # e.g. "5" or "X"
    cis_start = window["cis_start"]
    cis_end = window["cis_end"]

    # --- Step 1: extract cis-chr data and filter to cis window ---
    df = extract_cis_chr_data(
        tar_path=config["tar"],
        member=config["member"],
        regenie_chrom=config["regenie_chrom"],
        cis_start=cis_start,
        cis_end=cis_end,
    )
    print(f"  {protein}: {len(df)} variants in cis window", flush=True)
    if df.empty:
        print(f"  {protein}: no variants in cis window - SKIPPED", flush=True)
        return None

    # --- Step 2: genome-wide significance filter (LOG10P > -log10(5e-8) ≈ 7.301) ---
    LOG10P_THRESHOLD = 7.30103   # -log10(5e-8)
    df = df[df["LOG10P"] > LOG10P_THRESHOLD].copy()
    print(f"  {protein}: {len(df)} genome-wide significant variants", flush=True)
    if df.empty:
        print(f"  {protein}: no GWS variants - SKIPPED", flush=True)
        return None

    # --- Step 3: compute pval from LOG10P; underflow to 0.0 is expected for strong signals ---
    df["pval"] = 10.0 ** (-df["LOG10P"])

    # --- Step 4: compute F-statistic; keep F > 10 ---
    df = df[df["SE"] > 0].copy()
    df["F"] = (df["BETA"] / df["SE"]) ** 2
    df = df[df["F"] > 10].copy()
    print(f"  {protein}: {len(df)} variants with F > 10", flush=True)
    if df.empty:
        print(f"  {protein}: no variants with F > 10 - SKIPPED", flush=True)
        return None

    # --- Step 5: join rsid from the cis-chromosome rsid map ---
    rsid_map = load_rsid_map(config["rsid_map"])
    df = df.merge(rsid_map[["ID", "rsid"]], on="ID", how="left")
    n_with_rsid = df["rsid"].notna().sum()
    print(f"  {protein}: {n_with_rsid}/{len(df)} variants have rsid", flush=True)

    # --- Step 6: flag palindromic instruments (A/T or C/G pairs) ---
    # Flag only — never prefer or substitute palindromic variants.
    df["palindromic"] = df.apply(
        lambda r: (str(r["ALLELE0"]).upper(), str(r["ALLELE1"]).upper()) in PALINDROME_PAIRS,
        axis=1,
    )

    # --- Step 7: LD-clump to independent signals ---
    clumped = plink_clump(df, protein, tmpdir)

    # --- Step 8: build output with A3 schema ---
    # pos = GENPOS (GRCh38), already filtered to cis window
    clumped["protein"] = protein
    clumped["chrom"] = chrom
    clumped["Name"] = clumped["ID"]
    clumped["pos"] = clumped["GENPOS"].astype(int)
    clumped["effectAllele"] = clumped["ALLELE1"]   # REGENIE tests ALLELE1
    clumped["otherAllele"] = clumped["ALLELE0"]
    clumped["eaf"] = clumped["A1FREQ"]             # frequency of ALLELE1
    clumped["beta"] = clumped["BETA"]
    clumped["se"] = clumped["SE"]

    out_cols = [
        "protein", "Name", "rsid", "chrom", "pos",
        "effectAllele", "otherAllele", "eaf",
        "beta", "se", "pval", "F", "palindromic",
    ]
    result = clumped[out_cols].copy()
    print(f"  {protein}: {len(result)} final instrument(s)", flush=True)
    return result


def main() -> None:
    """Main entry point: extract and write cis-pQTL instruments for all 5 proteins."""
    with open(WINDOWS_JSON) as fh:
        windows = json.load(fh)

    tmpdir = tempfile.mkdtemp(prefix="mr_ukbppp_")
    all_instruments = []

    try:
        for protein, config in PROTEIN_CONFIG.items():
            if protein not in windows:
                print(f"\n=== {protein}: no window entry in windows.json - SKIPPED ===", flush=True)
                continue
            window = windows[protein]
            result = process_protein(protein, config, window, tmpdir)
            if result is not None:
                all_instruments.append(result)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not all_instruments:
        raise RuntimeError("No instruments found for any protein - aborting.")

    combined = pd.concat(all_instruments, ignore_index=True)
    os.makedirs("data/processed/mr", exist_ok=True)
    combined.to_parquet(OUT_PARQUET, index=False)
    print(
        f"\nWrote {len(combined)} instruments across "
        f"{combined['protein'].nunique()} proteins to {OUT_PARQUET}",
        flush=True,
    )

    # --- Sanity summary ---
    print("\nPer-protein instrument counts:", flush=True)
    for prot, grp in combined.groupby("protein"):
        palin_n = grp["palindromic"].sum()
        print(
            f"  {prot}: {len(grp)} instruments "
            f"({palin_n} palindromic, pval range "
            f"{grp['pval'].min():.2e}..{grp['pval'].max():.2e})",
            flush=True,
        )

    print("\nFull instrument table:", flush=True)
    print(combined.to_string(), flush=True)


if __name__ == "__main__":
    main()
