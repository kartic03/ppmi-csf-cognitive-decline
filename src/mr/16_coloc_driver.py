#!/usr/bin/env python3
"""
MR step 16: Colocalization analysis (coloc.abf + SuSiE-coloc).

For each (protein, platform) pair (skipping TIMP1):
  1. Count n_signals from the platform instrument parquet.
  2. Decide method: coloc.susie if n_signals > 1, else coloc.abf.
  3. Extract ALL cis-window SNPs from the raw pQTL sumstats (no p-value filter).
  4. Extract PD GWAS SNPs in the same cis window.
  5. Harmonize alleles (drop palindromic A/T, C/G; drop mismatches).
  6. For coloc.susie: compute pairwise LD from 1000G EUR via plink.
  7. Call 16_coloc.R, parse PP4.
  8. Write data/processed/mr/coloc_results.json.

Pre-registered threshold: PP4 >= 0.8 = colocalization.

Nalls 2019 PD GWAS: ~33,674 cases + 449,056 controls; s = 33674 / 482730.

Run from the project root with:
  pixi run -e mr python src/mr/16_coloc_driver.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from io import StringIO

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RSCRIPT       = os.path.expanduser("~/r_coloc_env/bin/Rscript")
COLOC_R       = os.path.join(os.path.dirname(__file__), "16_coloc.R")
LD_BFILE      = "data/raw/mr/ld_1000g_eur/EUR"
WINDOWS_JSON  = "data/processed/mr/windows.json"
DEC_PAR       = "data/processed/mr/instruments_decode.parquet"
UKB_PAR       = "data/processed/mr/instruments_ukbppp.parquet"
PD_GWAS       = "data/raw/mr/nalls2019_pd_harmonised.tsv.gz"
DECODE_DIR    = "data/raw/mr/decode_pqtl"
UKB_DIR       = "data/raw/mr/ukbppp"
OUT_JSON      = "data/processed/mr/coloc_results.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SKIP_PROTEINS   = {"TIMP1"}
PALINDROME_PAIRS = {("A","T"),("T","A"),("C","G"),("G","C")}

PD_N_CASES    = 33674
PD_N_CONTROLS = 449056
PD_N          = PD_N_CASES + PD_N_CONTROLS
PD_S          = PD_N_CASES / PD_N  # ~0.0698

DECODE_FILES = {
    "PDGFRB": f"{DECODE_DIR}/3459_49_PDGFRB_PDGF_Rb.txt.gz",
    "ICAM1":  f"{DECODE_DIR}/4342_10_ICAM1_sICAM_1.txt.gz",
    "VCAM1":  f"{DECODE_DIR}/2967_8_VCAM1_VCAM_1.txt.gz",
    "MMP9":   f"{DECODE_DIR}/2579_17_MMP9_MMP_9.txt.gz",
    "MMP2":   f"{DECODE_DIR}/4160_49_MMP2_MMP_2.txt.gz",
    "TIMP1":  f"{DECODE_DIR}/2211_9_TIMP1_TIMP_1.txt.gz",
    # Positive control: GPNMB (known PD-colocalizing locus, chr7)
    "GPNMB":  "data/raw/mr/poscontrol/8289_8_GPNMB_GPNMB.txt.gz",
}

# GPNMB positive-control window: gene chr7:23,235,967-23,275,108 (GRCh38), cis +-500 kb
GPNMB_WINDOW = {
    "chrom": "7",
    "gene_start": 23235967,
    "gene_end": 23275108,
    "cis_start": 22735967,
    "cis_end": 23775108,
    "strand": 1,
}

# UKB-PPP published sample size used when REGENIE output lacks a per-SNP N column
UKB_N_DEFAULT = 54000

# UKB protein configs (derived from 12_ukbppp_instruments.py)
UKB_CONFIG = {
    "PDGFRB": {
        "tar":    f"{UKB_DIR}/PDGFRB_P09619_OID20268_v1_Cardiometabolic.tar",
        "regenie_chrom": "5",
        "member": "PDGFRB_P09619_OID20268_v1_Cardiometabolic/discovery_chr5_PDGFRB:P09619:OID20268:v1:Cardiometabolic.gz",
        "rsid_map": f"{UKB_DIR}/olink_rsid_map_mac5_info03_b0_7_chr5_patched_v2.tsv.gz",
    },
    "ICAM1": {
        "tar":    f"{UKB_DIR}/ICAM1_P05362_OID20411_v1_Cardiometabolic.tar",
        "regenie_chrom": "19",
        "member": "ICAM1_P05362_OID20411_v1_Cardiometabolic/discovery_chr19_ICAM1:P05362:OID20411:v1:Cardiometabolic.gz",
        "rsid_map": f"{UKB_DIR}/olink_rsid_map_mac5_info03_b0_7_chr19_patched_v2.tsv.gz",
    },
    "VCAM1": {
        "tar":    f"{UKB_DIR}/VCAM1_P19320_OID20396_v1_Cardiometabolic.tar",
        "regenie_chrom": "1",
        "member": "VCAM1_P19320_OID20396_v1_Cardiometabolic/discovery_chr1_VCAM1:P19320:OID20396:v1:Cardiometabolic.gz",
        "rsid_map": f"{UKB_DIR}/olink_rsid_map_mac5_info03_b0_7_chr1_patched_v2.tsv.gz",
    },
    "MMP9": {
        "tar":    f"{UKB_DIR}/MMP9_P14780_OID21103_v1_Neurology.tar",
        "regenie_chrom": "20",
        "member": "MMP9_P14780_OID21103_v1_Neurology/discovery_chr20_MMP9:P14780:OID21103:v1:Neurology.gz",
        "rsid_map": f"{UKB_DIR}/olink_rsid_map_mac5_info03_b0_7_chr20_patched_v2.tsv.gz",
    },
}


# ---------------------------------------------------------------------------
# Exposure data extraction
# ---------------------------------------------------------------------------

def _first_rsid(val) -> str | None:
    """Return the first rsid from a comma-separated rsids string."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s in ("NA", ".", "nan", ""):
        return None
    return s.split(",")[0].strip()


def read_decode_cis(protein: str, window: dict) -> pd.DataFrame | None:
    """Extract ALL cis-window SNPs from deCODE pQTL (no p-value filter).

    Returns DataFrame with columns: rsid, effectAllele, otherAllele, beta_exp, se_exp, N_exp
    or None if no SNPs pass basic QC.
    """
    filepath = DECODE_FILES[protein]
    chrom_decode = f"chr{window['chrom']}"
    lo, hi = window["cis_start"], window["cis_end"]

    awk_prog = f'NR==1||($1=="{chrom_decode}"&&$2>={lo}&&$2<={hi})'
    with subprocess.Popen(["zcat", filepath], stdout=subprocess.PIPE) as zcat_proc:
        awk = subprocess.run(
            ["awk", "-F", "\t", awk_prog],
            stdin=zcat_proc.stdout,
            capture_output=True, text=True,
        )
        zcat_proc.stdout.close()

    if not awk.stdout.strip():
        return None

    df = pd.read_csv(StringIO(awk.stdout), sep="\t")
    # Normalise column names
    df = df.rename(columns={
        "Beta": "beta_exp", "SE": "se_exp", "N": "N_exp",
        "Pval": "pval_raw",
    })

    # Drop multiallelic
    df = df[df["otherAllele"] != "!"].copy()
    # Drop zero or negative SE
    df = df[df["se_exp"] > 0].copy()

    if df.empty:
        return None

    df["rsid"] = df["rsids"].apply(_first_rsid)

    out_cols = ["rsid", "effectAllele", "otherAllele", "beta_exp", "se_exp", "N_exp"]
    return df[out_cols].copy()


def read_ukb_cis(protein: str, window: dict) -> pd.DataFrame | None:
    """Extract ALL cis-window SNPs from UKB-PPP REGENIE output (no p-value filter).

    Returns DataFrame with columns: rsid, effectAllele, otherAllele, beta_exp, se_exp, N_exp
    or None if no SNPs pass basic QC.
    """
    cfg = UKB_CONFIG[protein]
    regenie_chrom = cfg["regenie_chrom"]
    lo, hi = window["cis_start"], window["cis_end"]

    awk_prog = (
        f'NR==1 || ($1=="{regenie_chrom}" && $2>={lo} && $2<={hi})'
    )

    with subprocess.Popen(
        ["tar", "-xOf", cfg["tar"], cfg["member"]], stdout=subprocess.PIPE
    ) as tar_proc:
        with subprocess.Popen(
            ["gunzip", "-c"], stdin=tar_proc.stdout, stdout=subprocess.PIPE
        ) as gz_proc:
            awk = subprocess.run(
                ["awk", awk_prog],
                stdin=gz_proc.stdout, capture_output=True, text=True,
            )
            gz_proc.stdout.close()
        tar_proc.stdout.close()

    if not awk.stdout.strip():
        return None

    lines = awk.stdout.strip().splitlines()
    rows = [line.split() for line in lines]
    header = rows[0]
    if not rows[1:]:
        return None

    df = pd.DataFrame(rows[1:], columns=header)
    for col in ["GENPOS", "BETA", "SE", "A1FREQ", "LOG10P"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop zero or negative SE
    df = df[df["SE"] > 0].copy()
    if df.empty:
        return None

    # Compute N from the column if present; fall back to UKB-PPP sample size (~54 k)
    if "N" in df.columns:
        df["N_exp"] = pd.to_numeric(df["N"], errors="coerce")
        n_missing = df["N_exp"].isna().sum()
        if n_missing > 0:
            print(
                f"  WARNING: {n_missing} rows with missing N in UKB REGENIE; "
                f"filling with UKB-PPP sample size {UKB_N_DEFAULT:,}",
                file=sys.stderr, flush=True,
            )
        df["N_exp"] = df["N_exp"].fillna(UKB_N_DEFAULT).astype(int)
        # If all N values are zero (empty column), use the published default
        if (df["N_exp"] == 0).all():
            print(
                f"  WARNING: N column all-zero in UKB REGENIE; "
                f"using UKB-PPP sample size {UKB_N_DEFAULT:,}",
                file=sys.stderr, flush=True,
            )
            df["N_exp"] = UKB_N_DEFAULT
    else:
        print(
            f"  WARNING: No N column in UKB REGENIE; "
            f"using UKB-PPP sample size {UKB_N_DEFAULT:,}",
            file=sys.stderr, flush=True,
        )
        df["N_exp"] = UKB_N_DEFAULT

    # Join rsid
    rsid_map = pd.read_csv(cfg["rsid_map"], sep="\t", usecols=["ID","rsid"])
    rsid_map = rsid_map[rsid_map["rsid"].notna()].drop_duplicates(subset="ID", keep="first")
    df = df.merge(rsid_map[["ID","rsid"]], on="ID", how="left")

    df["effectAllele"] = df["ALLELE1"]
    df["otherAllele"]  = df["ALLELE0"]
    df["beta_exp"]     = df["BETA"]
    df["se_exp"]       = df["SE"]

    out_cols = ["rsid", "effectAllele", "otherAllele", "beta_exp", "se_exp", "N_exp"]
    return df[out_cols].copy()


# ---------------------------------------------------------------------------
# Harmonization
# ---------------------------------------------------------------------------

def harmonize(exp_df: pd.DataFrame, pd_df: pd.DataFrame) -> pd.DataFrame | None:
    """Merge exposure and outcome by rsid and align alleles.

    Drops:
      - Palindromic variants (A/T or C/G) -- cannot reliably strand-align.
      - Allele mismatches (incompatible alleles between datasets).

    Returns DataFrame with columns:
      snp, beta_exp, se_exp, N_exp, beta_out, se_out
    or None if the result is empty.

    Uses vectorized pandas operations (no iterrows) to avoid memory
    fragmentation issues on large DataFrames.
    """
    exp = exp_df.copy()
    exp = exp[exp["rsid"].notna()].copy()
    if exp.empty:
        return None

    merged = pd.merge(
        exp,
        pd_df[["rsid","effect_allele","other_allele","beta","standard_error"]],
        on="rsid", how="inner",
    )
    if merged.empty:
        return None

    # Normalise alleles to uppercase strings
    merged = merged.assign(
        ea_exp=merged["effectAllele"].str.upper(),
        oa_exp=merged["otherAllele"].str.upper(),
        ea_out=merged["effect_allele"].str.upper(),
        oa_out=merged["other_allele"].str.upper(),
    )

    # Drop palindromic variants
    palindrome_mask = [
        (ea, oa) in PALINDROME_PAIRS
        for ea, oa in zip(merged["ea_exp"], merged["oa_exp"])
    ]
    merged = merged[~pd.array(palindrome_mask)].copy()
    if merged.empty:
        return None

    # Identify direct matches and flipped matches
    direct = (merged["ea_exp"] == merged["ea_out"]) & (merged["oa_exp"] == merged["oa_out"])
    flipped = (merged["ea_exp"] == merged["oa_out"]) & (merged["oa_exp"] == merged["ea_out"])

    # Combine direct and flipped, orienting beta_out
    direct_rows = merged[direct].copy()
    direct_rows["beta_out_aligned"] = direct_rows["beta"].astype(float)

    flipped_rows = merged[flipped].copy()
    flipped_rows["beta_out_aligned"] = -flipped_rows["beta"].astype(float)

    combined = pd.concat([direct_rows, flipped_rows], ignore_index=True)
    if combined.empty:
        return None

    # Resolve N_exp: replace 0 / NaN with 0 (will be imputed later)
    n_exp_series = pd.to_numeric(combined["N_exp"], errors="coerce").fillna(0)
    n_exp_series = n_exp_series.where(n_exp_series > 0, 0).astype(int)

    result = pd.DataFrame({
        "snp":      combined["rsid"].values,
        "beta_exp": combined["beta_exp"].astype(float).values,
        "se_exp":   combined["se_exp"].astype(float).values,
        "N_exp":    n_exp_series.values,
        "beta_out": combined["beta_out_aligned"].values,
        "se_out":   combined["standard_error"].astype(float).values,
    })

    return result if len(result) > 0 else None


# ---------------------------------------------------------------------------
# LD matrix computation
# ---------------------------------------------------------------------------

def compute_ld(rsids: list[str], tag: str, tmpdir: str) -> tuple[str | None, list[str]]:
    """Produce a plink --r square LD file from 1000G EUR for the given rsids.

    Steps:
      1. Write rsid list and run plink --make-bed --extract.
      2. Run plink --r square on the subset bfile.

    Returns (ld_file_path, snp_list) where snp_list is the SNP order from the
    BIM file (matching rows/cols of the LD file). The LD file is passed directly
    to R, which reads it with scan() — avoiding a Python numpy load of potentially
    very large (N^2) float matrices that can cause SIGSEGV on some systems.

    Returns (None, []) on failure.
    """
    snp_file   = os.path.join(tmpdir, f"{tag}_snps.txt")
    sub_prefix = os.path.join(tmpdir, f"{tag}_sub")
    ld_prefix  = os.path.join(tmpdir, f"{tag}_ld")

    with open(snp_file, "w") as f:
        for r in rsids:
            f.write(r + "\n")

    # Step 1: subset bfile
    res = subprocess.run([
        "plink", "--bfile", LD_BFILE,
        "--extract", snp_file,
        "--make-bed", "--out", sub_prefix,
        "--silent",
    ], capture_output=True, text=True)

    bim_file = sub_prefix + ".bim"
    if not os.path.exists(bim_file):
        print(f"  plink make-bed failed for {tag}: {res.stderr[-200:]}", flush=True)
        return None, []

    bim = pd.read_csv(bim_file, sep="\t", header=None,
                      names=["chrom","rsid","cm","pos","a1","a2"])
    ld_snps = bim["rsid"].tolist()
    if len(ld_snps) < 2:
        return None, []

    # Step 2: compute LD matrix (plink writes tab-separated float values)
    res2 = subprocess.run([
        "plink", "--bfile", sub_prefix,
        "--r", "square",
        "--out", ld_prefix,
        "--silent",
    ], capture_output=True, text=True)

    ld_file = ld_prefix + ".ld"
    if not os.path.exists(ld_file):
        print(f"  plink --r square failed for {tag}: {res2.stderr[-200:]}", flush=True)
        return None, []

    # Return the raw plink LD file path; R reads it directly with scan() to
    # avoid loading the full N×N float64 matrix into Python (which can SIGSEGV
    # for N > 2000 on some systems due to numpy memory pressure).
    return ld_file, ld_snps


# ---------------------------------------------------------------------------
# Single-pair colocalization
# ---------------------------------------------------------------------------

def run_coloc(
    protein: str,
    platform: str,
    window: dict,
    n_signals: int,
    method: str,
    pd_df: pd.DataFrame,
    tmpdir: str,
) -> dict:
    """Run colocalization for one (protein, platform) pair.

    Returns a dict: {protein, platform, method, pp4, n_signals}.
    """
    print(f"\n=== {protein} / {platform} | {method} ({n_signals} signal(s)) ===", flush=True)

    # --- 1. Extract exposure sumstats ---
    if platform == "decode":
        exp_df = read_decode_cis(protein, window)
    else:
        exp_df = read_ukb_cis(protein, window)

    if exp_df is None or exp_df.empty:
        nodata_label = method + "_nodata"
        print(f"  No exposure SNPs in cis window - skipping (label: {nodata_label}).", flush=True)
        return {"protein": protein, "platform": platform,
                "method": nodata_label, "pp4": 0.0, "pp4_estimable": False,
                "n_signals": n_signals}

    print(f"  Exposure SNPs in cis window: {len(exp_df)}", file=sys.stderr, flush=True)

    # --- 2. Subset PD GWAS to cis window ---
    # NOTE: the Nalls 2019 harmonised file uses chromosome/base_pair_location
    # as the harmonised GRCh38 columns (GWAS Catalog convention). windows.json
    # also uses GRCh38 coordinates.
    chrom_int = int(window["chrom"]) if window["chrom"] != "X" else 23
    lo, hi = window["cis_start"], window["cis_end"]
    pd_sub = pd_df[
        (pd_df["chromosome"] == chrom_int) &
        (pd_df["base_pair_location"] >= lo) &
        (pd_df["base_pair_location"] <= hi)
    ].copy()

    print(
        f"  SNP counts: exposure={len(exp_df)}, GWAS in cis={len(pd_sub)}",
        file=sys.stderr, flush=True,
    )

    if pd_sub.empty:
        nodata_label = method + "_nodata"
        print(f"  No PD GWAS SNPs in cis window - skipping (label: {nodata_label}).", flush=True)
        return {"protein": protein, "platform": platform,
                "method": nodata_label, "pp4": 0.0, "pp4_estimable": False,
                "n_signals": n_signals}

    print(f"  PD GWAS SNPs in cis window: {len(pd_sub)}", flush=True)

    # --- 3. Harmonize ---
    harm = harmonize(exp_df, pd_sub)
    if harm is None or len(harm) < 5:
        n_harm = 0 if harm is None else len(harm)
        nodata_label = method + "_nodata"
        print(
            f"  Too few harmonized SNPs ({n_harm}) - skipping "
            f"(label: {nodata_label}).",
            flush=True,
        )
        return {"protein": protein, "platform": platform,
                "method": nodata_label, "pp4": 0.0, "pp4_estimable": False,
                "n_signals": n_signals}

    print(f"  Harmonized SNPs: {len(harm)}", flush=True)

    # Fill missing N_exp with a reasonable default
    median_n_exp = int(harm["N_exp"].replace(0, np.nan).median())
    if np.isnan(median_n_exp) or median_n_exp <= 0:
        median_n_exp = 35000
    harm["N_exp"] = harm["N_exp"].replace(0, median_n_exp)

    # --- 4. Write data files ---
    tag = f"{protein}_{platform}"
    meta_file = os.path.join(tmpdir, f"{tag}_meta.txt")
    data_file = os.path.join(tmpdir, f"{tag}_data.tsv")

    with open(meta_file, "w") as f:
        f.write(f"method\t{method}\n")
        f.write(f"N_exp\t{median_n_exp}\n")
        f.write(f"N_out\t{PD_N}\n")
        f.write(f"s_out\t{PD_S:.8f}\n")

    harm[["snp","beta_exp","se_exp","beta_out","se_out"]].to_csv(data_file, sep="\t", index=False)

    # --- 5. LD computation for SuSiE ---
    # compute_ld() returns the raw plink .ld file path (no Python matrix load)
    # so that R can read it with scan() directly, avoiding SIGSEGV from
    # loading large N×N float64 matrices into Python.
    extra_args = []
    if method == "coloc.susie":
        rsids = harm["snp"].dropna().tolist()
        ld_file_raw, ld_snps = compute_ld(rsids, tag, tmpdir)
        if ld_file_raw is not None and len(ld_snps) >= 10:
            ld_snps_file = os.path.join(tmpdir, f"{tag}_ld_snps.txt")
            with open(ld_snps_file, "w") as f:
                for s in ld_snps:
                    f.write(s + "\n")
            extra_args = [ld_file_raw, ld_snps_file]
            print(f"  LD matrix: {len(ld_snps)} x {len(ld_snps)}", flush=True)
        else:
            print(f"  Too few SNPs in 1000G panel; will fall back to coloc.abf in R", flush=True)

    # --- 6. Call R script ---
    cmd = [RSCRIPT, COLOC_R, meta_file, data_file] + extra_args
    r_result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if r_result.returncode != 0:
        print(f"  R script error (rc={r_result.returncode}):\n{r_result.stderr[-500:]}", flush=True)
        # Distinct method label + pp4_estimable=False: an R-level crash must
        # never be reportable as a genuine "no colocalization" zero.
        return {"protein": protein, "platform": platform,
                "method": f"{method}_error", "pp4": 0.0, "pp4_estimable": False,
                "n_signals": n_signals}

    # --- 7. Parse R output ---
    r_out = {}
    for line in r_result.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            r_out[k.strip()] = v.strip()

    used_method = r_out.get("method", method)

    def _num(key):
        """Parse an emitted numeric; 'NA'/missing/garbage -> None (not 0.0)."""
        raw = r_out.get(key, "NA")
        if raw == "NA":
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    pp4 = _num("pp4")
    # estimable is authoritative; a parse failure counts as NOT estimable.
    estimable = r_out.get("estimable", "").upper() == "TRUE" and pp4 is not None
    if pp4 is None:
        pp4 = 0.0
    pp4 = max(0.0, min(1.0, pp4))

    rec = {"protein": protein, "platform": platform,
           "method": used_method, "pp4": pp4, "pp4_estimable": estimable,
           "n_signals": n_signals}

    # Pre-registered PP0-PP4 decomposition, where the method produces one.
    for k in ("pp0", "pp1", "pp2", "pp3"):
        v = _num(k)
        if v is not None:
            rec[k] = max(0.0, min(1.0, v))

    if estimable:
        print(f"  Result: method={used_method}, PP4={pp4:.4f}", flush=True)
    else:
        print(f"  Result: method={used_method}, PP4 NOT ESTIMABLE "
              f"(no posterior computed; reported as 0.0 for field compatibility)",
              flush=True)

    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate colocalization for all protein/platform pairs."""
    with open(WINDOWS_JSON) as f:
        windows = json.load(f)

    dec_instr = pd.read_parquet(DEC_PAR)
    ukb_instr = pd.read_parquet(UKB_PAR)

    # Pre-load PD GWAS (17M rows; done once)
    print("Loading PD GWAS sumstats...", flush=True)
    pd_gwas = pd.read_csv(
        PD_GWAS, sep="\t",
        usecols=["chromosome","base_pair_location","effect_allele","other_allele",
                 "beta","standard_error","rsid"],
        dtype={"chromosome": "int16", "base_pair_location": "int32"},
    )
    print(f"  PD GWAS loaded: {len(pd_gwas):,} rows", flush=True)

    # --- Build sanity check ---
    # LRRK2 G2019S (rs34637584): GRCh38 = chr12:40,340,400; GRCh37 = chr12:40,614,434.
    # The Nalls harmonised file uses GWAS Catalog GRCh38 coordinates in the
    # chromosome / base_pair_location columns.
    lrrk2 = pd_gwas[pd_gwas["rsid"] == "rs34637584"]
    if lrrk2.empty:
        print(
            "  Build WARNING: rs34637584 (LRRK2 G2019S) not found in PD GWAS",
            file=sys.stderr, flush=True,
        )
    else:
        row = lrrk2.iloc[0]
        chrom_ok = int(row["chromosome"]) == 12
        # GRCh38 position is 40,340,400; GRCh37 is 40,614,434.
        # Accept anything within 500 kb of the GRCh38 position.
        pos_ok = abs(int(row["base_pair_location"]) - 40_340_400) < 500_000
        if chrom_ok and pos_ok:
            print(
                f"  Build check PASS: rs34637584 at "
                f"chr{int(row['chromosome'])}:{int(row['base_pair_location'])} "
                "(consistent with GRCh38)",
                flush=True,
            )
        else:
            print(
                f"  Build WARNING: rs34637584 at "
                f"chr{int(row['chromosome'])}:{int(row['base_pair_location'])} "
                "- unexpected for GRCh38 (check coordinate build)",
                file=sys.stderr, flush=True,
            )

    tmpdir = tempfile.mkdtemp(prefix="mr_coloc_")
    results = []

    try:
        # deCODE pairs
        for protein in sorted(dec_instr["protein"].unique()):
            if protein in SKIP_PROTEINS:
                print(f"\n=== {protein} / decode: SKIPPED (excluded protein) ===", flush=True)
                continue
            if protein not in windows:
                print(f"\n=== {protein} / decode: SKIPPED (no window) ===", flush=True)
                continue
            n_sig = int((dec_instr["protein"] == protein).sum())
            method = "coloc.susie" if n_sig > 1 else "coloc.abf"
            r = run_coloc(protein, "decode", windows[protein], n_sig, method, pd_gwas, tmpdir)
            results.append(r)

        # UKB-PPP pairs
        for protein in sorted(ukb_instr["protein"].unique()):
            if protein in SKIP_PROTEINS:
                print(f"\n=== {protein} / ukbppp: SKIPPED (excluded protein) ===", flush=True)
                continue
            if protein not in UKB_CONFIG:
                print(f"\n=== {protein} / ukbppp: SKIPPED (not in UKB_CONFIG) ===", flush=True)
                continue
            if protein not in windows:
                print(f"\n=== {protein} / ukbppp: SKIPPED (no window) ===", flush=True)
                continue
            n_sig = int((ukb_instr["protein"] == protein).sum())
            method = "coloc.susie" if n_sig > 1 else "coloc.abf"
            r = run_coloc(protein, "ukbppp", windows[protein], n_sig, method, pd_gwas, tmpdir)
            results.append(r)

        # --- GPNMB positive control ---
        # GPNMB is a known PD-colocalizing locus (chr7, GRCh38).
        # 2 deCODE instruments -> coloc.susie. Expected PP4 > 0.5.
        # Result is flagged positive_control=true so it is excluded from the
        # main BBB colocalization table but validates the whole pipeline.
        print(
            "\n=== GPNMB / decode (POSITIVE CONTROL) | coloc.susie (2 signals) ===",
            flush=True,
        )
        r_gpnmb = run_coloc(
            "GPNMB", "decode", GPNMB_WINDOW, 2, "coloc.susie", pd_gwas, tmpdir
        )
        r_gpnmb["positive_control"] = True
        results.append(r_gpnmb)
        print(
            f"  GPNMB positive-control PP4 = {r_gpnmb['pp4']:.4f}",
            flush=True,
        )
        if not r_gpnmb.get("pp4_estimable", True):
            print(
                "  PIPELINE WARNING: GPNMB PP4 is NOT ESTIMABLE (no posterior "
                "computed). The coloc gate is not satisfied and the BBB "
                "non-colocalization result cannot be credited.",
                file=sys.stderr, flush=True,
            )
        elif r_gpnmb["pp4"] < 0.5:
            print(
                "  PIPELINE WARNING: GPNMB PP4 < 0.5 - the coloc pipeline may "
                "be broken (build/LD/units bug). BBB non-colocalization result "
                "is potentially an artifact.",
                file=sys.stderr, flush=True,
            )
        else:
            print(
                "  PIPELINE OK: GPNMB PP4 >= 0.5 - pipeline validated. "
                "The BBB non-colocalization result is credible, but note that "
                "for rows where PP4 is NOT ESTIMABLE the correct reading is "
                "'no overlapping credible sets', not 'PP4 = 0'.",
                flush=True,
            )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    os.makedirs("data/processed/mr", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} colocalization results to {OUT_JSON}", flush=True)
    print("\nSummary:", flush=True)
    n_not_est = 0
    for r in results:
        if r.get("pp4_estimable", True):
            pp4_str = f"PP4={r['pp4']:.4f}     "
        else:
            pp4_str = "PP4=NOT ESTIMABLE"
            n_not_est += 1
        print(f"  {r['protein']:8s} {r['platform']:8s}  method={r['method']:<22s}  "
              f"{pp4_str}  n_signals={r['n_signals']}", flush=True)

    if n_not_est:
        print(
            f"\nNOTE: {n_not_est}/{len(results)} rows have NO computable PP4. "
            "Their stored pp4=0.0 is a field-compatibility placeholder, NOT a "
            "posterior. Report these as 'no overlapping credible sets under "
            "SuSiE fine-mapping; PP4 not estimable' -- never as 'PP4 = 0'.",
            flush=True,
        )


if __name__ == "__main__":
    main()
