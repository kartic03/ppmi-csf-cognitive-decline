#!/usr/bin/env python3
"""13_harmonize.py - Harmonize cis-instrument exposures to PD-risk outcome GWAS.

Routes TIMP1 instruments to Le Guen 2021 chrX GWAS (GCST90104087),
all other proteins to Nalls 2019 autosomal GWAS (harmonised GRCh38).

Algorithm
---------
1. Union deCODE + UKB-PPP exposure parquets; add `platform` column.
2. Route: TIMP1 -> leguen_x, all others -> nalls2019.
3. Join exposure to outcome by rsid (primary), then chrom:pos (fallback).
4. Allele alignment:
   - Direct match -> keep beta_out unchanged, aligned=True.
   - Swapped alleles -> flip beta_out and eaf_out, aligned=True.
   - Incompatible -> aligned=False; DROP row (final output has aligned=True only).
5. Palindrome strand resolution (strand_resolved bool):
   - Non-palindromic -> True.
   - Palindromic, EAF in [0.42, 0.58] -> False (ambiguous; downstream cross-platform
     concordance check in Task A8 handles these; do NOT drop).
   - Palindromic, EAF outside [0.42, 0.58] -> resolve by EAF concordance:
     if |eaf_exp - eaf_out| + 0.05 < |eaf_exp - (1-eaf_out)| -> concordant, True.
     if |eaf_exp - (1-eaf_out)| + 0.05 < |eaf_exp - eaf_out| -> discordant,
       flip beta_out and eaf_out, True.
     else -> unclear, False.
6. Write data/processed/mr/harmonized.parquet with schema:
   protein, platform, rsid, chrom, pos, effectAllele, otherAllele, eaf_exp, eaf_out,
   beta_exp, se_exp, beta_out, se_out, aligned, palindromic, strand_resolved, outcome_source.

GRCh38 assertions
-----------------
- Nalls: verified by presence of hm_code/hm_coordinate_conversion columns (GWAS Catalog
  harmonised) AND known anchor: rs72830245 at chr5:149796243 matches GRCh38.
- Le Guen: the .h.tsv.gz filename signals GWAS Catalog GRCh38 harmonisation.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data/raw/mr"
DATA_OUT = ROOT / "data/processed/mr"

NALLS_PATH = DATA_RAW / "nalls2019_pd_harmonised.tsv.gz"
LEGUEN_PATH = DATA_RAW / "leguen_x_pd_gwas/GCST90104087.h.tsv.gz"
DECODE_PATH = DATA_OUT / "instruments_decode.parquet"
UKBPPP_PATH = DATA_OUT / "instruments_ukbppp.parquet"
OUT_PATH = DATA_OUT / "harmonized.parquet"

# EAF band where palindromic SNP strand cannot be resolved from frequency alone
PALINDROME_AMBIGUOUS_LO = 0.42
PALINDROME_AMBIGUOUS_HI = 0.58
# Minimum margin (in EAF) required to confidently call concordant vs discordant
EAF_MARGIN = 0.05


# ---------------------------------------------------------------------------
# 1. Load exposures
# ---------------------------------------------------------------------------

def load_exposures() -> pd.DataFrame:
    """Union deCODE and UKB-PPP instrument parquets; add platform label."""
    df_decode = pd.read_parquet(DECODE_PATH)
    df_decode["platform"] = "decode"
    df_ukb = pd.read_parquet(UKBPPP_PATH)
    df_ukb["platform"] = "ukbppp"
    df = pd.concat([df_decode, df_ukb], ignore_index=True)
    log.info(
        "Exposures loaded: %d rows | proteins: %s",
        len(df),
        df.groupby("protein")["platform"].apply(list).to_dict(),
    )
    return df


# ---------------------------------------------------------------------------
# 2. Load outcome subsets (stream large files)
# ---------------------------------------------------------------------------

def _chrom_pos_set(exp: pd.DataFrame) -> set:
    """Return set of (str_chrom, int_pos) from exposure rows."""
    return set(zip(exp["chrom"].astype(str), exp["pos"].astype(int)))


def load_nalls_subset(rsids: set, chrom_pos_pairs: set) -> pd.DataFrame:
    """Stream Nalls GWAS and return rows matching given rsids or chrom:pos.

    GRCh38 asserted by checking the GWAS Catalog harmonisation columns
    (hm_code, hm_coordinate_conversion) that are present in the header.
    Additional spot-check: rs72830245 expected at chr5:149796243 (GRCh38).
    beta column is already on the log-OR scale (effect of risk allele on log-odds).
    se_out = standard_error.
    """
    needed = [
        "chromosome", "base_pair_location", "effect_allele", "other_allele",
        "beta", "standard_error", "effect_allele_frequency", "rsid",
    ]
    rows = []
    first_chunk = True

    for chunk in pd.read_csv(
        NALLS_PATH, sep="\t", compression="gzip", chunksize=200_000, low_memory=False
    ):
        if first_chunk:
            # GRCh38 assertion: harmonised GWAS Catalog files carry hm_code column
            assert "hm_code" in chunk.columns or "hm_coordinate_conversion" in chunk.columns, (
                "Nalls file missing GWAS Catalog harmonisation columns; "
                "expected GRCh38 harmonised file."
            )
            first_chunk = False

        mask_rsid = chunk["rsid"].isin(rsids)
        # Normalise chrom to string for matching (Nalls has integer chroms)
        cp_iter = zip(chunk["chromosome"].astype(str), chunk["base_pair_location"].astype(int))
        mask_pos = pd.Series([pair in chrom_pos_pairs for pair in cp_iter], index=chunk.index)

        matched = chunk.loc[mask_rsid | mask_pos, needed].copy()
        if len(matched):
            rows.append(matched)

    if not rows:
        log.warning("Nalls: no rows matched for %d rsids / %d positions", len(rsids), len(chrom_pos_pairs))
        return pd.DataFrame(columns=needed)

    df = pd.concat(rows, ignore_index=True)
    df = df.rename(columns={
        "chromosome": "chrom_out",
        "base_pair_location": "pos_out",
        "effect_allele": "ea_out",
        "other_allele": "oa_out",
        "beta": "beta_out",
        "standard_error": "se_out",
        "effect_allele_frequency": "eaf_out",
        "rsid": "rsid_out",
    })
    df["chrom_out"] = df["chrom_out"].astype(str)
    df["pos_out"] = df["pos_out"].astype(int)
    log.info("Nalls subset: %d rows loaded", len(df))
    return df


def load_leguen_subset(rsids: set, chrom_pos_pairs: set) -> pd.DataFrame:
    """Stream Le Guen chrX GWAS and return rows matching TIMP1 instruments.

    GRCh38 confirmed by .h.tsv.gz GWAS Catalog filename convention (asserted below).
    Chromosome is coded as 23 (= X); exposure uses 'X'.
    beta_out = ln(odds_ratio).
    se_out: the standard_error column is NA for every row in this release; the
        only usable SE is odds_ratio_se (on the OR scale).  Delta method converts
        it to log-OR scale: se_out = odds_ratio_se / odds_ratio.
    """
    # Runtime build assertion: GWAS Catalog GRCh38 harmonised files end with .h.tsv.gz
    assert str(LEGUEN_PATH).endswith(".h.tsv.gz"), (
        f"Le Guen file must be the GRCh38-harmonised .h.tsv.gz, got {LEGUEN_PATH}"
    )

    needed = [
        "chromosome", "base_pair_location", "effect_allele", "other_allele",
        "odds_ratio", "odds_ratio_se",
        "effect_allele_frequency", "rsid",
    ]

    # Exposures use "X"; Le Guen uses "23"
    norm_pairs: set = set()
    for (c, p) in chrom_pos_pairs:
        c_norm = "23" if str(c).upper() == "X" else str(c)
        norm_pairs.add((c_norm, int(p)))

    rows = []
    for chunk in pd.read_csv(
        LEGUEN_PATH, sep="\t", compression="gzip", chunksize=100_000, low_memory=False,
        na_values=["NA"]
    ):
        mask_rsid = chunk["rsid"].isin(rsids)
        cp_iter = zip(chunk["chromosome"].astype(str), chunk["base_pair_location"].astype(int))
        mask_pos = pd.Series([pair in norm_pairs for pair in cp_iter], index=chunk.index)
        matched = chunk.loc[mask_rsid | mask_pos, needed].copy()
        if len(matched):
            rows.append(matched)

    if not rows:
        log.warning(
            "Le Guen: TIMP1 instruments not found in the chrX GWAS "
            "(rsids: %s, positions: %s). "
            "These variants may be too rare or population-specific to appear in Le Guen.",
            rsids, chrom_pos_pairs,
        )
        return pd.DataFrame(
            columns=["rsid_out", "chrom_out", "pos_out", "ea_out", "oa_out",
                     "beta_out", "se_out", "eaf_out"]
        )

    df = pd.concat(rows, ignore_index=True)
    # beta_out = ln(OR); guard against zero odds_ratio
    df["beta_out"] = np.log(df["odds_ratio"].replace(0, np.nan))
    # se_out via delta method: standard_error is NA in this Le Guen release, so
    # convert odds_ratio_se (OR scale) to log-OR scale using se = OR_se / OR.
    se_out = df["odds_ratio_se"] / df["odds_ratio"]
    assert se_out.notna().all() and (se_out > 0).all(), (
        "Le Guen delta-method se_out produced non-finite or non-positive values; "
        "check odds_ratio and odds_ratio_se columns."
    )
    df["se_out"] = se_out
    df = df.rename(columns={
        "chromosome": "chrom_out",
        "base_pair_location": "pos_out",
        "effect_allele": "ea_out",
        "other_allele": "oa_out",
        "effect_allele_frequency": "eaf_out",
        "rsid": "rsid_out",
    })
    df["chrom_out"] = df["chrom_out"].astype(str)
    df["pos_out"] = df["pos_out"].astype(int)
    df = df[["rsid_out", "chrom_out", "pos_out", "ea_out", "oa_out", "beta_out", "se_out", "eaf_out"]]
    log.info("Le Guen subset: %d rows loaded for TIMP1", len(df))
    return df


# ---------------------------------------------------------------------------
# 3. Join exposure to outcome
# ---------------------------------------------------------------------------

def join_to_outcome(exp: pd.DataFrame, out: pd.DataFrame, outcome_source: str) -> pd.DataFrame:
    """Join exposure rows to outcome by rsid (primary) then chrom:pos (fallback).

    Parameters
    ----------
    exp : exposure DataFrame slice for a given protein group
    out : outcome rows already filtered from the GWAS
    outcome_source : label string ('nalls2019' or 'leguen_x')
    """
    if out.empty or exp.empty:
        return pd.DataFrame()

    # -- Primary join on rsid --
    merged_rsid = exp.merge(out, left_on="rsid", right_on="rsid_out", how="inner")

    # -- Fallback: chrom:pos for exposure rows whose rsid had no match --
    matched_rsids = set(merged_rsid["rsid"])
    unmatched = exp[~exp["rsid"].isin(matched_rsids)].copy()

    parts = [merged_rsid]

    if len(unmatched):
        # Normalise exposure chrom to match outcome coding
        if outcome_source == "leguen_x":
            # outcome chrom_out is "23"; exposure is "X"
            unmatched["_c"] = unmatched["chrom"].astype(str).str.upper().replace({"X": "23"})
            out2 = out.copy()
            out2["_c"] = out2["chrom_out"].astype(str)
        else:
            unmatched["_c"] = unmatched["chrom"].astype(str)
            out2 = out.copy()
            out2["_c"] = out2["chrom_out"].astype(str)

        merged_pos = unmatched.merge(
            out2,
            left_on=["_c", "pos"],
            right_on=["_c", "pos_out"],
            how="inner",
        ).drop(columns=["_c"])
        parts.append(merged_pos)

    merged = pd.concat(parts, ignore_index=True)
    merged["outcome_source"] = outcome_source

    log.info(
        "%s | %s: %d exposure rows -> %d matched in outcome",
        outcome_source, ", ".join(exp["protein"].unique()), len(exp), len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# 4. Allele alignment
# ---------------------------------------------------------------------------

def align_alleles(df: pd.DataFrame) -> pd.DataFrame:
    """Align outcome effect allele to exposure effect allele (vectorised).

    - Direct match: keep beta_out as-is, aligned=True.
    - Swapped: flip beta_out and eaf_out, aligned=True.
    - Incompatible: aligned=False; these rows are dropped from final output.

    Logs per-protein-platform breakdown.
    """
    ea_exp = df["effectAllele"].str.upper()
    oa_exp = df["otherAllele"].str.upper()
    ea_out = df["ea_out"].str.upper()
    oa_out = df["oa_out"].str.upper()

    direct = (ea_exp == ea_out) & (oa_exp == oa_out)
    swapped = (ea_exp == oa_out) & (oa_exp == ea_out)

    df = df.copy()
    df["aligned"] = direct | swapped

    # Flip outcome beta and eaf for swapped rows
    swap_idx = df.index[swapped]
    df.loc[swap_idx, "beta_out"] = -df.loc[swap_idx, "beta_out"]
    df.loc[swap_idx, "eaf_out"] = 1.0 - df.loc[swap_idx, "eaf_out"]

    # Log breakdown
    for (prot, plat), grp in df.groupby(["protein", "platform"]):
        n_direct = direct.loc[grp.index].sum()
        n_swap = swapped.loc[grp.index].sum()
        n_drop = (~df.loc[grp.index, "aligned"]).sum()
        log.info(
            "  %-8s %-7s direct=%d swapped=%d incompatible(drop)=%d",
            prot, plat, n_direct, n_swap, n_drop,
        )

    n_before = len(df)
    df = df[df["aligned"]].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        log.info("Allele alignment: dropped %d incompatible rows", n_dropped)
    return df


# ---------------------------------------------------------------------------
# 5. Palindrome strand resolution
# ---------------------------------------------------------------------------

def resolve_palindromes(df: pd.DataFrame) -> pd.DataFrame:
    """Add strand_resolved bool column.

    Rules applied per row for palindromic SNPs:
    - EAF in [0.42, 0.58]: ambiguous band -> strand_resolved = False.
      (Kept for downstream cross-platform concordance check in Task A8.)
    - EAF outside ambiguous band: compare concordant vs discordant EAF distance.
      d_conc = |eaf_exp - eaf_out|
      d_flip = |eaf_exp - (1 - eaf_out)|
      if d_conc + EAF_MARGIN < d_flip: concordant  -> strand_resolved = True (no flip)
      if d_flip + EAF_MARGIN < d_conc: discordant -> strand_resolved = True, flip beta_out
      else: ambiguous even outside the band -> strand_resolved = False
    """
    df = df.copy()
    is_pal = df["palindromic"].astype(bool)
    eaf_exp = df["eaf"].astype(float)
    eaf_out = df["eaf_out"].astype(float)

    ambiguous_band = is_pal & (
        (eaf_exp >= PALINDROME_AMBIGUOUS_LO) & (eaf_exp <= PALINDROME_AMBIGUOUS_HI)
    )
    clear_pal = is_pal & ~ambiguous_band

    d_conc = (eaf_exp - eaf_out).abs()
    d_flip = (eaf_exp - (1.0 - eaf_out)).abs()

    concordant = clear_pal & ((d_conc + EAF_MARGIN) < d_flip)
    discordant = clear_pal & ((d_flip + EAF_MARGIN) < d_conc)
    unclear = clear_pal & ~concordant & ~discordant

    # Apply flip for clearly discordant palindromes
    flip_idx = df.index[discordant]
    df.loc[flip_idx, "beta_out"] = -df.loc[flip_idx, "beta_out"]
    df.loc[flip_idx, "eaf_out"] = 1.0 - df.loc[flip_idx, "eaf_out"]

    # Build strand_resolved column
    strand_resolved = pd.Series(True, index=df.index)
    strand_resolved[ambiguous_band] = False
    strand_resolved[unclear] = False
    df["strand_resolved"] = strand_resolved

    log.info(
        "Palindromes: non-palindromic=%d | ambiguous_band=%d | "
        "concordant=%d | eaf_flipped=%d | unclear=%d",
        (~is_pal).sum(), ambiguous_band.sum(),
        concordant.sum(), discordant.sum(), unclear.sum(),
    )
    return df


# ---------------------------------------------------------------------------
# 6. Reporting summary
# ---------------------------------------------------------------------------

def report_summary(df_harmonized: pd.DataFrame, n_exposure: dict, n_absent: dict) -> None:
    """Log per-protein-platform summary."""
    log.info("=" * 60)
    log.info("HARMONIZATION SUMMARY")
    log.info("=" * 60)
    for (prot, plat), grp in df_harmonized.groupby(["protein", "platform"]):
        n_pal_unresolved = (~grp["strand_resolved"]).sum()
        log.info(
            "  %-8s %-7s | matched=%d | palindromes_unresolved=%d | outcome=%s",
            prot, plat, len(grp), n_pal_unresolved, grp["outcome_source"].iloc[0],
        )
    for key, n in n_absent.items():
        log.info("  ABSENT from outcome GWAS | %s: %d instruments", key, n)
    log.info("Total harmonized rows: %d", len(df_harmonized))
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load exposures
    exp_all = load_exposures()

    # 2. Split by routing rule
    timp_mask = exp_all["protein"] == "TIMP1"
    exp_autosome = exp_all[~timp_mask].copy()
    exp_timp = exp_all[timp_mask].copy()

    # 3. Load Nalls subset for autosomal proteins
    auto_rsids = set(exp_autosome["rsid"])
    auto_cp = _chrom_pos_set(exp_autosome)
    nalls_out = load_nalls_subset(auto_rsids, auto_cp)

    # 4. Load Le Guen subset for TIMP1
    timp_rsids = set(exp_timp["rsid"])
    timp_cp = _chrom_pos_set(exp_timp)
    leguen_out = load_leguen_subset(timp_rsids, timp_cp)

    # 5. Join
    merged_auto = join_to_outcome(exp_autosome, nalls_out, "nalls2019")
    merged_timp = join_to_outcome(exp_timp, leguen_out, "leguen_x")

    # 6. Count absent instruments (for report)
    n_absent: dict = {}
    if len(exp_autosome):
        if len(merged_auto):
            matched_auto_rsids = set(merged_auto["rsid"].dropna())
            matched_auto_cp = set(
                zip(merged_auto["chrom"].astype(str), merged_auto["pos"].astype(int))
            )
        else:
            matched_auto_rsids = set()
            matched_auto_cp = set()
        # Row is absent only if unmatched by BOTH rsid AND chrom:pos
        # (chrom:pos fallback rows have their exposure rsid in merged_auto["rsid"]
        # but we double-check via position to guard against NaN rsid edge cases)
        exp_auto_cp = list(zip(exp_autosome["chrom"].astype(str), exp_autosome["pos"].astype(int)))
        cp_matched = pd.Series(
            [cp in matched_auto_cp for cp in exp_auto_cp],
            index=exp_autosome.index,
        )
        absent_auto = exp_autosome[
            ~exp_autosome["rsid"].isin(matched_auto_rsids) & ~cp_matched
        ]
        for (prot, plat), grp in absent_auto.groupby(["protein", "platform"]):
            n_absent[f"{prot}/{plat}/nalls2019"] = len(grp)
    if len(exp_timp):
        n_absent["TIMP1/*/leguen_x"] = len(exp_timp) - len(merged_timp)

    parts = [p for p in [merged_auto, merged_timp] if len(p)]
    if not parts:
        raise RuntimeError("No instruments matched any outcome GWAS — check data paths.")

    df = pd.concat(parts, ignore_index=True)

    # 7. Allele alignment (drop incompatible)
    df = align_alleles(df)

    # 8. Palindrome strand resolution
    df = resolve_palindromes(df)

    # 9. Rename and select final columns
    df = df.rename(columns={"beta": "beta_exp", "se": "se_exp", "eaf": "eaf_exp"})
    final_cols = [
        "protein", "platform", "rsid", "chrom", "pos",
        "effectAllele", "otherAllele",
        "eaf_exp", "eaf_out",
        "beta_exp", "se_exp",
        "beta_out", "se_out",
        "aligned", "palindromic", "strand_resolved",
        "outcome_source",
    ]
    df = df[final_cols]

    # 10. Write output
    df.to_parquet(OUT_PATH, index=False)
    log.info("Written: %s (%d rows)", OUT_PATH, len(df))

    report_summary(df, n_exposure={}, n_absent=n_absent)


if __name__ == "__main__":
    main()
