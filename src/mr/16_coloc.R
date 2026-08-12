#!/usr/bin/env Rscript
# 16_coloc.R -- Colocalization analysis for one (protein, platform) pair.
#
# Usage:
#   Rscript 16_coloc.R <meta_file> <data_tsv> [<ld_ld_file> <ld_snps_file>]
#
# meta_file: tab-separated key=value pairs, one per line:
#   method    coloc.susie   (or coloc.abf)
#   N_exp     35287
#   N_out     482730
#   s_out     0.069767
#
# data_tsv: tab-separated with header:
#   snp  beta_exp  se_exp  beta_out  se_out
#
# ld_ld_file: plink --r square output (whitespace-separated, no header, no row names)
# ld_snps_file: one rsid per line matching rows/cols of ld_ld_file
#
# Output (stdout):
#   method=coloc.susie
#   pp4=0.923

suppressPackageStartupMessages({
  library(coloc)
  library(susieR)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript 16_coloc.R <meta_file> <data_tsv> [<ld_ld_file> <ld_snps_file>]")
}

meta_file <- args[1]
data_file <- args[2]

# --- Read metadata ---
meta_lines <- readLines(meta_file)
meta <- list()
for (line in meta_lines) {
  parts <- strsplit(line, "\t")[[1]]
  if (length(parts) == 2) {
    meta[[parts[1]]] <- parts[2]
  }
}

method  <- meta[["method"]]
N_exp   <- as.integer(meta[["N_exp"]])
N_out   <- as.integer(meta[["N_out"]])
s_out   <- as.numeric(meta[["s_out"]])

# --- Read SNP data ---
dat <- read.table(data_file, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
snp      <- dat$snp
beta_exp <- as.numeric(dat$beta_exp)
se_exp   <- as.numeric(dat$se_exp)
beta_out <- as.numeric(dat$beta_out)
se_out   <- as.numeric(dat$se_out)

# Remove rows with any NA
ok <- !is.na(beta_exp) & !is.na(se_exp) & !is.na(beta_out) & !is.na(se_out) &
      se_exp > 0 & se_out > 0
if (sum(ok) < 2) {
  cat("method=coloc.abf\n")
  cat("pp4=0\n")
  quit(save = "no", status = 0)
}
snp      <- snp[ok]
beta_exp <- beta_exp[ok]
se_exp   <- se_exp[ok]
beta_out <- beta_out[ok]
se_out   <- se_out[ok]

# --- Build coloc datasets ---
dataset1 <- list(
  snp     = snp,
  beta    = beta_exp,
  varbeta = se_exp^2,
  N       = N_exp,
  # sdY=1 is an approximation for Olink NPX (log2-normalised protein abundance).
  # True sdY could be estimated from MAF + varbeta where MAF is available, but
  # NPX distributions are study-specific; sdY~1 is the accepted approximation
  # for SomaScan/Olink pQTLs and is a known limitation of this analysis.
  sdY     = 1,
  type    = "quant"
)
dataset2 <- list(
  snp     = snp,
  beta    = beta_out,
  varbeta = se_out^2,
  N       = N_out,
  s       = s_out,
  type    = "cc"
)

# --- coloc.abf fallback function ---
run_abf <- function(d1, d2, method_label = "coloc.abf") {
  tryCatch({
    res <- coloc.abf(d1, d2)
    pp4 <- as.numeric(res$summary["PP.H4.abf"])
    estimable <- !is.na(pp4)
    if (is.na(pp4)) pp4 <- 0.0
    # coloc.abf already computes the full PP.H0-H4 decomposition; report it.
    # DESIGN_SPEC_v3_npjPD.md:150 pre-registered "report PP0 through PP4
    # separately" -- previously only PP4 was extracted.
    getpp <- function(k) {
      v <- suppressWarnings(as.numeric(res$summary[k]))
      if (length(v) == 0 || is.na(v)) NA_real_ else v
    }
    list(method = method_label, pp4 = pp4, estimable = estimable,
         pp0 = getpp("PP.H0.abf"), pp1 = getpp("PP.H1.abf"),
         pp2 = getpp("PP.H2.abf"), pp3 = getpp("PP.H3.abf"))
  }, error = function(e) {
    message("coloc.abf failed: ", conditionMessage(e))
    # Distinct label + estimable=FALSE so a FAILED run can never be read as a
    # genuine "no colocalization" result.  Previously this returned pp4=0.0
    # under the unchanged method label, making failure indistinguishable from
    # a real zero posterior.
    list(method = paste0(method_label, "_error"), pp4 = 0.0, estimable = FALSE,
         pp0 = NA_real_, pp1 = NA_real_, pp2 = NA_real_, pp3 = NA_real_)
  })
}

# --- Main routing ---
if (method == "coloc.abf") {
  out <- run_abf(dataset1, dataset2)

} else if (method == "coloc.susie") {
  if (length(args) < 4) {
    message("No LD files provided for coloc.susie; falling back to coloc.abf")
    out <- run_abf(dataset1, dataset2, "coloc.abf_fallback")
  } else {
    ld_file   <- args[3]
    snps_file <- args[4]

    ld_snps <- readLines(snps_file)
    n_snps  <- length(ld_snps)

    # Read LD matrix using scan() rather than read.table().
    # read.table(sep="\t") on matrices with thousands of columns triggers
    # a stack overflow inside type.convert.default (R SIGSEGV, rc=-11).
    # scan() reads all whitespace-separated values into a flat numeric vector
    # and is robust for arbitrarily large matrices.
    ld_vals <- scan(ld_file, quiet = TRUE)

    if (length(ld_vals) != n_snps * n_snps) {
      message(
        "LD file has ", length(ld_vals), " values but expected ",
        n_snps, "^2 = ", n_snps * n_snps, "; falling back to coloc.abf"
      )
      out <- run_abf(dataset1, dataset2, "coloc.abf_fallback")
    } else {
      R_mat <- matrix(ld_vals, nrow = n_snps, ncol = n_snps, byrow = TRUE)
      rownames(R_mat) <- ld_snps
      colnames(R_mat) <- ld_snps

      # Align data SNPs to LD SNPs
      common <- intersect(snp, ld_snps)

      if (length(common) < 10) {
        message("Too few SNPs in LD panel (", length(common), "); falling back to coloc.abf")
        out <- run_abf(dataset1, dataset2, "coloc.abf_fallback")
      } else {
        idx_data <- match(common, snp)
        idx_ld   <- match(common, ld_snps)

        R_sub <- R_mat[idx_ld, idx_ld, drop = FALSE]

        # Ensure symmetry and diagonal = 1
        R_sub <- (R_sub + t(R_sub)) / 2
        diag(R_sub) <- 1.0

        # Regularize LD to ensure positive semi-definiteness
        lambda <- 0.01
        R_sub <- (1 - lambda) * R_sub + lambda * diag(nrow(R_sub))

        d1 <- list(
          snp     = common,
          beta    = beta_exp[idx_data],
          varbeta = se_exp[idx_data]^2,
          N       = N_exp,
          sdY     = 1,
          type    = "quant",
          LD      = R_sub
        )
        d2 <- list(
          snp     = common,
          beta    = beta_out[idx_data],
          varbeta = se_out[idx_data]^2,
          N       = N_out,
          s       = s_out,
          type    = "cc",
          LD      = R_sub
        )

        out <- tryCatch({
          s1 <- runsusie(d1)
          s2 <- runsusie(d2)
          coloc_res <- coloc.susie(s1, s2)
          if (is.null(coloc_res) || is.null(coloc_res$summary) ||
              nrow(coloc_res$summary) == 0) {
            # SuSiE ran but found no overlapping credible sets between the two
            # traits.  There is NO posterior to report: PP4 is NOT ESTIMABLE,
            # not zero.  pp4 is held at 0.0 only for backward compatibility of
            # the numeric field; estimable=FALSE is the authoritative signal and
            # downstream reporting MUST use it.  Do not describe this state as
            # "no colocalization (PP4 = 0)" -- see the reporting note at the
            # bottom of this file.
            message("coloc.susie returned empty summary (no overlapping CS); PP4 NOT ESTIMABLE")
            list(method = "coloc.susie_noCS", pp4 = 0.0, estimable = FALSE,
                 pp0 = NA_real_, pp1 = NA_real_, pp2 = NA_real_, pp3 = NA_real_)
          } else {
            pp4 <- max(coloc_res$summary$PP.H4.abf, na.rm = TRUE)
            estimable <- !is.na(pp4) && is.finite(pp4)
            if (!estimable) pp4 <- 0.0
            # coloc.susie's summary is per credible-set pair; take the same row
            # the reported PP4 came from so the decomposition is self-consistent.
            pp <- list(pp0 = NA_real_, pp1 = NA_real_, pp2 = NA_real_, pp3 = NA_real_)
            if (estimable) {
              i <- which.max(coloc_res$summary$PP.H4.abf)
              getcs <- function(k) {
                if (!k %in% names(coloc_res$summary)) return(NA_real_)
                v <- suppressWarnings(as.numeric(coloc_res$summary[[k]][i]))
                if (length(v) == 0 || is.na(v)) NA_real_ else v
              }
              pp <- list(pp0 = getcs("PP.H0.abf"), pp1 = getcs("PP.H1.abf"),
                         pp2 = getcs("PP.H2.abf"), pp3 = getcs("PP.H3.abf"))
            }
            list(method = "coloc.susie", pp4 = as.numeric(pp4), estimable = estimable,
                 pp0 = pp$pp0, pp1 = pp$pp1, pp2 = pp$pp2, pp3 = pp$pp3)
          }
        }, error = function(e) {
          message("SuSiE failed: ", conditionMessage(e),
                  "; falling back to coloc.abf")
          run_abf(dataset1, dataset2, "coloc.abf_fallback")
        })
      }  # end common >= 10
    }  # end scan size OK
  }  # end args >= 4
} else {
  stop(paste("Unknown method:", method))
}

# --- Output ---
# REPORTING NOTE (read before quoting any PP4 from this pipeline)
# ---------------------------------------------------------------
# pp4 = 0.0 with estimable=FALSE does NOT mean "the two signals do not
# colocalize".  It means no posterior could be computed -- either SuSiE found
# no overlapping credible sets (method coloc.susie_noCS) or the run errored
# (method suffix _error).  The correct prose is:
#   "no overlapping credible sets under SuSiE fine-mapping; PP4 not estimable"
# NOT "no colocalization (PP4 = 0)".  Only rows with estimable=TRUE carry a
# genuine posterior that can be compared against the pre-registered PP4 >= 0.8
# threshold.
emit <- function(k, v) {
  if (is.null(v) || (length(v) == 1 && is.na(v))) {
    cat(paste0(k, "=NA"), "\n")
  } else {
    cat(paste0(k, "=", format(v, scientific = FALSE, digits = 10)), "\n")
  }
}
cat(paste0("method=", out$method), "\n")
emit("pp4", out$pp4)
cat(paste0("estimable=", if (isTRUE(out$estimable)) "TRUE" else "FALSE"), "\n")
emit("pp0", out$pp0)
emit("pp1", out$pp1)
emit("pp2", out$pp2)
emit("pp3", out$pp3)
