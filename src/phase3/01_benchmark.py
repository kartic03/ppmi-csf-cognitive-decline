"""
Phase 3, task C1: FAIR established-vs-high-dim+AI benchmark (RQ4).

Research question (pre-registered):
    Do established CSF markers + clinical (Ridge) beat the full NULISA 132-plex
    CNS panel + modern AI (ElasticNet / HistGradientBoosting / TabPFN) at
    predicting the continuous EB MoCA-slope outcome?

Pre-registered expectation: NO — high-dim adds nothing meaningful over
established markers (this null is a positive scientific finding, not a failure).

Three arms, all through the SAME nested-CV harness (identical folds, seeds,
and EB-outcome refitting as Phase 2):

  ARM A  "established" : clinical (10) + established CSF markers (5)
                         (NEFL, pSNCA-129, pTau181, Abeta42/40 ratio, ABeta42)
                         Model: Ridge (penalised linear, the Phase 2 RQ1 model)

  ARM B  "high-dim"    : clinical (10) + full NULISA 132-plex CNS panel (131 proteins)
                         Models: ElasticNet, HistGradientBoostingRegressor, TabPFNRegressor

  ARM C  "both"        : ARM A features + full NULISA 131 (replaces NEFL/pSNCA from
                         established since they are already in the NULISA panel) + curated
                         CSF extras (pTau181, ab_ratio, ABeta42)
                         Models: ElasticNet, HistGradientBoostingRegressor, TabPFNRegressor

Analytic set (all arms): subjects who are slope-eligible AND have complete
established CSF (the binding constraint, N=607, the same analytic set as the
phase 2 RQ1 analysis). All arms use IDENTICAL
analytic_ids so fold partitions are guaranteed identical across arms.

Equivalence test:
    Pre-specified margin = 0.02 (a gain of <0.02 R2 is not a meaningful
    predictive improvement). Verdict: equivalent if the upper bound of the
    per-seed difference repeat-range is below margin. Also reported: the
    MINIMUM DETECTABLE DIFFERENCE (spread of the difference repeat-range), so
    a null finding is "we could have detected a gain of MDD and did not."

Output: data/processed/phase3/benchmark.json

Usage:
    # CANONICAL run = the default. On a CUDA box this auto-selects GPU + full
    # TabPFN ensemble and writes the reported benchmark.json:
    ~/bench_gpu2/bin/python src/phase3/01_benchmark.py

    # Force CPU (n_estimators=1) explicitly, e.g. on a CPU-only box:
    TABPFN_DEVICE=cpu ~/bench_env/bin/python src/phase3/01_benchmark.py

Env note: TabPFN device + ensemble size are set by TABPFN_DEVICE / TABPFN_N_ESTIMATORS
(see _make_tabpfn_factory); device defaults to CUDA when available, else CPU. Single-
thread BLAS/OMP pins are applied in-process at module load (HGBM's OpenMP threads
otherwise race -> non-deterministic + can crash); all sklearn grids run n_jobs=1 to
avoid SIGSEGV from nested loky+BLAS.
"""
import json
import os
import sys

# Pin single-thread BLAS/OpenMP BEFORE numpy/torch import so the default run is
# deterministic without relying on the operator exporting these. HistGBM's
# OpenMP threads otherwise race (float add-order) -> the HGBM arms become
# non-deterministic across runs and can intermittently corrupt memory. These
# only take effect if set before the numerical libs load. setdefault lets an
# explicit outer export still win.
for _thr_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thr_var, "1")

# Eagerly import torch (and force its nn submodules to load) in the main thread
# BEFORE any TabPFN fit. TabPFN v3 builds its ensemble during fit and can trigger
# a re-entrant torch-submodule import from a worker context, which surfaces as
# "AttributeError: 'super' object has no attribute 'torch'". Pre-loading here
# avoids that race. Non-fatal if torch is absent (the TabPFN arm falls back to
# Ridge in _make_tabpfn_factory).
try:
    import torch  # noqa: F401
    import torch.nn as _torch_nn  # noqa: F401
    _ = _torch_nn.Linear  # touch the container/linear modules to fully load them
except Exception:
    torch = None

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CURATED = os.path.join(_ROOT, "data", "processed", "phase1", "curated_cut.parquet")
_OUTCOME = os.path.join(_ROOT, "data", "processed", "phase2", "outcome.parquet")
_NULISA = os.path.join(
    _ROOT, "data", "raw", "ALL Proteomic Analysis", "converted",
    "PPMI_Project_282_NULISAseq_CNSDiseasePanel_NPQCounts_20260120.csv",
)
_DATA_OUT_DIR = os.path.join(_ROOT, "data", "processed", "phase3")
_JSON_OUT = os.path.join(_DATA_OUT_DIR, "benchmark.json")

# ---------------------------------------------------------------------------
# Import Phase 2 harness (reuse identical CV discipline)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(_HERE, "..", "phase2"))
from cv import nested_cv, fit_eb_params, eb_slopes, load_moca  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EQUIVALENCE_MARGIN: float = 0.02  # pre-specified meaningful R2 gain threshold

CLINICAL_COLS = [
    "age", "SEX", "EDUCYRS", "moca", "updrs3_score",
    "duration_yrs", "LEDD", "APOE_e4", "gba_status", "MIA_PUTAMEN_BILAT",
]

# Established CSF markers from the Phase 2 RQ1 model (B3)
ESTABLISHED_CSF_CORE = ["NEFL", "pSNCA_129", "pTau181", "ab_ratio", "ABeta42"]

# Hyperparameter grids
RIDGE_ALPHA_GRID = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
ELASTICNET_GRID = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.5, 0.9],
}
# Note: HGBM is run without inner-CV grid search (param_grid=None is passed in the
# arms list below). With 50 outer folds and a grid of 8 combinations x 5 inner
# folds = 2000 HGBM fits, runtime would exceed 30 minutes. Instead HGBM uses
# fixed "out of box" hyperparameters (learning_rate=0.1, max_iter=200, max_depth=4,
# l2_regularization=1.0), which are robust for tabular data with N~480 and p~141.
# This still gives HGBM a genuine shot — it is a well-tuned default, not a
# degenerate fallback.
HGBM_FIXED_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 200,
    "max_depth": 4,
    "l2_regularization": 1.0,
    "random_state": 0,
}


# ---------------------------------------------------------------------------
# Public helper functions (exported for tests)
# ---------------------------------------------------------------------------

def compute_min_detectable_diff(per_seed_diffs):
    """
    Minimum detectable difference: the spread (range width) of the per-seed
    difference repeat-range.

    ⚠ NAMING CAVEAT (2026-08-04). "Minimum detectable difference" is a
    misnomer for what this computes. In the power-analysis literature an MDD is
    a function of sample size and variance, and it IMPROVES (gets smaller) as
    you collect more information. This quantity is the WIDTH of the observed
    per-seed range, so it gets WORSE (larger) as more seeds are added — it
    measures CV/compute jitter, not statistical resolution. Two consequences
    for reporting: (a) do not describe it as a power quantity or compare it to
    a literature MDD; (b) a small value here reflects seed stability, not
    precision. Consider renaming to `per_seed_range_width` in any published
    table. The equivalence verdict itself does not depend on this value; it is
    gated on the range bounds versus the margin.

    Interpretation: "The CV cannot reliably distinguish differences smaller
    than MDD from sampling noise." A null finding combined with a small MDD
    is strong evidence; a large MDD means the resolution is too coarse.
    """
    diffs = [d for d in per_seed_diffs if np.isfinite(d)]
    if len(diffs) >= 2:
        return float(np.max(diffs) - np.min(diffs))
    if len(diffs) == 1:
        return 0.0
    return float("nan")


def build_equivalence_verdict(best_diff, range_lo, range_hi, margin, mdd):
    """
    Equivalence / non-inferiority verdict.

    Parameters
    ----------
    best_diff : float
        Canonical (seed 0) difference: best high-dim R2 - established R2.
    range_lo  : float
        Lower bound of the per-seed difference repeat-range.
    range_hi  : float
        Upper bound of the per-seed difference repeat-range.
    margin    : float
        Pre-specified threshold for a meaningful predictive gain.
    mdd       : float
        Minimum detectable difference (width of the difference repeat-range).

    Returns
    -------
    dict with keys: best_highdim_minus_established, range_lo, range_hi,
                    equivalent, margin, min_detectable_diff, verdict.
    """
    # Three-way verdict (TOST-style), driven by the two ENDS of the per-seed
    # difference range so each claim is gated by the bound that supports it:
    #   equivalent : range_hi < margin  -> high-dim never meaningfully exceeds.
    #   superior   : range_lo >= margin -> EVERY seed's gain clears the margin.
    #   inconclusive: the range straddles the margin (lo < margin <= hi), so a
    #                 "genuinely beats" claim is not supported at every seed.
    # equivalent and superior are mutually exclusive (range_lo <= range_hi).
    equivalent = bool(np.isfinite(range_hi) and range_hi < margin)
    superior = bool(np.isfinite(range_lo) and range_lo >= margin)
    if equivalent:
        status = "equivalent"
    elif superior:
        status = "superior"
    else:
        status = "inconclusive"

    mdd_val = float(mdd) if np.isfinite(float(mdd)) else None

    mdd_str = f"{mdd_val:.4f}" if mdd_val is not None else "nan"
    if status == "equivalent":
        verdict = (
            f"EQUIVALENT: high-dim/AI does not meaningfully exceed established "
            f"(upper range {range_hi:.4f} < margin {margin:.4f}; "
            f"MDD {mdd_str} - we could have detected a gain of {mdd_str} R2 and did not)"
        )
    elif status == "superior":
        verdict = (
            f"NOT EQUIVALENT (high-dim/AI superior): every-seed gain clears the "
            f"margin (lower range {range_lo:.4f} >= margin {margin:.4f}; "
            f"best diff = {best_diff:.4f})"
        )
    else:
        verdict = (
            f"INCONCLUSIVE: the difference range straddles the margin "
            f"(range [{range_lo:.4f}, {range_hi:.4f}] vs margin {margin:.4f}; "
            f"best diff = {best_diff:.4f}) - a genuine gain is not established at every seed"
        )

    return {
        "best_highdim_minus_established": float(best_diff),
        "range_lo": float(range_lo),
        "range_hi": float(range_hi),
        "equivalent": equivalent,
        "superior": superior,
        "status": status,
        "margin": float(margin),
        "min_detectable_diff": mdd_val,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Core benchmark runner (exposed for tests with injected builders/data)
# ---------------------------------------------------------------------------

def run_benchmark_on_data(
    established_builder,
    highdim_builder,
    both_builder,
    moca_long,
    analytic_ids,
    cv_config=None,
    model_overrides=None,
):
    """
    Run the seven-arm benchmark on pre-built feature builders.

    All seven arms receive the SAME analytic_ids and cv_config, guaranteeing
    identical fold partitions and identical EB-outcome computation.

    Parameters
    ----------
    established_builder : callable (subject_ids) -> pd.DataFrame
        ARM A features: clinical + established CSF markers.
    highdim_builder : callable (subject_ids) -> pd.DataFrame
        ARM B features: clinical + full NULISA 131-plex.
    both_builder : callable (subject_ids) -> pd.DataFrame
        ARM C features: clinical + NULISA 131 + curated CSF extras.
    moca_long : pd.DataFrame
        Longitudinal MoCA table (PATNO, EVENT_ID, age_at_visit, moca).
    analytic_ids : array-like
        Pre-computed intersection of subjects with all required features.
        Passed to every arm, guaranteeing identical fold splits.
    cv_config : dict or None
        n_outer (default 5), n_inner (default 5), seeds (default 0..9).
    model_overrides : dict or None
        Inject alternative model factories for testing. Supported key:
        "tabpfn" -> callable () -> sklearn estimator.
        Example: {"tabpfn": lambda: Ridge()} stubs TabPFN with Ridge.

    Returns
    -------
    dict with:
        "result"   : dict (JSON-ready benchmark result)
        "_oof_ids" : dict arm_name -> sorted list[int]
                     Canonical-seed OOF subject IDs for each arm. Used by
                     tests to verify fold-identity across arms.
    """
    model_overrides = model_overrides or {}

    # TabPFN: CPU-only, no hyperparameter tuning
    default_tabpfn = _make_tabpfn_factory()
    tabpfn_factory = model_overrides.get("tabpfn", default_tabpfn)

    # Arms: (name, builder, model_factory, param_grid)
    # TabPFN is hyperparameter-free: param_grid=None
    arms = [
        ("established_ridge",
         established_builder, lambda: Ridge(), RIDGE_ALPHA_GRID),
        # Fairness arm: TabPFN on the SAME 15 established features. Lets us
        # separate the model-class effect (TabPFN vs Ridge on identical inputs)
        # from the feature-set effect (131-plex vs established). Without this,
        # the best-high-dim-vs-established_ridge contrast conflates the two.
        ("established_tabpfn",
         established_builder, tabpfn_factory, None),
        ("highdim_elasticnet",
         highdim_builder, lambda: ElasticNet(max_iter=5000), ELASTICNET_GRID),
        # HGBM: no inner-CV grid (param_grid=None); uses fixed robust defaults
        # defined in HGBM_FIXED_PARAMS. See note near that constant for rationale.
        ("highdim_histgbm",
         highdim_builder,
         lambda: HistGradientBoostingRegressor(**HGBM_FIXED_PARAMS),
         None),
        ("highdim_tabpfn",
         highdim_builder, tabpfn_factory, None),
        ("both_elasticnet",
         both_builder, lambda: ElasticNet(max_iter=5000), ELASTICNET_GRID),
        ("both_histgbm",
         both_builder,
         lambda: HistGradientBoostingRegressor(**HGBM_FIXED_PARAMS),
         None),
        ("both_tabpfn",
         both_builder, tabpfn_factory, None),
    ]

    arm_results = {}
    oof_ids = {}

    for arm_name, builder, model_factory, param_grid in arms:
        print(f"  arm: {arm_name} ...", flush=True)
        res = nested_cv(
            feature_builder=builder,
            moca_long=moca_long,
            model_factory=model_factory,
            param_grid=param_grid,
            cv_config=cv_config,
            analytic_ids=analytic_ids,
        )
        arm_results[arm_name] = {
            "r2": float(res["r2_mean"]),
            "repeat_lo": float(res["r2_repeat_lo"]),
            "repeat_hi": float(res["r2_repeat_hi"]),
            "per_seed_r2": [float(x) for x in res["per_seed_r2"]],
            "n_subjects": int(res["n_subjects"]),
        }
        oof_ids[arm_name] = sorted(res["oof_truth"].index.astype(int).tolist())
        print(
            f"    R2={res['r2_mean']:.4f} "
            f"[{res['r2_repeat_lo']:.4f}, {res['r2_repeat_hi']:.4f}]",
            flush=True,
        )

    # Key contrast: best high-dim vs established_ridge.
    # established_tabpfn is an established-feature arm (same 15 inputs as
    # established_ridge, different model), so it must NOT be eligible as the
    # "best high-dim" arm — exclude it alongside established_ridge.
    established_arms = {"established_ridge", "established_tabpfn"}
    est = arm_results["established_ridge"]
    highdim_names = [k for k in arm_results if k not in established_arms]
    best_arm = max(highdim_names, key=lambda k: arm_results[k]["r2"])
    best = arm_results[best_arm]

    canonical_diff = best["r2"] - est["r2"]

    # Per-seed differences: arm with best canonical R2 vs established
    paired = [
        (b, e)
        for b, e in zip(best["per_seed_r2"], est["per_seed_r2"])
        if np.isfinite(b) and np.isfinite(e)
    ]
    per_seed_diffs = [b - e for b, e in paired]

    if per_seed_diffs:
        range_lo = float(np.min(per_seed_diffs))
        range_hi = float(np.max(per_seed_diffs))
    else:
        range_lo = range_hi = float("nan")

    mdd = compute_min_detectable_diff(per_seed_diffs)
    verdict_dict = build_equivalence_verdict(
        canonical_diff, range_lo, range_hi, EQUIVALENCE_MARGIN, mdd
    )

    # Fairness decomposition: hold the model class fixed (TabPFN everywhere) to
    # separate "TabPFN beats Ridge" (model class) from "131-plex beats the
    # established panel" (feature set). Only computed when established_tabpfn ran.
    fairness = None
    if "established_tabpfn" in arm_results:
        est_tab = arm_results["established_tabpfn"]["r2"]
        fairness = {
            "established_tabpfn_r2": est_tab,
            # TabPFN vs Ridge on the SAME 15 established features:
            "model_class_effect": est_tab - est["r2"],
            # Adding the 131-plex while holding the model = TabPFN:
            "featureset_effect_highdim": arm_results["highdim_tabpfn"]["r2"] - est_tab,
            "featureset_effect_both": arm_results["both_tabpfn"]["r2"] - est_tab,
        }

    result = {
        "margin": EQUIVALENCE_MARGIN,
        "min_detectable_diff": float(mdd) if np.isfinite(mdd) else None,
        "best_highdim_arm": best_arm,
        "established_r2": est["r2"],
        "arms": {
            arm: {
                "r2": v["r2"],
                "repeat_lo": v["repeat_lo"],
                "repeat_hi": v["repeat_hi"],
                "n_subjects": v["n_subjects"],
            }
            for arm, v in arm_results.items()
        },
        "key_contrast": verdict_dict,
        "fairness_contrast": fairness,
    }

    return {"result": result, "_oof_ids": oof_ids}


# ---------------------------------------------------------------------------
# TabPFN factory (separated so it can be tested without importing tabpfn)
# ---------------------------------------------------------------------------

def _make_tabpfn_factory():
    """
    Return a factory for TabPFNRegressor (hyperparameter-free), configured by
    two environment variables so the same committed code path serves both the
    safe default and the canonical run:

      TABPFN_DEVICE        "cuda" or "cpu". Default: "cuda" when a CUDA device
                           is available, else "cpu".
      TABPFN_N_ESTIMATORS  integer override for the ensemble size. If unset,
                           defaults to 1 on CPU and TabPFN's full default
                           ensemble (~32 passes) on CUDA.

    The default IS the canonical run: on a CUDA box it auto-selects GPU + full
    ensemble (the more stable estimate, what benchmark.json reports); on a
    CPU-only box it falls back to n_estimators=1 (~20s/fit vs ~128s for the full
    ensemble, a single deterministic in-context pass). Thread pins are applied
    in-process at module load, so the canonical run is just:
        python src/phase3/01_benchmark.py   (in a CUDA-capable env)
    Set TABPFN_DEVICE=cpu to force CPU explicitly.
    """
    default_device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    device = os.environ.get("TABPFN_DEVICE", default_device).strip().lower() or default_device
    n_est_env = os.environ.get("TABPFN_N_ESTIMATORS", "").strip()

    def factory():
        # Import INSIDE the factory (2026-08-04). The failure must surface when a
        # TabPFN model is actually constructed, not when the factory is built --
        # otherwise callers that legitimately never use this arm (e.g. tests that
        # pass model_overrides={"tabpfn": stub}) would be broken by an env that
        # merely lacks TabPFN.
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as exc:
            # FAIL LOUDLY. This previously fell back to Ridge(alpha=1.0), which
            # meant that in an environment without TabPFN every arm named
            # "*_tabpfn" would silently be a Ridge fit, written to
            # benchmark.json under the TabPFN label and reported as a
            # foundation-model result. A missing dependency must never be able
            # to masquerade as a scientific finding.
            raise ImportError(
                "TabPFN is required for the *_tabpfn arms of this benchmark and "
                "is not importable. Refusing to silently substitute Ridge, which "
                "would mislabel Ridge results as TabPFN results in "
                "benchmark.json. Install TabPFN in the GPU env (~/bench_gpu2) "
                "and set TABPFN_TOKEN, or override this arm explicitly."
            ) from exc

        if True:
            kwargs = dict(
                device=device,
                random_state=0,
                ignore_pretraining_limits=True,
                show_progress_bar=False,
            )
            if n_est_env:
                kwargs["n_estimators"] = int(n_est_env)
                kwargs["auto_scale_n_estimators"] = False
            elif device == "cpu":
                # CPU feasibility: single deterministic pass.
                kwargs["n_estimators"] = 1
                kwargs["auto_scale_n_estimators"] = False
            # else (cuda, no override): TabPFN's full default ensemble.
            return TabPFNRegressor(**kwargs)

    return factory


# ---------------------------------------------------------------------------
# Data loading helpers (real PPMI data)
# ---------------------------------------------------------------------------

def _load_curated_bl(curated_path, slope_ids):
    """Load baseline PD subjects from curated_cut.parquet."""
    curated = pd.read_parquet(curated_path)
    bl = curated[
        (curated["COHORT"] == 1) & (curated["EVENT_ID"] == "BL")
    ].copy()
    bl["gba_status"] = bl["subgroup"].str.contains("GBA", na=False).astype(float)
    bl = bl.set_index("PATNO")
    return bl[bl.index.isin(slope_ids)]


def _load_full_nulisa_wide(nulisa_path, slope_ids):
    """
    Load all 131 NULISA CNS 132-plex targets for slope-eligible subjects.

    Pivots to a wide matrix (PATNO x Target) with log1p-transformed NPQ counts.
    Missing proteins (QC failures for a specific subject) remain as NaN
    and are handled by the in-fold median imputer.
    """
    nulisa = pd.read_csv(nulisa_path)
    samples = nulisa[
        (nulisa["SampleType"] == "Sample") & (nulisa["SampleQC"] == "passed")
    ].dropna(subset=["PATNO"]).copy()
    samples["PATNO"] = samples["PATNO"].astype(int)
    samples = samples[samples["PATNO"].isin(slope_ids)]
    # Earliest clinical event per PATNO per Target
    samples_sorted = samples.sort_values("CLINICAL_EVENT")
    earliest = samples_sorted.groupby(["PATNO", "Target"], as_index=False).first()
    earliest["NPQ"] = np.log1p(earliest["NPQ"].clip(lower=0.0))
    wide = earliest.pivot(index="PATNO", columns="Target", values="NPQ")
    wide.columns.name = None
    return wide


def _load_established_csf(curated_path, nulisa_path, slope_ids):
    """
    Build the established CSF feature DataFrame (ARM A augmentation over clinical).

    Columns:
      NEFL       : log1p(NULISA NPQ)
      pSNCA_129  : log1p(NULISA NPQ for pSNCA-129)
      pTau181    : log1p(IU_pTau181_CSF from curated BL)
      ab_ratio   : IU_ABeta42_CSF / IU_ABeta40_CSF
      ABeta42    : log1p(IU_ABeta42_CSF)

    Returns pd.DataFrame indexed by PATNO. Only subjects with ALL five
    markers are included (CSF-complete, N=607).
    """
    # NULISA NEFL + pSNCA-129
    nulisa = pd.read_csv(nulisa_path)
    samples = nulisa[
        (nulisa["SampleType"] == "Sample") & (nulisa["SampleQC"] == "passed")
    ].dropna(subset=["PATNO"]).copy()
    samples["PATNO"] = samples["PATNO"].astype(int)
    samples = samples[
        (samples["PATNO"].isin(slope_ids)) &
        (samples["Target"].isin(["NEFL", "pSNCA-129"]))
    ]
    samples_sorted = samples.sort_values("CLINICAL_EVENT")
    earliest = samples_sorted.groupby(["PATNO", "Target"], as_index=False).first()
    earliest["NPQ"] = np.log1p(earliest["NPQ"].clip(lower=0.0))
    nulisa_wide = earliest.pivot(index="PATNO", columns="Target", values="NPQ")
    nulisa_wide.columns.name = None
    if "pSNCA-129" in nulisa_wide.columns:
        nulisa_wide = nulisa_wide.rename(columns={"pSNCA-129": "pSNCA_129"})

    # Curated BL CSF markers
    curated = pd.read_parquet(curated_path)
    bl = curated[
        (curated["COHORT"] == 1) & (curated["EVENT_ID"] == "BL")
    ].set_index("PATNO")
    bl = bl[bl.index.isin(slope_ids)]

    bl_csf = bl[["IU_pTau181_CSF", "IU_ABeta42_CSF", "IU_ABeta40_CSF"]].copy()
    mask = (bl_csf["IU_ABeta42_CSF"] > 0) & (bl_csf["IU_ABeta40_CSF"] > 0)
    bl_csf["ab_ratio"] = np.where(
        mask,
        bl_csf["IU_ABeta42_CSF"] / bl_csf["IU_ABeta40_CSF"],
        np.nan,
    )
    bl_csf["pTau181"] = np.log1p(bl_csf["IU_pTau181_CSF"].clip(lower=0.0))
    bl_csf["ABeta42"] = np.log1p(bl_csf["IU_ABeta42_CSF"].clip(lower=0.0))
    bl_csf = bl_csf.drop(columns=["IU_pTau181_CSF", "IU_ABeta42_CSF", "IU_ABeta40_CSF"])

    # Join and require all five CSF markers
    csf_df = nulisa_wide.join(bl_csf, how="inner")
    csf_core = ["NEFL", "pSNCA_129", "pTau181", "ab_ratio", "ABeta42"]
    csf_df = csf_df[csf_core].dropna()
    return csf_df


# ---------------------------------------------------------------------------
# Feature builder factories (real data)
# ---------------------------------------------------------------------------

def _make_established_builder(bl_clinical, csf_df):
    """
    ARM A: clinical (10) + established CSF (5) = 15 features.

    Both DataFrames are pre-loaded at the outer scope (no re-reading per fold).
    Only subjects in the intersection are returned.
    """
    def builder(subject_ids):
        ids = sorted(int(i) for i in subject_ids)
        clin = bl_clinical.loc[bl_clinical.index.intersection(ids), CLINICAL_COLS]
        csf_sub = csf_df.loc[csf_df.index.intersection(clin.index)]
        return clin.join(csf_sub, how="inner")
    return builder


def _make_highdim_builder(bl_clinical, nulisa_wide):
    """
    ARM B: clinical (10) + full NULISA 131-plex = 141 features.

    Missing proteins for a given subject appear as NaN; the in-fold
    SimpleImputer fills them with the training-fold median.
    """
    def builder(subject_ids):
        ids = sorted(int(i) for i in subject_ids)
        clin = bl_clinical.loc[bl_clinical.index.intersection(ids), CLINICAL_COLS]
        # reindex: keep all clin subjects, NaN for missing NULISA proteins
        nulisa_sub = nulisa_wide.reindex(clin.index)
        return clin.join(nulisa_sub, how="left")
    return builder


def _make_both_builder(bl_clinical, nulisa_wide, bl_curated_csf):
    """
    ARM C: clinical (10) + NULISA 131 + curated CSF extras (pTau181, ab_ratio,
    ABeta42) = 144 features.

    NEFL and pSNCA-129 are already in the NULISA wide matrix, so no duplication.
    Curated CSF extras add pTau181, ab_ratio, and ABeta42.
    """
    csf_extras = ["pTau181", "ab_ratio", "ABeta42"]

    def builder(subject_ids):
        ids = sorted(int(i) for i in subject_ids)
        clin = bl_clinical.loc[bl_clinical.index.intersection(ids), CLINICAL_COLS]
        nulisa_sub = nulisa_wide.reindex(clin.index)
        extras = bl_curated_csf.loc[bl_curated_csf.index.intersection(clin.index), csf_extras]
        X = clin.join(nulisa_sub, how="left").join(extras, how="left")
        return X
    return builder


# ---------------------------------------------------------------------------
# Main orchestrator (real PPMI data)
# ---------------------------------------------------------------------------

def run_benchmark(cv_config=None, curated_path=None, nulisa_path=None,
                  outcome_path=None, moca_long=None):
    """
    Run the full Phase 3 RQ4 benchmark on real PPMI data.

    Parameters
    ----------
    cv_config : dict or None
        n_outer (5), n_inner (5), seeds (0..9). Default: same as Phase 2.
    curated_path, nulisa_path, outcome_path : str or None
        Override default data paths.
    moca_long : pd.DataFrame or None
        Pre-loaded longitudinal MoCA table. If None, loaded via load_moca().

    Returns
    -------
    dict : full benchmark result (JSON-ready).
    """
    curated_path = curated_path or _CURATED
    nulisa_path = nulisa_path or _NULISA
    outcome_path = outcome_path or _OUTCOME

    if cv_config is None:
        # 5 seeds (matching the nested_cv default) rather than the 10 used in
        # Phase 2 increment tests. This keeps the two TabPFN arms within ~16 min
        # on CPU (25 outer folds × 20 s/fit × 2 arms) while still providing a
        # meaningful repeat-range for the equivalence test. The 5-seed repeat-
        # range is wider than a 10-seed range, which is conservative (makes it
        # harder to claim equivalence, not easier).
        cv_config = {
            "n_outer": 5,
            "n_inner": 5,
            "seeds": list(range(5)),
        }

    if moca_long is None:
        moca_long = load_moca(curated_path)

    # Slope-eligible subjects
    outcome_df = pd.read_parquet(outcome_path)
    slope_ids = set(outcome_df["subject_id"].astype(int))
    print(f"Slope-eligible subjects: {len(slope_ids)}")

    # Load data frames (once, outside the fold loop)
    print("Loading clinical block ...")
    bl_clinical = _load_curated_bl(curated_path, slope_ids)

    print("Loading established CSF block ...")
    csf_df = _load_established_csf(curated_path, nulisa_path, slope_ids)

    print("Loading full NULISA 131-plex ...")
    nulisa_wide = _load_full_nulisa_wide(nulisa_path, slope_ids)
    print(f"  NULISA wide: {nulisa_wide.shape[0]} subjects x {nulisa_wide.shape[1]} proteins")

    # Curated CSF extras for ARM C
    curated = pd.read_parquet(curated_path)
    bl_raw = curated[
        (curated["COHORT"] == 1) & (curated["EVENT_ID"] == "BL")
    ].set_index("PATNO")
    bl_raw = bl_raw[bl_raw.index.isin(slope_ids)]
    bl_csf_extras = bl_raw[["IU_pTau181_CSF", "IU_ABeta42_CSF", "IU_ABeta40_CSF"]].copy()
    mask = (bl_csf_extras["IU_ABeta42_CSF"] > 0) & (bl_csf_extras["IU_ABeta40_CSF"] > 0)
    bl_csf_extras["ab_ratio"] = np.where(
        mask, bl_csf_extras["IU_ABeta42_CSF"] / bl_csf_extras["IU_ABeta40_CSF"], np.nan
    )
    bl_csf_extras["pTau181"] = np.log1p(bl_csf_extras["IU_pTau181_CSF"].clip(lower=0.0))
    bl_csf_extras["ABeta42"] = np.log1p(bl_csf_extras["IU_ABeta42_CSF"].clip(lower=0.0))
    bl_csf_extras = bl_csf_extras.drop(
        columns=["IU_pTau181_CSF", "IU_ABeta42_CSF", "IU_ABeta40_CSF"]
    )

    # Analytic subject set: intersection across all three arms
    # Binding constraint: established CSF (requires NEFL, pSNCA-129, pTau181,
    # ab_ratio, ABeta42) — this is the same as the B3/RQ1 CSF-complete set.
    clin_ids = set(bl_clinical.index)
    csf_ids = set(csf_df.index)
    analytic_ids = np.array(sorted(
        set(int(i) for i in slope_ids) & clin_ids & csf_ids
    ))
    print(f"Analytic N (all arms, intersection): {len(analytic_ids)}")

    # Build feature builders
    established_builder = _make_established_builder(bl_clinical, csf_df)
    highdim_builder = _make_highdim_builder(bl_clinical, nulisa_wide)
    both_builder = _make_both_builder(bl_clinical, nulisa_wide, bl_csf_extras)

    # Run benchmark
    print("\n=== Running seven-arm benchmark ===")
    output = run_benchmark_on_data(
        established_builder=established_builder,
        highdim_builder=highdim_builder,
        both_builder=both_builder,
        moca_long=moca_long,
        analytic_ids=analytic_ids,
        cv_config=cv_config,
    )

    result = output["result"]

    # Record the CV configuration actually used, so the seed count in any
    # write-up can be checked against the run instead of being asserted.
    result["cv_config"] = cv_config

    # Print summary
    print("\n=== RESULTS ===")
    print(f"Analytic N: {len(analytic_ids)}")
    print(f"Equivalence margin: {result['margin']}")
    print(f"Min detectable diff: {result['min_detectable_diff']}")
    print()
    for arm, v in result["arms"].items():
        print(
            f"  {arm:<26} R2={v['r2']:+.4f}  "
            f"[{v['repeat_lo']:+.4f}, {v['repeat_hi']:+.4f}]"
        )
    print()
    c = result["key_contrast"]
    print(f"Best high-dim arm: {result['best_highdim_arm']}")
    print(
        f"Key contrast (high-dim - established): "
        f"{c['best_highdim_minus_established']:+.4f}  "
        f"[{c['range_lo']:+.4f}, {c['range_hi']:+.4f}]"
    )
    print(f"Equivalent: {c['equivalent']}")
    print(f"Verdict: {c['verdict']}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _provenance():
    """Environment fingerprint recorded alongside every result.

    The originally committed benchmark.json carried NO version, seed, device or
    git information, so when the GPU env (~/bench_gpu2) was later lost there was
    no way to tell which TabPFN produced it -- making the result unreproducible
    in principle, not just in practice.  TabPFN results are model-version
    dependent, so this block is required for any RQ4 number to be checkable.
    """
    import platform
    import subprocess

    def _ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    prov = {
        "python": platform.python_version(),
        "numpy": _ver("numpy"),
        "pandas": _ver("pandas"),
        "sklearn": _ver("sklearn"),
        "scipy": _ver("scipy"),
        "tabpfn": _ver("tabpfn"),
        "tabdpt": _ver("tabdpt"),
        "torch": _ver("torch"),
        # Filled from the result's own cv_config below, so the recorded seed
        # count can never drift from the run (METHODS_AND_RESULTS.md claimed
        # 10 seeds for a 5-seed run precisely because nothing recorded it).
        "seeds": None,
        "thread_env": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS", "TABPFN_DEVICE", "TABPFN_N_ESTIMATORS")
        },
    }

    try:
        import torch
        prov["cuda_available"] = torch.cuda.is_available()
        prov["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        prov["cuda_available"] = None

    try:
        prov["git_sha"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True,
            text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        prov["git_sha"] = None

    return prov


def main():
    result = run_benchmark()

    # Attach provenance so this run is checkable later.  Do not remove.
    prov = _provenance()
    cv = result.get("cv_config") if isinstance(result, dict) else None
    if isinstance(cv, dict):
        prov["seeds"] = cv.get("seeds")
        prov["n_outer"] = cv.get("n_outer")
        prov["n_inner"] = cv.get("n_inner")
    prov["n_seeds"] = len(prov["seeds"]) if prov.get("seeds") else None
    result["provenance"] = prov

    os.makedirs(_DATA_OUT_DIR, exist_ok=True)
    with open(_JSON_OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {_JSON_OUT}")
    print("Provenance:", json.dumps(result["provenance"], indent=2))
    return result


if __name__ == "__main__":
    main()
