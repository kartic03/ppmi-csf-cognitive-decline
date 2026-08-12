"""
Phase 2, step 02: clinical baseline model (task B2).

Predictors (Block 1 from B0 report):
  age, SEX, EDUCYRS, moca (at BL), updrs3_score, duration_yrs, LEDD,
  APOE_e4, gba_status (binarised from subgroup), MIA_PUTAMEN_BILAT

Source: data/processed/phase1/curated_cut.parquet (BL rows, COHORT==1)
        data/processed/phase2/outcome.parquet (slope-eligible subject IDs)

Missingness strategy:
  Median imputation within each outer fold (train only, then transform test).
  APOE_e4: 8/816 missing (~1%), imputed.
  updrs3_score: 53/816 missing (~6.5%), imputed.
  MIA_PUTAMEN_BILAT: 34/816 missing (~4.2%), imputed.

Models:
  Primary   : Ridge regression with alpha tuned by inner CV (5-fold).
  Comparator: HistGradientBoostingRegressor with default params (pre-specified).

Output: data/processed/phase2/baseline_cv.json
  Per-seed and per-fold R2 for both models; N, folds, repeats.

Usage:
    pixi run python src/phase2/02_baseline.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

# cv.py is in the same directory; insert it into sys.path so downstream tasks
# (B3, B4) can use the same one-liner without an importlib dance.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv import nested_cv, load_moca  # noqa: E402  (after sys.path insert)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_IN = os.path.join(_ROOT, "data", "processed", "phase1", "curated_cut.parquet")
_OUTCOME = os.path.join(_ROOT, "data", "processed", "phase2", "outcome.parquet")
_DATA_OUT_DIR = os.path.join(_ROOT, "data", "processed", "phase2")
_JSON_OUT = os.path.join(_DATA_OUT_DIR, "baseline_cv.json")

# ---------------------------------------------------------------------------
# Clinical predictor columns (Block 1 from B0 report)
# ---------------------------------------------------------------------------
CLINICAL_COLS = [
    "age",               # age at BL visit (continuous, fully observed)
    "SEX",               # 0=female, 1=male (fully observed)
    "EDUCYRS",           # years of education (1 missing, imputed)
    "moca",              # baseline MoCA score (1 missing, imputed)
    "updrs3_score",      # MDS-UPDRS-III motor score (53 missing, imputed)
    "duration_yrs",      # disease duration at BL (fully observed)
    "LEDD",              # levodopa equivalent daily dose (fully observed)
    "APOE_e4",           # count of e4 alleles (0/1/2); 8 missing, imputed
    "gba_status",        # GBA carrier binary (derived from subgroup; complete)
    "MIA_PUTAMEN_BILAT", # DaTSCAN bilateral putamen SBR (34 missing, imputed)
]

# Ridge alpha search grid
RIDGE_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# HGBM: pre-specified comparator (default hyperparameters)
# max_iter=200, max_leaf_nodes=31, learning_rate=0.1 (sklearn defaults)
HGBM_MAX_ITER = 200
HGBM_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def make_clinical_feature_builder(parquet_path=None, outcome_path=None):
    """
    Returns a feature builder closure for the clinical block.

    The builder restricts output to subjects that are:
      - slope-eligible (listed in outcome.parquet)
      - have a BL record in curated_cut.parquet (COHORT==1, EVENT_ID=='BL')

    The returned DataFrame is indexed by PATNO and has exactly the columns
    listed in CLINICAL_COLS. Missing values are left as NaN; imputation
    happens inside each CV fold (train only) in the harness.

    Parameters
    ----------
    parquet_path : str or None
        Path to curated_cut.parquet. Defaults to the project-standard path.
    outcome_path : str or None
        Path to outcome.parquet (slope-eligible subject IDs). Defaults to
        the project-standard path.

    Returns
    -------
    callable : (subject_ids) -> pd.DataFrame indexed by PATNO
    """
    parquet_path = parquet_path or _DATA_IN
    outcome_path = outcome_path or _OUTCOME

    cut = pd.read_parquet(parquet_path)
    bl = cut[(cut["COHORT"] == 1) & (cut["EVENT_ID"] == "BL")].copy()
    # Binarise GBA status from the subgroup column
    bl["gba_status"] = bl["subgroup"].str.contains("GBA", na=False).astype(float)
    bl = bl.set_index("PATNO")

    # Slope-eligible subject IDs (from B1 outcome)
    slope_ids = set(pd.read_parquet(outcome_path)["subject_id"].astype(int))

    def builder(subject_ids):
        ids = set(int(i) for i in np.asarray(list(subject_ids)))
        valid = ids & slope_ids & set(bl.index)
        return bl.loc[sorted(valid), CLINICAL_COLS].copy()

    return builder


# ---------------------------------------------------------------------------
# Main run function (callable from tests and __main__)
# ---------------------------------------------------------------------------

def run_baseline(cv_config=None, parquet_path=None, outcome_path=None):
    """
    Run the clinical baseline nested CV for ridge and HGBM.

    Parameters
    ----------
    cv_config : dict or None
        Passed to nested_cv. If None uses the production defaults:
        n_outer=5, n_inner=5, seeds=[0,1,2,3,4,5,6,7,8,9] (10 repeats).
    parquet_path : str or None
        curated_cut.parquet path (defaults to project standard).
    outcome_path : str or None
        outcome.parquet path (defaults to project standard).

    Returns
    -------
    dict : combined results for ridge and HGBM plus metadata.
    """
    # nested_cv and load_moca are imported at module level from cv.py.
    # cv.py internally handles loading 01_outcome.py (digit-prefixed).
    moca = load_moca(parquet_path)

    # Feature builder
    builder = make_clinical_feature_builder(parquet_path, outcome_path)

    # Determine analytic IDs upfront (avoids re-loading inside harness)
    outcome_df = pd.read_parquet(outcome_path or _OUTCOME)
    all_slope_ids = outcome_df["subject_id"].astype(int).tolist()
    X_full = builder(all_slope_ids)
    analytic_ids = X_full.index.tolist()

    print(f"Analytic N (clinical block, slope-eligible with BL data): {len(analytic_ids)}")

    if cv_config is None:
        cv_config = {
            "n_outer": 5,
            "n_inner": 5,
            "seeds": list(range(10)),  # 10 repeats for stable CI
        }

    print(f"CV config: {cv_config}")

    # --- Ridge ---
    print("\nRunning Ridge nested CV...")
    ridge_result = nested_cv(
        feature_builder=builder,
        moca_long=moca,
        model_factory=lambda: Ridge(),
        param_grid={"alpha": RIDGE_ALPHA_GRID},
        scoring="r2",
        cv_config=cv_config,
        analytic_ids=analytic_ids,
    )
    print(
        f"  Ridge: pooled OOF R2 = {ridge_result['r2_mean']:.4f} "
        f"repeat-range [{ridge_result['r2_repeat_lo']:.4f}, {ridge_result['r2_repeat_hi']:.4f}]"
    )
    print(f"  Per-fold R2: {[f'{r:.3f}' for r in ridge_result['per_fold_r2']]}")
    print(f"  Fold-mean R2 (reference): {ridge_result['r2_fold_mean']:.4f}")

    # --- HGBM (pre-specified comparator, default params) ---
    print("\nRunning HGBM nested CV...")
    hgbm_result = nested_cv(
        feature_builder=builder,
        moca_long=moca,
        model_factory=lambda: HistGradientBoostingRegressor(
            max_iter=HGBM_MAX_ITER,
            random_state=HGBM_RANDOM_STATE,
        ),
        param_grid=None,  # pre-specified: no inner tuning
        scoring="r2",
        cv_config=cv_config,
        analytic_ids=analytic_ids,
    )
    print(
        f"  HGBM : pooled OOF R2 = {hgbm_result['r2_mean']:.4f} "
        f"repeat-range [{hgbm_result['r2_repeat_lo']:.4f}, {hgbm_result['r2_repeat_hi']:.4f}]"
    )
    print(f"  Per-fold R2: {[f'{r:.3f}' for r in hgbm_result['per_fold_r2']]}")
    print(f"  Fold-mean R2 (reference): {hgbm_result['r2_fold_mean']:.4f}")

    combined = {
        "ridge": {
            "r2_mean": ridge_result["r2_mean"],
            "r2_fold_mean": ridge_result["r2_fold_mean"],
            "r2_repeat_lo": ridge_result["r2_repeat_lo"],
            "r2_repeat_hi": ridge_result["r2_repeat_hi"],
            "per_fold_r2": ridge_result["per_fold_r2"],
            "per_seed_r2": ridge_result["per_seed_r2"],
            "n_repeats": ridge_result["n_repeats"],
        },
        "hgbm": {
            "r2_mean": hgbm_result["r2_mean"],
            "r2_fold_mean": hgbm_result["r2_fold_mean"],
            "r2_repeat_lo": hgbm_result["r2_repeat_lo"],
            "r2_repeat_hi": hgbm_result["r2_repeat_hi"],
            "per_fold_r2": hgbm_result["per_fold_r2"],
            "per_seed_r2": hgbm_result["per_seed_r2"],
            "n_repeats": hgbm_result["n_repeats"],
        },
        "n_subjects": len(analytic_ids),
        "n_folds": cv_config.get("n_outer", 5),
        "n_repeats": len(cv_config.get("seeds", list(range(10)))),
    }
    return combined


# ---------------------------------------------------------------------------
# Entry point: production run, writes JSON
# ---------------------------------------------------------------------------

def main():
    result = run_baseline()

    os.makedirs(_DATA_OUT_DIR, exist_ok=True)
    with open(_JSON_OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {_JSON_OUT}")

    print("\n=== CLINICAL BASELINE SUMMARY ===")
    for model in ("ridge", "hgbm"):
        r = result[model]
        print(
            f"  {model.upper():5s}: pooled OOF R2 = {r['r2_mean']:.4f}  "
            f"repeat-range (N={r['n_repeats']}) [{r['r2_repeat_lo']:.4f}, {r['r2_repeat_hi']:.4f}]"
        )
    print(
        f"  N = {result['n_subjects']}, "
        f"outer folds = {result['n_folds']}, "
        f"repeats = {result['n_repeats']}"
    )

    print("\nPre-committed kill-gate check:")
    ridge_r2 = result["ridge"]["r2_mean"]
    hgbm_r2 = result["hgbm"]["r2_mean"]
    print(
        f"  Ridge ({ridge_r2:.4f}) >= HGBM ({hgbm_r2:.4f}): "
        f"{'PASS (linear >= nonlinear as expected)' if ridge_r2 >= hgbm_r2 else 'NOTE: nonlinear leads'}"
    )
    return result


if __name__ == "__main__":
    main()
