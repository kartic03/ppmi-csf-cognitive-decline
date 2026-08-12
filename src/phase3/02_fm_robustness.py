"""
Foundation-model robustness + calibration check (RQ4 follow-up, EXPLORATORY).

Two go/no-go questions before committing to a "foundation models unlock
high-dimensional proteomic signal" positive paper, both answered on the PPMI
data already in hand (NO external cohort — that is B5/PDBP, still blocked):

  (a) Does the proteome feature-set gain (both_FM - established_FM) hold WITH
      MAINTAINED CALIBRATION of the predicted MoCA slope? A gain that comes with
      a wrecked calibration slope is not a usable clinical model.
  (b) Does the gain REPLICATE with a SECOND foundation model (TabDPT), i.e. is it
      a property of in-context tabular foundation models, not a TabPFN artifact?

Reuses the merged, leak-safe `nested_cv` harness (in-fold median-impute + scale,
train-only preprocessing, identical outer folds across arms) and the benchmark's
feature builders, so the leak-safety guarantee carries over unchanged. Runs on
GPU by default (TABPFN_DEVICE / torch auto-select cuda).

Output: data/processed/phase3/fm_robustness.json
"""
import importlib.util
import json
import os
import sys

# Single-thread BLAS/OMP before numpy/torch (HGBM-style OpenMP races + determinism).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

try:
    import torch  # noqa: F401
    import torch.nn as _torch_nn  # noqa: F401
    _ = _torch_nn.Linear
except Exception:
    torch = None

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "phase2"))
from cv import nested_cv, load_moca  # noqa: E402

# Import the merged benchmark module (numeric filename) to reuse its loaders.
_spec = importlib.util.spec_from_file_location(
    "bench_mod", os.path.join(_HERE, "01_benchmark.py")
)
bm = importlib.util.module_from_spec(_spec)
sys.modules["bench_mod"] = bm
_spec.loader.exec_module(bm)

DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"


# ---------------------------------------------------------------------------
# Foundation-model factories
# ---------------------------------------------------------------------------

def tabpfn_factory():
    """Reuse the reviewed TabPFN factory (auto-selects cuda + full ensemble)."""
    return bm._make_tabpfn_factory()


class TabDPTSklearn(BaseEstimator, RegressorMixin):
    """sklearn-style wrapper so nested_cv's fit(X,y)/predict(X) interface works.

    TabDPTRegressor.predict takes inference-time knobs (n_ensembles, context_size)
    that have no sklearn analogue; we fix them and expose a plain predict(X).
    """

    def __init__(self, device=None, n_ensembles=8, context_size=1024, seed=0):
        self.device = device
        self.n_ensembles = n_ensembles
        self.context_size = context_size
        self.seed = seed

    def fit(self, X, y):
        from tabdpt import TabDPTRegressor
        # faiss (TabDPT's context-retrieval backend) spawns its own OpenMP threads
        # that ignore the BLAS pins and race with torch -> intermittent SIGSEGV,
        # especially on the wide (141-feature) arm. Force faiss single-threaded.
        try:
            import faiss
            faiss.omp_set_num_threads(1)
        except Exception:
            pass
        dev = self.device or DEVICE
        # compile=False: torch.compile/inductor autotune is fragile here ("not
        # enough SMs") and adds per-fit overhead with no accuracy benefit.
        self.model_ = TabDPTRegressor(device=dev, compile=False, verbose=False)
        self.model_.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X):
        Xa = np.asarray(X, dtype=np.float64)
        # predict signature verified by smoke test; fall back if kwargs differ.
        try:
            return self.model_.predict(
                Xa, n_ensembles=self.n_ensembles,
                context_size=self.context_size, seed=self.seed,
            )
        except TypeError:
            return self.model_.predict(Xa)


def tabdpt_factory():
    def f():
        return TabDPTSklearn(device=DEVICE)
    return f


# ---------------------------------------------------------------------------
# Calibration of the predicted MoCA slope (regression calibration)
# ---------------------------------------------------------------------------

def calibration(truth, pred):
    """OLS of observed ~ predicted. Ideal: slope=1, intercept=0, CITL=0.

    calibration_slope < 1  -> predictions over-dispersed (over-confident spread).
    calibration_in_the_large (CITL) = mean(pred) - mean(truth).
    """
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(truth) & np.isfinite(pred)
    truth, pred = truth[m], pred[m]
    if len(truth) < 3 or float(np.std(pred)) < 1e-12:
        return {"slope": float("nan"), "intercept": float("nan"),
                "citl": float("nan"), "n": int(len(truth))}
    slope, intercept = np.polyfit(pred, truth, 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "citl": float(np.mean(pred) - np.mean(truth)), "n": int(len(truth))}


# ---------------------------------------------------------------------------
# Data (replicates run_benchmark's loading via the merged helpers)
# ---------------------------------------------------------------------------

def load_data():
    curated, nulisa, outcome = bm._CURATED, bm._NULISA, bm._OUTCOME
    moca = load_moca(curated)
    odf = pd.read_parquet(outcome)
    slope_ids = set(odf["subject_id"].astype(int))

    bl_clinical = bm._load_curated_bl(curated, slope_ids)
    csf_df = bm._load_established_csf(curated, nulisa, slope_ids)
    nulisa_wide = bm._load_full_nulisa_wide(nulisa, slope_ids)

    cur = pd.read_parquet(curated)
    bl_raw = cur[(cur["COHORT"] == 1) & (cur["EVENT_ID"] == "BL")].set_index("PATNO")
    bl_raw = bl_raw[bl_raw.index.isin(slope_ids)]
    ex = bl_raw[["IU_pTau181_CSF", "IU_ABeta42_CSF", "IU_ABeta40_CSF"]].copy()
    mask = (ex["IU_ABeta42_CSF"] > 0) & (ex["IU_ABeta40_CSF"] > 0)
    ex["ab_ratio"] = np.where(mask, ex["IU_ABeta42_CSF"] / ex["IU_ABeta40_CSF"], np.nan)
    ex["pTau181"] = np.log1p(ex["IU_pTau181_CSF"].clip(lower=0.0))
    ex["ABeta42"] = np.log1p(ex["IU_ABeta42_CSF"].clip(lower=0.0))
    ex = ex.drop(columns=["IU_pTau181_CSF", "IU_ABeta42_CSF", "IU_ABeta40_CSF"])

    established_builder = bm._make_established_builder(bl_clinical, csf_df)
    both_builder = bm._make_both_builder(bl_clinical, nulisa_wide, ex)

    analytic_ids = np.array(sorted(
        set(int(i) for i in slope_ids) & set(bl_clinical.index) & set(csf_df.index)
    ))
    return moca, established_builder, both_builder, analytic_ids


def run_arm(builder, factory, grid, moca, ids, cv_config):
    res = nested_cv(
        feature_builder=builder, moca_long=moca, model_factory=factory,
        param_grid=grid, cv_config=cv_config, analytic_ids=ids,
    )
    truth = res["oof_truth"]
    pred = res["oof_predictions"].reindex(truth.index)
    return {
        "r2": float(res["r2_mean"]),
        "repeat_lo": float(res["r2_repeat_lo"]),
        "repeat_hi": float(res["r2_repeat_hi"]),
        "per_seed_r2": [float(x) for x in res["per_seed_r2"]],
        "calibration": calibration(truth.values, pred.values),
    }


CV_CONFIG = {"n_outer": 5, "n_inner": 5, "seeds": list(range(10))}


def _increment(arms, fm):
    e, bo = arms[f"established_{fm}"], arms[f"both_{fm}"]
    diffs = [x - y for x, y in zip(bo["per_seed_r2"], e["per_seed_r2"])
             if np.isfinite(x) and np.isfinite(y)]
    return {
        "established_r2": e["r2"], "both_r2": bo["r2"],
        "featureset_gain": bo["r2"] - e["r2"],
        "range_lo": float(np.min(diffs)) if diffs else float("nan"),
        "range_hi": float(np.max(diffs)) if diffs else float("nan"),
        "established_cal_slope": e["calibration"]["slope"],
        "both_cal_slope": bo["calibration"]["slope"],
    }


def run_fm(which):
    """Run established_ridge + the two arms for ONE foundation model, in its own
    process (TabPFN and TabDPT never co-import -> isolates intermittent segfaults
    and any library conflict). Writes fm_robustness_<which>.json."""
    assert which in ("tabpfn", "tabdpt")
    fm_factory = tabpfn_factory if which == "tabpfn" else tabdpt_factory
    moca, est_b, both_b, ids = load_data()
    print(f"[{which}] Device={DEVICE} | analytic N={len(ids)}", flush=True)
    plan = [
        ("established_ridge", est_b, lambda: Ridge(), bm.RIDGE_ALPHA_GRID),
        (f"established_{which}", est_b, fm_factory(), None),
        (f"both_{which}", both_b, fm_factory(), None),
    ]
    # Per-arm checkpoint cache: an intermittent SIGSEGV in one arm must not
    # discard completed arms. On rerun, cached arms are loaded and skipped, so
    # repeated invocations converge (only the failed arm reruns).
    cache_dir = os.path.join(bm._DATA_OUT_DIR, "_fm_arm_cache")
    os.makedirs(cache_dir, exist_ok=True)
    arms = {}
    for name, builder, factory, grid in plan:
        cache_path = os.path.join(cache_dir, f"{name}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                arms[name] = json.load(f)
            print(f"  arm {name} (cached)", flush=True)
        else:
            print(f"  arm {name} ...", flush=True)
            arms[name] = run_arm(builder, factory, grid, moca, ids, CV_CONFIG)
            with open(cache_path, "w") as f:
                json.dump(arms[name], f, indent=2)
        a, c = arms[name], arms[name]["calibration"]
        print(f"    R2={a['r2']:+.4f} [{a['repeat_lo']:+.4f},{a['repeat_hi']:+.4f}] "
              f"| cal slope={c['slope']:.3f} int={c['intercept']:+.4f} citl={c['citl']:+.4f}",
              flush=True)
    out = {"device": DEVICE, "n": int(len(ids)), "arms": arms,
           "increment": _increment(arms, which)}
    path = os.path.join(bm._DATA_OUT_DIR, f"fm_robustness_{which}.json")
    os.makedirs(bm._DATA_OUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}", flush=True)
    return out


def combine():
    """Merge the two per-FM partials, compute replication summary."""
    margin = 0.02
    parts = {}
    for fm in ("tabpfn", "tabdpt"):
        with open(os.path.join(bm._DATA_OUT_DIR, f"fm_robustness_{fm}.json")) as f:
            parts[fm] = json.load(f)
    arms = {}
    for fm in ("tabpfn", "tabdpt"):
        arms.update(parts[fm]["arms"])
    out = {
        "n": parts["tabpfn"]["n"], "margin": margin, "arms": arms,
        "increments": {fm: parts[fm]["increment"] for fm in ("tabpfn", "tabdpt")},
        "established_ridge_consistency": {
            "tabpfn_run": parts["tabpfn"]["arms"]["established_ridge"]["r2"],
            "tabdpt_run": parts["tabdpt"]["arms"]["established_ridge"]["r2"],
        },
    }
    print("\n=== FEATURE-SET GAIN (proteome over established, model held fixed) ===")
    for fm in ("tabpfn", "tabdpt"):
        i = out["increments"][fm]
        print(f"  {fm:>8}: established {i['established_r2']:.4f} -> both {i['both_r2']:.4f} "
              f"| gain {i['featureset_gain']:+.4f} [{i['range_lo']:+.4f},{i['range_hi']:+.4f}] "
              f"| cal slope est {i['established_cal_slope']:.3f} / both {i['both_cal_slope']:.3f}")
    g = {fm: out["increments"][fm]["featureset_gain"] for fm in ("tabpfn", "tabdpt")}
    print(f"\n  Replication: TabPFN gain {g['tabpfn']:+.4f}, TabDPT gain {g['tabdpt']:+.4f}, "
          f"margin {margin}.")
    print(f"  Both gains clear margin: {g['tabpfn'] >= margin and g['tabdpt'] >= margin}")
    print(f"  established_ridge consistency across runs: "
          f"{out['established_ridge_consistency']['tabpfn_run']:.4f} vs "
          f"{out['established_ridge_consistency']['tabdpt_run']:.4f}")
    path = os.path.join(bm._DATA_OUT_DIR, "fm_robustness.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tabpfn"
    if mode == "combine":
        combine()
    else:
        run_fm(mode)
