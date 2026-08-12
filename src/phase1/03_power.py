"""
Phase 1, step 03: sample-size / power gate (DESIGN_SPEC_v2 Section 11; review C5).

Continuous-outcome prediction-model sample size (Riley et al. 2019, Stat Med, Part 2).
Criterion 1 (global shrinkage S >= 0.9):  n1 = p / ((S-1) * ln(1 - R2/S)).
Equivalently, the MAX number of predictor parameters supportable at a given n:
   p_max(n) = n * (S-1) * ln(1 - R2/S).
We also report the adjusted-R2 optimism at our actual n (Ezekiel):
   R2_adj = 1 - (1-R2)(n-1)/(n-p-1);  optimism = R2 - R2_adj;  heuristic shrinkage = R2_adj/R2.

Our outcome (Part II EB slope) has reliability ~0.68, which caps achievable R2; so plausible
adjusted R2 for baseline biomarkers predicting the slope is modest (~0.10-0.25). We evaluate
across that range. N: Set A (complete-case incl. Q-albumin) = 287; Set B (general) = 641.
"""
import numpy as np
import pandas as pd

S = 0.9
NS = {"Set A (BBB, n=287)": 287, "Set B (general, n=641)": 641}
R2S = [0.10, 0.15, 0.20, 0.25]


def p_max(n, R2, S=0.9):
    return n * (S - 1) * np.log(1 - R2 / S)


def riley_n1(p, R2, S=0.9):
    return p / ((S - 1) * np.log(1 - R2 / S))


def adj_r2(n, p, R2):
    return 1 - (1 - R2) * (n - 1) / (n - p - 1)


print("=== Max predictor parameters supportable (Riley 2019, shrinkage>=0.9) ===")
rows = []
for label, n in NS.items():
    rows.append({"set": label, **{f"R2={r}": round(p_max(n, r), 1) for r in R2S}})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== Required n (Riley criterion 1) for candidate predictor counts ===")
rows = []
for p in [5, 8, 12, 20]:
    rows.append({"p_predictors": p, **{f"R2={r}": int(np.ceil(riley_n1(p, r))) for r in R2S}})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== Optimism / heuristic shrinkage at our ACTUAL n ===")
rows = []
for label, n in NS.items():
    for p in [8, 12, 20]:
        for R2 in [0.15, 0.25]:
            ar = adj_r2(n, p, R2)
            rows.append({"set": label, "p": p, "R2_app": R2,
                         "R2_adj": round(ar, 3), "optimism": round(R2 - ar, 3),
                         "shrinkage": round(ar / R2, 3)})
print(pd.DataFrame(rows).to_string(index=False))

print("""
=== Interpretation (gate) ===
- Set A (287): supports only a SMALL number of predictors. At a plausible R2~0.15 it
  supports ~p<=4-5; at R2~0.25, ~p<=9. So the complete-case BBB test must be PARSIMONIOUS
  (few predictors) and heavily regularized, and is better reported as effect-size-with-CI
  (estimation) than as a high-dimensional model. This matches review C5.
- Set B (641): comfortable up to ~p=10-12 at R2~0.15, more at higher R2. The general
  multimodal model lives here.
- Action: kill-gate baselines use few, pre-specified predictors + penalization; the BBB
  increment is an estimation (delta with CI), not a many-feature model on Set A.
""")
