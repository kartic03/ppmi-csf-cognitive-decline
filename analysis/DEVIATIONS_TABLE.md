# Pre-registration deviations and analysis inventory

**Purpose.** A single dated record of every departure from the pre-registered plan, with its reason and its effect on the claim. Individually each deviation below is defensible; scattered through a Discussion they would read as drift. Collected here they are an audit trail.

**Pre-registration:** OSF DOI 10.17605/OSF.IO/P9TMB (osf.io/p9tmb), registered 2026-07-17, under project osf.io/6rka8. The registration text and its addendum are available on OSF at that DOI.

**SCOPE OF THE REGISTRATION — read before writing the word "pre-registered" anywhere (added 2026-08-05).** The OSF registration covers the **external PDBP test only**. Its own text is explicit: *"PPMI is the discovery cohort; PDBP is the confirmatory cohort and is unseen at the time of this registration… This registration fixes the primary estimand… before any PDBP outcome-modelling contact."* Registration is stamped **2026-07-17**; the PPMI kill-gates, cohort construction and the entire MR pipeline date from **2026-06-23 to 2026-06-29** (git history). **Therefore: no PPMI-side result in this paper is pre-registered, and none may be described as such.** `DESIGN_REVIEW_v3.md:9` reached this conclusion at design time ("Pre-registration is post-hoc on PPMI; reserve true pre-registration for the unseen PDBP test") and the project has largely honoured it — one violation was found and corrected on 2026-08-05 (`AMYLOID_REFRAME.md:57`, which called the BBB null "pre-registered").

This scoping is **correct practice, not a weakness**, and should be stated plainly in the Methods: the discovery analyses are exploratory, the confirmatory test is registered and unrun. A referee comparing OSF timestamps against the analysis record will find exactly that, and the paper should say it first.

**Standing rule.** Any analysis conceived after the data were in hand is labelled post hoc and hypothesis-generating. No analysis is described as pre-specified unless it appears in a registration time-stamped before the data were examined. A reviewer who checks OSF timestamps against data-access dates will see any violation, and for a project rebuilt after a fabrication rejection that is a disproportionate risk for zero gain.

---

## A. Deviations

| # | Pre-registered | What was actually done | Reason | Effect on the claim |
|---|---|---|---|---|
| 1 | External validation in PDBP (B5) as the confirmatory test | **NOT RUN.** | Requires a Terra/GCP billing account that could not be obtained. Phase 0 feasibility scan WAS completed (free, BigQuery sandbox). | No external-validation claim is made anywhere. All estimates are PPMI discovery. `POSITIONING_TABLE.md:38` rule applied; five "externally validated" claims struck 2026-08-04. |
| 2 | Primary estimand scoped as a **CSF Aβ42/40 ratio** increment | Cannot be computed in the external cohort. | AMP-PD v4 CSF assays are only Aβ42, Tau, p-Tau. **No Aβ40 exists in the release** (verified by direct query, 2026-08-04). | The pre-registered external estimand is untestable as written. Any future B5 must use a reduced block; see #3. |
| 3 | 15-feature frozen model (10 clinical + 5 CSF) | Externally testable block reduces to **{Aβ42, pTau181}**. | NfL exists only in the Olink neurology panel (398 subjects vs ~1,100, different platform); pSer129 α-synuclein absent entirely. | Internally the reduced block is worth dR² **+0.0378 [+0.0202, +0.0440]** vs +0.0581 for the full block — 65%, still above the +0.02 SESOI. |
| 4 | Positive control must recover on **BOTH** pQTL platforms before any BBB null is reported (`DESIGN_SPEC_v3_npjPD.md:150`; added as red-team fix C3) | Gate applied on **deCODE only**. | GPNMB recovers on deCODE (OR 1.49) and fails on UKB-PPP (OR 0.994 [0.896, 1.103]) from **4 instruments vs deCODE's 2** — so it is not instrument-poverty. | **The UKB-PPP arm of the null is UNGATED** and is reported as descriptive. Cross-platform concordance language must be softened or supported by a UKB-detectable control. |
| 5 | Colocalization by SuSiE-coloc, reporting PP0–PP4 | SuSiE returned **no overlapping credible sets** for 6 of 8 non-control cells, so no decomposition. A **`coloc.abf` companion** was added at those loci. | The pre-registered method could not deliver the pre-registered quantity. ABF needs no credible sets. | Companion, not replacement; SuSiE remains primary. ABF gives PDGFRβ **PP4 = 0.009/0.008 with PP1 ≈ 0.75**. Caveat: ABF assumes ONE causal variant per trait and these loci have 5–10. |
| 6 | RQ4 equivalence test of high-dimensional proteomics + foundation models | Equivalence **rejected**, then the proteome claim was **withdrawn by the authors**. | The TabPFN gain did not replicate on a second foundation model (TabDPT: negative gain, calibration 1.46–1.51). | Reframed as evidence FOR parsimony. Reported as a bounded supplement, not a headline. |
| 7 | Clinical comparator = the pre-registered 10-feature block | An **enriched 14-feature comparator** is additionally reported (adds `COG_COMPOSITE_INT`, `gds`, `stai`, `rem`; caudate SBR replaces putamen). | The 10-feature block contains only ONE cognitive predictor and is below the field's floor; it also **fails Riley's minimum sample size** at N=607 (needs 986–1254), whereas the enriched block passes (524–568). | Both are reported. Increment against the strong comparator is **+0.0456** (vs +0.0581 against the thin one) and survives on all three outcome definitions. |
| 8 | B5 feasibility gate: total eligible **N ≥ 150** (≥200 preferred), calibration-slope 95% CI width ≤ 0.6 | The two criteria are **mutually inconsistent**. | Archer (PMID 33150684): the stated CI width needs **n ≥ 293** (full model) or **n ≥ 357** (reduced block). | The N-gate must be raised to ~300–360 before any enclave work. Passing at N=180 would not deliver the promised precision. |
| 10 | PRS negative control (rank 11 of the research plan) | **NOT RUN.** | The project holds no PPMI genotypes; the only genetic files on disk are 1000G LD reference panels downloaded for the MR, which are reference data, not participant data. Needs the LONI imputed-genotype release + Nalls 2019 (PMID 31701892) weights. | **Must never be described as null — it is unrun.** APOE ε4 and GBA are not substitutes: both have established effects on cognitive decline, which makes them positive controls, not negative ones. |
| 11 | *(not pre-registered)* APOE ε4 as a positive control | Run, and it **did not fire** (dR² −0.0003). | APOE ε4 has **no marginal association in this cohort** — conversion 27.8% vs 28.6%, Fisher p=0.86, with the slope difference running opposite to expectation. Flat on the binary endpoint, under dose coding, and in long-follow-up subjects too. | Report as a **cohort characteristic, not a dropped arm**. It is not evidence of an insensitive outcome — GBA fires on the same harness. |
| 9 | *(not pre-registered)* Parsimony comparison against Hlavnička 2025 | **Post-hoc refit added 2026-08-04**, after the deep-research pass identified PMID 40274837 as the comparator an npj PD reviewer would raise. | Their four questionnaire predictors are free; the paper's premise is that an LP is worth its cost. Unanswered, this is an obvious reviewer objection. | Labelled post hoc and hypothesis-generating. Reported as *their predictors, our cohort, our endpoint* — **not a replication**. Their 0.70 ± 0.10 is never quoted as a number being beaten. |

---

## B. Corrections to previously reported numbers

| Was reported as | Correct value | Note |
|---|---|---|
| Calibration deviation described as "7.6% of **outcome SD**", instability as "47% of **outcome SD**" | **The percentages are right; the denominator is mislabelled.** Both use the **residual SD of the clinical+CSF model (0.3502)**, not the outcome SD: 0.026472/0.3502 = 7.6% and 0.163407/0.3502 = 46.7%. Against the actual outcome SD on the analysis set (0.3749, n=607) they are **7.1%** and **43.6%** | Investigated twice on 2026-08-05. My first pass called 7.6% unreproducible and "corrected" it to 7.1% — that was premature: the two documented percentages are mutually consistent under a common denominator, which is what gave the game away. **Fix the label, not the arithmetic.** For the manuscript, report the absolute values (0.0265 and 0.163 MoCA points/year) and name whichever denominator is used |
| Bootstrap CI described as **300 replicates** | **B = 200** (`bootstrap_ci.json`: `B_requested` = `B_usable` = 200) | caught 2026-08-05 while figure captions were being written from the artifacts. The interval itself, [+0.0120, +0.0923], is unchanged and correct |
| *(data-integrity issue found 2026-08-05 while building figures)* Two artifacts report the **same nominal analysis** with **different numbers** | **Use `estimand_audit.json`. Do not use `enriched_baseline_check.json`.** | Both claim n=607, identical `cv_config` (5×5, seeds 0–9) and identical thin/enriched column lists, yet disagree: enriched R² **0.21065 vs 0.21317**, ΔR² **+0.04560 vs +0.04266**; thin ΔR² **+0.05813 vs +0.05080**. **`estimand_audit.json` reproduces the committed `increment.json` headline to the last digit on its thin control arm (0.06921477 / 0.12734763 / 0.05813286); `enriched_baseline_check.json` does not.** An artifact that fails to reproduce a known committed value on its control arm cannot be trusted on its test arm. **ROOT CAUSE FOUND AND VERIFIED 2026-08-05: a mislabelled estimator.** `enriched_baseline_check.json`'s control-arm ΔR² is **0.05079575757612287**, which equals `increment.json` `rq1_csf.perm_observed_dr2_fixed` **exactly, to the last digit (delta 0.000e+00)** — that is the *fixed-alpha permutation* estimator, not the tuned nested-CV estimator the file's own `cv_config` declares. So the whole file is on the fixed-alpha estimator while claiming 5×5 nested CV with alpha tuning. Not seed noise, not a rounding artifact: a wrong estimator under a correct-looking label. This is the same family as the documented `param_grid=None` trap. **The manuscript's enriched-baseline numbers (R² 0.2106, ΔR² +0.0456) are correct as written**; the risk was that a downstream consumer reading the wrong file would silently substitute 0.21317 / +0.04266. |
| The increment's concentration in short-follow-up subjects is "largely administrative", and this "explains the temporal-era transport failure as arithmetic, not biology" | **Both withdrawn 2026-08-04.** Splitting within enrolment phase 1 makes the concentration **stronger** (gap +0.1126 vs +0.0703, 160%), and the administratively censored group shows **no increment (−0.0153)**. It is **genuine early exit**, and the cross-wave transport failure is real | confirmed on RMSE and on unshrunken `ols_slope`. Replacement framing: the model predicts decline mainly in subjects who leave the study — **informative censoring** |
| GPNMB cross-platform discordance "a SomaScan-vs-Olink epitope difference is a candidate explanation" | **tested 2026-08-04 and NOT supported** — 0/8 instruments coding on either platform. The likelier cause is that the platforms instrument different variation (no shared rsID; strongest instrument on each nearly independent of the other's) | the wording in `15_poscontrol.py` and any draft must be updated: the epitope hypothesis is no longer merely untested, it has been tested and failed. **The gate is still not restored.** |
| "no colocalization (PP4 = 0)" | **PP4 not estimable** under SuSiE for 6/8 cells; ABF companion gives PDGFRβ 0.009/0.008 | `pp4 = 0.0` was a `coloc.susie_noCS` sentinel, not a posterior |
| "PP4 = 0" for MMP9 | **0.018 (deCODE), 0.007 (UKB-PPP)** | genuine computed posteriors, not placeholders |
| "two positive controls" | **one exposure on one platform, verified two ways** (MR and coloc, both deCODE) | not two independent controls |
| RQ4 "10 seeds × 5 outer × 5 inner" | **5 seeds** | `01_benchmark.py` sets `seeds = range(5)`; provenance block now records it |
| TabPFN "Deterministic (identical across runs)" | reproduces to ~1e-9 **within** a fixed environment, but the repo's two committed artifacts disagree at ~5e-5 | they were produced under different conditions |
| Frozen-bundle commit `f81999f` | **`b6f1976`** | the bundle's own `meta.git_commit`; f81999f is an earlier commit |
| L3 "5x2 nested CV" | **5 outer × 5 inner, 10 seeds** | |
| Amyloid ratio "carries the increment on its own" | **largest single contributor, ~73%** | 10-seed ranges do not overlap the full block's |
| Noise floor "dR² = −0.0093" | **a single random draw, not a floor.** Over 20 draws: mean −0.0040, SD 0.0067, **95th percentile +0.0078**, max **+0.0133** | corrected 2026-08-04. A second draw of the same null gave +0.0066. Quote the 95th percentile as the bar; the CSF increment is 9.3 SD above the noise mean |
| CSF increment stated without an endpoint qualifier | **specific to the continuous decline-rate outcome.** On the dichotomised conversion endpoint, same 324 subjects and folds, CSF adds **−0.003 AUC** | added 2026-08-04. Not a reproduction failure — dichotomising discards magnitude and timing, so the estimands genuinely differ — but every statement of the increment must now name its outcome |

---

## C. Reporting conventions

1. **Bracketed ranges in the source artifacts are cross-seed repeat ranges, NOT confidence intervals** (`src/phase2/cv.py:35-38` says so). The proper subject-level bootstrap CI on the RQ1 increment is **[+0.0120, +0.0923]** — about 5× wider than the repeat range, excludes 0, does **not** exclude the +0.02 SESOI.
2. **Never print `p = 0`.** The permutation estimator lacks the (b+1)/(m+1) correction; report `p < 1/n_perm`.
3. **Clinical+CSF is the paired N=607 pair (0.069 → 0.127).** Never "0.076 + 0.058" — different cohorts.
4. **Marginal conformal coverage is not evidence about the model** (split conformal guarantees it by construction). Report sharpness — 90% interval width **1.036 MoCA pts/yr**, ~3× the outcome SD — and Mondrian group coverage instead.
5. **`min_detectable_diff` in `benchmark.json` is a misnomer**: it is the WIDTH of the per-seed range, so it worsens with more seeds and measures compute jitter, not statistical resolution.

---

## D. Confirmatory vs exploratory

**Confirmatory:** RQ1, the CSF increment over clinical predictors on the EB MoCA-slope outcome, in PPMI.

**Exploratory / hypothesis-generating** (nominal p, no confirmatory interpretation, no family-wise correction claimed):
the within-CSF decomposition (9 subsets); the enriched-comparator arm; the RQ4 benchmark (8 arms); the equity audit (~9 subgroup cells); the two internal-external splits; the MR panel (8 protein/platform cells plus colocalization); the trial-enrichment panel; all age- and visit-count-stratified analyses.

**Scope conditions that are results, not caveats:** the model is age-scoped (calibration slope 0.271 in the youngest tertile, where decline is near-flat) and its increment concentrates in subjects with shorter observation windows.

---

*Every figure traceable to a JSON artifact under `data/processed/`.*
