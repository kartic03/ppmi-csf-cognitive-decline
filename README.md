# Where CSF markers do and do not predict cognitive decline in early Parkinson's disease

Analysis code and derived results for a study of how much an established cerebrospinal fluid marker panel adds to a clinical baseline when predicting the rate of cognitive decline in early Parkinson's disease, and where that added value stops.

Data come from the Parkinson's Progression Markers Initiative (PPMI). The analytic sample is 607 participants with complete baseline CSF assays and at least three longitudinal MoCA assessments.

## What the analysis found

The CSF panel adds ΔR² = 0.058 over a pre-registered ten-variable clinical baseline and 0.046 over a stronger fourteen-variable one. That increment sits 9.3 standard deviations above a random-feature null and survives leave-one-site-out cross-validation.

It is also bounded in three ways, and the boundaries are the point of the paper:

- **Endpoint.** On a dichotomised conversion endpoint, scored on the same participants and the same folds, the panel adds nothing (ΔAUC −0.003).
- **Age.** The increment reverses below age 58 (ΔR² −0.051), and calibration collapses in that same tertile (slope 0.271, against 1.086 and 1.119 in the other two).
- **Attrition.** It reaches 0.119 only in participants who enrolled early and then left the study, and is absent in those who stayed — a pattern consistent with informative censoring rather than administrative censoring.

A two-sample Mendelian randomisation of blood–brain barrier proteins is included as a separate lane. Genetically predicted circulating PDGFRβ shows no causal effect on PD susceptibility.

## Layout

```
src/                  analysis pipeline (phases 1-3) and the Mendelian randomisation lane
analysis/             deviations table and the figure set
analysis/figures/     six figures, one generating script each, plus figstyle.py
data/processed/       derived result artifacts (JSON/CSV) that the figures read
results/tables/       summary tables
pixi.toml, pixi.lock  pinned Python environment
```

## Reproducing the figures

Each figure script reads its values from the artifacts in `data/processed/` at run time and asserts agreement with committed reference values before rendering, so a figure fails rather than drawing a stale number. Paths are resolved relative to the repository root, so no configuration is needed.

```bash
pixi install
pixi run python analysis/figures/fig1_increment_vs_null.py
```

The figures reproduce pixel for pixel across matplotlib 3.11.0 and 3.11.1.

`analysis/figures/artifact_integrity_sweep.py` re-runs the integrity audit described below.

## What is not here

**No participant-level PPMI data.** PPMI's data use agreement does not permit redistribution. Data are available to qualified researchers at [ppmi-info.org](https://www.ppmi-info.org/access-data-specimens/download-data) after registration and approval. Only derived result artifacts are included.

GWAS summary statistics used in the Mendelian randomisation are public at their original sources and are not mirrored here.

The colocalisation step needs R 4.5.3 with `coloc` 5.2.3 and `susieR` 0.14.2, pinned separately in `src/mr/r_coloc_env.lock`.

## Two things to read before quoting any number

`analysis/DEVIATIONS_TABLE.md` records every departure from the original analysis plan, and the scope of the pre-registration. **The OSF registration (doi:10.17605/OSF.IO/P9TMB) covers only an external replication that has not been run.** No PPMI-side result here is pre-registered, and none should be described as such.

`analysis/figures/ARTIFACT_INTEGRITY_SWEEP.md` tests every artifact used by a figure against the committed anchors, and grades each one PASS, ROUNDED, UNANCHORABLE or FAIL. Three artifacts cannot be anchored to any independently committed value, and one file (`data/processed/phase2/enriched_baseline_check.json`) runs a different estimator from the one its own configuration declares. It is retained, carries a `_SUPERSEDED` key, and nothing in `src/` reads it.

## Citation

Manuscript in preparation. Pre-registration: [doi:10.17605/OSF.IO/P9TMB](https://doi.org/10.17605/OSF.IO/P9TMB).

Data used in the preparation of this work were obtained from the PPMI database. PPMI is a public-private partnership funded by The Michael J. Fox Foundation for Parkinson's Research and its industry partners.
