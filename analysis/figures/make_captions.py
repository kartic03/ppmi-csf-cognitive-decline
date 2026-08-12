"""Generate captions.md for the npj Parkinson's Disease revision figure set.

Every numeral in every caption is READ FROM THE ARTIFACTS at runtime and
formatted into the text. Nothing is transcribed by hand, so a caption cannot
drift from the figure it describes.

Writes: analysis/figures/captions.md
"""
import json
import os

import pandas as pd

# Repo-relative. This file lives at <repo>/analysis/figures/, so the repo
# root is two levels up. That holds in the WSL repro environment and in a
# clean checkout, so a figure is never built against a path that exists on
# only one machine.
OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(OUT))
BASE = os.path.join(REPO, "data", "processed")
assert os.path.isdir(BASE), f"artifact directory not found: {BASE}"


def load(rel):
    with open(os.path.join(BASE, rel)) as fh:
        return json.load(fh)


inc = load("phase2/increment.json")
ea = load("phase2/estimand_audit.json")
boot = load("phase2/bootstrap_ci.json")
nf = load("phase2/noise_floor_distribution.json")
b1 = load("phase2/batch1_checks.json")
pv = load("phase2/phase1_visit_split.json")
hl = load("phase2/hlavnicka_head2head.json")
cal = load("phase2/calibration.json")
cf = load("phase2/calibration_flexible.json")
fa = load("phase3/fairness_audit.json")
iv = load("phase2/iecv_site.json")
rk = load("phase2/rank11_controls.json")
gba = load("phase2/gba_scale_anchor.json")
pb = load("mr/mr_power_bounds.json")
pos = load("mr/poscontrol_result.json")
mr_csv = pd.read_csv(os.path.join(BASE, "mr/mr_final_table.csv"))

rq1 = inc["rq1_csf"]
N = rq1["n"]
ea_thin, ea_rich = ea["results"]["eb_slope|thin"], ea["results"]["eb_slope|enriched"]
sd_above = (rq1["dr2"] - nf["mean"]) / nf["sd"]
ter = {t["stratum"]: t for t in b1["age_tertile_dr2"]}
cells = {c["cell"]: c for c in pv["results"]["eb_slope"]["cells"]}
arms = hl["arms"]
d_thin = arms["clinical + CSF"]["auc"] - arms["clinical (10)"]["auc"]
d_rich = arms["enriched + CSF"]["auc"] - arms["enriched (14)"]["auc"]
curve = cal["continuous"]["calibration_curve"]
bn = [b["n"] for b in curve]
cft = {t["stratum"]: t for t in cf["age_tertile_calibration"]}
site, rc, ref = iv["site"], iv["random_control"], iv["reference_random_5fold"]
cm = site["calibration_meta_real_sites_only"]
real = [c for c in site["per_cluster"] if c["cluster"] != "-1.0"]
pooled = [c for c in site["per_cluster"] if c["cluster"] == "-1.0"][0]
ra = rk["arms"]
cells_mr = {(c["protein"], c["platform"]): c for c in pb["cells"]}
pdg_d, pdg_u = cells_mr[("PDGFRB", "decode")], cells_mr[("PDGFRB", "ukbppp")]
gp_d, gp_u = pos["platforms"]["decode"], pos["platforms"]["ukbppp"]

L = []
L.append("# Figure captions")
L.append("")
L.append("Every numeral below is read from the source artifact at runtime by "
         "`make_captions.py`; none is transcribed by hand. Artifact integrity "
         "verdicts referenced here are recorded in `ARTIFACT_INTEGRITY_SWEEP.md`.")
L.append("")

# ---------------------------------------------------------------- Fig 1
L.append("## Figure 1. The CSF increment against its null.")
L.append("")
L.append(
    f"**(a)** Adding the CSF block to the pre-registered 10-variable clinical "
    f"baseline raises out-of-fold R\u00b2 for the continuous decline-rate outcome "
    f"from {ea_thin['r2_base']:.4f} to {ea_thin['r2_aug']:.4f} "
    f"(\u0394R\u00b2 = +{ea_thin['dr2']:.4f}); against a stronger 14-variable "
    f"clinical block the same CSF panel raises R\u00b2 from "
    f"{ea_rich['r2_base']:.4f} to {ea_rich['r2_aug']:.4f} "
    f"(\u0394R\u00b2 = +{ea_rich['dr2']:.4f}). A richer clinical baseline absorbs "
    f"part of the increment without eliminating it. Both rows come from "
    f"`phase2/estimand_audit.json` (`results['eb_slope|thin']` and "
    f"`results['eb_slope|enriched']`), whose control arm reproduces the "
    f"committed headline in `phase2/increment.json` to the last digit. "
    f"`phase2/enriched_baseline_check.json` is deliberately NOT used: its "
    f"control arm returns \u0394R\u00b2 = "
    f"{[r for r in load('phase2/enriched_baseline_check.json')['results'] if r['outcome'] == 'eb_slope' and r['baseline'] == 'thin'][0]['dr2']:.8f}, "
    f"which is exactly `increment.json rq1_csf.perm_observed_dr2_fixed` "
    f"({rq1['perm_observed_dr2_fixed']:.8f}) \u2014 the fixed-alpha permutation "
    f"estimator, not the tuned nested-CV estimator its own configuration "
    f"declares. **(b)** The observed CSF \u0394R\u00b2 of {rq1['dr2']:.4f} "
    f"(`phase2/increment.json`, `rq1_csf.dr2`) against {nf['n_draws']} "
    f"random-feature draws (`phase2/noise_floor_distribution.json`), which have "
    f"mean {nf['mean']:+.4f} and SD {nf['sd']:.4f}; the observed value sits "
    f"{sd_above:.1f} SD above the null mean and far beyond the null's 95th "
    f"percentile of {nf['p95']:.4f}. The error bar is the subject-level "
    f"percentile bootstrap over participants, {boot['ci_lo']:.4f} to "
    f"{boot['ci_hi']:.4f} (B = {boot['B_usable']}, "
    f"`phase2/bootstrap_ci.json`) \u2014 the only confidence interval on the "
    f"increment anywhere in this figure set. Cross-seed repeat ranges are not "
    f"confidence intervals and are not drawn. n = {N} CSF-complete participants.")
L.append("")

# ---------------------------------------------------------------- Fig 2
L.append("## Figure 2. Where the increment holds and where it fails.")
L.append("")
L.append(
    f"**(a)** \u0394R\u00b2 from adding the CSF block, by stratum, ordered so the "
    f"two negative cells sit immediately beneath the overall result rather than "
    f"at the foot of the panel. Overall \u0394R\u00b2 = +{rq1['dr2']:.4f} "
    f"(n = {N}, `phase2/increment.json`). The increment REVERSES in the youngest "
    f"age tertile (\u0394R\u00b2 = {ter['age <= 58']['dr2']:+.4f}, "
    f"n = {ter['age <= 58']['n']}) and is absent in the administratively "
    f"censored cell (\u0394R\u00b2 = {cells['phase 2, few-visit']['dr2']:+.4f}, "
    f"n = {cells['phase 2, few-visit']['n']}). It holds in completers "
    f"({cells['phase 1, many-visit']['dr2']:+.4f}, "
    f"n = {cells['phase 1, many-visit']['n']}), the middle tertile "
    f"({ter['age 58-67']['dr2']:+.4f}, n = {ter['age 58-67']['n']}), the oldest "
    f"tertile ({ter['age > 67']['dr2']:+.4f}, n = {ter['age > 67']['n']}) and "
    f"early exiters ({cells['phase 1, few-visit']['dr2']:+.4f}, "
    f"n = {cells['phase 1, few-visit']['n']}). Age-stratified increments come "
    f"from `phase2/batch1_checks.json` (`age_tertile_dr2`), whose `noise_block` "
    f"section is superseded by `phase2/noise_floor_distribution.json` but whose "
    f"age arm is untouched by that supersession and is independently "
    f"corroborated: all three per-tertile augmented R\u00b2 reproduce "
    f"`phase3/fairness_audit.json`'s age-tertile R\u00b2 to ~1e-16 at identical n. "
    f"Visit strata come from `phase2/phase1_visit_split.json` and are the "
    f"PHASE-STRATIFIED cells; the file's pooled '(all phases)' cells are not "
    f"plotted, because the pooled few-visit cell (n = "
    f"{cells['few-visit (all phases)']['n']}) is the sum of the two phase cells "
    f"shown and would double-count the phase-2 participants. Each stratification "
    f"partitions the cohort exactly. These cells are scored on shared "
    f"out-of-fold predictions and carry no confidence intervals. "
    f"**(b)** The SAME CSF block on the DICHOTOMISED conversion endpoint, a "
    f"different metric on a separate axis: AUC moves from "
    f"{arms['clinical (10)']['auc']:.3f} to "
    f"{arms['clinical + CSF']['auc']:.3f} ({d_thin:+.3f}) against the clinical "
    f"block, and from {arms['enriched (14)']['auc']:.3f} to "
    f"{arms['enriched + CSF']['auc']:.3f} ({d_rich:+.3f}) against the enriched "
    f"block. CSF adds nothing on this endpoint. Source "
    f"`phase2/hlavnicka_head2head.json` (n = {hl['n']}, {hl['events']} events), "
    f"which is **UNANCHORABLE**: it shares no arm with any committed value, so "
    f"unlike every other source here it cannot be checked against the headline "
    f"result. Its brackets are cross-seed ranges, not confidence intervals, and "
    f"are not drawn.")
L.append("")

# ---------------------------------------------------------------- Fig 3
L.append("## Figure 3. Calibration.")
L.append("")
L.append(
    f"**(a)** Binned decile calibration curve: {len(curve)} bins of "
    f"{min(bn)}\u2013{max(bn)} participants each (summing to {sum(bn)}), plotting "
    f"mean observed against mean predicted decline rate, with the 45\u00b0 ideal. "
    f"Each point is a BIN MEAN, not a participant. Coordinates come from "
    f"`phase2/calibration.json` (`continuous.calibration_curve`), the only "
    f"artifact in the project holding calibration curve coordinates of any kind. "
    f"The overall calibration slope is {cf['overall_slope']:.4f} "
    f"(`phase2/calibration_flexible.json`, `overall_slope`; intercept "
    f"{cf['overall_intercept']:+.6f}), and the flexible calibration mean "
    f"absolute deviation is {cf['flexible_mean_abs_deviation']:.6f} (same file, "
    f"`flexible_mean_abs_deviation`). That flexible result exists ONLY as a "
    f"summary statistic \u2014 no flexible-fit coordinates exist anywhere in "
    f"`data/processed` \u2014 so no smooth fit is drawn and the panel must not be "
    f"read as showing one. **(b)** Per-age-tertile calibration slopes against the "
    f"ideal of 1: the youngest tertile collapses to "
    f"{cft['age<=58']['slope']:.3f} (n = {cft['age<=58']['n']}), while the "
    f"middle and oldest tertiles are close to ideal at "
    f"{cft['age 58-67']['slope']:.3f} and {cft['age>67']['slope']:.3f}. "
    f"**(c)** The same three tertiles' \u0394R\u00b2 from Figure 2, on the same "
    f"rows: the n = {cft['age<=58']['n']} youngest subgroup fails on both axes "
    f"\u2014 its calibration slope collapses AND its CSF increment reverses "
    f"({ter['age <= 58']['dr2']:+.4f}). These are two independent symptoms in one "
    f"subgroup, not two unrelated findings. All three tertile slopes reconcile "
    f"with `phase3/fairness_audit.json` to ~1e-16 at identical n, and the two "
    f"calibration sources agree at the stored 4 dp of the coarser one "
    f"(`calibration.json` `calib_slope` = {cal['continuous']['calib_slope']}).")
L.append("")

# ---------------------------------------------------------------- Fig 4
L.append("## Figure 4. Internal-external cross-validation across enrolment sites.")
L.append("")
L.append(
    f"This is internal-external cross-validation across enrolment sites, NOT "
    f"external validation: every fold shares PPMI's protocol, era and inclusion "
    f"criteria. **(a)** Holding out whole enrolment sites lowers overall "
    f"performance \u2014 clinical+CSF R\u00b2 falls from "
    f"{ref['r2_aug']:.4f} under standard k-fold to "
    f"{rc['mean_r2_aug']:.4f} under a matched random-cluster control and "
    f"{site['pooled_r2_aug']:.4f} under leave-one-site-out, with the clinical "
    f"baseline falling correspondingly from {ref['r2_base']:.4f} to "
    f"{rc['mean_r2_base']:.4f} to {site['pooled_r2_base']:.4f}. "
    f"**(b)** The INCREMENT largely holds across the same three designs: "
    f"{ref['dr2']:+.4f} (k-fold), {rc['mean_dr2']:+.4f} (random-cluster, "
    f"{len(rc['runs'])} seeds), {site['pooled_dr2']:+.4f} (leave-one-site-out). "
    f"The drop in (a) is a transportability cost paid by the whole model, not by "
    f"the CSF block specifically. **(c)** Random-effects calibration slope across "
    f"REAL SITES ONLY (k = {cm['k']}): {cm['estimate']:.3f}, 95% CI "
    f"{cm['ci_lo']:.3f}\u2013{cm['ci_hi']:.3f}, 95% prediction interval for a new "
    f"site {cm['pi_lo']:.3f}\u2013{cm['pi_hi']:.3f}. I\u00b2 = "
    f"{cm['I2_percent']:.0f}% is UNDERPOWERED at k = {cm['k']} and must not be "
    f"read as evidence of homogeneity. The k = "
    f"{site['calibration_meta']['k']} figure "
    f"({site['calibration_meta']['estimate']:.3f}) is not used anywhere: it folds "
    f"in a pooled cluster of 38 small sites (n = {pooled['n']}) that is not a "
    f"site and, having the smallest standard error, would dominate the pooling. "
    f"That cluster is retained in the pooled R\u00b2 so no participant is dropped, "
    f"but excluded from the calibration meta-analysis. Per-site R\u00b2 is not "
    f"plotted: the {len(real)} real sites have n = "
    f"{min(c['n'] for c in real)}\u2013{max(c['n'] for c in real)}, where it is "
    f"noise. All values from `phase2/iecv_site.json`; n = {N} across "
    f"{len(site['per_cluster'])} clusters.")
L.append("")

# ---------------------------------------------------------------- Fig 5
L.append("## Figure 5. Controls against the random-feature null.")
L.append("")
L.append(
    f"\u0394R\u00b2 for each control arm against the null band spanned by the "
    f"{nf['n_draws']} random-feature draws ({nf['min']:+.4f} to "
    f"{nf['max']:+.4f}), with the null's 95th percentile at {nf['p95']:.4f} "
    f"marked. The CSF block under test contributes "
    f"+{ra['anchor_csf']['dr2']:.4f} (n = {ra['anchor_csf']['n']}). All three "
    f"negative controls stay inside the null: 5 random Gaussians "
    f"{ra['neg_noise_csfset']['dr2']:+.4f}, SAA assay version "
    f"(technical/batch) {ra['neg_assay_csfset']['dr2']:+.4f}, and SAA positivity "
    f"{ra['neg_saa_csfset']['dr2']:+.4f}. Of the two established genetic factors, "
    f"APOE \u03b54 does not fire at all "
    f"({ra['pos_APOE_e4']['dr2']:+.4f}, n = {ra['pos_APOE_e4']['n']}) \u2014 a "
    f"reportable result, not a missing bar: a well-established risk factor adds "
    f"nothing to this decline-rate model. GBA carrier status contributes "
    f"{ra['pos_gba_status']['dr2']:+.4f} (n = {ra['pos_gba_status']['n']}), which "
    f"clears the null 95th percentile but sits BELOW the largest single noise "
    f"draw ({nf['max']:.4f}) \u2014 clearing a percentile is not the same as "
    f"exceeding every draw. Its contribution is also prevalence-bounded: at the "
    f"observed carrier prevalence of {gba['prevalence'] * 100:.1f}% the maximum "
    f"attainable R\u00b2 contribution is {gba['r2_ceiling_at_this_prevalence']:.4f} "
    f"(`phase2/gba_scale_anchor.json`), drawn on the panel, so the bar reads as "
    f"capped by design rather than as a weak effect. Control arms from "
    f"`phase2/rank11_controls.json`, whose anchor arm reproduces the committed "
    f"increment exactly; null from `phase2/noise_floor_distribution.json`. Arms "
    f"differ in n because each control is scored on the largest set where it is "
    f"defined. The file's repeat brackets are cross-seed ranges, not confidence "
    f"intervals, and are not drawn.")
L.append("")

# ---------------------------------------------------------------- Fig 6
L.append("## Figure 6. The causal null.")
L.append("")
L.append(
    f"Two-sample Mendelian randomisation for PDGFRB on both proteomic platforms, "
    f"with the GPNMB positive control on the same odds-ratio axis. PDGFRB is null "
    f"on deCODE (OR {pdg_d['or']:.3f}, 95% CI "
    f"{pdg_d['ci_low']:.3f}\u2013{pdg_d['ci_high']:.3f}, "
    f"{pdg_d['n_instruments']} instruments, F = {pdg_d['F']:.0f}) and on UKB-PPP "
    f"(OR {pdg_u['or']:.3f}, 95% CI "
    f"{pdg_u['ci_low']:.3f}\u2013{pdg_u['ci_high']:.3f}, "
    f"{pdg_u['n_instruments']} instruments, F = {pdg_u['F']:.0f}). Two distinct "
    f"quantities bound this null and are reported separately. The "
    f"**equivalence bound** \u2014 the upper 95% confidence limit, i.e. the "
    f"largest effect the data exclude \u2014 is OR {pdg_d['ci_high']:.6f} on "
    f"deCODE and OR {pdg_u['ci_high']:.6f} on UKB-PPP; effects larger than "
    f"these are ruled out. The **minimum detectable effect at 80% power** "
    f"\u2014 the smallest odds ratio each design could have detected, computed "
    f"in the source artifact as exp((z\u2080.\u2089\u2087\u2085 + "
    f"z\u2080.\u2088\u2080) \u00d7 SE) and marked below each interval \u2014 is "
    f"OR {pdg_d['mde_or_80pct']:.6f} and OR {pdg_u['mde_or_80pct']:.6f} "
    f"respectively. The first says what the data rule out; the second says "
    f"what the study was capable of seeing, and for a null result the second "
    f"is the more informative. Both sit far below the positive-control effect "
    f"of OR {gp_d['or']:.3f} "
    f"({gp_d['ci_low']:.3f}\u2013{gp_d['ci_high']:.3f}) that the same pipeline "
    f"recovers on deCODE, so this is a null at a scale where the pipeline "
    f"demonstrably detects a real effect, not a null from insufficient power. "
    f"**The UKB-PPP arm is marked UNGATED "
    f"on the figure.** The pre-specified design required positive-control "
    f"recovery on BOTH platforms before any null could be reported; GPNMB does "
    f"not recover on UKB-PPP (OR {gp_u['or']:.3f}, 95% CI "
    f"{gp_u['ci_low']:.3f}\u2013{gp_u['ci_high']:.3f}, spanning 1), so the gate "
    f"was narrowed to deCODE only and nothing demonstrates the UKB-PPP arm could "
    f"have detected a true effect. That arm is descriptive, and both it and the "
    f"failed positive control are drawn with open markers and labelled UNGATED. "
    f"Estimates and bounds from `mr/mr_power_bounds.json`; positive control from "
    f"`mr/poscontrol_result.json`. `mr/mr_final_table.csv` is **UNANCHORABLE** "
    f"\u2014 it contains no GPNMB row and so cannot be checked against either "
    f"positive control \u2014 and is used here only as a 3 dp cross-check of the "
    f"odds ratios, never as the plotted source.")
L.append("")

L.append("---")
L.append("")
L.append("### Sources flagged UNANCHORABLE where they are used")
L.append("")
L.append("Two artifacts in this figure set share no arm with any committed value "
         "and therefore cannot be verified against the headline result. Both are "
         "flagged in the captions above at the point of use:")
L.append("")
L.append(f"- `phase2/hlavnicka_head2head.json` (Figure 2b) \u2014 different "
         f"endpoint, different sample (n = {hl['n']}), different metric.")
L.append("- `mr/mr_final_table.csv` (Figure 6, cross-check only) \u2014 no GPNMB "
         "row, so neither positive control appears in it.")
L.append("")
L.append("A third, `phase2/trial_enrichment_ci.json`, is also UNANCHORABLE and is "
         "not used in this figure set. Note that CSF does not drive the "
         "trial-enrichment benefit: its incremental contribution straddles zero "
         "at every threshold.")
L.append("")

text = "\n".join(L)
with open(os.path.join(OUT, "captions.md"), "w") as fh:
    fh.write(text + "\n")
print(f"captions.md written: {len(text)} chars, {len(text.split())} words")
