"""Artifact integrity sweep for the npj Parkinson's Disease revision figure set.

For every artifact named in the figure brief, ask three questions:

  1. Does it contain an arm that SHOULD equal a known committed value?
  2. If so, does it? (asserted to 1e-6; actual delta always recorded)
  3. If not, say so explicitly and mark it UNANCHORABLE -- an artifact nobody
     can check is a different risk from one that checks out, and both must be
     visible.

ANCHORS
  phase2/increment.json rq1_csf
      r2_base = 0.06921477353696548
      r2_aug  = 0.12734763266201798
      dr2     = 0.058132859125052505
      n       = 607
  mr/poscontrol_result.json
      deCODE  OR = 1.491785
      UKB-PPP OR = 0.994015

Anchor values are READ FROM THE ANCHOR FILES at runtime, not hardcoded; the
literals above are documentation only and are themselves verified below.

VERDICTS
  ANCHOR        this file defines the anchor
  PASS          overlapping arm, agrees within 1e-6
  ROUNDED       overlapping arm stored at reduced precision; agrees at its
                stored precision but fails a strict 1e-6 assert
  FAIL          overlapping arm, does not agree
  UNANCHORABLE  no arm overlaps any committed value -- cannot be checked

Reports only. Nothing is fixed or regenerated.
Writes: analysis/figures/ARTIFACT_INTEGRITY_SWEEP.md
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
TOL = 1e-6


def load(rel):
    with open(os.path.join(BASE, rel)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- anchors
inc = load("phase2/increment.json")
pos = load("mr/poscontrol_result.json")

A = {"r2_base": inc["rq1_csf"]["r2_base"],
     "r2_aug": inc["rq1_csf"]["r2_aug"],
     "dr2": inc["rq1_csf"]["dr2"],
     "n": inc["rq1_csf"]["n"]}
P = {"decode_or": pos["platforms"]["decode"]["or"],
     "ukbppp_or": pos["platforms"]["ukbppp"]["or"]}

# the documented literals must match what the anchor files actually contain
assert abs(A["r2_base"] - 0.06921477353696548) < 1e-15
assert abs(A["r2_aug"] - 0.12734763266201798) < 1e-15
assert abs(A["dr2"] - 0.058132859125052505) < 1e-15
assert A["n"] == 607
assert abs(P["decode_or"] - 1.491785) < 1e-15
assert abs(P["ukbppp_or"] - 0.994015) < 1e-15
assert abs(pos["or"] - P["decode_or"]) < 1e-15, "poscontrol headline OR is the deCODE arm"

rows = []       # value-level checks
files = []      # one verdict per artifact


def chk(artifact, path, anchor_name, anchor_val, actual, stored_dp=None):
    """Record one value-level comparison. stored_dp = decimal places the
    artifact stores, when it holds a deliberately rounded copy."""
    delta = actual - anchor_val
    strict = abs(delta) <= TOL
    if strict:
        verdict = "PASS"
    elif stored_dp is not None and round(anchor_val, stored_dp) == round(actual, stored_dp):
        verdict = "ROUNDED"
    else:
        verdict = "FAIL"
    rows.append({"artifact": artifact, "field": path, "anchor": anchor_name,
                 "anchor_value": anchor_val, "actual": actual, "delta": delta,
                 "strict_1e6": strict, "verdict": verdict})
    return verdict


def worst(*verdicts):
    order = ["PASS", "ROUNDED", "FAIL"]
    return max(verdicts, key=lambda v: order.index(v)) if verdicts else "PASS"


def record(artifact, verdict, anchorable, note):
    files.append({"artifact": artifact, "verdict": verdict,
                  "anchorable": anchorable, "note": note})


# ================================================================ phase2
# --- increment.json: the anchor itself
record("phase2/increment.json", "ANCHOR", "n/a (defines the anchor)",
       "rq1_csf is the committed headline all other R2 arms are tested against.")

# --- bootstrap_ci.json
b = load("phase2/bootstrap_ci.json")
v = chk("phase2/bootstrap_ci.json", "point_dr2", "dr2", A["dr2"], b["point_dr2"])
record("phase2/bootstrap_ci.json", v, "yes -- point_dr2 vs anchor dr2",
       f"n={b['n']} matches anchor n={A['n']}: {b['n'] == A['n']}. "
       "ci_lo/ci_hi are the bootstrap resample spread and have no committed "
       "counterpart, so they are not independently checkable.")

# --- noise_floor_distribution.json
nf = load("phase2/noise_floor_distribution.json")
v = chk("phase2/noise_floor_distribution.json", "csf_increment", "dr2",
        A["dr2"], nf["csf_increment"], stored_dp=4)
record("phase2/noise_floor_distribution.json", v,
       "partly -- csf_increment is a 4 dp restatement of the anchor dr2",
       f"n_subjects={nf['n_subjects']} matches anchor n: "
       f"{nf['n_subjects'] == A['n']}. The 20 dr2_draws are the null itself "
       "and have no committed counterpart -- the null distribution is "
       "UNANCHORABLE and rests on the correctness of its own run.")

# --- enriched_baseline_check.json
ebc = load("phase2/enriched_baseline_check.json")
ebc_thin = {f"{r['outcome']}|{r['baseline']}": r for r in ebc["results"]}["eb_slope|thin"]
v = worst(
    chk("phase2/enriched_baseline_check.json", "results[eb_slope|thin].r2_base",
        "r2_base", A["r2_base"], ebc_thin["r2_base"]),
    chk("phase2/enriched_baseline_check.json", "results[eb_slope|thin].r2_aug",
        "r2_aug", A["r2_aug"], ebc_thin["r2_aug"]),
    chk("phase2/enriched_baseline_check.json", "results[eb_slope|thin].dr2",
        "dr2", A["dr2"], ebc_thin["dr2"]))
coincide = [k for k, x in inc["rq1_csf"].items()
            if isinstance(x, float) and abs(x - ebc_thin["dr2"]) < 1e-9]
record("phase2/enriched_baseline_check.json", v,
       "yes -- eb_slope|thin is the control arm",
       f"Declares the same n ({ebc['n']}), the same cv_config and the same "
       f"column lists as estimand_audit.json, yet disagrees on all four shared "
       f"arms. Its control-arm dr2 ({ebc_thin['dr2']:.8f}) is exactly equal to "
       f"increment.json rq1_csf.{coincide[0] if coincide else '?'}, i.e. the "
       "fixed-alpha permutation estimator, not the tuned nested-CV estimator "
       "its cv_config declares. Mislabelled estimator, not seed noise. "
       "EXCLUDED from the figure set.")

# --- estimand_audit.json
ea = load("phase2/estimand_audit.json")
ea_thin = ea["results"]["eb_slope|thin"]
v = worst(
    chk("phase2/estimand_audit.json", "results[eb_slope|thin].r2_base",
        "r2_base", A["r2_base"], ea_thin["r2_base"]),
    chk("phase2/estimand_audit.json", "results[eb_slope|thin].r2_aug",
        "r2_aug", A["r2_aug"], ea_thin["r2_aug"]),
    chk("phase2/estimand_audit.json", "results[eb_slope|thin].dr2",
        "dr2", A["dr2"], ea_thin["dr2"]))
record("phase2/estimand_audit.json", v,
       "yes -- eb_slope|thin is the control arm",
       f"n={ea['n']} matches anchor. Reproduces the committed control arm to "
       "the last digit, so its enriched arm and its age/visit strata are used "
       "as the trusted source. NOTE: the lo/hi fields on its arms are "
       "cross-seed repeat ranges (dr2_repeat_lo/hi), NOT confidence intervals, "
       "and must never be drawn as error bars.")

# --- phase1_visit_split.json
pv = load("phase2/phase1_visit_split.json")
fs = pv["results"]["eb_slope"]["full_sample"]
v = worst(
    chk("phase2/phase1_visit_split.json", "results.eb_slope.full_sample.r2_base",
        "r2_base", A["r2_base"], fs["r2_base"]),
    chk("phase2/phase1_visit_split.json", "results.eb_slope.full_sample.r2_aug",
        "r2_aug", A["r2_aug"], fs["r2_aug"]),
    chk("phase2/phase1_visit_split.json", "results.eb_slope.full_sample.dr2",
        "dr2", A["dr2"], fs["dr2"]))
record("phase2/phase1_visit_split.json", v,
       "yes -- results.eb_slope.full_sample is the pooled arm",
       f"n_full={pv['results']['eb_slope']['n_full']} matches anchor. The six "
       "per-cell rows are scored on shared out-of-fold predictions (the file "
       "says so itself) and have no committed counterpart, so the CELLS are "
       "unanchorable even though the pooled arm passes.")

# --- hlavnicka_head2head.json
hl = load("phase2/hlavnicka_head2head.json")
record("phase2/hlavnicka_head2head.json", "UNANCHORABLE", "no",
       f"Different endpoint (dichotomised conversion), different sample "
       f"(n={hl['n']}, {hl['events']} events) and a different metric (AUC). "
       "No arm overlaps the R2 anchor or the MR anchor. Nothing in this file "
       "can be checked against a committed value. Its own caveats state the "
       "brackets are cross-seed ranges, not CIs, and that the H4 arm is the "
       "most favourable of four fainting-item choices.")

# --- iecv_site.json
iv = load("phase2/iecv_site.json")
ref = iv["reference_random_5fold"]
v = worst(
    chk("phase2/iecv_site.json", "reference_random_5fold.r2_base", "r2_base",
        A["r2_base"], ref["r2_base"], stored_dp=4),
    chk("phase2/iecv_site.json", "reference_random_5fold.r2_aug", "r2_aug",
        A["r2_aug"], ref["r2_aug"], stored_dp=4),
    chk("phase2/iecv_site.json", "reference_random_5fold.dr2", "dr2",
        A["dr2"], ref["dr2"], stored_dp=4))
record("phase2/iecv_site.json", v,
       "partly -- reference_random_5fold is a 4 dp restatement of the anchor",
       f"n_pooled={iv['site']['n_pooled']} matches anchor. The reference arm "
       "agrees at 4 dp. The leave-one-site-out and random-cluster arms are the "
       "file's own results and have no committed counterpart -- UNANCHORABLE. "
       "Primary calibration is calibration_meta_real_sites_only (k=10); the "
       "k=11 figure includes the pooled 38-site cluster and must not be used.")

# --- rank11_controls.json
rk = load("phase2/rank11_controls.json")
ac = rk["arms"]["anchor_csf"]
v = worst(
    chk("phase2/rank11_controls.json", "arms.anchor_csf.r2_base", "r2_base",
        A["r2_base"], ac["r2_base"]),
    chk("phase2/rank11_controls.json", "arms.anchor_csf.r2_aug", "r2_aug",
        A["r2_aug"], ac["r2_aug"]),
    chk("phase2/rank11_controls.json", "arms.anchor_csf.dr2", "dr2",
        A["dr2"], ac["dr2"]))
record("phase2/rank11_controls.json", v,
       "yes -- arms.anchor_csf is a deliberate anchor arm",
       f"anchor_csf n={ac['n']} matches anchor. The seven control arms are the "
       "file's own results and are unanchorable individually, but the file "
       "carries its anchor arm precisely, which is the strongest form of "
       "self-check in the set. Its repeat_lo/hi are cross-seed ranges, not CIs.")

# --- gba_scale_anchor.json
gs = load("phase2/gba_scale_anchor.json")
v_csf = chk("phase2/gba_scale_anchor.json", "csf_dr2", "dr2", A["dr2"],
            gs["csf_dr2"], stored_dp=4)
gba_rank = rk["arms"]["pos_gba_status"]["dr2"]
v_gba = chk("phase2/gba_scale_anchor.json", "observed_dr2",
            "rank11_controls arms.pos_gba_status.dr2", gba_rank,
            gs["observed_dr2"], stored_dp=4)
record("phase2/gba_scale_anchor.json", worst(v_csf, v_gba),
       "partly -- csf_dr2 is a 4 dp restatement of the anchor dr2",
       f"Scored on the full cohort (n={gs['n']}), not the CSF-complete set, so "
       "its n does not match the anchor by design. Its observed_dr2 is a 4 dp "
       "restatement of rank11_controls pos_gba_status.dr2 (secondary "
       "cross-check, shown above). The prevalence ceiling "
       f"({gs['r2_ceiling_at_this_prevalence']:.6f}) is derived only within "
       "this file and is UNANCHORABLE.")

# --- trial_enrichment_ci.json
te = load("phase2/trial_enrichment_ci.json")
record("phase2/trial_enrichment_ci.json", "UNANCHORABLE", "no",
       f"Declares n={te['n']} (matches the anchor cohort size) but reports "
       "percentage sample-size reductions, not R2. No arm overlaps any "
       "committed value. Its own csf_incremental medians straddle zero at "
       "every threshold (excludes_zero = "
       f"{sorted({v['excludes_zero'] for v in te['csf_incremental'].values()})}), "
       "which is the reportable finding.")

# --- calibration_flexible.json
cf = load("phase2/calibration_flexible.json")
fa = load("phase3/fairness_audit.json")
cal = load("phase2/calibration.json")
v_x1 = chk("phase2/calibration_flexible.json", "overall_slope",
           "fairness_audit overall.cal_slope", fa["overall"]["cal_slope"],
           cf["overall_slope"])
v_x2 = chk("phase2/calibration_flexible.json", "overall_slope",
           "calibration.json continuous.calib_slope",
           cal["continuous"]["calib_slope"], cf["overall_slope"], stored_dp=4)
record("phase2/calibration_flexible.json", worst(v_x1, v_x2),
       "no against the committed anchor -- yes against two sibling artifacts",
       "Contains no R2 arm, so it cannot be tested against increment.json. "
       "Its overall_slope agrees with fairness_audit.json overall.cal_slope to "
       "1e-6 and with calibration.json continuous.calib_slope at its stored "
       "4 dp. The per-age-tertile slopes (including the youngest at "
       f"{cf['age_tertile_calibration'][0]['slope']:.6f}) reconcile with "
       "fairness_audit age_tertile -- cross-checked below.")

# per-tertile cross-check between the two calibration sources
fa_ter = fa["axes"]["age_tertile"]
name_map = {"age<=58": "age<=58", "age 58-67": "age_58-67", "age>67": "age>67"}
for t in cf["age_tertile_calibration"]:
    chk("phase2/calibration_flexible.json",
        f"age_tertile_calibration[{t['stratum']}].slope",
        f"fairness_audit age_tertile[{name_map[t['stratum']]}].cal_slope",
        fa_ter[name_map[t["stratum"]]]["cal_slope"], t["slope"])

# --- fairness_audit.json
v = chk("phase3/fairness_audit.json", "overall.r2", "r2_aug", A["r2_aug"],
        fa["overall"]["r2"])
record("phase3/fairness_audit.json", v,
       "yes -- overall.r2 is the augmented (clinical+CSF) model R2",
       f"n={fa['n']} matches anchor. overall.r2 reproduces the committed "
       "r2_aug, which anchors the whole subgroup table. Subgroup R2 and "
       "calibration slopes have no committed counterpart individually.")

# --- batch1_checks.json
# Not in the original figure brief, but it became a figure source: it is the
# ONLY artifact in data/processed carrying age-tertile dR2, which Figure 2
# needs. It is therefore audited here like any other source.
b1 = load("phase2/batch1_checks.json")
v_nb = chk("phase2/batch1_checks.json", "noise_block.r2_base", "r2_base",
           A["r2_base"], b1["noise_block"]["r2_base"])
# the age arm carries no baseline counterpart, so it is corroborated laterally
# against fairness_audit, which itself PASSES against the committed r2_aug
_fa_ter = fa["axes"]["age_tertile"]
_b1_map = {"age <= 58": "age<=58", "age 58-67": "age_58-67", "age > 67": "age>67"}
_age_v = []
for t in b1["age_tertile_dr2"]:
    k = _b1_map[t["stratum"]]
    assert t["n"] == _fa_ter[k]["n"], "tertile n mismatch between the two files"
    _age_v.append(chk("phase2/batch1_checks.json",
                      f"age_tertile_dr2[{t['stratum']}].r2_aug",
                      f"fairness_audit age_tertile[{k}].r2",
                      _fa_ter[k]["r2"], t["r2_aug"]))
_nb_dr2 = b1["noise_block"]["dr2"]
record("phase2/batch1_checks.json", worst(v_nb, *_age_v),
       "yes -- noise_block.r2_base vs anchor; age arm vs fairness_audit",
       "PARTIALLY SUPERSEDED, AND THE SCOPE MATTERS. "
       f"noise_floor_distribution.json states it replaces the single-draw "
       f"{_nb_dr2:.4f} figure in this file -- that is `noise_block.dr2` "
       f"({_nb_dr2:.15g}), so the supersession lands on the noise_block "
       "section ONLY. It does not touch age_tertile_dr2. "
       "The age arm is the only source of age-stratified increments in "
       "data/processed and is used for Figure 2. It is corroborated laterally: "
       "all three per-tertile r2_aug reproduce fairness_audit.json's "
       "age_tertile r2 to ~1e-16 at identical n (203/202/202), and "
       "fairness_audit itself PASSES against the committed r2_aug. The file's "
       "own noise_block.r2_base also reproduces the committed anchor exactly. "
       "Do NOT use noise_block for the null -- use "
       "noise_floor_distribution.json's 20 draws.")

# --- calibration.json
# Not in the original figure brief, but it became a figure source: it is the
# ONLY artifact in data/processed holding calibration CURVE COORDINATES, which
# Figure 3 needs. calibration_flexible.json carries slopes and a summary
# deviation but no coordinates of any kind.
_cc = cal["continuous"]["calibration_curve"]
_bin_sum = sum(b["n"] for b in _cc)
v_cal = chk("phase2/calibration.json", "continuous.calib_slope",
            "calibration_flexible overall_slope", cf["overall_slope"],
            cal["continuous"]["calib_slope"], stored_dp=4)
assert _bin_sum == A["n"], "calibration curve bins must partition the cohort"
record("phase2/calibration.json", v_cal,
       "partly -- calib_slope vs calibration_flexible overall_slope",
       "THE ONLY SOURCE OF CALIBRATION CURVE COORDINATES IN THE PROJECT -- no "
       "other artifact in data/processed holds observed-vs-predicted points of "
       "any kind. "
       f"Contains no R2 arm, so it cannot be tested against increment.json. "
       f"continuous.n = {cal['continuous']['n']} matches the anchor n, and its "
       f"{len(_cc)} decile bins sum to exactly {_bin_sum}, so the curve "
       "partitions the cohort. Its calib_slope agrees with "
       "calibration_flexible.json's overall_slope at its stored 4 dp "
       f"(delta {cal['continuous']['calib_slope'] - cf['overall_slope']:+.3e}). "
       "USED FOR FIGURE 3 PANEL A. The curve it provides is a BINNED DECILE "
       "curve; the 'flexible' calibration result exists only as "
       "flexible_mean_abs_deviation in calibration_flexible.json, with no "
       "coordinates anywhere in data/processed. The two must not be conflated "
       "and no smooth flexible fit is drawn in the figure set. Its `binary` "
       "section (n=548) is a different endpoint and is not used.")

# ================================================================ mr
# --- mr_power_bounds.json
pb = load("mr/mr_power_bounds.json")
v = chk("mr/mr_power_bounds.json", "gpnmb_positive_control_or",
        "poscontrol deCODE OR", P["decode_or"], pb["gpnmb_positive_control_or"])
record("mr/mr_power_bounds.json", v,
       "partly -- deCODE positive control only",
       "Carries the deCODE positive-control OR exactly. It stores NO UKB-PPP "
       f"positive-control field, so the UKB-PPP anchor (OR={P['ukbppp_or']}) "
       "cannot be checked against this file -- consistent with the gate "
       "deviation recorded in poscontrol_result.json, where GPNMB recovers on "
       "deCODE but not on UKB-PPP. The eight per-protein cells are the file's "
       "own estimates and are unanchorable individually.")

# --- mr_final_table.csv
mr = pd.read_csv(os.path.join(BASE, "mr/mr_final_table.csv"))
pb_cells = {(c["protein"], c["platform"]): c for c in pb["cells"]}
csv_checks = 0
for _, r in mr.iterrows():
    for plat, col in (("decode", "decode_or"), ("ukbppp", "ukbppp_or")):
        cell = pb_cells.get((r["protein"], plat))
        if cell is None or pd.isna(r[col]):
            continue
        chk("mr/mr_final_table.csv", f"{r['protein']}.{col}",
            f"mr_power_bounds cells[{r['protein']}|{plat}].or",
            cell["or"], float(r[col]), stored_dp=3)
        csv_checks += 1
record("mr/mr_final_table.csv", "UNANCHORABLE", "no against the MR anchor",
       f"Contains no GPNMB row, so neither positive-control OR appears in it "
       f"and it cannot be tested against poscontrol_result.json. Its {csv_checks} "
       "protein-by-platform ORs are 3 dp restatements of mr_power_bounds.json "
       "and agree at that precision (cross-check below), which makes the two "
       "MR files mutually consistent but jointly unanchored to the positive "
       "control.")

# ---------------------------------------------------------------- report
df = pd.DataFrame(rows)
fdf = pd.DataFrame(files)
counts = fdf["verdict"].value_counts().to_dict()

L = []
L.append("# Artifact integrity sweep")
L.append("")
L.append("Every artifact named in the figure brief, tested against the committed "
         "anchors before any of it is plotted. Reports only -- nothing here was "
         "fixed or regenerated.")
L.append("")
L.append("Generated by `analysis/figures/artifact_integrity_sweep.py`. All deltas "
         "are computed at runtime from the artifacts themselves.")
L.append("")
L.append("## Anchors")
L.append("")
L.append("| anchor | field | value |")
L.append("|---|---|---|")
for k in ("r2_base", "r2_aug", "dr2"):
    L.append(f"| `phase2/increment.json` `rq1_csf` | {k} | {A[k]:.17g} |")
L.append(f"| `phase2/increment.json` `rq1_csf` | n | {A['n']} |")
L.append(f"| `mr/poscontrol_result.json` | deCODE OR | {P['decode_or']:.6f} |")
L.append(f"| `mr/poscontrol_result.json` | UKB-PPP OR | {P['ukbppp_or']:.6f} |")
L.append("")
L.append("## Verdict key")
L.append("")
L.append("| verdict | meaning |")
L.append("|---|---|")
L.append("| **ANCHOR** | this file defines the anchor |")
L.append("| **PASS** | has an overlapping arm; agrees within 1e-6 |")
L.append("| **ROUNDED** | has an overlapping arm stored at reduced precision; "
         "agrees at its stored precision, fails a strict 1e-6 assert |")
L.append("| **FAIL** | has an overlapping arm; does not agree |")
L.append("| **UNANCHORABLE** | no arm overlaps any committed value -- cannot be "
         "checked at all |")
L.append("")
L.append(f"**Summary of {len(fdf)} artifacts:** "
         + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) + ".")
L.append("")
L.append("## Per-artifact verdict")
L.append("")
L.append("| artifact | anchorable? | verdict |")
L.append("|---|---|---|")
for r in files:
    L.append(f"| `{r['artifact']}` | {r['anchorable']} | **{r['verdict']}** |")
L.append("")
L.append("## Value-level checks")
L.append("")
L.append("| artifact | field | compared against | expected | actual | delta | "
         "<=1e-6 | verdict |")
L.append("|---|---|---|---|---|---|---|---|")
for r in rows:
    L.append(f"| `{r['artifact'].split('/')[-1]}` | `{r['field']}` | {r['anchor']} | "
             f"{r['anchor_value']:.10g} | {r['actual']:.10g} | {r['delta']:+.3e} | "
             f"{'yes' if r['strict_1e6'] else 'no'} | **{r['verdict']}** |")
L.append("")
L.append("## Notes per artifact")
L.append("")
for r in files:
    L.append(f"**`{r['artifact']}` -- {r['verdict']}**")
    L.append("")
    L.append(r["note"])
    L.append("")

L.append("## The one failure, in full")
L.append("")
L.append("`phase2/enriched_baseline_check.json` and `phase2/estimand_audit.json` "
         "score the same four arms on the same declared cohort with the same "
         "declared cross-validation configuration and the same declared column "
         "lists. They disagree on all four.")
L.append("")
ea_arms = {k: v for k, v in ea["results"].items()
           if "|" in k and not k.startswith("strata")}
ebc_arms = {f"{r['outcome']}|{r['baseline']}": r for r in ebc["results"]}
L.append("| arm | field | estimand_audit | enriched_baseline_check | difference |")
L.append("|---|---|---|---|---|")
for arm in sorted(set(ea_arms) & set(ebc_arms)):
    for f in ("r2_base", "r2_aug", "dr2"):
        L.append(f"| `{arm}` | {f} | {ea_arms[arm][f]:.8f} | "
                 f"{ebc_arms[arm][f]:.8f} | {ea_arms[arm][f] - ebc_arms[arm][f]:+.8f} |")
L.append("")
L.append(f"Declared `n` identical: {ea['n'] == ebc['n']} "
         f"({ea['n']} vs {ebc['n']}). "
         f"Declared cross-validation config identical: {ea['cv'] == ebc['cv_config']}. "
         f"Thin column list identical: {ea['thin'] == ebc['thin_cols']}. "
         f"Enriched column list identical: {ea['enriched'] == ebc['enriched_cols']}.")
L.append("")
L.append(f"`enriched_baseline_check.json` control-arm dr2 = "
         f"{ebc_thin['dr2']:.8f}, which is exactly "
         f"`increment.json rq1_csf.{coincide[0] if coincide else '?'}` "
         f"= {inc['rq1_csf'][coincide[0]] if coincide else float('nan'):.8f}. "
         "The disagreement is therefore an estimator substitution -- the "
         "fixed-alpha permutation estimator reported under a nested-CV label -- "
         "not run-to-run variation. It is one-directional: the file's dr2 is "
         "lower on all four arms.")
L.append("")

L.append("## What this means for the figure set")
L.append("")
L.append("1. `enriched_baseline_check.json` is excluded. Figure 1's enriched "
         "baseline is taken from `estimand_audit.json`, which reproduces the "
         "committed control arm exactly.")
L.append("1b. `batch1_checks.json` was not in the original brief but became a "
         "figure source, since it is the only artifact holding age-tertile "
         "dR2. It is partially superseded and the SCOPE is section-level: "
         "`noise_floor_distribution.json` replaces its `noise_block` figure "
         "only, and leaves `age_tertile_dr2` untouched. The age arm is "
         "corroborated against `fairness_audit.json` to ~1e-16 at identical n. "
         "Figure 2 uses the age arm; the null everywhere comes from "
         "`noise_floor_distribution.json`.")
L.append("2. Four artifacts carry a deliberate anchor arm and pass to 1e-6: "
         "`bootstrap_ci`, `estimand_audit`, `phase1_visit_split`, "
         "`rank11_controls`, plus `fairness_audit` on the augmented R2. These "
         "are the best-evidenced files in the set.")
L.append("3. Three artifacts hold 4 dp restatements of the anchor and are "
         "consistent at that precision but cannot support a stricter claim: "
         "`noise_floor_distribution`, `iecv_site`, `gba_scale_anchor`.")
L.append("4. Three artifacts are UNANCHORABLE outright -- "
         "`hlavnicka_head2head`, `trial_enrichment_ci`, `mr_final_table.csv`. "
         "They are not wrong; they are uncheckable. Nothing in the manuscript "
         "should lean on them the way it can lean on the anchored files, and "
         "the AUC head-to-head in particular carries the paper's negative "
         "result on the dichotomised endpoint.")
L.append("5. The MR arm is anchored only on deCODE. The UKB-PPP positive "
         "control is absent from `mr_power_bounds.json` because GPNMB does not "
         "recover there -- so the UKB-PPP side of the PDGFRB null is ungated, "
         "exactly as `poscontrol_result.json` records. Figure 6 must mark it.")
L.append("6. Two quantities the figures rely on have no committed counterpart "
         "anywhere: the 20-draw random-feature null and the per-site IECV "
         "results. Both rest entirely on the correctness of their own run.")
L.append("")

text = "\n".join(L)
with open(os.path.join(OUT, "ARTIFACT_INTEGRITY_SWEEP.md"), "w") as fh:
    fh.write(text + "\n")

print(fdf.to_string(index=False))
print()
print(df[["artifact", "field", "delta", "strict_1e6", "verdict"]].to_string(index=False))
print()
print("verdict counts:", counts)
