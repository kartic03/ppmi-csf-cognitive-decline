"""FIG 2 - Where the increment holds and where it fails. (CENTREPIECE)

Panel A: forest of dR2 by stratum, line at zero, failures placed where they
         cannot be missed rather than buried at the bottom.
Panel B: the DICHOTOMISED conversion endpoint, as a delta in AUC. Separate
         sub-panel, separate axis. An AUC delta is not an R2 delta and the two
         must never share a scale.

Every number is read from JSON at runtime. Nothing is hardcoded.

PROVENANCE
  overall dR2 .............. phase2/increment.json rq1_csf
  age tertiles ............. phase2/batch1_checks.json age_tertile_dr2
  visit / censoring strata . phase2/phase1_visit_split.json results.eb_slope.cells
  conversion endpoint AUC .. phase2/hlavnicka_head2head.json arms

ON batch1_checks.json  (see revision/figures/ARTIFACT_INTEGRITY_SWEEP.md)
  It is the ONLY artifact in data/processed carrying age-tertile dR2.
  It is PARTIALLY superseded and the scope is section-level:
  noise_floor_distribution.json replaces its `noise_block` single-draw figure
  (dr2 = -0.009283) ONLY. age_tertile_dr2 is untouched by that supersession.
  The age arm is corroborated laterally -- all three per-tertile r2_aug
  reproduce fairness_audit.json's age_tertile r2 to ~1e-16 at identical n.

ON hlavnicka_head2head.json
  UNANCHORABLE: different endpoint, different sample (n=324), different metric.
  No arm overlaps any committed value. Its own brackets are cross-seed ranges,
  NOT confidence intervals, and are therefore NOT drawn as intervals here.
"""
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_figure_style, bbox_check, panel_letter, set_frame

# Repo-relative. This file lives at <repo>/revision/figures/, so the repo
# root is two levels up. That holds in the WSL repro environment and in a
# clean checkout, so a figure is never built against a path that exists on
# only one machine.
OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(OUT))
BASE = os.path.join(REPO, "data", "processed")
assert os.path.isdir(BASE), f"artifact directory not found: {BASE}"
os.makedirs(OUT, exist_ok=True)


def load(rel):
    with open(os.path.join(BASE, rel)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- data
inc = load("phase2/increment.json")
b1 = load("phase2/batch1_checks.json")
pv = load("phase2/phase1_visit_split.json")
hl = load("phase2/hlavnicka_head2head.json")
fa = load("phase3/fairness_audit.json")

rq1 = inc["rq1_csf"]

# integrity gates -- fail loudly rather than plot a drifted number
_fa_ter = fa["axes"]["age_tertile"]
_map = {"age <= 58": "age<=58", "age 58-67": "age_58-67", "age > 67": "age>67"}
for t in b1["age_tertile_dr2"]:
    k = _map[t["stratum"]]
    assert t["n"] == _fa_ter[k]["n"]
    assert abs(t["r2_aug"] - _fa_ter[k]["r2"]) < 1e-9, (
        f"batch1 age tertile {t['stratum']} r2_aug no longer matches fairness_audit")
_fs = pv["results"]["eb_slope"]["full_sample"]
assert abs(_fs["dr2"] - rq1["dr2"]) < 1e-9, "visit-split pooled arm drifted from anchor"

cells = {c["cell"]: c for c in pv["results"]["eb_slope"]["cells"]}
ter = {t["stratum"]: t for t in b1["age_tertile_dr2"]}

# VISIT CELLS ARE THE PHASE-STRATIFIED ONES ONLY.
# The file also carries pooled "(all phases)" cells. They are NOT plotted here:
# "few-visit (all phases)" (n=344) is exactly "phase 1, few-visit" (n=219) plus
# "phase 2, few-visit" (n=125), so plotting it beside the phase-2 cell would
# count those 125 subjects twice AND reintroduce the phase confound that the
# phase-1 split exists to break. "many-visit (all phases)" is numerically
# identical to "phase 1, many-visit" (phase 2 contributes no many-visit
# subjects, n=0, "too small to score"), but is sourced from the phase-
# stratified set anyway so all three visit rows share one basis.
# The pooled view belongs in a supplementary panel labelled as confounded.
V_FEW1 = cells["phase 1, few-visit"]
V_MANY1 = cells["phase 1, many-visit"]
V_FEW2 = cells["phase 2, few-visit"]

# Rows are ORDERED SO THE FAILURES LEAD, not so the successes do. The overall
# result sits at the top as the reference; the two negative cells follow
# immediately beneath it; the positive strata ascend below them, so the
# largest positive sits furthest from the reference row.
# (label, n, dr2, group, source_tag)
rows = [
    ("Overall (pre-registered)", rq1["n"], rq1["dr2"], "overall", "increment"),

    ("Youngest tertile (age \u2264 58)", ter["age <= 58"]["n"],
     ter["age <= 58"]["dr2"], "age", "batch1"),
    ("Administratively censored\n(phase 2, few-visit)", V_FEW2["n"],
     V_FEW2["dr2"], "visit", "visitsplit"),

    ("Completers\n(phase 1, many-visit)", V_MANY1["n"], V_MANY1["dr2"],
     "visit", "visitsplit"),
    ("Middle tertile (age 58\u201367)", ter["age 58-67"]["n"],
     ter["age 58-67"]["dr2"], "age", "batch1"),
    ("Oldest tertile (age > 67)", ter["age > 67"]["n"],
     ter["age > 67"]["dr2"], "age", "batch1"),
    ("Early exit\n(phase 1, few-visit)", V_FEW1["n"], V_FEW1["dr2"],
     "visit", "visitsplit"),
]

# --- mutual-exclusivity gate -------------------------------------------
# Each stratification family must PARTITION the cohort, never over-cover it.
# A double-counted cell (e.g. plotting a pooled cell beside one of its parts)
# fails here rather than rendering.
_N = rq1["n"]
for _grp in ("age", "visit"):
    _tot = sum(r[1] for r in rows if r[3] == _grp)
    assert _tot <= _N, (f"{_grp} rows sum to n={_tot} > cohort {_N} -- strata "
                        "overlap; a pooled cell is being plotted beside its parts")
assert sum(r[1] for r in rows if r[3] == "visit") == _N, "visit cells must partition"
assert sum(r[1] for r in rows if r[3] == "age") == _N, "age tertiles must partition"
assert V_FEW1["n"] + V_FEW2["n"] == cells["few-visit (all phases)"]["n"], (
    "pooled few-visit cell is no longer the sum of its phase parts")

# panel B: AUC on the dichotomised endpoint -- clinical vs clinical+CSF
arms = hl["arms"]
aucB = [("clinical (10)", arms["clinical (10)"]),
        ("clinical + CSF", arms["clinical + CSF"]),
        ("enriched (14)", arms["enriched (14)"]),
        ("enriched + CSF", arms["enriched + CSF"])]
d_thin = arms["clinical + CSF"]["auc"] - arms["clinical (10)"]["auc"]
d_rich = arms["enriched + CSF"]["auc"] - arms["enriched (14)"]["auc"]

# ---------------------------------------------------------------- style
apply_figure_style(sizes=(8, 7, 6))
POS = "#0072B2"     # increment above zero
NEG = "#D55E00"     # increment below zero (alarm hue, reserved)
OVER = "#333333"    # the overall reference row
GREY = "#8C8C8C"

fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.42], hspace=0.55,
                      left=0.335, right=0.90, top=0.915, bottom=0.115)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])

# ---------------------------------------------------------------- panel A
y = np.arange(len(rows))[::-1]
axA.axvline(0.0, color="#4D4D4D", lw=1.0, zorder=2)

for yi, (lab, n, dr, grp, src) in zip(y, rows):
    neg = dr < 0
    col = OVER if grp == "overall" else (NEG if neg else POS)
    ms = 8.0 if grp == "overall" else 6.5
    axA.plot([0, dr], [yi, yi], color=col, lw=1.2, alpha=0.45, zorder=3)
    axA.plot([dr], [yi], marker="o", ms=ms, color=col, zorder=4, clip_on=False)
    off = -9 if neg else 9
    ha = "right" if neg else "left"
    axA.annotate(f"{dr:+.4f}", xy=(dr, yi), xytext=(off, 0),
                 textcoords="offset points", ha=ha, va="center", color=col,
                 fontsize=mpl.rcParams["legend.fontsize"],
                 fontweight="bold" if grp == "overall" or neg else "normal")

axA.set_yticks(y)
axA.set_yticklabels([f"{lab}\nn = {n}" for lab, n, *_ in rows])
axA.set_xlabel("$\\Delta R^2$ from adding the CSF block, continuous "
               "decline-rate outcome\n"
               "\u2190 CSF hurts        0        CSF helps \u2192")
axA.set_xlim(-0.105, 0.165)
axA.set_ylim(-0.7, len(rows) - 0.3)
axA.set_title("The CSF increment reverses in the youngest tertile and in the\n"
              "administratively censored cell", loc="left")
set_frame(axA, "open")
axA.tick_params(axis="y", length=0)
axA.spines["left"].set_visible(False)

# Call out the two failures. Both notes sit to the RIGHT of the zero line, in
# the empty band between the negative marks and the positive ones, so neither
# collides with its own value label (which is placed left of a negative mark).
for lab_needle, note in (("Youngest", "CSF makes prediction worse here"),
                         ("Administratively",
                          "increment absent under administrative censoring")):
    i = next(k for k, r in enumerate(rows) if r[0].startswith(lab_needle))
    # No leader line. The note sits on its own row in otherwise empty space, so
    # the row is unambiguous without one -- and any leader long enough to reach
    # from the mark to the text ends up terminating inside the text box.
    axA.annotate(note, xy=(0.012, y[i]), ha="left", va="center", color=NEG,
                 fontsize=mpl.rcParams["xtick.labelsize"])

# Direction of goodness is carried by the axis label itself (set above), so it
# cannot collide with it. Left of zero = CSF hurts; right of zero = CSF helps.
panel_letter(axA, "a", dx=-0.395, dy=1.10)

# ---------------------------------------------------------------- panel B
yB = np.arange(len(aucB))[::-1]
for yi, (lab, a) in zip(yB, aucB):
    has_csf = "CSF" in lab
    col = POS if has_csf else GREY
    axB.plot([a["auc"]], [yi], marker="s" if has_csf else "o", ms=6.0,
             mfc=col if has_csf else "white", mec=col, mew=1.4, zorder=3,
             clip_on=False)
    axB.annotate(f"{a['auc']:.3f}", xy=(a["auc"], yi), xytext=(9, 0),
                 textcoords="offset points", ha="left", va="center", color=col,
                 fontsize=mpl.rcParams["legend.fontsize"])

axB.set_yticks(yB)
axB.set_yticklabels([lab for lab, _ in aucB])
axB.set_xlabel("AUC on the dichotomised conversion endpoint "
               "(a DIFFERENT metric \u2014 not $\\Delta R^2$)")
axB.set_xlim(0.60, 0.70)
axB.set_ylim(-0.75, len(aucB) - 0.35)
axB.set_title("On the dichotomised endpoint the same CSF block adds nothing",
              loc="left")
set_frame(axB, "open")
axB.tick_params(axis="y", length=0)
axB.spines["left"].set_visible(False)

# The two deltas are bracketed against the pairs they compare, at the right of
# the panel, where there is empty space and no tick label to collide with.
for (base_lab, csf_lab, d) in (("clinical (10)", "clinical + CSF", d_thin),
                               ("enriched (14)", "enriched + CSF", d_rich)):
    i_b = next(k for k, (l, _) in enumerate(aucB) if l == base_lab)
    i_c = next(k for k, (l, _) in enumerate(aucB) if l == csf_lab)
    yb, yc = yB[i_b], yB[i_c]
    # bracket sits just right of ITS OWN pair, not at a shared far-right x,
    # so it cannot be read as spanning the wrong rows
    xbr = max(dict(aucB)[base_lab]["auc"], dict(aucB)[csf_lab]["auc"]) + 0.0125
    axB.plot([xbr, xbr + 0.003, xbr + 0.003, xbr], [yb, yb, yc, yc],
             color=NEG, lw=0.7, clip_on=False, zorder=3)
    axB.annotate(f"{d:+.3f} AUC", xy=(xbr + 0.005, (yb + yc) / 2),
                 ha="left", va="center", color=NEG,
                 fontsize=mpl.rcParams["legend.fontsize"], clip_on=False)
panel_letter(axB, "b", dx=-0.395, dy=1.22)

fig.text(0.335, 0.008,
         f"a: n = {rq1['n']} CSF-complete. Strata are scored on shared "
         f"out-of-fold predictions and carry no CIs.   "
         f"b: n = {hl['n']}, {hl['events']} events.",
         fontsize=mpl.rcParams["xtick.labelsize"], color="#4D4D4D", ha="left")

assert not bbox_check(fig), "layout QA gate failed -- fix before saving"
fig.savefig(os.path.join(OUT, "fig2_where_it_holds_and_fails.png"), dpi=300)
fig.savefig(os.path.join(OUT, "fig2_where_it_holds_and_fails.pdf"))

print("FIG 2 -- value : source")
for lab, n, dr, grp, src in rows:
    tag = {"increment": "phase2/increment.json rq1_csf",
           "batch1": "phase2/batch1_checks.json age_tertile_dr2",
           "visitsplit": "phase2/phase1_visit_split.json results.eb_slope.cells"}[src]
    print(f"  a {lab.replace(chr(10), ' '):46} n={n:4d} dr2={dr:+.8f}   <- {tag}")
for lab, a in aucB:
    print(f"  b {lab:46} auc={a['auc']:.8f}   <- phase2/hlavnicka_head2head.json arms")
print(f"  b delta AUC clinical  {d_thin:+.8f} (derived)")
print(f"  b delta AUC enriched  {d_rich:+.8f} (derived)")
