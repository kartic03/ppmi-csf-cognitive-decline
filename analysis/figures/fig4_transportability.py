"""FIG 4 - Internal-external cross-validation across enrolment sites.

Panel A: R2 for the clinical and clinical+CSF models under three resampling
         schemes -- leave-one-site-out, a matched random-cluster control, and
         standard k-fold -- so the reader sees overall performance drop.
Panel B: the corresponding dR2, so the reader sees that the INCREMENT largely
         holds even as overall performance falls.
Panel C: random-effects calibration slope with its 95% prediction interval,
         REAL SITES ONLY (k = 10).

Every number is read from JSON at runtime. Nothing is hardcoded.

PROVENANCE  (all from phase2/iecv_site.json)
  leave-one-site-out ....... site.pooled_r2_base / pooled_r2_aug / pooled_dr2
  random-cluster control ... random_control.mean_r2_base / mean_r2_aug / mean_dr2
  standard k-fold .......... reference_random_5fold
  calibration slope ........ site.calibration_meta_real_sites_only  (k = 10)

THREE REPORTING RULES ENFORCED HERE
  1. This is NOT external validation. It is internal-external cross-validation
     across enrolment sites: it shares PPMI's protocol, era and inclusion
     criteria. The title says so and the word "external validation" is never
     used alone.
  2. Calibration uses calibration_meta_real_sites_only (k = 10). The k = 11
     figure folds in cluster '-1.0', which is 38 small sites glued together --
     not a site -- and at n = 289 it would dominate the pooling. A runtime
     assert below refuses the k = 11 value.
  3. I^2 = 0% is annotated as UNDERPOWERED at k = 10, never as evidence of
     homogeneity.

PER-SITE R2 IS DELIBERATELY NOT PLOTTED IN THE MAIN PANELS.
  The 10 real sites have n = 24-75. Per-site R2 at those cluster sizes is
  noise, and the file says so itself. It is summarised as a range in the
  footnote rather than drawn as if interpretable.
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
iv = load("phase2/iecv_site.json")
inc = load("phase2/increment.json")

site = iv["site"]
rc = iv["random_control"]
ref = iv["reference_random_5fold"]
cm = site["calibration_meta_real_sites_only"]          # k = 10, PRIMARY
rq1 = inc["rq1_csf"]

# --- runtime gates -------------------------------------------------------
# 1. the calibration meta-analysis MUST be the real-sites-only one. If this
#    ever silently becomes the k=11 pooled-cluster figure, the figure stops.
assert cm["k"] == 10, f"calibration meta must be k=10 real sites, got k={cm['k']}"
assert cm is not site["calibration_meta"], "must not use the k=11 pooled figure"
assert site["calibration_meta"]["k"] == 11, "k=11 arm is the one being excluded"

# 2. the clusters must partition the cohort, and the pooled small-site cluster
#    must be the one excluded from calibration -- k_real = k_all - 1
_clusters = site["per_cluster"]
_real = [c for c in _clusters if c["cluster"] != "-1.0"]
_pooled = [c for c in _clusters if c["cluster"] == "-1.0"]
assert len(_pooled) == 1, "expected exactly one pooled small-site cluster"
assert len(_real) == cm["k"], (
    f"{len(_real)} real sites but calibration k={cm['k']}")
assert sum(c["n"] for c in _clusters) == rq1["n"], "clusters must partition cohort"
assert site["n_pooled"] == rq1["n"]

# 3. the k-fold reference arm must still be the committed anchor at its stored
#    precision (it is a 4 dp restatement -- the ROUNDED entry in the sweep)
for _k, _anchor in (("r2_base", rq1["r2_base"]), ("r2_aug", rq1["r2_aug"]),
                    ("dr2", rq1["dr2"])):
    _dp = len(str(ref[_k]).split(".")[-1])
    assert round(_anchor, _dp) == round(ref[_k], _dp), (
        f"reference_random_5fold.{_k} no longer restates the committed anchor")

# 4. dR2 must equal aug - base within each scheme
for _lab, _b, _a, _d in (("site", site["pooled_r2_base"], site["pooled_r2_aug"],
                          site["pooled_dr2"]),
                         ("random", rc["mean_r2_base"], rc["mean_r2_aug"],
                          rc["mean_dr2"])):
    assert abs((_a - _b) - _d) < 1e-9, f"{_lab}: dr2 != aug - base"

schemes = [
    ("Leave-one-site-out\n(internal-external CV)", site["pooled_r2_base"],
     site["pooled_r2_aug"], site["pooled_dr2"], True),
    ("Matched random-cluster\ncontrol (5 seeds)", rc["mean_r2_base"],
     rc["mean_r2_aug"], rc["mean_dr2"], False),
    ("Standard k-fold\n(reference)", ref["r2_base"], ref["r2_aug"],
     ref["dr2"], False),
]
seed_dr2 = np.array([r["pooled_dr2"] for r in rc["runs"]], float)
n_real = len(_real)
n_min = min(c["n"] for c in _real)
n_max = max(c["n"] for c in _real)

# ---------------------------------------------------------------- style
apply_figure_style(sizes=(8, 7, 6))
CLIN = "#8C8C8C"
CSF = "#0072B2"
FOCAL = "#0072B2"
ALARM = "#D55E00"
IDEAL = "#666666"

# LAYOUT: two gridspecs rather than one 2-row grid, for the same reason as
# Figure 3. In a shared grid the space between panel a's xlabel and the two-line
# titles beneath it is hspace -- a fraction of the MEAN axes height -- so it
# changes whenever either row is resized, and it closed up here. Declaring the
# two row bands outright fixes the gap at ~0.8 in and keeps it there.
#
# Panel a's band is also deliberately SHORTER than an even split. It holds three
# lollipop rows; given more height they simply drift apart, which reads as three
# unrelated results rather than one comparison.
#
# Both rows share left=0.225, so panel a and panel b are left-aligned on the
# same edge and their y labels sit in one column.
fig = plt.figure(figsize=(7.1, 6.3))
gsTop = fig.add_gridspec(1, 1, left=0.225, right=0.975, top=0.915, bottom=0.585)
gsBot = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.42,
                         left=0.225, right=0.975, top=0.440, bottom=0.190)
axA = fig.add_subplot(gsTop[0, 0])
axB = fig.add_subplot(gsBot[0, 0])
axC = fig.add_subplot(gsBot[0, 1])

# ---------------------------------------------------------------- panel A
yA = np.arange(len(schemes))[::-1]
for yi, (lab, rb, ra, dr, focal) in zip(yA, schemes):
    axA.plot([rb, ra], [yi, yi], color=CSF, lw=1.4, alpha=0.45, zorder=2)
    axA.plot([rb], [yi], marker="o", ms=6.5, mfc="white", mec=CLIN, mew=1.4,
             zorder=3, clip_on=False)
    axA.plot([ra], [yi], marker="o", ms=6.5, color=CSF, zorder=3, clip_on=False)
    axA.annotate(f"{ra:.4f}", xy=(ra, yi), xytext=(8, 0),
                 textcoords="offset points", ha="left", va="center", color=CSF,
                 fontsize=mpl.rcParams["legend.fontsize"])
    axA.annotate(f"{rb:.4f}", xy=(rb, yi), xytext=(-8, 0),
                 textcoords="offset points", ha="right", va="center", color=CLIN,
                 fontsize=mpl.rcParams["legend.fontsize"])

axA.set_yticks(yA)
axA.set_yticklabels([s[0] for s in schemes])
axA.set_xlabel("Out-of-fold $R^2$ (continuous decline-rate outcome)")
axA.set_xlim(0.0, 0.163)
axA.set_ylim(-0.62, len(schemes) - 0.32)
axA.set_title("Overall performance drops when whole enrolment sites are held out",
              loc="left")
set_frame(axA, "open")
axA.tick_params(axis="y", length=0)
axA.spines["left"].set_visible(False)

_top = yA[0]
axA.annotate("clinical only", xy=(schemes[0][1], _top), xytext=(0, 15),
             textcoords="offset points", ha="center", va="bottom", color=CLIN,
             fontsize=mpl.rcParams["legend.fontsize"])
axA.annotate("clinical + CSF", xy=(schemes[0][2], _top), xytext=(0, 15),
             textcoords="offset points", ha="center", va="bottom", color=CSF,
             fontsize=mpl.rcParams["legend.fontsize"])

# ---------------------------------------------------------------- panel B
yB = np.arange(len(schemes))[::-1]
axB.axvline(0.0, color="#4D4D4D", lw=1.0, zorder=1)
for yi, (lab, rb, ra, dr, focal) in zip(yB, schemes):
    axB.plot([0.0, dr], [yi, yi], color=FOCAL, lw=1.2, alpha=0.45, zorder=2)
    axB.plot([dr], [yi], marker="o", ms=7.5 if focal else 6.0, color=FOCAL,
             zorder=3, clip_on=False)
    axB.annotate(f"{dr:+.4f}", xy=(dr, yi), xytext=(0, 11),
                 textcoords="offset points", ha="center", va="bottom",
                 color=FOCAL, fontsize=mpl.rcParams["legend.fontsize"],
                 fontweight="bold" if focal else "normal")

# Panel b carries its OWN row labels, worded to match panel a's rows. It sits
# under a full-width panel a rather than beside it, so row identity cannot be
# inherited by alignment -- and a panel cropped out of the figure, or scaled
# down in a journal column, must still say which design each bar belongs to.
axB.set_yticks(yB)
axB.set_yticklabels(["Leave-one-\nsite-out", "Matched random-\ncluster control",
                     "Standard\nk-fold"])
axB.set_xlim(-0.012, 0.088)
axB.set_ylim(-0.62, len(schemes) - 0.32)
axB.set_xlabel("$\\Delta R^2$ from the CSF block")
axB.set_title("...but the increment\nlargely holds", loc="left")
set_frame(axB, "open")
axB.tick_params(axis="y", length=0)
axB.spines["left"].set_visible(False)

# ---------------------------------------------------------------- panel C
axC.axvline(1.0, color=IDEAL, lw=1.0, ls=(0, (4, 2)), zorder=1)
yc = 0.0
axC.plot([cm["pi_lo"], cm["pi_hi"]], [yc, yc], color=FOCAL, lw=1.4, alpha=0.40,
         solid_capstyle="butt", zorder=2)
for xb in (cm["pi_lo"], cm["pi_hi"]):
    axC.plot([xb, xb], [yc - 0.075, yc + 0.075], color=FOCAL, lw=1.2, alpha=0.65,
             zorder=2)
axC.errorbar([cm["estimate"]], [yc],
             xerr=[[cm["estimate"] - cm["ci_lo"]], [cm["ci_hi"] - cm["estimate"]]],
             fmt="o", ms=7.5, color=FOCAL, ecolor=FOCAL, elinewidth=1.8,
             capsize=3.5, capthick=1.8, zorder=4)

axC.annotate(f"{cm['estimate']:.3f}", xy=(cm["estimate"], yc), xytext=(0, 12),
             textcoords="offset points", ha="center", va="bottom", color=FOCAL,
             fontsize=mpl.rcParams["legend.fontsize"], fontweight="bold")

# The two intervals must be SELF-DESCRIBING: thick inner = 95% CI on the
# pooled estimate, thin outer = 95% prediction interval for a NEW site. A
# drawn key beneath the row shows the geometry itself, so the reader never has
# to match a bare number to a mark. (Labels placed at the interval ENDS
# collided with the estimate marker and the ideal line.)
# NO separate key row: a second drawn interval beneath the data row reads as a
# second RESULT, not as a legend. The two intervals are named in text keyed to
# their line weights instead.
# Anchored in AXES fraction and wrapped to three short lines. Anchored to the
# prediction-interval end in DATA coordinates, as it was, the single long line
# started at x = pi_lo and ran rightwards across the ideal=1 rule and out past
# the axes -- the collision gate does not catch either, because it exempts
# full-span reference rules and only tests containment against the whole saved
# canvas, not the panel. Axes-fraction anchoring cannot overflow the panel, and
# it aligns this key with the I2 note below it on the same left edge.
axC.annotate(f"thick: 95% CI {cm['ci_lo']:.3f}\u2013{cm['ci_hi']:.3f}\n"
             f"thin: 95% prediction interval\n"
             f"    for a new site {cm['pi_lo']:.3f}\u2013{cm['pi_hi']:.3f}",
             xy=(0.02, 0.60), xycoords="axes fraction",
             ha="left", va="top", color=FOCAL,
             fontsize=mpl.rcParams["xtick.labelsize"], linespacing=1.5)
axC.annotate("ideal = 1", xy=(1.0, 0.40), xytext=(4, 0),
             textcoords="offset points", ha="left", va="center", color=IDEAL,
             fontsize=mpl.rcParams["xtick.labelsize"])
# Readouts live INSIDE the axes: panel c's single row leaves the whole upper
# and lower band empty, and text parked below the axes collides with the
# figure footnote.
axC.annotate(f"$I^2$ = {cm['I2_percent']:.0f}% here is UNDERPOWERED\n"
             f"at k = {cm['k']} \u2014 not evidence\nof homogeneity",
             xy=(0.02, 0.28), xycoords="axes fraction", ha="left", va="top",
             color=ALARM, fontsize=mpl.rcParams["xtick.labelsize"],
             linespacing=1.5)

axC.set_yticks([])
axC.set_ylim(-1.75, 0.60)
axC.set_xlim(0.40, 1.19)
axC.set_xlabel("Random-effects calibration slope")
axC.set_title(f"Calibration across sites\n(real sites only, k = {cm['k']})",
              loc="left")
set_frame(axC, "open")
axC.spines["left"].set_visible(False)

# Panel letters in FIGURE coordinates -- see the note in fig3_calibration.py.
# Through ax.transAxes each letter needed its own dx to clear that panel's y
# labels (-0.305 for a against -0.10 for b and c), which is exactly how they
# came to sit at three different distances from their titles.
_L = dict(fontsize=mpl.rcParams["font.size"] + 1, fontweight="bold",
          va="top", ha="left")
fig.text(0.026, 0.988, "a", **_L)
fig.text(0.026, 0.520, "b", **_L)
fig.text(0.612, 0.520, "c", **_L)

# Footnote is WRAPPED. A single long line forces savefig's tight bbox to
# expand the canvas well past the intended column width.
fig.text(0.026, 0.075,
         f"n = {rq1['n']} across {len(_clusters)} clusters. Per-site $R^2$ is not "
         f"shown: the {n_real} real sites have n = {n_min}\u2013{n_max}, where it is "
         f"noise.\nThe pooled small-site cluster (n = {_pooled[0]['n']}, 38 sites) "
         f"is kept in the pooled $R^2$ but excluded from the calibration "
         f"meta-analysis.",
         fontsize=mpl.rcParams["xtick.labelsize"], color="#4D4D4D", ha="left",
         va="top", linespacing=1.5)

assert not bbox_check(fig), "layout QA gate failed -- fix before saving"
fig.savefig(os.path.join(OUT, "fig4_transportability.png"), dpi=300)
fig.savefig(os.path.join(OUT, "fig4_transportability.pdf"))

print("FIG 4 -- value : source   (all phase2/iecv_site.json)")
for lab, rb, ra, dr, _ in schemes:
    print(f"  a/b {lab.replace(chr(10),' '):44} base={rb:.10f} aug={ra:.10f} dr2={dr:+.10f}")
print(f"  c calibration slope k={cm['k']}  est={cm['estimate']:.10f} "
      f"CI=({cm['ci_lo']:.10f},{cm['ci_hi']:.10f})")
print(f"  c prediction interval        ({cm['pi_lo']:.10f},{cm['pi_hi']:.10f})")
print(f"  c I2={cm['I2_percent']}%  tau2={cm['tau2']}  k={cm['k']}  "
      f"<- calibration_meta_real_sites_only (k=11 arm NOT used)")
print(f"  footnote: {n_real} real sites, n {n_min}-{n_max}; "
      f"pooled cluster n={_pooled[0]['n']}")
print(f"  random-control per-seed dr2: {[round(float(v), 6) for v in seed_dr2]}")
