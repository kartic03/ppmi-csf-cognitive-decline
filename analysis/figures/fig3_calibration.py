"""FIG 3 - Calibration.

Panel A: BINNED DECILE calibration curve (observed vs predicted) against the
         45-degree ideal. These are ten decile-bin MEANS, not individual
         observations, and the panel says so.
Panel B: per-age-tertile calibration slopes against the ideal slope of 1.
Panel C: the same three tertiles' dR2 from Figure 2, so the youngest-tertile
         failure is visibly the SAME subgroup failing on two independent axes.

Every number is read from JSON at runtime. Nothing is hardcoded.

PROVENANCE
  binned decile curve ...... phase2/calibration.json continuous.calibration_curve
  overall slope/intercept .. phase2/calibration_flexible.json
  flexible mean abs dev .... phase2/calibration_flexible.json
                             flexible_mean_abs_deviation
  per-tertile slopes ....... phase2/calibration_flexible.json
                             age_tertile_calibration
  per-tertile dR2 (panel c)  phase2/batch1_checks.json age_tertile_dr2

TWO DISTINCT QUANTITIES, DELIBERATELY NOT CONFLATED
  The curve in panel A is a BINNED DECILE curve from calibration.json. The
  "flexible" calibration result exists in calibration_flexible.json ONLY as a
  summary statistic (mean absolute deviation), with no coordinates anywhere in
  data/processed. No smooth flexible fit is drawn, and the panel must never be
  labelled a flexible calibration curve. The two quantities are annotated
  separately and sourced separately.

ON calibration.json  (see analysis/figures/ARTIFACT_INTEGRITY_SWEEP.md)
  Not in the original figure brief; audited into the sweep as a figure source.
  Its continuous.calib_slope agrees with calibration_flexible.json's
  overall_slope at its stored 4 dp (delta ~4.9e-06) -- a ROUNDED verdict, not
  a strict PASS.
"""
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_figure_style, bbox_check, panel_letter, set_frame

# Repo-relative. This file lives at <repo>/analysis/figures/, so the repo
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
cal = load("phase2/calibration.json")
cf = load("phase2/calibration_flexible.json")
fa = load("phase3/fairness_audit.json")
b1 = load("phase2/batch1_checks.json")
inc = load("phase2/increment.json")

N = inc["rq1_csf"]["n"]
curve = cal["continuous"]["calibration_curve"]
mad = cf["flexible_mean_abs_deviation"]
slope_all, int_all = cf["overall_slope"], cf["overall_intercept"]

# --- runtime gates -------------------------------------------------------
# 1. the binned curve must account for every subject exactly once
_bin_n = sum(b["n"] for b in curve)
assert _bin_n == N, f"decile bins sum to {_bin_n}, expected the full {N}"
assert cal["continuous"]["n"] == N

# 2. the two calibration sources must still agree at the stored precision of
#    the coarser one (calibration.json stores 4 dp) -- this is the ROUNDED
#    relationship recorded in the sweep, asserted rather than assumed
_stored = cal["continuous"]["calib_slope"]
_dp = len(str(_stored).split(".")[-1])
assert round(slope_all, _dp) == round(_stored, _dp), (
    f"calibration.json calib_slope {_stored} no longer agrees with "
    f"calibration_flexible overall_slope {slope_all} at {_dp} dp")

# 3. per-tertile slopes must still reconcile with fairness_audit to 1e-9
_map = {"age<=58": "age<=58", "age 58-67": "age_58-67", "age>67": "age>67"}
_fa_ter = fa["axes"]["age_tertile"]
for t in cf["age_tertile_calibration"]:
    k = _map[t["stratum"]]
    assert t["n"] == _fa_ter[k]["n"], f"tertile n mismatch for {t['stratum']}"
    assert abs(t["slope"] - _fa_ter[k]["cal_slope"]) < 1e-9, (
        f"tertile {t['stratum']} slope no longer matches fairness_audit")

# 4. tertiles must partition the cohort
assert sum(t["n"] for t in cf["age_tertile_calibration"]) == N

# panel c: same three tertiles, dR2 from the Figure 2 source
_b1map = {"age<=58": "age <= 58", "age 58-67": "age 58-67", "age>67": "age > 67"}
_ter_dr2 = {t["stratum"]: t for t in b1["age_tertile_dr2"]}
for t in cf["age_tertile_calibration"]:
    assert _ter_dr2[_b1map[t["stratum"]]]["n"] == t["n"], (
        "tertile n differs between calibration_flexible and batch1_checks")

ters = list(cf["age_tertile_calibration"])          # youngest -> oldest
labels = {"age<=58": "Youngest\n(age \u2264 58)",
          "age 58-67": "Middle\n(age 58\u201367)",
          "age>67": "Oldest\n(age > 67)"}

# ---------------------------------------------------------------- style
apply_figure_style(sizes=(8, 7, 6))
DATA = "#0072B2"
IDEAL = "#666666"
ALARM = "#D55E00"
GREY = "#8C8C8C"

# LAYOUT. Panel a is square -- equal aspect is not optional for a calibration
# plot, because the identity line has to render at 45 degrees for deviations to
# be read honestly. A square tiles badly against the short, wide panels below
# it, and two decisions follow from that:
#
#   * The two rows get SEPARATE gridspecs instead of one 2-row grid. In a shared
#     grid the row gap is hspace, a fraction of the MEAN axes height, so it
#     shrinks as soon as one row is made taller -- which is how panel a's xlabel
#     ended up almost touching the two-line titles beneath it. Separate specs
#     let the gap be stated outright and stay put.
#   * The canvas is sized so that square + readout very nearly fills the width.
#     Widening the figure does NOT enlarge the square, whose side is capped by
#     the row height; it only opens dead canvas to the right of the readout.
fig = plt.figure(figsize=(5.9, 6.9))
gsTop = fig.add_gridspec(1, 1, left=0.165, right=0.985, top=0.925, bottom=0.520)
gsBot = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30,
                         left=0.165, right=0.985, top=0.383, bottom=0.157)
axA = fig.add_subplot(gsTop[0, 0])
axB = fig.add_subplot(gsBot[0, 0])
axC = fig.add_subplot(gsBot[0, 1])

# ---------------------------------------------------------------- panel A
xp = np.array([b["mean_pred"] for b in curve])
yo = np.array([b["mean_obs"] for b in curve])
bn = np.array([b["n"] for b in curve])

lim_lo = min(xp.min(), yo.min()) - 0.055
lim_hi = max(xp.max(), yo.max()) + 0.055
axA.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=IDEAL, lw=1.0,
         ls=(0, (4, 2)), zorder=1)
# label sits on the line itself, in the empty lower-right triangle
_lx = lim_lo + 0.42 * (lim_hi - lim_lo)
axA.annotate("ideal (45\u00b0)", xy=(_lx, _lx), xytext=(8, -8),
             textcoords="offset points", ha="left", va="top", color=IDEAL,
             fontsize=mpl.rcParams["legend.fontsize"])

axA.plot(xp, yo, color=DATA, lw=0.9, alpha=0.55, zorder=2)
axA.plot(xp, yo, marker="o", ls="none", ms=6.0, color=DATA, zorder=3)

axA.set_xlim(lim_lo, lim_hi)
axA.set_ylim(lim_lo, lim_hi)
axA.set_aspect("equal", adjustable="box")
# Left-anchor the shrunken square. The default anchor is centre, which splits
# the surplus width into two dead margins; anchoring west collects it into ONE
# band on the right, which the readout then occupies.
axA.set_anchor("W")
axA.set_xlabel("Mean predicted decline rate (decile bin)")
axA.set_ylabel("Mean observed\ndecline rate (decile bin)")
axA.set_title("Overall calibration is close to ideal; each point is a decile "
              "bin mean,\nnot a participant", loc="left")
set_frame(axA, "open")

# the two quantities, named and sourced separately so neither reads as the other
# Panel A is square (equal aspect), so the readout goes in the empty margin to
# its right rather than over the data.
# The readout is SPREAD down the band beside the square instead of stacked at
# its top. A square panel is far taller than ten lines of 6 pt text, so a single
# block leaves a rectangle of dead canvas beneath it -- which was the largest
# empty region in the previous version. Three blocks pinned to the top, middle
# and lower third of the same band fill the column and read as a key aligned to
# the panel rather than as a paragraph that ran out.
for _yf, _txt in (
    (1.00, f"{len(curve)} decile bins\n"
           f"n = {bn.min()}\u2013{bn.max()} per bin"),
    (0.62, f"overall calibration slope\n"
           f"    {slope_all:.4f}   (ideal 1)"),
    (0.24, f"flexible calibration\nmean |deviation|\n"
           f"    {mad:.6f}\n"
           f"    summary statistic \u2014\n"
           f"    no smooth fit drawn"),
):
    axA.annotate(_txt, xy=(1.045, _yf), xycoords="axes fraction",
                 ha="left", va="top", color="#333333",
                 fontsize=mpl.rcParams["legend.fontsize"], linespacing=1.45,
                 annotation_clip=False)

# ---------------------------------------------------------------- panel B
yb = np.arange(len(ters))[::-1]
axB.axvline(1.0, color=IDEAL, lw=1.0, ls=(0, (4, 2)), zorder=1)
axB.annotate("ideal = 1", xy=(1.0, len(ters) - 0.45), xytext=(3, 0),
             textcoords="offset points", ha="left", va="center", color=IDEAL,
             fontsize=mpl.rcParams["xtick.labelsize"])
for yi, t in zip(yb, ters):
    focal = t["stratum"] == "age<=58"
    col = ALARM if focal else DATA
    axB.plot([1.0, t["slope"]], [yi, yi], color=col, lw=1.2, alpha=0.45, zorder=2)
    axB.plot([t["slope"]], [yi], marker="o", ms=7.5 if focal else 6.0,
             color=col, zorder=3, clip_on=False)
    # Labels stay ABOVE the marker, but a marker sitting close to the ideal=1
    # rule gets its label left-ALIGNED so the text grows away from the rule
    # instead of straddling it -- which is what "1.086" and "1.119" did when
    # every label was centred. The collision gate cannot catch this: figstyle
    # treats a full-span reference rule as background furniture that labels may
    # legitimately cross, so it has to be handled at placement.
    # Moving these labels fully outside the lollipop instead is worse -- it
    # drives the youngest tertile's label into the y tick labels, which the gate
    # DOES catch.
    _near_rule = abs(t["slope"] - 1.0) < 0.25
    axB.annotate(f"{t['slope']:.3f}", xy=(t["slope"], yi),
                 xytext=(6 if _near_rule else 0, 10),
                 textcoords="offset points",
                 ha="left" if _near_rule else "center", va="bottom", color=col,
                 fontsize=mpl.rcParams["legend.fontsize"],
                 fontweight="bold" if focal else "normal")

axB.set_yticks(yb)
axB.set_yticklabels([f"{labels[t['stratum']]}\nn = {t['n']}" for t in ters])
axB.set_xlim(0.10, 1.32)
axB.set_ylim(-0.62, len(ters) - 0.30)
axB.set_xlabel("Calibration slope")
axB.set_title("Calibration collapses in\nthe youngest tertile", loc="left")
set_frame(axB, "open")
axB.tick_params(axis="y", length=0)
axB.spines["left"].set_visible(False)

# ---------------------------------------------------------------- panel C
axC.axvline(0.0, color="#4D4D4D", lw=1.0, zorder=1)
for yi, t in zip(yb, ters):
    d = _ter_dr2[_b1map[t["stratum"]]]["dr2"]
    focal = t["stratum"] == "age<=58"
    col = ALARM if d < 0 else DATA
    axC.plot([0.0, d], [yi, yi], color=col, lw=1.2, alpha=0.45, zorder=2)
    axC.plot([d], [yi], marker="o", ms=7.5 if focal else 6.0, color=col,
             zorder=3, clip_on=False)
    # Panel c needs no such adjustment: every marker clears the zero rule by
    # more than half a label width, so centred-above does not straddle it.
    axC.annotate(f"{d:+.4f}", xy=(d, yi), xytext=(0, 10),
                 textcoords="offset points", ha="center", va="bottom", color=col,
                 fontsize=mpl.rcParams["legend.fontsize"],
                 fontweight="bold" if focal else "normal")

axC.set_yticks(yb)
axC.set_yticklabels([])
axC.set_xlim(-0.088, 0.150)
axC.set_ylim(-0.62, len(ters) - 0.30)
axC.set_xlabel("$\\Delta R^2$ from CSF (same tertiles, Fig 2)")
axC.set_title("...and the increment\nreverses there too", loc="left")
set_frame(axC, "open")
axC.tick_params(axis="y", length=0)
axC.spines["left"].set_visible(False)

# Tie b and c together: ONE subgroup, TWO independent failure modes. The note
# sits on the youngest-tertile row itself, spanning the gutter between the two
# panels, so the connection is read off the figure rather than the caption.
# Placed inside panel c on the youngest-tertile row, where the axis is empty.
# NOTE: an annotation parked in the gutter between panels gets OVER-PAINTED by
# the neighbouring axes patch -- which bbox_check cannot see, because occlusion
# is not a text-on-text collision. Keep it inside an axes.
_y_young = yb[0]
axC.annotate(f"same n = {ters[0]['n']} subgroup,\ntwo independent failures",
             xy=(0.146, _y_young), ha="right", va="center", color=ALARM,
             fontsize=mpl.rcParams["xtick.labelsize"], linespacing=1.4)

# Panel letters in FIGURE coordinates. Placed through ax.transAxes they land at
# the axes top, which for a TWO-LINE title is level with the second line, not
# the first; and the dx needed to clear each panel's y labels differs, so the
# letters drift out of vertical and horizontal alignment with one another.
# Pinning them to the figure fixes both, and the layout above states the row
# tops outright, so these coordinates are derived from it rather than guessed.
_L = dict(fontsize=mpl.rcParams["font.size"] + 1, fontweight="bold",
          va="top", ha="left")
fig.text(0.024, 0.995, "a", **_L)
fig.text(0.024, 0.452, "b", **_L)
fig.text(0.578, 0.452, "c", **_L)

# Footnote WRAPPED: a single long line forces savefig's tight bbox to widen the
# canvas past the figure width, reopening the dead margin this layout removes.
fig.text(0.024, 0.056,
         f"n = {N}. a: bins from calibration.json; the flexible deviation is from "
         f"calibration_flexible.json and is\na summary statistic, not a drawn fit. "
         f"   b, c: tertiles partition the same {N}.",
         fontsize=mpl.rcParams["xtick.labelsize"], color="#4D4D4D", ha="left",
         va="top", linespacing=1.5)

assert not bbox_check(fig), "layout QA gate failed -- fix before saving"
fig.savefig(os.path.join(OUT, "fig3_calibration.png"), dpi=300)
fig.savefig(os.path.join(OUT, "fig3_calibration.pdf"))

print("FIG 3 -- value : source")
print(f"  a decile bins: {len(curve)}, n/bin {bn.min()}-{bn.max()}, sum {bn.sum()}"
      "   <- phase2/calibration.json continuous.calibration_curve")
print(f"  a overall slope   {slope_all:.10f}   <- phase2/calibration_flexible.json overall_slope")
print(f"  a overall intcpt  {int_all:.10f}   <- phase2/calibration_flexible.json overall_intercept")
print(f"  a flexible MAD    {mad:.10f}   <- phase2/calibration_flexible.json flexible_mean_abs_deviation")
print(f"  a (cross-check) calibration.json calib_slope {_stored} agrees at {_dp} dp")
for t in ters:
    d = _ter_dr2[_b1map[t["stratum"]]]["dr2"]
    print(f"  b {t['stratum']:10} n={t['n']:3d} slope={t['slope']:.10f}"
          "   <- phase2/calibration_flexible.json age_tertile_calibration")
    print(f"  c {t['stratum']:10} n={t['n']:3d} dr2  ={d:+.10f}"
          "   <- phase2/batch1_checks.json age_tertile_dr2")
