"""FIG 5 - Controls against the random-feature null.

One panel: dR2 for each control arm, plotted against the null band from the
20 random-feature draws. Negative controls (SAA positivity, SAA assay version,
random Gaussians) should sit inside the null. Established positive controls
(GBA carrier, APOE e4) are the interesting cases.

Every number is read from JSON at runtime. Nothing is hardcoded.

PROVENANCE
  all control arms ......... phase2/rank11_controls.json arms
  null band / p95 / max .... phase2/noise_floor_distribution.json
  GBA prevalence ceiling ... phase2/gba_scale_anchor.json
                             r2_ceiling_at_this_prevalence

TWO ANNOTATIONS THE PANEL MUST CARRY
  1. GBA clears the null 95th percentile but sits BELOW the largest single
     noise draw -- clearing a percentile is not the same as exceeding every
     draw, and the figure shows both lines so the reader can see it.
  2. GBA's contribution is PREVALENCE-BOUNDED. At its observed carrier
     prevalence the maximum attainable R2 contribution is the ceiling from
     gba_scale_anchor.json. Drawn as a bound so the bar reads as capped by
     design rather than as a weak effect.

  APOE e4 not firing is a REPORTABLE RESULT, not an empty row. It is drawn
  with a visible zero-stub and labelled, never left blank.

NOTE ON INTERVALS
  rank11_controls.json carries repeat_lo / repeat_hi on every arm. Those are
  CROSS-SEED REPEAT RANGES, not confidence intervals, and the file says so.
  They are NOT drawn as error bars. The only interval anywhere in this figure
  set that is a real CI is the subject-level bootstrap in Figure 1.
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
rk = load("phase2/rank11_controls.json")
nf = load("phase2/noise_floor_distribution.json")
gba = load("phase2/gba_scale_anchor.json")
inc = load("phase2/increment.json")

arms = rk["arms"]
rq1 = inc["rq1_csf"]
p95, nmax, nmin = nf["p95"], nf["max"], nf["min"]
ceiling = gba["r2_ceiling_at_this_prevalence"]

# --- runtime gates -------------------------------------------------------
# 1. the anchor arm must still BE the committed increment -- this file carries
#    a deliberate anchor arm, so a drift here means the controls moved too
for _k in ("r2_base", "r2_aug", "dr2"):
    assert abs(arms["anchor_csf"][_k] - rq1[_k]) < 1e-9, (
        f"rank11_controls anchor_csf.{_k} no longer matches the committed anchor")

# 2. the two claims annotated on the panel must be TRUE of the plotted values.
#    If GBA ever stops clearing p95, or starts exceeding the largest draw, the
#    annotation would become false -- so assert both rather than trusting them.
_gba_dr2 = arms["pos_gba_status"]["dr2"]
assert _gba_dr2 > p95, "annotation claims GBA clears the null p95"
assert _gba_dr2 < nmax, "annotation claims GBA sits below the largest noise draw"
assert _gba_dr2 < ceiling, "GBA dr2 must lie under its own prevalence ceiling"

# 3. gba_scale_anchor must be describing the same GBA arm (it stores a 4 dp
#    copy of it) -- guards against the two files drifting apart
_dp = len(str(gba["observed_dr2"]).split(".")[-1])
assert round(_gba_dr2, _dp) == round(gba["observed_dr2"], _dp), (
    "gba_scale_anchor.observed_dr2 no longer restates rank11 pos_gba_status.dr2")

# 4. every negative control must sit inside the null band, which is the whole
#    point of calling them negative controls
for _key in ("neg_saa_csfset", "neg_assay_csfset", "neg_noise_csfset",
             "neg_saa_max", "neg_assay_max"):
    assert arms[_key]["dr2"] < p95, (
        f"{_key} is labelled a negative control but clears the null p95")

# rows, ordered so the null-consistent controls group together and the two
# established genetic factors sit at the bottom where the annotations go
# (key, display label, kind)
rows = [
    ("anchor_csf", "CSF block (the result under test)", "anchor"),
    ("neg_noise_csfset", "5 random Gaussians", "neg"),
    ("neg_assay_csfset", "SAA assay version (technical/batch)", "neg"),
    ("neg_saa_csfset", "SAA positivity", "neg"),
    ("pos_APOE_e4", "APOE \u03b54 carrier", "pos"),
    ("pos_gba_status", "GBA carrier", "pos"),
]

# ---------------------------------------------------------------- style
apply_figure_style(sizes=(8, 7, 6))
ANCHOR = "#333333"
NEGC = "#8C8C8C"
POSC = "#0072B2"
NULLBAND = "#C9C9C9"
REF = "#D55E00"
CEIL = "#009E73"

# Axes fills the canvas: the previous geometry left roughly a quarter of the
# figure empty between the x-axis label and the footnote.
fig = plt.figure(figsize=(7.2, 4.0))
ax = fig.add_axes([0.325, 0.255, 0.585, 0.665])

y = np.arange(len(rows))[::-1]

# --- the null, as a band plus its two reference lines --------------------
# The band spans the ACTUAL extent of the 20 draws (min to max). Ending it at
# p95 while a dotted line for the largest draw sits outside it reads as a band
# that stops short of its own contents.
ax.axvspan(nmin, nmax, color=NULLBAND, alpha=0.42, lw=0, zorder=0)
ax.axvline(0.0, color="#9A9A9A", lw=0.8, zorder=1)
ax.axvline(p95, color=REF, lw=1.1, ls=(0, (4, 2)), zorder=2)
ax.axvline(nmax, color=REF, lw=0.9, ls=(0, (1, 2)), alpha=0.85, zorder=2)

# --- control arms --------------------------------------------------------
for yi, (key, lab, kind) in zip(y, rows):
    a = arms[key]
    d = a["dr2"]
    col = {"anchor": ANCHOR, "neg": NEGC, "pos": POSC}[kind]
    ax.plot([0.0, d], [yi, yi], color=col, lw=1.2, alpha=0.45, zorder=3)
    # a near-zero arm still gets a visible mark: not firing is a RESULT
    ax.plot([d], [yi], marker="o", ms=7.5 if kind == "anchor" else 6.0,
            color=col, zorder=4, clip_on=False)
    # GBA's value label goes to the LEFT of its mark. To the right sits the
    # prevalence-ceiling tick and its arrow; directly above, the largest-noise-
    # draw rule at nmax passes through the text.
    if key == "pos_gba_status":
        ax.annotate(f"{d:+.4f}", xy=(d, yi), xytext=(-9, 9),
                    textcoords="offset points", ha="right", va="bottom",
                    color=col, fontsize=mpl.rcParams["legend.fontsize"])
    else:
        off = 9 if d >= 0 else -9
        ha = "left" if d >= 0 else "right"
        ax.annotate(f"{d:+.4f}", xy=(d, yi), xytext=(off, 0),
                    textcoords="offset points", ha=ha, va="center", color=col,
                    fontsize=mpl.rcParams["legend.fontsize"],
                    fontweight="bold" if kind == "anchor" else "normal")

# --- GBA prevalence ceiling ---------------------------------------------
y_gba = y[[r[0] for r in rows].index("pos_gba_status")]
ax.plot([ceiling, ceiling], [y_gba - 0.34, y_gba + 0.34], color=CEIL, lw=1.6,
        zorder=5)
ax.annotate(f"ceiling {ceiling:.4f}", xy=(ceiling, y_gba + 0.36),
            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
            color=CEIL, fontsize=mpl.rcParams["xtick.labelsize"])
# span from the observed GBA value to its ceiling, so "bounded" is visible
ax.annotate("", xy=(ceiling, y_gba), xytext=(_gba_dr2, y_gba),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=CEIL,
                            shrinkA=3, shrinkB=0))

ax.set_yticks(y)
ax.set_yticklabels([f"{lab}\nn = {arms[key]['n']}" for key, lab, _ in rows])
ax.set_xlabel("$\\Delta R^2$ over the clinical baseline "
              "(continuous decline-rate outcome)")
ax.set_xlim(-0.011, 0.075)
ax.set_ylim(-0.95, len(rows) - 0.35)
ax.set_title("Negative controls stay inside the random-feature null; GBA "
             "clears it\nbut is bounded by carrier prevalence", loc="left")
set_frame(ax, "open")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)

# --- reference-line labels ----------------------------------------------
_top = len(rows) - 0.42
ax.annotate(f"null 95th pct {p95:.4f}", xy=(p95, _top), xytext=(-4, 0),
            textcoords="offset points", ha="right", va="center", color=REF,
            fontsize=mpl.rcParams["xtick.labelsize"])
ax.annotate(f"largest single noise draw {nmax:.4f}", xy=(nmax, _top),
            xytext=(5, 0), textcoords="offset points", ha="left", va="center",
            color=REF, alpha=0.9, fontsize=mpl.rcParams["xtick.labelsize"])

# --- the APOE result, stated rather than left blank ----------------------
y_apoe = y[[r[0] for r in rows].index("pos_APOE_e4")]
# No leader lines on these two notes: a leader spanning half the panel crosses
# the null band and the other rows' marks. Each note sits on its own row, which
# is unambiguous without one.
ax.annotate("does not fire \u2014 an established risk factor\n"
            "adds nothing to the decline-rate model",
            xy=(0.030, y_apoe), ha="left", va="center", color=POSC,
            fontsize=mpl.rcParams["xtick.labelsize"], linespacing=1.4)

# --- the GBA nuance, stated on the panel ---------------------------------
# One block, not two: the bottom row has no vertical room for stacked notes,
# and the ceiling mark is already labelled at the tick itself.
ax.annotate("clears the null 95th percentile, but sits below the\n"
            "largest single noise draw \u2014 and its contribution is\n"
            f"prevalence-bounded at {ceiling:.4f} "
            f"({gba['prevalence'] * 100:.1f}% carriers)",
            xy=(0.030, y_gba), ha="left", va="center", color=POSC,
            fontsize=mpl.rcParams["xtick.labelsize"], linespacing=1.4)

fig.text(0.325, 0.088,
         f"Null band spans the {nf['n_draws']} random-feature draws "
         f"({nmin:+.4f} to {nmax:+.4f}). Brackets in the source file are "
         f"cross-seed repeat ranges,\nnot confidence intervals, and are not "
         f"drawn. Arms differ in n because each control is scored on the "
         f"largest set where it is defined.",
         fontsize=mpl.rcParams["xtick.labelsize"], color="#4D4D4D", ha="left",
         va="top", linespacing=1.5)

assert not bbox_check(fig), "layout QA gate failed -- fix before saving"
fig.savefig(os.path.join(OUT, "fig5_controls.png"), dpi=300)
fig.savefig(os.path.join(OUT, "fig5_controls.pdf"))

print("FIG 5 -- value : source")
for key, lab, kind in rows:
    a = arms[key]
    print(f"  {lab:38} n={a['n']:4d} dr2={a['dr2']:+.10f}"
          "   <- phase2/rank11_controls.json arms")
print(f"  null mean/p95/max  {nf['mean']:+.10f} / {p95:.10f} / {nmax:.10f}"
      "   <- phase2/noise_floor_distribution.json")
print(f"  GBA ceiling        {ceiling:.10f}  at prevalence "
      f"{gba['prevalence']:.10f}   <- phase2/gba_scale_anchor.json")
print(f"  GBA clears p95: {_gba_dr2 > p95}   below largest draw: {_gba_dr2 < nmax}")
