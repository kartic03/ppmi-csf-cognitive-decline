"""FIG 1 - The CSF increment against its null.

Panel A: clinical baseline R2 vs clinical+CSF R2, pre-registered 10-variable
         block and enriched 14-variable block.
Panel B: 20 random-feature draws (the null) as a strip, with the observed CSF
         dR2 (subject-level bootstrap CI) and the 95th percentile of the null.

Every number is read from JSON at runtime. Nothing is hardcoded.

PROVENANCE
  Panel A, both rows ....... phase2/estimand_audit.json
                             results['eb_slope|thin']     (10-variable block)
                             results['eb_slope|enriched'] (14-variable block)
  Panel B, null strip ...... phase2/noise_floor_distribution.json
  Panel B, observed dR2 .... phase2/increment.json rq1_csf.dr2
  Panel B, error bar ....... phase2/bootstrap_ci.json (subject-level percentile)

NOT USED: phase2/enriched_baseline_check.json. Its control arm
(eb_slope|thin, dr2=0.05079576) does not reproduce the committed
increment.json rq1_csf headline (dr2=0.05813286) despite declaring the same
n, cv_config and column lists; that value is instead exactly equal to
increment.json rq1_csf.perm_observed_dr2_fixed, i.e. the fixed-alpha
permutation estimator rather than the tuned nested-CV one. estimand_audit.json
reproduces the committed control arm to the last digit and is used instead.
See revision/checks/estimand_vs_enriched_baseline_diff.md.
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
ea = load("phase2/estimand_audit.json")
boot = load("phase2/bootstrap_ci.json")
noise = load("phase2/noise_floor_distribution.json")

rq1 = inc["rq1_csf"]
ea_thin = ea["results"]["eb_slope|thin"]
ea_rich = ea["results"]["eb_slope|enriched"]

# integrity gate: the block plotted as "pre-registered" must BE the committed one
for f in ("r2_base", "r2_aug", "dr2"):
    assert abs(ea_thin[f] - rq1[f]) < 1e-9, (
        f"estimand_audit eb_slope|thin.{f} does not reproduce increment.json rq1_csf.{f}")

# Panel A rows: (label, r2_base, r2_aug, dr2, is_headline)
rowsA = [
    ("Pre-registered clinical block\n(10 variables)",
     ea_thin["r2_base"], ea_thin["r2_aug"], ea_thin["dr2"], True),
    ("Enriched clinical block\n(14 variables)",
     ea_rich["r2_base"], ea_rich["r2_aug"], ea_rich["dr2"], False),
]

draws = np.asarray(noise["dr2_draws"], float)
null_mean, null_sd, null_p95 = noise["mean"], noise["sd"], noise["p95"]
obs = boot["point_dr2"]
ci_lo, ci_hi = boot["ci_lo"], boot["ci_hi"]
sd_above = (obs - null_mean) / null_sd
n_subj = rq1["n"]

assert len(draws) == noise["n_draws"]
assert abs(obs - rq1["dr2"]) < 1e-12, "bootstrap point must match increment.json dR2"

# ---------------------------------------------------------------- style
apply_figure_style(sizes=(8, 7, 6))
CLIN = "#8C8C8C"      # clinical-only marks
CSF = "#0072B2"       # anything carrying the CSF block (focal)
NULLC = "#9E9E9E"     # random-feature draws
REF = "#D55E00"       # reference threshold line

fig = plt.figure(figsize=(7.2, 4.5))
gs = fig.add_gridspec(2, 1, height_ratios=[0.62, 1.0], hspace=0.70,
                      left=0.30, right=0.965, top=0.90, bottom=0.11)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])

# ---------------------------------------------------------------- panel A
# DELIBERATELY NO ERROR BARS IN PANEL A.
# estimand_audit.json carries dr2_repeat_lo / dr2_repeat_hi on every arm, but
# those are CROSS-SEED REPEAT RANGES, not confidence intervals. Drawing them as
# intervals is precisely the error the Panel B footnote exists to prevent. The
# only legitimate interval on the increment is the subject-level bootstrap, and
# it appears once, on the pre-registered row, in Panel B where it is labelled.
yA = np.arange(len(rowsA))[::-1]
for y, (lab, rb, ra, dr, headline) in zip(yA, rowsA):
    axA.plot([rb, ra], [y, y], color=CSF, lw=1.4, alpha=0.55,
             solid_capstyle="butt", zorder=1)
    axA.plot([rb], [y], marker="o", ms=6.5, mfc="white", mec=CLIN, mew=1.4,
             zorder=3, clip_on=False)
    axA.plot([ra], [y], marker="o", ms=6.5, color=CSF, zorder=3, clip_on=False)
    axA.annotate(f"$\\Delta R^2$ = +{dr:.4f}", xy=(ra, y), xytext=(7, 0),
                 textcoords="offset points", va="center", ha="left",
                 fontsize=mpl.rcParams["legend.fontsize"],
                 color=CSF, fontweight="bold" if headline else "normal")

axA.set_yticks(yA)
axA.set_yticklabels([r[0] for r in rowsA])
axA.set_xlabel("Out-of-fold $R^2$ for the continuous decline-rate outcome")
axA.set_xlim(0.0, 0.325)
axA.set_ylim(-0.55, len(rowsA) - 0.10)
axA.set_title("A stronger clinical baseline absorbs part of the CSF increment,\n"
              "but does not remove it", loc="left")
set_frame(axA, "open")
axA.tick_params(axis="y", length=0)

# No legend box. A floating key in a 2-row dot-and-line panel reads as an
# annotation on whichever row it sits nearest. The two marker types are
# direct-labelled once, on the top row, above the marks they name.
_top_rb, _top_ra = rowsA[0][1], rowsA[0][2]
axA.annotate("clinical only", xy=(_top_rb, yA[0]), xytext=(0, 13),
             textcoords="offset points", ha="center", va="bottom", color=CLIN,
             fontsize=mpl.rcParams["legend.fontsize"])
axA.annotate("clinical + CSF", xy=(_top_ra, yA[0]), xytext=(0, 13),
             textcoords="offset points", ha="center", va="bottom", color=CSF,
             fontsize=mpl.rcParams["legend.fontsize"])
panel_letter(axA, "a", dx=-0.335, dy=1.10)

# ---------------------------------------------------------------- panel B
rng = np.random.default_rng(0)
y_null, y_obs = 1.0, 0.0

axB.axvline(0.0, color="#B0B0B0", lw=0.8, zorder=0)
axB.plot(draws, y_null + rng.uniform(-0.11, 0.11, draws.size), marker="o",
         ls="none", ms=4.2, mfc="none", mec=NULLC, mew=1.0, zorder=2)
axB.plot([null_mean, null_mean], [y_null - 0.20, y_null + 0.20], color="#4D4D4D",
         lw=1.8, zorder=3)

axB.axvline(null_p95, color=REF, lw=1.1, ls=(0, (4, 2)), zorder=1)
axB.annotate(f"95th percentile of the null = {null_p95:.4f}",
             xy=(null_p95, y_null + 0.40), xytext=(4, 0),
             textcoords="offset points", ha="left", va="center", color=REF,
             fontsize=mpl.rcParams["legend.fontsize"])

axB.errorbar([obs], [y_obs], xerr=[[obs - ci_lo], [ci_hi - obs]], fmt="o",
             ms=7.5, color=CSF, ecolor=CSF, elinewidth=1.5, capsize=3.5,
             capthick=1.5, zorder=4)
axB.annotate(f"observed CSF $\\Delta R^2$ = {obs:.4f}\n"
             f"{sd_above:.1f} SD above the null mean\n"
             f"bootstrap CI {ci_lo:.4f} to {ci_hi:.4f}",
             xy=(obs, y_obs), xytext=(0, -14), textcoords="offset points",
             ha="center", va="top", color=CSF,
             fontsize=mpl.rcParams["legend.fontsize"])

axB.set_yticks([y_obs, y_null])
axB.set_yticklabels(["observed\nCSF block", "20 random-feature\ndraws (null)"])
axB.set_ylim(-1.05, 1.80)
axB.set_xlim(-0.034, 0.108)
axB.set_xlabel("$\\Delta R^2$ over the pre-registered clinical baseline")
axB.set_title("The increment sits far outside the random-feature null", loc="left")
set_frame(axB, "open")
axB.tick_params(axis="y", length=0)
axB.annotate("null mean", xy=(null_mean, y_null + 0.22), xytext=(-16, 7),
             textcoords="offset points", ha="right", va="bottom", color="#4D4D4D",
             fontsize=mpl.rcParams["xtick.labelsize"],
             arrowprops=dict(arrowstyle="-", lw=0.6, color="#4D4D4D",
                             shrinkA=0, shrinkB=1))
panel_letter(axB, "b", dx=-0.335, dy=1.16)

fig.text(0.30, 0.005, f"n = {n_subj} CSF-complete participants; "
         f"interval is the subject-level percentile bootstrap "
         f"(B = {boot['B_usable']}), not a cross-seed range.",
         fontsize=mpl.rcParams["xtick.labelsize"], color="#4D4D4D", ha="left")

assert not bbox_check(fig), "layout QA gate failed -- fix before saving"
fig.savefig(os.path.join(OUT, "fig1_increment_vs_null.png"), dpi=300)
fig.savefig(os.path.join(OUT, "fig1_increment_vs_null.pdf"))

print("FIG 1 -- value : source")
for lab, rb, ra, dr, _ in rowsA:
    tag = lab.replace("\n", " ")
    print(f"  A {tag:52} r2_base={rb:.8f} r2_aug={ra:.8f} dr2={dr:.8f}"
          "   <- phase2/estimand_audit.json")
print(f"  B observed dR2                                       {obs:.8f}"
      "   <- phase2/increment.json rq1_csf.dr2")
print(f"  B bootstrap CI                                       "
      f"({ci_lo:.8f}, {ci_hi:.8f})  <- phase2/bootstrap_ci.json")
print(f"  B null mean / sd / p95                               "
      f"{null_mean:.8f} / {null_sd:.8f} / {null_p95:.8f}"
      "   <- phase2/noise_floor_distribution.json")
print(f"  B SD above null mean (derived)                       {sd_above:.4f}")
print(f"  n = {n_subj}   <- phase2/increment.json rq1_csf.n")
print("  NOT USED: phase2/enriched_baseline_check.json "
      "(control arm fails to reproduce committed headline)")
