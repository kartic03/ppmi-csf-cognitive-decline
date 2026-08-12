"""FIG 6 - The causal null.

MR forest for PDGFRB on both platforms with 95% CIs, each arm's 80%-power
minimum detectable OR, and the GPNMB positive control on the SAME odds-ratio
axis -- so the null visibly excludes effects far smaller than one the pipeline
demonstrably detects.

TWO DISTINCT QUANTITIES, NOT ONE
  ci_high      -- the EQUIVALENCE BOUND: the largest OR the data exclude.
                  This licenses "we exclude effects larger than OR x".
  mde_or_80pct -- the MINIMUM DETECTABLE EFFECT at 80% power: the smallest OR
                  this design could have detected, computed by the source
                  artifact as exp((z0.975 + z0.80) * SE). A statement about
                  the DESIGN, not about what the data rule out.
  Both are reported. Neither is called by the other's name -- the project's
  prose reserves "equivalence bound" for ci_high, and the figure must agree.

Every number is read from CSV/JSON at runtime. Nothing is hardcoded.

PROVENANCE
  PDGFRB OR + CI ........... mr/mr_power_bounds.json cells (PDGFRB x platform)
  equivalence bound ........ mr/mr_power_bounds.json cells[].ci_high
  minimum detectable effect  mr/mr_power_bounds.json cells[].mde_or_80pct
  GPNMB positive control ... mr/poscontrol_result.json platforms
  instrument counts, F ..... mr/mr_power_bounds.json cells
  cross-check of ORs ....... mr/mr_final_table.csv (3 dp restatement)

THE GATE DEVIATION, MARKED ON THE FIGURE ITSELF
  The pre-specified design required positive-control recovery on BOTH
  platforms before any null could be reported. GPNMB recovers on deCODE
  (OR ~1.49, CI excludes 1) but NOT on UKB-PPP (OR ~0.99, CI spans 1). The
  gate was narrowed to deCODE, so THE UKB-PPP ARM OF THE PDGFRB NULL IS
  UNGATED: on that platform nothing demonstrates the pipeline could have
  detected a true effect. That arm is drawn open/hatched and labelled
  "ungated" on the panel, not only in the caption.

ON mr_final_table.csv
  UNANCHORABLE (see ARTIFACT_INTEGRITY_SWEEP.md): it contains no GPNMB row,
  so it cannot be checked against either positive control. It is used here
  ONLY as a 3 dp cross-check of the ORs, never as the plotted source.
"""
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
pb = load("mr/mr_power_bounds.json")
pos = load("mr/poscontrol_result.json")
mr_csv = pd.read_csv(os.path.join(BASE, "mr/mr_final_table.csv"))

cells = {(c["protein"], c["platform"]): c for c in pb["cells"]}
pdg_dec = cells[("PDGFRB", "decode")]
pdg_ukb = cells[("PDGFRB", "ukbppp")]
gp_dec = pos["platforms"]["decode"]
gp_ukb = pos["platforms"]["ukbppp"]

# --- runtime gates -------------------------------------------------------
# 1. the positive control must still behave as the gate deviation describes:
#    recovers on deCODE (CI excludes 1), fails on UKB-PPP (CI spans 1).
#    If this ever flips, the "ungated" mark on the figure becomes wrong.
assert gp_dec["ci_low"] > 1.0, "deCODE positive control must exclude OR = 1"
assert gp_ukb["ci_low"] < 1.0 < gp_ukb["ci_high"], (
    "UKB-PPP positive control must span OR = 1 -- this is why that arm is ungated")
assert pos["recovers_both_platforms"] is False
assert pos["gate_applied"] == "decode_only"

# 2. mr_power_bounds' headline positive-control OR must be the deCODE arm
assert abs(pb["gpnmb_positive_control_or"] - gp_dec["or"]) < 1e-9

# 3. both PDGFRB arms must be nulls whose CI spans 1
for _lab, _c in (("deCODE", pdg_dec), ("UKB-PPP", pdg_ukb)):
    assert _c["ci_low"] < 1.0 < _c["ci_high"], f"PDGFRB {_lab} CI must span 1"
    assert _c["verdict"] == "null"

# 4. the minimum detectable effect must be BELOW the positive-control effect --
#    that is the whole argument of the figure: this design could have detected
#    effects far smaller than one the pipeline demonstrably picks up
for _lab, _c in (("deCODE", pdg_dec), ("UKB-PPP", pdg_ukb)):
    assert _c["mde_or_80pct"] < gp_dec["or"], (
        f"PDGFRB {_lab} MDE must sit below the positive-control OR")
    assert _c["excludes_gpnmb_scale"] == "YES"

# 5. the CSV must still restate the same ORs at its stored 3 dp
_row = mr_csv.loc[mr_csv["protein"] == "PDGFRB"].iloc[0]
for _col, _c in (("decode_or", pdg_dec), ("ukbppp_or", pdg_ukb)):
    assert round(float(_row[_col]), 3) == round(_c["or"], 3), (
        f"mr_final_table.csv {_col} no longer restates mr_power_bounds")
assert "GPNMB" not in set(mr_csv["protein"]), (
    "csv gained a GPNMB row -- revisit its UNANCHORABLE verdict in the sweep")

# rows: positive control first (it licenses everything below), then the two
# PDGFRB arms.  (label, or, lo, hi, mde, kind, n_instr, F)
rows = [
    ("GPNMB positive control\n(deCODE)", gp_dec["or"], gp_dec["ci_low"],
     gp_dec["ci_high"], None, "poscontrol_ok",
     gp_dec["n_instruments"], None),
    ("GPNMB positive control\n(UKB-PPP)", gp_ukb["or"], gp_ukb["ci_low"],
     gp_ukb["ci_high"], None, "poscontrol_fail",
     gp_ukb["n_instruments"], None),
    ("PDGFRB (deCODE)", pdg_dec["or"], pdg_dec["ci_low"], pdg_dec["ci_high"],
     pdg_dec["mde_or_80pct"], "null_gated", pdg_dec["n_instruments"],
     pdg_dec["F"]),
    ("PDGFRB (UKB-PPP)", pdg_ukb["or"], pdg_ukb["ci_low"], pdg_ukb["ci_high"],
     pdg_ukb["mde_or_80pct"], "null_ungated", pdg_ukb["n_instruments"],
     pdg_ukb["F"]),
]

# ---------------------------------------------------------------- style
apply_figure_style(sizes=(8, 7, 6))
NULLC = "#0072B2"
POSOK = "#009E73"
FAILC = "#D55E00"
MDEC = "#666666"

fig = plt.figure(figsize=(7.2, 4.2))
ax = fig.add_axes([0.315, 0.265, 0.480, 0.615])

y = np.arange(len(rows))[::-1]
ax.axvline(1.0, color="#4D4D4D", lw=1.0, zorder=2)

for yi, (lab, orv, lo, hi, mde, kind, ninstr, F) in zip(y, rows):
    col = {"poscontrol_ok": POSOK, "poscontrol_fail": FAILC,
           "null_gated": NULLC, "null_ungated": NULLC}[kind]
    ungated = kind in ("null_ungated", "poscontrol_fail")
    ax.plot([lo, hi], [yi, yi], color=col, lw=1.6, zorder=3,
            alpha=0.85 if not ungated else 0.60)
    for xb in (lo, hi):
        ax.plot([xb, xb], [yi - 0.10, yi + 0.10], color=col, lw=1.3,
                zorder=3, alpha=0.85 if not ungated else 0.60)
    # open marker = ungated / failed arm; filled = gated
    ax.plot([orv], [yi], marker="o" if kind.startswith("null") else "D",
            ms=7.0, mfc="white" if ungated else col, mec=col, mew=1.6,
            zorder=4, clip_on=False)
    # MINIMUM DETECTABLE EFFECT (not the equivalence bound -- that is ci_high,
    # already shown as the CI's upper limit). Drawn BELOW the interval, not on
    # it: on the row line it lands inside the value label whenever the MDE
    # falls right of the CI.
    if mde is not None:
        ax.plot([mde], [yi - 0.20], marker="|", ms=9, mew=1.6, color=MDEC,
                zorder=5)
        ax.annotate(f"{mde:.3f}", xy=(mde, yi - 0.20), xytext=(0, -8),
                    textcoords="offset points", ha="center", va="top",
                    color=MDEC, fontsize=mpl.rcParams["xtick.labelsize"])
        _xtext = max(hi, mde)
    else:
        _xtext = hi
    ax.annotate(f"{orv:.3f}  ({lo:.3f}\u2013{hi:.3f})", xy=(_xtext, yi),
                xytext=(11, 0), textcoords="offset points", ha="left",
                va="center", color=col,
                fontsize=mpl.rcParams["legend.fontsize"])

ax.set_yticks(y)
ax.set_yticklabels([
    (f"{r[0]}\nUNGATED" if r[5] in ("null_ungated", "poscontrol_fail") else r[0])
    for r in rows])
ax.set_xscale("log")
ax.set_xlim(0.86, 2.05)
ax.set_ylim(-0.72, len(rows) - 0.52)
ax.set_xticks([0.9, 1.0, 1.2, 1.5, 2.0])
ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(
    lambda v, _: f"{v:g}"))
ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_xlabel("Odds ratio for Parkinson's disease per SD of plasma protein "
              "(log scale)")
ax.set_title("PDGFRB is null on both platforms, at a scale where the pipeline\n"
             "detects the positive control", loc="left")
set_frame(ax, "open")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)

# --- the ungated arm, marked ON the figure -------------------------------
y_ukb = y[[r[5] for r in rows].index("null_ungated")]
y_gpf = y[[r[5] for r in rows].index("poscontrol_fail")]
# The UNGATED tag is folded INTO the row label (see set_yticklabels above) and
# the tick label is recoloured, so the mark travels with the row and cannot be
# separated from it by a layout change or a panel crop.
for _tick, (_lab, *_rest) in zip(ax.get_yticklabels(), rows):
    if _rest[4] in ("null_ungated", "poscontrol_fail"):
        _tick.set_color(FAILC)

ax.annotate("positive control does NOT recover here \u2014 so the\n"
            "UKB-PPP null below is descriptive, not gated",
            xy=(1.055, y_gpf - 0.36), ha="left", va="center", color=FAILC,
            fontsize=mpl.rcParams["xtick.labelsize"], linespacing=1.4)

# --- equivalence-bound key ----------------------------------------------
# NOT labelled "equivalence bound". mde_or_80pct is a MINIMUM DETECTABLE
# EFFECT -- a power statement about the design, computed by the artifact as
# exp((z0.975 + z0.80) * SE). The equivalence bound proper is the CI UPPER
# LIMIT (ci_high), which is what the data exclude. Attaching one term to both
# quantities is the error this wording avoids.
ax.annotate("|  minimum detectable effect at 80% power (what this design "
            "could have seen)",
            xy=(0.0, -0.185), xycoords="axes fraction", ha="left", va="top",
            color=MDEC, fontsize=mpl.rcParams["xtick.labelsize"],
            annotation_clip=False)

fig.text(0.275, 0.055,
         f"Instruments: GPNMB {gp_dec['n_instruments']} (deCODE), "
         f"{gp_ukb['n_instruments']} (UKB-PPP); PDGFRB "
         f"{pdg_dec['n_instruments']} (deCODE, F = {pdg_dec['F']:.0f}), "
         f"{pdg_ukb['n_instruments']} (UKB-PPP, F = {pdg_ukb['F']:.0f}).\n"
         f"Pre-specified gate required recovery on BOTH platforms; it was "
         f"narrowed to deCODE, leaving the UKB-PPP arm ungated.",
         fontsize=mpl.rcParams["xtick.labelsize"], color="#4D4D4D", ha="left",
         va="top", linespacing=1.5)

assert not bbox_check(fig), "layout QA gate failed -- fix before saving"
fig.savefig(os.path.join(OUT, "fig6_causal_null.png"), dpi=300)
fig.savefig(os.path.join(OUT, "fig6_causal_null.pdf"))

print("FIG 6 -- value : source")
for lab, orv, lo, hi, mde, kind, ninstr, F in rows:
    src = ("mr/poscontrol_result.json platforms" if kind.startswith("poscontrol")
           else "mr/mr_power_bounds.json cells")
    mdes = f" mde80={mde:.6f}" if mde is not None else ""
    print(f"  {lab.replace(chr(10), ' '):36} OR={orv:.6f} "
          f"CI=({lo:.6f},{hi:.6f}){mdes}   <- {src}")
print(f"  gate: recovers_both_platforms={pos['recovers_both_platforms']}, "
      f"gate_applied={pos['gate_applied']!r}   <- mr/poscontrol_result.json")
print(f"  csv cross-check (3 dp): decode_or={float(_row['decode_or'])}, "
      f"ukbppp_or={float(_row['ukbppp_or'])}   <- mr/mr_final_table.csv")
