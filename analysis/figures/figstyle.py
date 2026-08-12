"""Shared publication-figure mechanics for the npj Parkinson's Disease revision.

Sets a role-mapped font-size ladder, outward ticks, frameless legends,
300-dpi raster output and Type-42 embedded fonts for vector output. This is
mechanics, not a house look: frame, font and the size ladder are parameters.

Import from any fig*.py in this directory:
    from figstyle import apply_figure_style, set_frame, panel_letter
"""

META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    """Set matplotlib rcParams for publication-grade output. Call once before plotting.

    frame : 'open' (bottom+left spines) | 'boxed' (all four) | 'none'
    font  : sans-serif family name; None = system default sans-serif
    sizes : (base, secondary, tick) — titles/axis-labels, legend/annotation, ticks
    grid  : whether to draw axes.grid
    """
    import matplotlib as mpl

    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    base, secondary, tick = sizes
    boxed = frame == "boxed"
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    if font:
        mpl.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]


def set_frame(ax, style="open"):
    """Set spine visibility on an existing axes. style in {'open','boxed','none'}."""
    show = {"open": (False, False, True, True),
            "boxed": (True, True, True, True),
            "none": (False, False, False, False)}[style]
    for side, vis in zip(("top", "right", "bottom", "left"), show):
        ax.spines[side].set_visible(vis)
        if vis:
            ax.spines[side].set_linewidth(0.6)
    ax.tick_params(direction="out", length=0 if style == "none" else 3, width=0.6)


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    """Bold panel letter outside the top-left of the axes box."""
    import matplotlib.pyplot as plt

    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize, va="bottom", ha="left")


def _offview_ticklabels(fig):
    """Tick-label Text objects whose tick lies outside its axes' view limits.

    The locator manufactures these; matplotlib never renders them, but they
    still report a window_extent — often far off-canvas. Counting them as
    'outside the figure' is a false positive, so they are excluded.
    """
    dead = set()
    for ax in fig.axes:
        for getlim, getlabels in ((ax.get_xlim, ax.get_xticklabels),
                                  (ax.get_ylim, ax.get_yticklabels)):
            lo, hi = sorted(getlim())
            for t in getlabels(which="both"):
                try:
                    v = float(t.get_text().replace("\u2212", "-"))
                except ValueError:
                    continue
                if not (lo <= v <= hi):
                    dead.add(t)
    return dead


def bbox_check(fig, verbose=True):
    """Geometric overlap check: visible text boxes must not collide with each
    other or with a spine that is not their own axes' tick spine, and every
    text box must lie inside the canvas that will actually be SAVED.

    Two subtleties this handles, both of which produce false positives if
    ignored:
      * off-view tick labels (see _offview_ticklabels) are excluded;
      * with savefig.bbox='tight' the saved canvas is the tight bbox, which is
        larger than fig.bbox, so containment is tested against the tight box.
    """
    import matplotlib as mpl

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dead = _offview_ticklabels(fig)
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and t not in dead]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes
              for s in ax.spines.values() if s.get_visible()]
    ticklabels = {ax: set(ax.get_xticklabels(which="both")
                          + ax.get_yticklabels(which="both")) for ax in fig.axes}
    findings = []
    for i, (a, ba) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if ba.overlaps(bb):
                findings.append(("text/text", a.get_text()[:40], b.get_text()[:40]))
    for t, bt in texts:
        for s, bs in spines:
            if bt.overlaps(bs) and t not in ticklabels.get(s.axes, ()):
                findings.append(("text/spine", t.get_text()[:40],
                                 f"{s.axes.get_label() or 'ax'}:{s.spine_type}"))

    # text vs DRAWN ARTISTS (lines, markers, reference rules, arrow patches).
    # Without this the gate is blind to a label sitting on top of a plotted
    # mark -- it only ever saw text-on-text and text-on-spine. A tight bbox
    # around a long diagonal line over-reports, so only near-axis-aligned
    # segments (vertical rules, horizontal interval bars, small markers) are
    # tested, which is where real label collisions occur.
    import matplotlib.collections as mcoll
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    for ax in fig.axes:
        arts = [a for a in ax.get_children()
                if isinstance(a, (mlines.Line2D, mpatches.FancyArrowPatch,
                                  mcoll.PathCollection, mcoll.LineCollection))
                and a.get_visible()]
        for t, bt in texts:
            if t.axes is not ax:
                continue
            for a in arts:
                try:
                    ba = a.get_window_extent(r)
                except Exception:
                    continue
                if ba.width <= 0 and ba.height <= 0:
                    continue
                # Two collidable shapes: THIN (rule, interval bar) or COMPACT
                # (a marker -- roughly square, sized by markersize). A
                # min()-only test silently excludes every marker, which is the
                # class that overpaints value labels.
                slim = (min(ba.width, ba.height) <= 12
                        or max(ba.width, ba.height) <= 40)
                # A full-span reference rule (axvline/axhline) is BACKGROUND
                # furniture: labels legitimately sit across it, and flagging
                # every one buries the real findings. Only local artists --
                # data marks, short rules, arrows -- are treated as collidable.
                axb = ax.get_window_extent(r)
                spans_axis = (ba.height >= 0.92 * axb.height
                              or ba.width >= 0.92 * axb.width)
                if slim and not spans_axis and bt.overlaps(ba):
                    findings.append(("text/artist", t.get_text()[:40],
                                     f"{type(a).__name__} "
                                     f"x[{ba.x0:.0f},{ba.x1:.0f}] "
                                     f"y[{ba.y0:.0f},{ba.y1:.0f}]"))
    # containment is tested against the canvas that will actually be written
    if str(mpl.rcParams.get("savefig.bbox")) == "tight":
        tb = fig.get_tightbbox(r)               # inches
        dpi = fig.dpi
        canvas = mpl.transforms.Bbox.from_extents(
            tb.x0 * dpi, tb.y0 * dpi, tb.x1 * dpi, tb.y1 * dpi)
    else:
        canvas = fig.bbox
    pad = 0.5  # px, tolerates antialias rounding
    for t, bt in texts:
        if (bt.x0 < canvas.x0 - pad or bt.x1 > canvas.x1 + pad
                or bt.y0 < canvas.y0 - pad or bt.y1 > canvas.y1 + pad):
            findings.append(("outside-saved-canvas", t.get_text()[:40],
                             f"text x[{bt.x0:.0f},{bt.x1:.0f}] y[{bt.y0:.0f},{bt.y1:.0f}] "
                             f"vs canvas x[{canvas.x0:.0f},{canvas.x1:.0f}] "
                             f"y[{canvas.y0:.0f},{canvas.y1:.0f}]"))
    if verbose:
        print(f"[bbox_check] {len(findings)} finding(s)")
        for f in findings:
            print("   ", f)
    return findings
