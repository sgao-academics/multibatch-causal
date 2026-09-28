# -*- coding: utf-8 -*-
"""Shared drawing language for the MultiBatch figure set.

Extracted from ``gen_fig1_main.py`` so that every later figure can be drawn on
exactly the same grid instead of re-declaring the tokens: one muted cool
palette, all type set in indigo rather than black, a faint rounded card behind
each panel, capsules resting on a very light track, markers carrying a soft
drop shadow rather than a hard outline, and rounded annotation containers.

``gen_fig1_main.py`` keeps its own private copy of these helpers on purpose --
it is already signed off, and re-plumbing it would risk changing a figure that
has been verified.  This module exists so the remaining panels reuse the same
language without touching it.
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import offset_copy

MM = 1.0 / 25.4  # mm -> inch
W = 174.0 * MM   # canvas width, matching every other figure in the package

# --------------------------------------------------------------------------
# Palette, taken from the reference swatch strip.  mm note: three of these
# codes arrived seven digits long in the original artwork; the values below are
# the six-digit readings and are the ones every figure now shares.
# --------------------------------------------------------------------------
PAL = {
    "mist":   "#4D6EAF",   # pale mist blue
    "peri":   "#7A9AC3",   # periwinkle blue
    "lilac":  "#B0B9CB",   # lilac lavender
    "orchid": "#D0A0B6",   # orchid pink
    "moss":   "#8FB578",   # soft moss green
    "violet": "#9A5A9F",   # muted violet purple
    "indigo": "#3F3A5B",   # indigo plum
}

# Monotone cool ramp for the matrices; the floor is a faint tint rather than
# pure white, so an all-zero matrix still reads as a matrix and not as missing
# artwork.
CMAP = LinearSegmentedColormap.from_list(
    "mist_blue", ["#E9EEF5", "#C6D3E5", "#A9BFD9", "#7A9AC3", "#5A6A94", "#3F3A5B"])

# Signed ramp, for matrices whose cells carry a sign (edge weights, correlations).
# A diverging map is unavoidable there -- collapsing sign into one hue would hide
# the one thing those panels are about, e.g. the sign-flipped rewire in Fig. S2.
# The poles are the two colours the forest plot already uses for the two
# directions (mist = protective, orchid = risk), so the supplementary panels speak
# the same language as the main text instead of importing a second convention.
# Both poles are pushed to a deeper tint of the same hue so the two ends differ in
# luminance as well as in hue and stay separable in greyscale.
CMAP_DIV = LinearSegmentedColormap.from_list(
    "mist_orchid", ["#33507F", "#4D6EAF", "#A9BFD9", "#F4F6FA",
                    "#E7C6D3", "#D0A0B6", "#A8638A"])

# Three categorical series for line panels.  Drawn from the same swatch strip as
# everything else, with light/dark alternation so the set survives greyscale.
SERIES = [("mist", "o"), ("orchid", "s"), ("moss", "^")]

# --- design tokens ---------------------------------------------------------
INK = "#3F3A5B"      # all text, from the palette's deepest colour
TICK = "#5B6478"     # tick labels, one step lighter than the text
GREY = "#8A93A6"     # de-emphasised labels
GRID = "#EDF1F7"     # grid
TRACK = "#EFF3F8"    # capsule track behind bars
FRAME = "#C9D2E0"    # spines
BOXFC = "#FBFCFE"    # rounded annotation container fill
BOXEC = "#DDE4EF"    # rounded annotation container edge
CARD_FC = "#FAFBFE"  # faint rounded card behind each panel
CARD_EC = "#E7EDF6"
SHADOW = "#AEB8CC"   # drop shadow under markers

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType, not Type 3
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    # Symbols with no Arial glyph at all are taken from Computer Modern Symbol on
    # purpose.  The only one in this figure set is the "greater than or similar
    # to" in the driver-entry note, and the manuscript writes that same relation
    # as \gtrsim in its text and in the Table 9 caption -- so the fallback makes
    # the artwork agree with the prose instead of fighting it.  (matplotlib
    # accepts only cm, stix and stixsans here.)
    "mathtext.fallback": "cm",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.6,
    "axes.edgecolor": FRAME, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": TICK, "ytick.color": TICK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 400, "savefig.dpi": 400, "text.usetex": False,
})


def tidy(ax, grid=None):
    """One axis language for every panel: open frame, thin ticks, faint grid."""
    ax.tick_params(direction="out", length=2.4, width=0.6)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def card(fig, ax, pad=0.011, radius=0.016, z=-4):
    """A faint rounded container behind a panel -- the reference artwork's habit."""
    bb = ax.get_position()
    fig.add_artist(FancyBboxPatch(
        (bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad, bb.height + 2 * pad,
        boxstyle="round,pad=0,rounding_size=%.4f" % radius,
        transform=fig.transFigure, fc=CARD_FC, ec=CARD_EC, lw=0.7, zorder=z))


def px_per_data(ax):
    """Display pixels per data unit, in x and y, after the figure has been drawn.

    Must be called only once ``set_xlim``/``set_ylim`` are final: before that
    the axes still carry their default 0-1 ranges and the ratio comes out
    wrong by more than an order of magnitude.
    """
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((1.0, 1.0))
    return (p1[0] - p0[0]), (p1[1] - p0[1])


def capsule(ax, x0, x1, y, h, color, z=3):
    """A horizontal capsule (rounded-end bar) in data coordinates.

    The cap radius is chosen so the ends stay circular on screen regardless of
    how the x and y axes are scaled -- a rounded rectangle drawn straight in
    data units would come out elliptical here.
    """
    sx, sy = px_per_data(ax)
    d_px = abs(h) * sy
    rad = (d_px / 2.0) / sx
    w = x1 - x0
    if w <= 2.0 * rad:
        rad = w / 2.0
    ax.add_patch(Rectangle((x0 + rad, y - h / 2.0), max(w - 2.0 * rad, 1e-9), h,
                           fc=color, ec="none", zorder=z))
    if w > 1e-9:
        d_pt = d_px * 72.0 / ax.figure.dpi
        ax.scatter([x0 + rad, x1 - rad], [y, y], s=d_pt * d_pt,
                   c=color, edgecolors="none", zorder=z)


def dots(ax, x, y, s, color, z=4, shadow=True, ec="white", lw=0.6,
         marker="o", alpha=1.0):
    """Markers with a soft drop shadow instead of a heavy outline."""
    if shadow:
        tr = offset_copy(ax.transData, fig=ax.figure, x=0.9, y=-0.9, units="points")
        ax.scatter(x, y, s=s * 1.04, c=SHADOW, alpha=0.32, edgecolors="none",
                   marker=marker, zorder=z - 0.5, transform=tr, clip_on=True)
    return ax.scatter(x, y, s=s, c=color, edgecolors=ec, linewidths=lw,
                      marker=marker, zorder=z, alpha=alpha)


def haloed_text(ax, x, y, s, size=8, color=INK, halo="white", hw=2.2, z=5, **kw):
    """Text with a soft halo, drawn as two stacked objects.

    Path effects on text rasterise the glyphs to outlines, which would break any
    later text extraction from the PDF, so only the underlying halo copy is
    stroked (that copy is what stops a track showing through the letters); the
    copy on top is ordinary text and stays extractable.
    """
    ax.text(x, y, s, fontsize=size, color=halo, zorder=z,
            path_effects=[pe.withStroke(linewidth=hw, foreground=halo)], **kw)
    return ax.text(x, y, s, fontsize=size, color=color, zorder=z + 0.1, **kw)


def haloed_text_pt(ax, x, y, s, dx=0.0, dy=0.0, size=8, color=INK, halo="white",
                   hw=2.2, z=5, **kw):
    """``haloed_text`` placed by a fixed offset in points rather than data units.

    Needed whenever the thing the text has to clear is a marker whose size is
    itself physical: a data-unit offset would have to be recomputed for every
    point and would still fail on the axis with the smallest span, where one
    millimetre of marker is a large fraction of the range.  Offsets in points
    clear the marker whatever the axis scaling is.
    """
    ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=size, color=halo, zorder=z,
                path_effects=[pe.withStroke(linewidth=hw, foreground=halo)], **kw)
    return ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points",
                       fontsize=size, color=color, zorder=z + 0.1, **kw)


def panel_letters(fig, pairs, dx=-0.027, dy=0.007, size=11):
    """Panel letters in figure coordinates.

    Axes-fraction offsets push the left-hand column's letters off the canvas,
    so the position is taken from each axes' figure-space bounding box.
    """
    for ax, s in pairs:
        bb = ax.get_position()
        fig.text(bb.x0 + dx, bb.y1 + dy, s + ")", fontsize=size,
                 fontweight="bold", ha="left", va="bottom", color=INK)


def two_slope(lo, hi, centre=0.0):
    """Normaliser that puts the neutral colour exactly on ``centre``.

    A plain ``vmin/vmax`` pair only puts white at zero when the data happen to be
    symmetric.  Correlation matrices are not: the co-expression block runs to
    +1 but only to about -0.6, so a plain [-0.6, 1] range paints +0.2 in the
    same tint as -0.1 and every sign reading in the panel goes wrong.
    """
    return TwoSlopeNorm(vcenter=centre, vmin=lo, vmax=hi)


def heatgrid(ax, M, cmap=None, vmin=-1.0, vmax=1.0, norm=None, lw=0.5,
             bad="#EDF1F7"):
    """A matrix drawn as a grid of cells, with a hairline between cells.

    The hairline is what makes a small matrix read as a table of values rather
    than as a photograph: at the size these panels print, a bare imshow of a
    10 x 10 weight matrix is a smudge and the reader cannot see where one cell
    ends and the next begins.  Grid lines are drawn in white on the cell edges,
    never through the cell centres, so they cannot be mistaken for data.
    """
    cm = CMAP if cmap is None else cmap
    if bad is not None:
        cm = cm.copy()
        cm.set_bad(bad)
    if norm is None:
        im = ax.imshow(M, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto",
                       interpolation="nearest", origin="upper", zorder=1)
    else:
        im = ax.imshow(M, cmap=cm, norm=norm, aspect="auto",
                       interpolation="nearest", origin="upper", zorder=1)
    nr, nc = M.shape
    ax.set_xticks(np.arange(-0.5, nc, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, nr, 1.0), minor=True)
    ax.grid(which="minor", color="white", lw=lw)
    ax.tick_params(which="minor", length=0)
    ax.set_axisbelow(False)
    for sp in ax.spines.values():
        sp.set_color(FRAME)
        sp.set_linewidth(0.6)
    return im
