# -*- coding: utf-8 -*-
"""Figure 4: independent checks on the inferred per-cancer structure.

This plate replaces three separate figures.  The DepMap cross-platform check (old
Figure 8), the baseline comparison (old Figure 9) and the pathway scan (old Figure
4) were three expressions of one argument -- re-examine the structure from outside
and the same sparse, immune-axis answer comes back -- and none of them filled a
page on its own.  Two of the three also began with a bar chart whose bars sat at
zero, which draws a measured zero as an empty space; a zero that has no shape is
indistinguishable from a value that was never measured.

Panels
------
a  platform     eight network edges that map onto the DepMap cell-line panel
                ($n = 1{,}684$), Spearman $r$ between the two genes of each edge.
                Six of the eight are paralogous family members, whose expression is
                co-regulated by shared ancestry rather than by a relation seen in
                tumour tissue; the two that are not are the two lowest, which is
                the point of the panel.
b  method       estimator swap (NOTEARS against the non-DAG ensemble GENIE3,
                under both counting conventions) and aggregation swap (33
                per-cancer fits against one pooled fit).
c  pathway      MSigDB C2 over-representation of the per-cohort networks.

Drawing language comes from `_figstyle`, so this plate shares its tokens with
Figures 1-3 and 5-7: the muted palette, indigo type, rounded cards, capsules on a
light track, markers with a soft shadow, one 8 pt lettering size.

Journal requirements applied here
---------------------------------
* 174 mm wide -- the printed text width, so the point sizes are print sizes.
* Nothing below 8 pt.
* Sans-serif Arial throughout, including the maths text; TrueType, never Type 3.
* Every encoded distinction survives greyscale: filled against open markers, and
  the paralogue block is separated by a rule as well as by colour.
* No title inside the artwork.

Output: figures/Fig4.pdf / .png
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _figstyle import (PAL, INK, GREY, TICK, TRACK, GRID, FRAME, MM,
                       CARD_FC, CARD_EC, tidy, card, capsule, dots,
                       haloed_text_pt)

ROOT = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

sc = json.load(open(os.path.join(RES, "_depmap_scatter.json"), encoding="utf-8"))
pairs, expr = sc["pairs"], sc["expr"]
n_lines = len(sc["models"])
B = json.load(open(os.path.join(RES, "_baseline_stats.json"), encoding="utf-8"))
NT, G3, OV, PL = B["notears"], B["genie3"], B["overlap"], B["pooled"]
per = json.load(open(os.path.join(RES, "_enrichment_percancer.json"),
                     encoding="utf-8"))["per_cancer"]
glo = json.load(open(os.path.join(RES, "_enrichment_global.json"), encoding="utf-8"))

# --------------------------------------------------------------------------
# Assertions: the claims the caption will make have to hold in the data, and
# the paralogue split has to be the one the manuscript names.
# --------------------------------------------------------------------------
PARA = {("MAGEA6", "MAGEA3"), ("CEACAM6", "CEACAM5"), ("SERPINB3", "SERPINB4"),
        ("KRT6C", "KRT6A"), ("KRT6B", "KRT6C"), ("GSTA1", "GSTA2")}
if len(pairs) != 8:
    print("ABORT: expected 8 DepMap edges, found %d" % len(pairs))
    sys.exit(1)
_npar = sum(1 for p in pairs if (p["A"], p["B"]) in PARA)
if _npar != 6:
    print("ABORT: expected 6 paralogous edges, found %d" % _npar)
    sys.exit(1)


def find(cancer, name):
    return next((e for e in per[cancer]["enrichment"] if e["set"] == name), None)


for _c, _n in [("LUAD", "REACTOME_SURFACTANT_METABOLISM"),
               ("LUAD", "REACTOME_TOLL_LIKE_RECEPTOR_CASCADES"),
               ("BRCA", "REACTOME_DEVELOPMENTAL_BIOLOGY"),
               ("BRCA", "ONDER_CDH1_TARGETS_1_DN"),
               ("BRCA", "ONDER_CDH1_TARGETS_2_DN")]:
    if find(_c, _n) is None:
        print("ABORT: missing %s / %s" % (_c, _n))
        sys.exit(1)
if [e for e in per["CHOL"]["enrichment"] if e["p"] < 0.05]:
    print("ABORT: CHOL unexpectedly has significant sets")
    sys.exit(1)
_gnames = {e["set"] for e in glo["global_enrichment"]}
for _n in ["HSIAO_LIVER_SPECIFIC_GENES", "SWEET_LUNG_CANCER_KRAS_UP"]:
    if _n not in _gnames:
        print("ABORT: global set %s missing" % _n)
        sys.exit(1)

# The panel-a argument is that the paralogous edges are the high ones.  If that
# ever stops being true the panel is drawing a different claim than the caption.
_ord = sorted(pairs, key=lambda p: -p["expr_corr"])
_low2 = {(p["A"], p["B"]) for p in _ord[-2:]}
if _low2 & PARA:
    print("ABORT: a paralogous edge is among the two lowest, panel a would mislead")
    sys.exit(1)
print("claim checks passed: 8 DepMap edges, 6 paralogous, both lowest non-paralogous;")
print("                     LUAD/BRCA named sets present, CHOL 0 significant, globals present")

FS = 8.0
TOPN = 6

#   cohort   records                        n_test  colour            note
COH = [("LUAD",   per["LUAD"]["enrichment"],   22,   PAL["mist"],
        "Toll-like receptor\nand surfactant\nmetabolism"),
       ("BRCA",   per["BRCA"]["enrichment"],   23,   PAL["orchid"],
        "developmental and\nCDH1 targets"),
       ("CHOL",   per["CHOL"]["enrichment"],   94,   PAL["moss"],
        "flat; none of 40\nsets significant"),
       ("pooled", glo["global_enrichment"],    1248, PAL["peri"],
        "pooled pan-cancer\nnetwork")]
DRAWN = [sorted(recs, key=lambda e: -float(e["enrichment"]))[:TOPN] for _, recs, _, _, _ in COH]
N_SIG = [sum(1 for e in recs if float(e["p"]) < 0.05) for _, recs, _, _, _ in COH]

# --------------------------------------------------------------------------
# 174 x 228 mm.  bbox_inches="tight" is deliberately NOT used: it grows the page
# by whatever sticks out and shrinks the lettering with it; here the page is the
# canvas, so the canvas check at the end is a hard gate.
# --------------------------------------------------------------------------
W_MM, H_MM = 174.0, 228.0
fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
W_IN, H_IN = fig.get_size_inches()


def card_box(x0, y0, x1, y1, radius=0.014, z=-4):
    """A rounded card in figure coordinates, so a panel that needs a wide label
    column gets a card that covers the column too (ax.get_position() would only
    cover the axes box)."""
    fig.add_artist(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=%.4f" % radius,
        transform=fig.transFigure, fc=CARD_FC, ec=CARD_EC, lw=0.7, zorder=z))


# ======================================================================
# a) platform -- DepMap cross-platform concordance
# ======================================================================
AXA = [0.248, 0.662, 0.212, 0.278]
axa = fig.add_axes(AXA)
order = sorted(pairs, key=lambda p: -p["expr_corr"])
ypos = np.arange(len(order))[::-1]

n_crispr = 0
for yi, p in zip(ypos, order):
    ispar = (p["A"], p["B"]) in PARA
    col = PAL["lilac"] if ispar else PAL["mist"]
    r = p["expr_corr"]
    axa.plot([0, r], [yi, yi], color=TRACK, lw=2.8, solid_capstyle="round", zorder=1)
    if ispar:
        # open marker: the whole distinction has to survive greyscale
        dots(axa, [r], [yi], 48, "white", z=5, ec=col, lw=1.2, shadow=False)
    else:
        dots(axa, [r], [yi], 48, col, z=5, ec="white", lw=0.8)
    axa.text(r + 0.035, yi, "%.2f" % r, va="center", ha="left",
             fontsize=FS, color=INK, zorder=7)
    # Second readout on the same row and the same axis -- both are correlations,
    # so they share a scale.  The whole transpose argument is visible here: the
    # expression correlation is high and the functional one is not.
    cr = p.get("crispr_corr")
    if cr is not None and cr == cr:
        n_crispr += 1
        dots(axa, [cr], [yi], 28, PAL["violet"], z=6, ec="white", lw=0.5,
             marker="s", shadow=False)

axa.set_yticks(ypos)
axa.set_yticklabels([r"$\it{%s}$ $\rightarrow$ $\it{%s}$" % (p["A"], p["B"])
                     for p in order], fontsize=FS)
axa.set_xlim(-0.05, 1.20)
axa.set_ylim(-0.62, 7.60)
axa.set_xticks([0, 0.5, 1.0])
axa.set_xticklabels(["0", "0.5", "1.0"], fontsize=FS)
axa.set_xlabel("Spearman $r$, %s cell lines" % format(n_lines, ","), fontsize=FS)
# rule, as well as colour, between the paralogous block and the two edges that
# are not: the two lowest are the only ones that could count as independent
axa.axhline(1.5, color=FRAME, lw=0.7, ls=(0, (3.0, 2.4)), zorder=2)
tidy(axa)
axa.tick_params(axis="y", length=0)
# The key goes under the axis rather than inside it: the inside upper-right corner
# is where the largest edge sits, and the two labels would otherwise land on the
# value of the row above the rule.  Open against filled carries the distinction on
# its own in greyscale.
_h_open = dots(axa, [], [], 46, "white", z=9, ec=PAL["lilac"], lw=1.2, shadow=False)
_h_fill = dots(axa, [], [], 46, PAL["mist"], z=9, ec="white", lw=0.8)
_h_crispr = dots(axa, [], [], 28, PAL["violet"], z=9, ec="white", lw=0.5,
                 marker="s", shadow=False)
axa.legend([_h_open, _h_fill, _h_crispr],
           ["paralogous", "not paralogous", "CRISPR"],
           loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=3,
           frameon=False, fontsize=FS, handletextpad=0.3, columnspacing=1.1,
           borderpad=0.0, borderaxespad=0.0, scatterpoints=1)

# ======================================================================
# b) method -- estimator swap (top) and aggregation swap (bottom)
# ======================================================================
axb1 = fig.add_axes([0.640, 0.772, 0.322, 0.170])
xs = [0, 1]
d = [NT["directed"]["pct_ge3"], G3["directed"]["pct_ge3"]]
u = [NT["undirected"]["pct_ge3"], G3["undirected"]["pct_ge3"]]
axb1.plot(xs, d, color=PAL["orchid"], lw=1.1, zorder=3, solid_capstyle="round")
axb1.plot(xs, u, color=PAL["mist"], lw=1.1, zorder=3, solid_capstyle="round")
dots(axb1, xs, d, 46, PAL["orchid"], z=5, ec="white", lw=0.8)
dots(axb1, xs, u, 46, PAL["mist"], z=5, ec="white", lw=0.8)
for x, v in zip(xs, d):
    haloed_text_pt(axb1, x, v, "%.1f" % v, dx=0, dy=-12, size=FS,
                   color=PAL["orchid"], ha="center", va="top")
for x, v in zip(xs, u):
    haloed_text_pt(axb1, x, v, "%.1f" % v, dx=0, dy=9, size=FS,
                   color=PAL["mist"], ha="center", va="bottom")
axb1.set_xlim(-0.62, 1.62)
axb1.set_ylim(0, 3.05)
axb1.set_xticks(xs)
axb1.set_xticklabels(["NOTEARS", "GENIE3"], fontsize=FS)
axb1.set_yticks([0, 1, 2, 3])
axb1.set_ylabel("pairs in $\\geq$3\ncohorts (%)", fontsize=FS, linespacing=1.4)
# Inside the axes, upper-left: that corner is empty because the two series start
# low and only rise.  A key hung off the right-hand edge runs off the canvas.
_h_u = dots(axb1, [], [], 40, PAL["mist"], z=9, ec="white", lw=0.8)
_h_d = dots(axb1, [], [], 40, PAL["orchid"], z=9, ec="white", lw=0.8)
axb1.legend([_h_u, _h_d], ["undirected", "directed"], loc="upper left",
            ncol=2, frameon=False, fontsize=FS, handletextpad=0.35,
            columnspacing=1.4, borderpad=0.0, borderaxespad=0.2, scatterpoints=1)
# The third quantity of the old baseline plate: how far the two estimators agree
# with each other, rather than how often each agrees with itself across cohorts.
axb1.text(1.58, 0.44, "%d pairs\nshared" % OV["undirected"]["shared_pairs"],
          ha="right", va="bottom", fontsize=FS, color=GREY, zorder=7,
          linespacing=1.45)
tidy(axb1, grid="y")

axb2 = fig.add_axes([0.640, 0.616, 0.322, 0.124])
tot, pooled = NT["total_edges"], PL["edges"]
fold = PL["fold_vs_per_cancer_total"]
lv = [np.log10(tot), np.log10(pooled)]
axb2.plot([0, 1], lv, color=PAL["violet"], lw=1.1, zorder=3, solid_capstyle="round")
dots(axb2, [0, 1], lv, 46, PAL["violet"], z=5, ec="white", lw=0.8)
axb2.annotate("", xy=(0.5, lv[0]), xytext=(0.5, lv[1]),
              arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.7,
                              shrinkA=2.5, shrinkB=2.5))
axb2.text(0.5, (lv[0] + lv[1]) / 2.0, "  %.0f$\\times$" % round(fold),
          ha="left", va="center", fontsize=FS, color=GREY)
haloed_text_pt(axb2, 0, lv[0], format(tot, ","), dx=-8, dy=0, size=FS,
               color=PAL["violet"], ha="right", va="center")
haloed_text_pt(axb2, 1, lv[1], format(pooled, ","), dx=8, dy=0, size=FS,
               color=PAL["violet"], ha="left", va="center")
axb2.set_xlim(-0.62, 1.62)
axb2.set_ylim(np.log10(19), np.log10(7200))
axb2.set_yticks(np.log10([100, 1000]))
axb2.set_yticklabels(["100", "1,000"], fontsize=FS)
axb2.set_xticks([0, 1])
axb2.set_xticklabels(["per-cancer\n(33 fits)", "pooled\n(1 fit)"],
                     fontsize=FS, linespacing=1.5)
axb2.set_ylabel("edges", fontsize=FS)
tidy(axb2, grid="y")

# ======================================================================
# c) pathway -- MSigDB C2 over-representation (design carried over unchanged)
# ======================================================================
axc = fig.add_axes([0.086, 0.091, 0.898, 0.409])
JIT = [-0.200, -0.120, -0.040, 0.040, 0.120, 0.200]
for xi, ((name, recs, nn, col, note), sel, ns) in enumerate(zip(COH, DRAWN, N_SIG)):
    axc.add_patch(FancyBboxPatch((xi - 0.40, 1.2), 0.80, 43.6,
                                 boxstyle="round,pad=0,rounding_size=1.3",
                                 fc=TRACK, ec="none", zorder=0))
    for j, e in enumerate(sel):
        if float(e["p"]) < 0.05:
            dots(axc, [xi + JIT[j]], [int(e["k"])], 52, col, z=5, shadow=True,
                 ec="white", lw=0.7)
        else:
            dots(axc, [xi + JIT[j]], [int(e["k"])], 52, "white", z=5, shadow=False,
                 ec=col, lw=1.2)
    capsule(axc, xi - 0.31, xi + 0.31,
            float(np.median([int(e["k"]) for e in sel])), 0.26, col, z=4)
    axc.text(xi, 37.5, note, ha="center", va="top", fontsize=FS, color=INK,
             linespacing=1.45, zorder=7)
    folds = sorted({float(e["enrichment"]) for e in sel}, reverse=True)
    axc.text(xi, 44.0, "fold " + " \u00b7 ".join("%.2f" % v for v in folds),
             ha="center", va="bottom", fontsize=FS, color=col, zorder=7)
    axc.text(xi, 41.0, "%d of %d pass $p<0.05$" % (ns, len(recs)),
             ha="center", va="bottom", fontsize=FS, color=GREY, zorder=7)

axc.set_xticks(range(len(COH)))
# linespacing is not cosmetic: at the default the two rows' bounding boxes touch
# and the overlap audit flags the name against the "$n$ = ..." row beneath it.
axc.set_xticklabels(["%s\n$n$ = %s" % (c[0], format(c[2], ",")) for c in COH],
                    fontsize=FS, linespacing=1.6)
axc.set_xlim(-0.62, len(COH) - 1 + 0.62)
axc.set_ylim(0, 46)
axc.set_yticks([0, 5, 10, 15, 20, 25, 30])
axc.set_ylabel("hit genes in the enriched set")
for _s in ("top", "right"):
    axc.spines[_s].set_visible(False)
tidy(axc, grid="y")
axc.tick_params(axis="x", length=0)

# ---- cards and panel letters ------------------------------------------------
card_box(0.058, 0.578, 0.505, 0.958)          # a, covering its label column
card_box(0.602, 0.578, 0.978, 0.958)          # b, covering both sub-axes
card(fig, axc, pad=0.008, radius=0.012)       # c
for _ax, _s in ((axa, "a"), (axb1, "b"), (axc, "c")):
    _bb = _ax.get_position()
    fig.text(_bb.x0 - 0.027, _bb.y1 + 0.0065, _s + ")", fontsize=11,
             fontweight="bold", ha="left", va="bottom", color=INK)

# ---- nothing may leave the canvas: the page IS the canvas here --------------
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_W, _H = fig.get_size_inches() * fig.dpi
_bad = []
for _t in fig.findobj(matplotlib.text.Text):
    if _t.axes is None or _t.get_figure() is not fig:
        continue
    if not _t.get_visible() or not _t.get_text().strip():
        continue
    _b = _t.get_window_extent(renderer=_rend)
    if _b.x0 < -12 or _b.y0 < -12 or _b.x1 > _W + 12 or _b.y1 > _H + 12:
        _bad.append((_t.get_text()[:34].replace("\n", " "), round(_b.x0, 1),
                     round(_b.x1, 1)))
if _bad:
    print("ABORT: %d text elements leave the canvas" % len(_bad))
    for b in _bad[:12]:
        print("   ", b)
    sys.exit(1)
print("canvas check passed: all text inside %.2f x %.2f in" % tuple(fig.get_size_inches()))

fig.savefig(os.path.join(FIG, "Fig4.pdf"), facecolor="white")
fig.savefig(os.path.join(FIG, "Fig4.png"), facecolor="white")
plt.close(fig)
print("  panel a: %d of %d edges carry a CRISPR readout" % (n_crispr, len(pairs)))
print("  panel b: %d undirected pairs shared by the two estimators" % OV["undirected"]["shared_pairs"])
print("Fig4 done: %d KB (pdf) / %d KB (png)"
      % (os.path.getsize(os.path.join(FIG, "Fig4.pdf")) // 1024,
         os.path.getsize(os.path.join(FIG, "Fig4.png")) // 1024))
