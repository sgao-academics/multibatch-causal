# -*- coding: utf-8 -*-
"""Figure 4: pathway over-representation of the inferred per-cancer causal networks.

One grouped dot plot, replacing the 2x2 grid of horizontal bars the plate used to
carry and the 2x2 lollipop that briefly replaced it.

Why neither the bars nor a lollipop can carry this data
-------------------------------------------------------
Among the twenty-four sets this figure draws, all but two have k = K: every gene
of the set that is testable in that cohort was hit.  When k = K the fold
enrichment reduces to N / n -- the cohort's gene count over the universe -- so it
stops depending on the set at all and reports only which cohort was scanned.
That is why the old bars arrived in blocks of identical length (four at 4.55, six
at 4.35, six at 1.06): four different pathways drawn as four identical slabs.  A
lollipop whose stem is that same fold inherits the defect.

What does vary is the number of hit genes, which runs 3-4 in the two cohorts with
a small testable panel and 14-28 in the pooled one.  So the hit count is the axis
of this figure and significance is the fill; the columns are ordered by how many
genes each cohort could test at all, which is also the quantity the degenerate
fold was tracking.

Drawing language
----------------
Same tokens as Figures 1-3, imported from `_figstyle`: the seven muted colours,
type in the palette's indigo, a faint rounded card behind the panel, a rounded
rail behind each column, markers with a soft drop shadow, capsules for the
median bars, and one 8 pt lettering size throughout.

Journal requirements applied here
---------------------------------
* 174 mm wide -- one of the four widths the journal sanctions and the printed
  text width, so the point sizes are the sizes that appear in print.
* Nothing below 8 pt.
* Sans-serif Arial throughout, including the maths text.
* Every encoded distinction survives greyscale: filled against open dots.
* No title inside the artwork.

Output: figures/Fig4.pdf / .png
"""
import os, sys, json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import (PAL, INK, GREY, TICK, TRACK, MM, tidy, card, capsule, dots)

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

per = json.load(open(os.path.join(RES, "_enrichment_percancer.json"), encoding="utf-8"))["per_cancer"]
glo = json.load(open(os.path.join(RES, "_enrichment_global.json"), encoding="utf-8"))


# ---- assertions: the manuscript's textual claims must hold in the data ----
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
print("claim checks passed: LUAD/BRCA named sets present, CHOL has 0 significant sets,")
print("                     global panel contains the two sets named in the text")

FS = 8.0          # the single lettering size of this plate
TOPN = 6          # sets drawn per cohort

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
K_MAX = max(int(e["k"]) for sel in DRAWN for e in sel)
_nk = sum(1 for sel in DRAWN for e in sel if int(e["k"]) == int(e["K"]))
print("hit counts run %d..%d over the %d sets drawn; k == K in %d of them (%.0f%%)"
      % (min(int(e["k"]) for sel in DRAWN for e in sel), K_MAX,
         sum(len(s) for s in DRAWN), _nk,
         100.0 * _nk / sum(len(s) for s in DRAWN)))
for i, ((name, recs, nn, _, _), ns) in enumerate(zip(COH, N_SIG)):
    folds = sorted({float(e["enrichment"]) for e in DRAWN[i]}, reverse=True)
    print("   %-7s n=%-5d  %d of %d sets at p<0.05  fold=%s"
          % (name, nn, ns, len(recs), " / ".join("%.2f" % v for v in folds)))

# 174 x 118 mm.  bbox_inches="tight" is deliberately NOT used: it would grow the
# page by whatever sticks out and shrink the lettering with it; the page is the
# canvas, so the canvas check below is a hard gate.
W_MM, H_MM = 174.0, 118.0
fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
ax = fig.add_axes([0.088, 0.118, 0.894, 0.790])

JIT = [-0.200, -0.120, -0.040, 0.040, 0.120, 0.200]
# The upper third of the axis is label space, not data space: the pooled column
# reaches k = 28, so the column notes have to start clear of that or the note sits
# on the dots.  Hence the 0-46 range for a 3-28 spread of points, with the notes
# at 37.5 and the fold/count strips above them.
for xi, ((name, recs, nn, col, note), sel, ns) in enumerate(zip(COH, DRAWN, N_SIG)):
    # rail: a very light rounded channel, so a column reads as a group
    ax.add_patch(FancyBboxPatch((xi - 0.40, 1.2), 0.80, 43.6,
                                boxstyle="round,pad=0,rounding_size=1.3",
                                fc=TRACK, ec="none", zorder=0))
    for j, e in enumerate(sel):
        sig = float(e["p"]) < 0.05
        if sig:
            dots(ax, [xi + JIT[j]], [int(e["k"])], 52, col, z=5, shadow=True,
                 ec="white", lw=0.7)
        else:
            dots(ax, [xi + JIT[j]], [int(e["k"])], 52, "white", z=5, shadow=False,
                 ec=col, lw=1.2)
    capsule(ax, xi - 0.31, xi + 0.31, float(np.median([int(e["k"]) for e in sel])),
            0.26, col, z=4)
    # column reading: what this cohort's network lands on, in the free upper half
    ax.text(xi, 37.5, note, ha="center", va="top", fontsize=FS, color=INK,
            linespacing=1.45, zorder=7)
    folds = sorted({float(e["enrichment"]) for e in sel}, reverse=True)
    ax.text(xi, 44.0, "fold " + " \u00b7 ".join("%.2f" % v for v in folds),
            ha="center", va="bottom", fontsize=FS, color=col, zorder=7)
    ax.text(xi, 41.0, "%d of %d pass $p<0.05$" % (ns, len(recs)),
            ha="center", va="bottom", fontsize=FS, color=GREY, zorder=7)

ax.set_xticks(range(len(COH)))
# linespacing is not cosmetic: at the default the two rows' bounding boxes touch,
# and the overlap audit flags the name against the "$n$ = ..." row beneath it.
ax.set_xticklabels(["%s\n$n$ = %s" % (c[0], format(c[2], ",")) for c in COH],
                   fontsize=FS, linespacing=1.6)
ax.set_xlim(-0.62, len(COH) - 1 + 0.62)
ax.set_ylim(0, 46)
ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
ax.set_ylabel("hit genes in the enriched set")
for _s in ("top", "right"):
    ax.spines[_s].set_visible(False)
tidy(ax, grid="y")
ax.tick_params(axis="x", length=0)

card(fig, ax, pad=0.008, radius=0.012)


# ---- nothing may leave the canvas: the page IS the canvas here ----
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
print("Fig4 done: %d KB (pdf) / %d KB (png)"
      % (os.path.getsize(os.path.join(FIG, "Fig4.pdf")) // 1024,
         os.path.getsize(os.path.join(FIG, "Fig4.png")) // 1024))
