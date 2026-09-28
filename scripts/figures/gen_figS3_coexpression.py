# -*- coding: utf-8 -*-
"""Figure S3: co-expression structure of the STRING-supported network.

Nine cohorts, the twelve highest-degree genes of the supported network, Spearman
rho within each cohort, lower triangle only.  Cells that correspond to a causal
edge we called in that cancer type are boxed, so the plate doubles as a check on
those calls.

Rebuilt on the shared drawing language (`_figstyle`).  Two things changed beyond
the palette:

*  The ramp is the paper's own signed ramp (mist against orchid) instead of a
  blue-white-red import, so the supplementary matrices and the main-text panels
  read as one system.  White sits exactly on rho = 0 rather than on the middle of
  the data range: the co-expression block reaches +1 but only about -0.6, so a
  plain range would paint +0.2 in the same tint as -0.1 and every sign reading in
  the plate would be off.
*  The canvas is 174 mm wide and the page includes it at the full text width, so
  the declared point sizes are the printed ones.  The previous version was drawn
  on the same canvas but included at 0.82\\textwidth, which printed the gene
  labels at 7.2 pt -- below the journal's floor -- and made the audit disagree
  with the page.

Output: figures/FigS3.pdf / .png   (canvas 174 x 210 mm)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _figstyle import (PAL, INK, GREY, FRAME, MM, CMAP_DIV,
                       heatgrid, card, two_slope)

ROOT = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

D = json.load(open(os.path.join(RES, "_corr_matrix.json"), encoding="utf-8"))
A = json.load(open(os.path.join(RES, "_string_channel_analysis.json"),
                   encoding="utf-8"))
OV = A["overlaps"]
MAT = D["matrices"]

TOPN = 12
NPANEL = 9
panels = [c for c in D["cancers"] if c in MAT][:NPANEL]

FS = 8.0            # gene labels, notes, tick labels: the printed floor
FT = 8.8            # cohort titles
W_MM, H_MM = 174.0, 210.0
W, H = W_MM * MM, H_MM * MM

# Layout, in mm from the top-left
TOP = 2.0
colw, rowy = 48.0, 54.0
gx, gy = 4.0, 5.0
lab_l, lab_b = 11.0, 11.0
title_h = 6.0
CBW = 5.0

# the ramp has to be centred on zero, and the floor of the range is taken from
# the data rather than assumed
_all = []
for c in panels:
    C = np.array(MAT[c]["rho"], dtype=float)[:TOPN, :TOPN]
    ii, jj = np.tril_indices_from(C, -1)
    _all.extend(C[ii, jj].tolist())
VMIN = np.floor(min(_all) * 10.0) / 10.0
VMAX = 1.0
NORM = two_slope(VMIN, VMAX)

fig = plt.figure(figsize=(W, H))
fig.patch.set_facecolor("white")


def rect(x_mm, y_top_mm, w_mm, h_mm):
    return [x_mm / W_MM, 1.0 - (y_top_mm + h_mm) / H_MM, w_mm / W_MM, h_mm / H_MM]


texts, titles, notes = [], [], []


def T(x_mm, y_mm, s, bucket=None, **kw):
    kw.setdefault("fontsize", FS)
    kw.setdefault("color", INK)
    kw.setdefault("ha", "left")
    kw.setdefault("va", "top")
    t = fig.text(x_mm / W_MM, 1.0 - y_mm / H_MM, s, **kw)
    texts.append(t)
    if bucket is not None:
        bucket.append(t)
    return t


matrix_axes = []


boxed = 0
for pi, cancer in enumerate(panels):
    r, c = divmod(pi, 3)
    x0 = 2.0 + c * (colw + gx)
    y0 = TOP + r * (rowy + gy)

    genes = MAT[cancer]["genes"][:TOPN]
    C = np.array(MAT[cancer]["rho"], dtype=float)[:TOPN, :TOPN]
    n = MAT[cancer]["n"]

    mw, mh = colw - lab_l - 2.0, rowy - title_h - lab_b - 2.0
    ax = fig.add_axes(rect(x0 + lab_l, y0 + title_h, mw, mh))
    matrix_axes.append(ax)

    Cm = np.ma.masked_where(~np.tril(np.ones_like(C, dtype=bool)), C)
    heatgrid(ax, Cm, cmap=CMAP_DIV, norm=NORM, lw=0.45, bad="#F7F9FC")

    for k, v in OV.items():
        ga, gb = k.split("|")
        if ga not in genes or gb not in genes or cancer not in v["cancers"]:
            continue
        ia, ib = genes.index(ga), genes.index(gb)
        ri, ci = max(ia, ib), min(ia, ib)
        ax.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1, fill=False,
                               edgecolor=PAL["violet"], lw=0.9, zorder=6))
        boxed += 1

    ax.set_xticks(range(TOPN))
    ax.set_xticklabels(genes, rotation=90, fontsize=FS, va="top", ha="center")
    ax.set_yticks(range(TOPN))
    ax.set_yticklabels(genes, fontsize=FS)
    ax.tick_params(length=1.4, pad=1.2, width=0.5)
    ax.set_xlim(-0.5, TOPN - 0.5)
    ax.set_ylim(TOPN - 0.5, -0.5)

    T(x0 + lab_l - 1.0, y0 + 0.6, "%s  (n=%d)" % (cancer, n),
      bucket=titles, fontsize=FT, fontweight="bold", color=INK)
    card(fig, ax, pad=0.0, radius=0.010)

# ------------------------------------------------------------------ colour bar
cb_x = 2.0 + 3 * (colw + gx)
cax = fig.add_axes(rect(cb_x, TOP + title_h + 6.0, CBW, 46.0))
sm = plt.cm.ScalarMappable(cmap=CMAP_DIV, norm=NORM)
sm.set_array([])
cb = fig.colorbar(sm, cax=cax)
cb.set_label("Spearman rho", fontsize=FS, labelpad=2.0, color=INK)
cb.ax.tick_params(labelsize=FS, length=1.6, pad=1.2, width=0.5)
cb.outline.set_linewidth(0.5)
cb.outline.set_edgecolor(FRAME)

# ------------------------------------------------------------------ notes
ybot = TOP + 3 * rowy + 2 * gy + 7.0
T(1.0, ybot - 6.0,
  "Boxed cells: a causal edge of ours in that cancer type that STRING also "
  "supports (%d cells across the nine panels)." % boxed, bucket=notes, color=GREY)
T(1.0, ybot + 1.0,
  "Genes are the %d highest-degree members of the STRING-supported network; the "
  "nine panels carry the most supported edges." % TOPN, bucket=notes, color=GREY)
T(1.0, ybot + 8.0,
  "Lower triangles only. Spearman rho within each cohort, on tumors complete for "
  "all %d genes." % TOPN, bucket=notes, color=GREY)
T(1.0, ybot + 15.0,
  "The Y-chromosome and the keratin genes each form a correlated block, and the "
  "two blocks are negatively correlated.", bucket=notes, color=GREY)
T(1.0, ybot + 22.0,
  "Mean off-diagonal rho for these twelve genes is %+.3f; the cohort mean is "
  "positive in all %d cancer types." % (float(np.mean(_all)), len(MAT)),
  bucket=notes, color=GREY)

# ------------------------------------------------------------------ checks
fig.canvas.draw()
rr = fig.canvas.get_renderer()
import matplotlib.transforms as mtr
inv = mtr.Affine2D().scale(25.4 / fig.dpi)

# ``inv`` maps display pixels to mm measured from the BOTTOM of the canvas, so
# every comparison below is converted back to millimetres from the top edge.
def top_mm(t):
    return H_MM - t.get_window_extent(renderer=rr).transformed(inv).y1


def bot_mm(t):
    return H_MM - t.get_window_extent(renderer=rr).transformed(inv).y0


gene_ticks = []
for ax_ in matrix_axes:
    gene_ticks += list(ax_.get_xticklabels()) + list(ax_.get_yticklabels())
cb_ticks = list(cb.ax.yaxis.get_ticklabels())

every = list(titles) + list(notes) + gene_ticks + cb_ticks

# A colourbar built on a TwoSlopeNorm carries one tick label that is never
# placed, and its window extent comes back as NaN.  That is not a rendering
# bug, but it poisons the intersection test: max(0.0, NaN) returns 0.0 while
# min(x, NaN) returns x, so the unplaced label would report a hit against every
# other text in the figure at its full size.  Drop non-finite boxes.
bad, boxes, skipped = [], [], 0
for t in every:
    if not t.get_text():
        continue
    bb = t.get_window_extent(renderer=rr).transformed(inv)
    if not (np.isfinite(bb.x0) and np.isfinite(bb.x1)
            and np.isfinite(bb.y0) and np.isfinite(bb.y1)):
        skipped += 1
        continue
    boxes.append((t.get_text()[:20], bb))
    if (bb.x0 < -0.6 or bb.x1 > W_MM + 0.6
            or H_MM - bb.y1 < -0.6 or H_MM - bb.y0 > H_MM + 0.6):
        bad.append((t.get_text()[:20], round(bb.x0, 1), round(top_mm(t), 1),
                    round(bb.x1, 1), round(bot_mm(t), 1)))

ovl = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i][1], boxes[j][1]
        ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
        iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
        if ix > 1.6 and iy > 1.6:
            ovl.append((boxes[i][0], boxes[j][0], round(ix, 1), round(iy, 1)))

# a note line must not run into the rotated gene labels of the panel above it
gene_bot = max(bot_mm(t) for t in gene_ticks)
note_top = min(top_mm(t) for t in notes)
print("canvas %.0f x %.0f mm | panels %d | genes %d | boxed cells %d"
      % (W_MM, H_MM, len(panels), TOPN, boxed))
print("rho range %.2f .. %.2f | ramp %.2f .. %.2f centred on 0"
      % (min(_all), max(_all), VMIN, VMAX))
print("gene-label cell pitch: %.2f mm across, %.2f mm down"
      % ((colw - lab_l - 2.0) / TOPN, (rowy - title_h - lab_b - 2.0) / TOPN))
print("lowest gene label bottom %.1f mm | first note top %.1f mm | gap %.1f mm"
      % (gene_bot, note_top, note_top - gene_bot))
print("outside canvas: %d %s" % (len(bad), bad[:6]))
print("unplaced tick labels skipped: %d" % skipped)
print("overlapping text pairs: %d %s" % (len(ovl), ovl[:6]))
assert not bad, "text outside the canvas"
assert not ovl, "overlapping text"
assert note_top - gene_bot > 0.5, "notes collide with the gene labels"

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, "FigS3." + ext), dpi=400, facecolor="white")
print("wrote figures/FigS3.pdf and .png  (%.1f KB / %.1f KB)"
      % (os.path.getsize(os.path.join(FIG, "FigS3.pdf")) / 1024.0,
         os.path.getsize(os.path.join(FIG, "FigS3.png")) / 1024.0))
