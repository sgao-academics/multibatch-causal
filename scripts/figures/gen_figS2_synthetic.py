# -*- coding: utf-8 -*-
"""Figure S2: two-stage decomposition on the sparse chain-structured DAG.

Four weight matrices, ground truth above, estimate below.  The panel answers one
question -- does Stage 2 recover the shared backbone, and does the difference
matrix isolate the batch that was actually rewired -- and it can only answer it
if the sign is visible, because Batch 1 rewires one shared edge by flipping its
sign relative to the backbone.

That constraint fixes the two drawing decisions:
*  the cells carry a divergence.  A single-hue ramp would draw the sign-flipped
  edge in the same tint as the edge it replaces, which is exactly the thing the
  panel exists to show.  The ramp is the one the main text already uses for its
  two directions (mist against orchid), not a fresh convention.
*  every annotated cell prints its number, and the number is coloured from the
  cell it sits on rather than from a fixed threshold, so legibility cannot
  depend on which end of the ramp a cell happens to fall.

Rebuilt on the shared drawing language (`_figstyle`); the previous version used
matplotlib's default RdBu, a 6.6 pt tick size and a canvas 18 mm narrower than
the printed measure, which printed the matrix annotations below the journal's
8 pt floor.

Output: figures/FigS2.pdf / .png   (canvas 174 x 192 mm)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _figstyle import (INK, TICK, GREY, FRAME, MM, CMAP_DIV, heatgrid, card)

ROOT = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

FS = 8.0          # every tick, annotation and label
FL = 11.0         # panel letters
W_MM, H_MM = 174.0, 192.0
W, H = W_MM * MM, H_MM * MM
TAU_DISP = 0.12   # print a number once the cell is this far from zero
TAU_EDGE = 0.20   # operating point the caption quotes

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
synth = json.load(open(os.path.join(RES, "synth_ckpt.json"), encoding="utf-8"))
dd = synth["d"]
W0_true = np.array(synth["W0_true"], dtype=float)
W_trues = [np.array(m, dtype=float) for m in synth["W_trues"]]
W1_true = W_trues[1]          # Batch 1: rewired edge + new edge

v6 = json.load(open(os.path.join(RES, "_v6_original_output.json"), encoding="utf-8"))
W0_rec = np.array(v6["W0"], dtype=float)
D1 = np.array(v6["Deltas"][2], dtype=float)
h0 = v6["h_W0"]

# The claims the caption makes, checked against the data before they are drawn.
n_gt = int((np.abs(W0_true) > TAU_EDGE).sum())
sk_true = set(zip(*np.where(np.abs(W0_true) > TAU_EDGE)))
sk_rec = set(zip(*np.where(np.abs(W0_rec) > TAU_EDGE)))
n_shared = len(sk_true & sk_rec)
# W is a DAG, so it is deliberately asymmetric; only the counts are checked.
assert n_gt == 11, "ground-truth backbone is not the 11-edge chain"
assert n_shared == 10, "recovered backbone does not carry 10 of the 11 shared edges"
assert abs(v6["total_ok"] - 14) < 1e-9 and abs(v6["total_gt"] - 15) < 1e-9, \
    "recall in the checkpoint is not 14 of 15"

# --------------------------------------------------------------------------
# Layout, mm from the top-left
# --------------------------------------------------------------------------
TOP = 6.5
MW = 75.0            # matrix side; cells come out square
LEFT, GAP = 7.0, 12.0
R1Y = TOP
R1LBL = 11.0         # tick row + axis label under row 1
R2Y = R1Y + MW + R1LBL
CBY = R2Y + MW + 12.0
CBH = 3.6
assert CBY + CBH + 8.0 <= H_MM, "canvas too short for the colour bar"

X1 = LEFT
X2 = LEFT + MW + GAP


def ax_rect(x0, y0, w, h):
    return [x0 / W_MM, 1.0 - (y0 + h) / H_MM, w / W_MM, h / H_MM]


fig = plt.figure(figsize=(W, H))
fig.patch.set_facecolor("white")


def text_colour(v):
    """Pick the annotation colour from the cell it is printed on."""
    t = np.clip((v + 1.0) / 2.0, 0.0, 1.0)
    r, g, b, _ = CMAP_DIV(t)
    return "white" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.56 else INK


def draw(ax, M, show_all):
    heatgrid(ax, M, cmap=CMAP_DIV, vmin=-1.0, vmax=1.0, lw=0.55)
    ax.set_xticks(range(dd))
    ax.set_yticks(range(dd))
    ax.set_xticklabels([str(i) for i in range(dd)], fontsize=FS)
    ax.set_yticklabels([str(i) for i in range(dd)], fontsize=FS)
    ax.tick_params(length=0, pad=1.6)
    ax.set_xlim(-0.5, dd - 0.5)
    ax.set_ylim(dd - 0.5, -0.5)
    cells = []
    for i in range(dd):
        for j in range(dd):
            v = M[i, j]
            if abs(v) < (0.0 if show_all else TAU_DISP) or v == 0.0:
                continue
            cells.append(ax.text(j, i, "%.2f" % v, ha="center", va="center",
                                 fontsize=FS, color=text_colour(v), zorder=4))
    return cells


axa = fig.add_axes(ax_rect(X1, R1Y, MW, MW))
axb = fig.add_axes(ax_rect(X2, R1Y, MW, MW))
axc = fig.add_axes(ax_rect(X1, R2Y, MW, MW))
axd = fig.add_axes(ax_rect(X2, R2Y, MW, MW))

cells_a = draw(axa, W0_true, True)
cells_b = draw(axb, W1_true, True)
cells_c = draw(axc, W0_rec, False)
cells_d = draw(axd, D1, False)
cell_texts = cells_a + cells_b + cells_c + cells_d

axc.set_xlabel("Source node")
axc.set_ylabel("Target node")
axd.set_xlabel("Source node")

# --------------------------------------------------------------------------
# Colour bar, shared by the four panels
# --------------------------------------------------------------------------
cbw = 56.0
cax = fig.add_axes(ax_rect((W_MM - cbw) / 2.0, CBY, cbw, CBH))
sm = plt.cm.ScalarMappable(cmap=CMAP_DIV, norm=plt.Normalize(-1.0, 1.0))
sm.set_array([])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_ticks([-1, -0.5, 0, 0.5, 1])
cb.ax.tick_params(labelsize=FS, length=1.8, pad=1.4, width=0.5)
cb.set_label("Edge weight", fontsize=FS, labelpad=2.0, color=INK)
for s in list(cb.ax.spines.values()) + [cb.outline]:
    s.set_linewidth(0.5)
    s.set_color(FRAME)

# --------------------------------------------------------------------------
# Cards and panel letters
# --------------------------------------------------------------------------
for ax in (axa, axb, axc, axd):
    card(fig, ax, pad=0.010)
for ax, s in zip((axa, axb, axc, axd), "abcd"):
    bb = ax.get_position()
    fig.text(bb.x0 - 0.022, bb.y1 + 0.010, s + ")", fontsize=FL,
             fontweight="bold", ha="left", va="bottom", color=INK)

# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------
fig.canvas.draw()
rr = fig.canvas.get_renderer()
import matplotlib.transforms as mtr
inv = mtr.Affine2D().scale(25.4 / fig.dpi)

every = []
for ax in (axa, axb, axc, axd) + (cax,):
    every += [t for t in ax.texts if t.get_text()]
    every += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
    every += [ax.xaxis.label, ax.yaxis.label]
# (cax is cb.ax, so its tick labels are already in the list above)

bad, boxes = [], []
for t in every:
    if not t.get_text():
        continue
    bb = t.get_window_extent(renderer=rr).transformed(inv)
    boxes.append((t.get_text()[:18], bb))
    if bb.x0 < -0.6 or bb.y0 < -0.6 or bb.x1 > W_MM + 0.6 or bb.y1 > H_MM + 0.6:
        bad.append((t.get_text()[:18], round(bb.x0, 1), round(bb.y0, 1),
                    round(bb.x1, 1), round(bb.y1, 1)))

ovl = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i][1], boxes[j][1]
        ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
        iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
        if ix > 1.2 and iy > 1.0:
            ovl.append((boxes[i][0], boxes[j][0], round(ix, 1), round(iy, 1)))

# the widest in-cell annotation has to fit inside one cell
cell_mm = MW / dd
_wb = [(t.get_text(), (t.get_window_extent(renderer=rr).transformed(inv).x1
                       - t.get_window_extent(renderer=rr).transformed(inv).x0))
       for t in cell_texts]
widest, widest_w = max(_wb, key=lambda p: p[1])
print("canvas %.0f x %.0f mm | d=%d | cells %.2f mm | FS %.1f pt"
      % (W_MM, H_MM, dd, cell_mm, FS))
print("annotated cells: a %d  b %d  c %d  d %d"
      % (len(cells_a), len(cells_b), len(cells_c), len(cells_d)))
print("ground-truth backbone %d edges | recovered %d of them above tau=%.2f | "
      "h(W0)=%.2e" % (n_gt, n_shared, TAU_EDGE, h0))
print("widest cell annotation %r is %.2f mm vs cell %.2f mm  -> %s"
      % (widest, widest_w, cell_mm,
         "fits" if widest_w < cell_mm - 0.3 else "TOO WIDE"))
print("outside canvas: %d %s" % (len(bad), bad[:6]))
print("overlapping text pairs: %d %s" % (len(ovl), ovl[:6]))
assert widest_w < cell_mm - 0.3, "cell annotation wider than the cell it sits in"

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, "FigS2." + ext), dpi=400, facecolor="white")
print("wrote figures/FigS2.pdf and .png  (%.1f KB / %.1f KB)"
      % (os.path.getsize(os.path.join(FIG, "FigS2.pdf")) / 1024.0,
         os.path.getsize(os.path.join(FIG, "FigS2.png")) / 1024.0))
