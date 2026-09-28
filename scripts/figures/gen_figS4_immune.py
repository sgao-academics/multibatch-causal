# -*- coding: utf-8 -*-
"""Figure S4: does each cohort's own hub gene track the immune microenvironment?

Two matrices, one per molecular layer: rows are the 28 TISIDB immune cell types,
columns the 30 TCGA cohorts carrying an immune-abundance profile, and each cell is
the Spearman rho between that cohort's own hub gene and that cell type, with
FDR-controlled significance marked.

Two structural changes from the previous version.

*  The panels are **side by side**, not stacked.  The binding constraint on this
  plate is the 28-row axis, which needs 28 x 2.9 mm before the row labels start
  to touch; stacked, that alone fills a page and pushes the caption over.  Side by
  side the two layers still only cost one label column each in height, the plate
  becomes 134 mm tall, and the reader compares expression against methylation
  across a 5 mm gap instead of down a page.  Both panels are labelled with the
  cohort names, because the two matrices sit at different x offsets: a label
  column drawn under the second panel alone would not line up with the first, and
  panel a would then have no way of naming its own columns.
*  The colour ramp is the paper's own (mist against orchid).  The previous version
  imported a blue-white-red map, which is the one convention this manuscript uses
  nowhere else.

The four note lines that used to sit inside the canvas stay in the LaTeX caption,
and the one title line that duplicated the caption's first sentence is gone -- the
journal asks that an illustration carry no title of its own.

Output: figures/FigS4.pdf / .png   (canvas 174 x 134 mm)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _figstyle import (INK, GREY, FRAME, TICK, BOXFC, BOXEC, MM, CMAP_DIV,
                       heatgrid, card)

BASE = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(BASE, "results")
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

FS = 8.0
FL = 11.0
W_MM, H_MM = 174.0, 134.0
W, H = W_MM * MM, H_MM * MM

CELL_SHORT = {
    "Act_CD8": "Activated CD8", "Tcm_CD8": "Central-mem. CD8",
    "Tem_CD8": "Effector-mem. CD8", "Act_CD4": "Activated CD4",
    "Tcm_CD4": "Central-mem. CD4", "Tem_CD4": "Effector-mem. CD4",
    "Tfh": "Tfh", "Tgd": "gd T", "Th1": "Th1", "Th17": "Th17", "Th2": "Th2",
    "Treg": "Treg", "Act_B": "Activated B", "Imm_B": "Immature B",
    "Mem_B": "Memory B", "NK": "NK", "CD56bright": "CD56 bright",
    "CD56dim": "CD56 dim", "MDSC": "MDSC", "NKT": "NKT",
    "Act_DC": "Activated DC", "pDC": "pDC", "iDC": "iDC",
    "Macrophage": "Macrophage", "Eosinophil": "Eosinophil", "Mast": "Mast",
    "Monocyte": "Monocyte", "Neutrophil": "Neutrophil",
}
LAYERS = [("expr", "a", "Hub-gene expression"),
          ("meth", "b", "Hub-gene methylation")]
IMMUNE_HUB = {"CD79A", "MZB1", "MGC29506", "FCRL5", "CD14", "CHGB", "CD19",
              "MS4A1"}

D = json.load(open(os.path.join(RES, "_immune_corr_all.json"), encoding="utf-8"))
CELLS, ORDER, R = D["cells"], D["order"], D["result"]
NC, NK = len(CELLS), len(ORDER)

M = {m: np.full((NC, NK), np.nan) for m, _, _ in LAYERS}
Q = {m: np.full((NC, NK), np.nan) for m, _, _ in LAYERS}
P = {m: np.full((NC, NK), np.nan) for m, _, _ in LAYERS}
for m, _, _ in LAYERS:
    for j, c in enumerate(ORDER):
        for i, cell in enumerate(CELLS):
            v = R.get(c, {}).get(m, {}).get(cell)
            if v is not None:
                M[m][i, j], P[m][i, j], Q[m][i, j] = v[0], v[1], v[2]

# --------------------------------------------------------------------------
# The claims the caption makes, checked before they are drawn.
# --------------------------------------------------------------------------
# The counts the caption quotes are the FDR-controlled ones -- the filled
# markers -- not the raw p < 0.05 ones; the two differ by about 5%.
n_expr = int(np.isfinite(P["expr"]).sum())
n_meth = int(np.isfinite(P["meth"]).sum())
sig_expr = int(np.nansum((Q["expr"] < 0.05).astype(float)))
pos_expr = int(np.nansum(((Q["expr"] < 0.05) & (M["expr"] > 0)).astype(float)))
sig_meth = int(np.nansum((Q["meth"] < 0.05).astype(float)))
neg_meth = int(np.nansum(((Q["meth"] < 0.05) & (M["meth"] < 0)).astype(float)))
n_bold = sum(1 for c in ORDER if R[c]["hub"].upper() in IMMUNE_HUB)

# the layer that is not drawn still has to agree with the sentence about it
cnv_m = np.full((NC, NK), np.nan)
cnv_q = np.full((NC, NK), np.nan)
for j, c in enumerate(ORDER):
    for i, cell in enumerate(CELLS):
        v = R.get(c, {}).get("cnv", {}).get(cell)
        if v is not None:
            cnv_m[i, j], cnv_q[i, j] = v[0], v[2]
n_cnv = int(np.isfinite(cnv_q).sum())
sig_cnv = int(np.nansum((cnv_q < 0.05).astype(float)))

print("expression : %d of %d pairs FDR-significant, %d of them positive"
      % (sig_expr, n_expr, pos_expr))
print("methylation: %d of %d pairs FDR-significant, %d of them negative"
      % (sig_meth, n_meth, neg_meth))
print("copy number: %d of %d pairs FDR-significant (reported, not drawn)"
      % (sig_cnv, n_cnv))
print("cohorts carrying an immune-lineage hub (bold): %d" % n_bold)

# every one of these is a number the caption states
assert (sig_expr, pos_expr) == (416, 306), "caption quotes 306 of 416 for expression"
assert (sig_meth, neg_meth) == (270, 175), "caption quotes 175 of 270 for methylation"
assert (sig_cnv, n_cnv) == (44, 756), "caption quotes 44 of 756 for copy number"
assert n_meth // NC == 21, "caption says methylation covers 21 cohorts"

# --------------------------------------------------------------------------
# Layout, mm from the top-left
# --------------------------------------------------------------------------
TOP = 4.5
LBL = 5.5            # the "a) Hub-gene expression" line
MH = 88.0            # matrix height: 28 rows -> 3.14 mm per row
GAPB = 5.0           # between the two matrices
LEFT, RIGHT = 27.5, 4.0
XT = 12.5            # rotated cohort labels
BLOCK = 21.0         # legend and colour bar band
BOT = 4.0
PW = (W_MM - LEFT - GAPB - RIGHT) / 2.0
MY = TOP + LBL


def rect(x0, y0, w, h):
    return [x0 / W_MM, 1.0 - (y0 + h) / H_MM, w / W_MM, h / H_MM]


fig = plt.figure(figsize=(W, H))
fig.patch.set_facecolor("white")

axes = []
for k, (mod, letter, title) in enumerate(LAYERS):
    x0 = LEFT + k * (PW + GAPB)
    ax = fig.add_axes(rect(x0, MY, PW, MH))
    axes.append(ax)

    heatgrid(ax, M[mod], cmap=CMAP_DIV, vmin=-1.0, vmax=1.0, lw=0.0,
             bad="#E8E8E8")
    ax.set_xlim(-0.5, NK - 0.5)
    ax.set_ylim(NC - 0.5, -0.5)
    ax.set_yticks(range(NC))
    ax.set_yticklabels([CELL_SHORT[c] for c in CELLS], fontsize=FS)
    # Both panels carry the cohort names.  The two matrices sit at different x
    # offsets -- one label column drawn under the second one alone does line up
    # with the first, so panel a would have no way of naming its own columns.
    ax.set_xticks(range(NK))
    ax.set_xticklabels(ORDER, fontsize=FS, rotation=90, va="top", ha="center")
    for lbl, c_ in zip(ax.get_xticklabels(), ORDER):
        if R[c_]["hub"].upper() in IMMUNE_HUB:
            lbl.set_fontweight("bold")
    ax.tick_params(length=1.2, pad=1.3, width=0.4)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color(FRAME)

    # Significance.  Both markers carry a ring in the opposite value, so one is
    # legible on a deep cell and the other on a pale one; a dark dot on a dark
    # cell (what the previous version drew) is not a mark at all.
    for i in range(NC):
        for j in range(NK):
            pv, qv = P[mod][i, j], Q[mod][i, j]
            if not np.isfinite(qv):
                continue
            if qv < 0.05:
                ax.plot(j, i, marker="o", ms=1.6, mfc=INK, mec="white",
                        mew=0.3, zorder=4)
            elif pv < 0.05:
                ax.plot(j, i, marker="o", ms=1.6, mfc="white", mec=INK,
                        mew=0.3, zorder=4)

    # Panel letter (11 pt, as in Figures 1-7) and the layer name beside it.
    fig.text((x0 - 5.5) / W_MM, 1.0 - (MY - 1.4) / H_MM, letter + ")",
             fontsize=FL, fontweight="bold", ha="left", va="bottom", color=INK)
    fig.text(x0 / W_MM, 1.0 - (MY - 2.2) / H_MM, title, fontsize=FS + 0.8,
             ha="left", va="bottom", color=INK, fontweight="bold")
    card(fig, ax, pad=0.0, radius=0.010)

# --------------------------------------------------------------------------
# Legend and colour bar, sharing one band under the matrices
# --------------------------------------------------------------------------
YB = MY + MH + XT
f_handles = [
    Line2D([], [], marker="o", ls="none", ms=3.0, mfc="white", mec=INK, mew=0.5),
    Line2D([], [], marker="o", ls="none", ms=3.0, mfc=INK, mec="white", mew=0.5),
]
leg = fig.legend(f_handles,
                 ["$p < 0.05$", "$p < 0.05$ and FDR $< 0.05$"],
                 loc="upper left", bbox_to_anchor=(1.0 / W_MM, 1.0 - (YB + 1.0) / H_MM),
                 frameon=False, fontsize=FS, handletextpad=0.4, labelspacing=0.45,
                 borderpad=0.0, borderaxespad=0.0)
fig.text(1.0 / W_MM, 1.0 - (YB + 1.0) / H_MM, "Significance", fontsize=FS,
         ha="left", va="bottom", color=INK, fontweight="bold")

cbw, cbx = 56.0, 100.0
cax = fig.add_axes(rect(cbx, YB + 3.0, cbw, 3.6))
sm = plt.cm.ScalarMappable(cmap=CMAP_DIV, norm=plt.Normalize(-1.0, 1.0))
sm.set_array([])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_ticks([-1, -0.5, 0, 0.5, 1])
cb.ax.tick_params(labelsize=FS, length=1.8, pad=1.4, width=0.5)
cb.set_label("Spearman rho", fontsize=FS, labelpad=2.0, color=INK)
for s in list(cb.ax.spines.values()) + [cb.outline]:
    s.set_linewidth(0.5)
    s.set_color(FRAME)

# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------
fig.canvas.draw()
r = fig.canvas.get_renderer()
import matplotlib.transforms as mtr
inv = mtr.Affine2D().scale(25.4 / fig.dpi)


def top_mm(t):
    return H_MM - t.get_window_extent(renderer=r).transformed(inv).y1


def bot_mm(t):
    return H_MM - t.get_window_extent(renderer=r).transformed(inv).y0


every = []
for ax in axes + [cax]:
    every += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
every += list(fig.texts) + list(leg.get_texts())

bad, boxes, skipped = [], [], 0
for t in every:
    if not t.get_text():
        continue
    bb = t.get_window_extent(renderer=r).transformed(inv)
    if not (np.isfinite(bb.x0) and np.isfinite(bb.x1)
            and np.isfinite(bb.y0) and np.isfinite(bb.y1)):
        skipped += 1
        continue
    boxes.append((t.get_text()[:18], bb))
    if (bb.x0 < -0.6 or bb.x1 > W_MM + 0.6
            or top_mm(t) < -0.6 or bot_mm(t) > H_MM + 0.6):
        bad.append((t.get_text()[:18], round(bb.x0, 1), round(top_mm(t), 1),
                    round(bb.x1, 1), round(bot_mm(t), 1)))

ov = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i][1], boxes[j][1]
        ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
        iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
        if ix > 1.4 and iy > 1.4:
            ov.append((boxes[i][0], boxes[j][0], round(ix, 1), round(iy, 1)))

# the row labels are the binding constraint: their pitch must clear the glyphs
pitch = MH / NC
ylab_w = max(t.get_window_extent(renderer=r).transformed(inv).x1
             for t in axes[0].get_yticklabels()) 
print("canvas %.0f x %.0f mm | %d cell types x %d cohorts | FS %.1f pt"
      % (W_MM, H_MM, NC, NK, FS))
print("row pitch %.2f mm | widest row label ends at x %.1f mm (room to %.1f)"
      % (pitch, ylab_w, LEFT))
print("column pitch %.2f mm" % (PW / NK))
print("outside canvas: %d %s" % (len(bad), bad[:6]))
print("unplaced labels skipped: %d" % skipped)
print("overlapping text pairs: %d %s" % (len(ov), ov[:6]))
assert not bad, "text outside the canvas"
assert not ov, "overlapping text"
assert ylab_w < LEFT - 0.8, "row labels run into the first matrix"
assert n_bold == 4, "the caption quotes four immune-lineage hubs"

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGDIR, "FigS4.%s" % ext), dpi=400, facecolor="white")
print("wrote figures/FigS4.pdf and .png  (%.1f KB / %.1f KB)"
      % (os.path.getsize(os.path.join(FIGDIR, "FigS4.pdf")) / 1024.0,
         os.path.getsize(os.path.join(FIGDIR, "FigS4.png")) / 1024.0))
