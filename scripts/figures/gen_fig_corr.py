# -*- coding: utf-8 -*-
"""Per-cancer co-expression matrices for the genes that carry STRING support
(shaped after reference Fig. 8).

Nine cancer types, the twelve highest-degree genes of the STRING-supported network, Spearman
rho within each cohort (lower triangle). Cells that correspond to a causal edge we called in
that cancer type are boxed, so the panel doubles as a check on those calls.

Reads results/_corr_matrix.json; nothing is recomputed here.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

D = json.load(open(os.path.join(RES, "_corr_matrix.json"), encoding="utf-8"))
A = json.load(open(os.path.join(RES, "_string_channel_analysis.json"), encoding="utf-8"))
OV = A["overlaps"]
MAT = D["matrices"]

TOPN = 12
NPANEL = 9
panels = [c for c in D["cancers"] if c in MAT][:NPANEL]

MM = 1.0 / 25.4
# 174 mm = the full-width figure area of the F&IG printed page, matching the submitted figures
# (gen_fig1_landscape.py: 6.85 in). Canvas width == printed width, so the font sizes below are
# the sizes that actually appear in print.
W, H = 174.0, 243.0
fig = plt.figure(figsize=(W * MM, H * MM))

# This panel was the last one still built with the matplotlib defaults, which means DejaVu Sans
# and Type 3 fonts -- the combination journals reject.  Set the same contract as the other
# figures: embedded TrueType, Arial (Helvetica fallback), Arial maths glyphs.
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType, not Type 3
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "text.usetex": False,
})

# The panel is placed at 0.82\textwidth in the supplementary file, so a declared point size
# prints smaller by that factor.  10.2 pt keeps even the smallest label (FS - 1.6) at 7.1 pt.
FS = 10.2
INK = "#1a1a1a"
GREY = "#8a8a8a"
CMAP = LinearSegmentedColormap.from_list("corr", ["#2E5AAC", "#8FB1E0", "#F5F5F5",
                                                  "#F0A08A", "#B40426"])
VMIN, VMAX = -0.6, 1.0
NOTE_H = 4.6


def rect(x_mm, y_top_mm, w_mm, h_mm):
    return [x_mm / W, 1.0 - (y_top_mm + h_mm) / H, w_mm / W, h_mm / H]


texts = []


def T(x_mm, y_mm, s, **kw):
    kw.setdefault("fontsize", FS)
    kw.setdefault("color", INK)
    kw.setdefault("ha", "left")
    kw.setdefault("va", "top")
    t = fig.text(x_mm / W, 1.0 - y_mm / H, s, **kw)
    texts.append(t)
    return t


# No title inside the artwork: the journal asks that illustrations carry no titles of their own,
# and the caption already names the panel.  Dropping it also frees the room the larger type below
# needs, which is what keeps every label above the 7 pt print floor.
TOP = 2.0

colw, rowy = 48.0, 62.0
gx, gy = 4.0, 5.0
lab_l, lab_b = 13.2, 9.0          # room for the gene labels, at the enlarged type size
title_h = 6.0

boxed = 0
tested = 0
allr = []

for pi, cancer in enumerate(panels):
    r, c = divmod(pi, 3)
    x0 = 2.0 + c * (colw + gx)
    y0 = TOP + r * (rowy + gy)

    genes = MAT[cancer]["genes"][:TOPN]
    C = np.array(MAT[cancer]["rho"], dtype=float)[:TOPN, :TOPN]
    n = MAT[cancer]["n"]
    ii, jj = np.tril_indices_from(C, -1)      # off-diagonal only: the diagonal is 1 by definition
    allr.extend(C[ii, jj].tolist())

    mw, mh = colw - lab_l - 2.0, rowy - title_h - lab_b - 2.0
    ax = fig.add_axes(rect(x0 + lab_l, y0 + title_h, mw, mh))

    Cm = np.ma.masked_where(~np.tril(np.ones_like(C, dtype=bool)), C)
    im = ax.imshow(Cm, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect="auto",
                   interpolation="nearest")

    # box the cells that match a causal edge called in this cancer type
    for k, v in OV.items():
        ga, gb = k.split("|")
        if ga not in genes or gb not in genes or cancer not in v["cancers"]:
            continue
        ia, ib = genes.index(ga), genes.index(gb)
        ri, ci = max(ia, ib), min(ia, ib)
        ax.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1, fill=False,
                               edgecolor="#111111", lw=0.85, zorder=5))
        boxed += 1

    k_ = TOPN
    ax.set_xticks(range(k_))
    ax.set_xticklabels(genes, rotation=90, fontsize=FS - 1.4)
    ax.set_yticks(range(k_))
    ax.set_yticklabels(genes, fontsize=FS - 1.4)
    ax.tick_params(length=1.6, pad=1.0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#999999")

    T(x0 + lab_l - 1.0, y0 + 0.4, "%s  (n=%d)" % (cancer, n), fontsize=FS - 0.6,
      fontweight="bold", color=INK)

# ------------------------------------------------------------------ colour bar
cb_w = 5.0
# the old expression put the colour bar at x = 175 mm on a 174 mm canvas, so it was drawn entirely
# off-canvas and never appeared; leave room for the bar plus its tick labels
cb_x = 2.0 + 3 * (colw + gx)
cax = fig.add_axes(rect(cb_x, TOP + title_h + 6.0, cb_w, 52.0))
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(VMIN, VMAX))
cb = fig.colorbar(sm, cax=cax)
cb.set_label("Spearman rho", fontsize=FS - 0.8, labelpad=1.5)
cb.ax.tick_params(labelsize=FS - 1.5, length=1.8)
cb.outline.set_linewidth(0.5)

# ------------------------------------------------------------------ notes
ybot = TOP + 3 * rowy + 2 * gy + 7.0
T(1.0, ybot - 6.0,
  "Boxed cells: a causal edge of ours in that cancer type that STRING also supports "
  "(%d cells across the nine panels)." % boxed, fontsize=FS - 1.6, color=GREY)
T(1.0, ybot + 1.0,
  "Genes are the %d highest-degree members of the STRING-supported network; the nine panels "
  "carry the most supported edges." % TOPN, fontsize=FS - 1.6, color=GREY)
T(1.0, ybot + 8.0,
  "Lower triangles only. Spearman rho within each cohort, on tumours complete for all "
  "%d genes." % TOPN, fontsize=FS - 1.6, color=GREY)
T(1.0, ybot + 15.0,
  "The Y-chromosome and the keratin genes each form a correlated block, and the two blocks "
  "are negatively correlated.", fontsize=FS - 1.6, color=GREY)
T(1.0, ybot + 22.0,
  "Mean off-diagonal rho for these twelve genes is %+.3f; the cohort mean is positive in all "
  "%d cancer types." % (float(np.mean(allr)), len(D["matrices"])),
  fontsize=FS - 1.6, color=GREY)

# ------------------------------------------------------------------ checks
fig.canvas.draw()
rr = fig.canvas.get_renderer()
import matplotlib.transforms as _mtr
# dpi_scale_trans.inverted() lands in INCHES while the bounds below are in MILLIMETRES, so the
# original check compared a 0-7 range against a 0-210 one and could never report anything.
inv = _mtr.Affine2D().scale(25.4 / fig.dpi)

every = list(texts)
for ax_ in fig.get_axes():
    every += list(ax_.texts)
    every += list(ax_.get_xticklabels()) + list(ax_.get_yticklabels())
    every += [ax_.xaxis.label, ax_.yaxis.label]

bad = []
for t in every:
    if not t.get_text():
        continue
    bb = t.get_window_extent(renderer=rr).transformed(inv)
    if bb.x0 < -0.6 or bb.y0 < -0.6 or bb.x1 > W + 0.6 or bb.y1 > H + 0.6:
        bad.append((t.get_text()[:20], round(bb.x0, 1), round(bb.y0, 1),
                    round(bb.x1, 1), round(bb.y1, 1)))

boxes = [(t.get_text()[:16], t.get_window_extent(renderer=rr).transformed(inv))
         for t in texts]
ovl = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i][1], boxes[j][1]
        ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
        iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
        if ix > 1.6 and iy > 1.6:
            ovl.append((boxes[i][0], boxes[j][0], round(ix, 1), round(iy, 1)))

# gene labels must stay inside their own panel and not run into the next cell
lab_bad = []
for ax_ in fig.get_axes():
    for t in list(ax_.get_xticklabels()) + list(ax_.get_yticklabels()):
        bb = t.get_window_extent(renderer=rr).transformed(inv)
        if bb.x1 > W + 0.6 or bb.x0 < -0.6 or bb.y1 > H + 0.6 or bb.y0 < -0.6:
            lab_bad.append((t.get_text(), round(bb.x0, 1), round(bb.y0, 1)))

scale = 174.0 / W          # canvas width is the printed width, so this is 1.0 by construction
print("canvas %.0f x %.0f mm | panels %d | genes %d | boxed cells %d" % (W, H, len(panels), TOPN, boxed))
print("outside canvas: %d %s" % (len(bad), bad[:6]))
print("overlapping fig.text pairs: %d %s" % (len(ovl), ovl[:6]))
print("gene labels out of canvas: %d %s" % (len(lab_bad), lab_bad[:6]))
print("font size FS=%.1f pt -> %.2f pt in the published layout (canvas %.0f mm vs the "
      "174.9 mm text block) | %.2f pt in the author PDF (131 mm)"
      % (FS, FS * 174.9 / W, W, FS * 131.0 / W))
print("mean rho across panels: %+.3f" % float(np.mean(allr)))

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, "fig_corr." + ext), dpi=400, facecolor="white")
print("wrote figures/fig_corr.pdf and .png")
