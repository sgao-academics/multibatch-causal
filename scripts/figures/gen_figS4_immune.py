# -*- coding: utf-8 -*-
"""Figure: does each cohort's own hub gene track the immune microenvironment?

Two stacked matrices, one per molecular layer, in the shape of the reference paper's Fig. 5:
rows are the 28 TISIDB immune cell types, columns are the 30 TCGA cohorts that carry an
immune-abundance profile, and each cell is the Spearman rho between that cohort's own hub gene
and that immune cell type, with FDR-controlled significance marked.

The two panels carry one result: hub expression correlates positively with immune abundance, hub
methylation negatively. The third layer that was drawn here -- copy number -- has almost no
signal (44 of 756 pairs significant, the matrix read as an empty grid), so it is reported as a
sentence in the caption instead of occupying a third of the canvas.

Data: results/_immune_corr_all.json   (see scripts/figures/_analyze_immune_all.py)

Layout notes.  Canvas 174 x 205 mm.  The figures of this manuscript are drawn on the journal's
printed measure (174 mm), so a figure placed at \textwidth in the supplementary file prints at
1:1 and the in-figure type of 7.6 pt is the printed size.  The 28-row matrices are the binding
constraint, so both panels share a single x tick row under the bottom panel only, and the four
note lines that used to sit inside the canvas now live in the LaTeX caption.
Writes a NEW pair of files; it never overwrites an existing figure.
"""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE, "results")
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# 174 mm = the full-width figure area of the F&IG printed page, matching the submitted figures
# (gen_fig1_landscape.py: 6.85 in). Canvas width == printed width, so font sizes are print sizes.
W_MM, H_MM = 174.0, 205.0
PRINT_SCALE = 174.0 / W_MM
PRINT_FLOOR = 5.5
FS = 7.6
MM = 72.0 / 25.4
W, H = W_MM * MM, H_MM * MM

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    # keep the maths glyphs in Arial too; the DejaVu default would otherwise supply the
    # rho, the <= comparison and the significance markers as a second, unmatched family
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "font.size": FS, "axes.linewidth": 0.45, "text.usetex": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "legend.frameon": False, "legend.fontsize": FS,
    "xtick.major.size": 1.5, "ytick.major.size": 1.5,
    "xtick.major.width": 0.45, "ytick.major.width": 0.45,
})
BODY, GREY, FAINT = "#1a1a1a", "#6E6E6E", "#d8d8d8"
CMAP = LinearSegmentedColormap.from_list("cwt", ["#3B4CC0", "#8FB1E0", "#F2F2F2",
                                                 "#F0A08A", "#B40426"])
CMAP.set_bad("#e8e8e8")          # cohorts with no data in that layer

CELL_SHORT = {
    "Act_CD8": "Activated CD8", "Tcm_CD8": "Central-mem. CD8", "Tem_CD8": "Effector-mem. CD8",
    "Act_CD4": "Activated CD4", "Tcm_CD4": "Central-mem. CD4", "Tem_CD4": "Effector-mem. CD4",
    "Tfh": "Tfh", "Tgd": "gd T", "Th1": "Th1", "Th17": "Th17", "Th2": "Th2", "Treg": "Treg",
    "Act_B": "Activated B", "Imm_B": "Immature B", "Mem_B": "Memory B", "NK": "NK",
    "CD56bright": "CD56 bright", "CD56dim": "CD56 dim", "MDSC": "MDSC", "NKT": "NKT",
    "Act_DC": "Activated DC", "pDC": "pDC", "iDC": "iDC", "Macrophage": "Macrophage",
    "Eosinophil": "Eosinophil", "Mast": "Mast", "Monocyte": "Monocyte",
    "Neutrophil": "Neutrophil",
}
# The copy-number layer is reported in the caption rather than drawn: 44 of its 756 pairs were
# significant and the matrix read as an empty grid. Dropping it lets the two remaining 28-row
# matrices use the full vertical budget.
LAYERS = [("expr", "a", "Hub-gene expression"),
          ("meth", "b", "Hub-gene methylation")]
# Hubs that are themselves immune-lineage markers, flagged in bold on the axis. The set matches
# the one the control analysis excludes (analyze_immune_control.py), so the bold labels and the
# caption's "excluding them" figure refer to the same cohorts.
IMMUNE_HUB = {"CD79A", "MZB1", "MGC29506", "FCRL5", "CD14", "CHGB", "CD19", "MS4A1"}

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

# ---------------- vertical budget ----------------
TOP = 14.0          # figure header line
PTITLE = 11.0       # panel title band
GAP = 9.0           # between panel blocks
# These vertical allowances are in points, not millimetres -- they are subtracted from H, which is
# in points. The rotated cohort labels need ~24 pt of height plus padding, and the legend needs two
# 8 pt lines; at the earlier 18 + 24 the legend landed inside the tick-label band.
XTICKS = 40.0       # rotated cohort labels under the bottom panel
NOTE = 30.0         # legend only; the four note lines moved into the LaTeX caption
BOT = XTICKS + NOTE + 8.0
LEFT, CBAR, CBGAP, RGAP = 76.0, 9.0, 5.0, 42.0
pw = W - LEFT - CBAR - CBGAP - RGAP
blk = PTITLE + 0.0
avail = H - TOP - BOT - 2 * PTITLE - 1 * GAP
ph = avail / 2.0

fig = plt.figure(figsize=(W / 72.0, H / 72.0))
axes, texts = [], []

t = fig.text(LEFT / W, (H - 7.0) / H,
             "Own-cohort hub gene versus immune-cell abundance, across %d TCGA cohorts" % NK,
             fontsize=FS, ha="left", va="top", color=BODY)
texts.append(t)

for k, (mod, letter, title) in enumerate(LAYERS):
    blk_top = H - TOP - k * (PTITLE + GAP + ph)
    ax_top = blk_top - PTITLE
    ax_bot = ax_top - ph
    ax = fig.add_axes([LEFT / W, ax_bot / H, pw / W, ph / H])
    axes.append(ax)
    ax.imshow(M[mod], cmap=CMAP, vmin=-1, vmax=1, aspect="auto",
              interpolation="nearest", origin="upper", zorder=1)
    ax.set_xlim(-0.5, NK - 0.5)
    ax.set_ylim(NC - 0.5, -0.5)
    ax.set_yticks(range(NC))
    ax.set_yticklabels([CELL_SHORT[c] for c in CELLS], fontsize=FS)
    if k == len(LAYERS) - 1:
        ax.set_xticks(range(NK))
        ax.set_xticklabels(ORDER, fontsize=FS, rotation=90, va="top", ha="center")
        for lbl, c_ in zip(ax.get_xticklabels(), ORDER):
            if R[c_]["hub"].upper() in IMMUNE_HUB:
                lbl.set_fontweight("bold")
    else:
        ax.set_xticks([])
    ax.tick_params(length=1.2, pad=1.2, width=0.4)
    for s in ax.spines.values():
        s.set_linewidth(0.45)
        s.set_color(BODY)
    # significance
    for i in range(NC):
        for j in range(NK):
            pv, qv = P[mod][i, j], Q[mod][i, j]
            if not np.isfinite(qv):
                continue
            if qv < 0.05:
                ax.plot(j, i, marker="o", ms=1.35, mfc=BODY, mec="none", zorder=3)
            elif pv < 0.05:
                ax.plot(j, i, marker="o", ms=1.35, mfc="none", mec=BODY, mew=0.32, zorder=3)
    t = fig.text(LEFT / W, (ax_top + 2.2) / H, title, fontsize=FS, ha="left", va="bottom",
                 color=BODY, fontweight="bold")
    texts.append(t)
    t = fig.text((LEFT - 5.0) / W, (ax_top + 2.2) / H, letter, fontsize=FS + 1.0, ha="right",
                 va="bottom", color=BODY, fontweight="bold")
    texts.append(t)

# colorbar beside the top panel
ax0_top = H - TOP - PTITLE
cax = fig.add_axes([(LEFT + pw + CBGAP) / W, (ax0_top - ph) / H, CBAR / W, ph / H])
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(vmin=-1, vmax=1))
cb = fig.colorbar(sm, cax=cax)
cb.set_ticks([-1, -0.5, 0, 0.5, 1])
cb.ax.tick_params(labelsize=FS, length=1.4, pad=1.2, width=0.4)
for s in list(cb.ax.spines.values()) + [cb.outline]:
    s.set_linewidth(0.45)
t = fig.text((LEFT + pw + CBGAP + CBAR / 2.0) / W, (ax0_top + 2.2) / H, "Spearman rho",
             fontsize=FS, ha="center", va="bottom", color=BODY)
texts.append(t)

# legend under the bottom panel; the explanatory lines moved into the LaTeX caption
y0 = H - TOP - 2 * PTITLE - 1 * GAP - 2 * ph - XTICKS - 2.0
t = fig.text(LEFT / W, y0 / H, "Significance", fontsize=FS, ha="left", va="top",
             color=BODY, fontweight="bold")
texts.append(t)
# The circle and bullet are plain Arial glyphs rather than $\circ$ / $\bullet$: mathtext has no
# such symbol in Arial and was pulling Cmsy10 in for them, putting a second family in the figure.
t = fig.text(LEFT / W, (y0 - 11.0) / H, "\u25cb  $p < 0.05$", fontsize=FS,
             ha="left", va="top", color=BODY)
texts.append(t)
t = fig.text((LEFT + 58.0) / W, (y0 - 11.0) / H,
             "\u25cf  $p < 0.05$ and FDR $< 0.05$", fontsize=FS,
             ha="left", va="top", color=BODY)
texts.append(t)

# ---------------- layout self-check ----------------
fig.canvas.draw()
r = fig.canvas.get_renderer()
Wpx, Hpx = fig.bbox.width, fig.bbox.height          # display units, not points
# A shrinking margin of 0.15 was too forgiving: a real collision between the legend and the
# rotated tick labels passed as clean. 0.08 still absorbs near-touching glyph boxes but reports
# anything a reader would call an overlap.
SHRINK = 0.08
boxes, oob = [], 0


def push(bb, label):
    global oob
    dx, dy = (bb.x1 - bb.x0) * SHRINK, (bb.y1 - bb.y0) * SHRINK
    x0, y0_, x1, y1 = bb.x0 + dx, bb.y0 + dy, bb.x1 - dx, bb.y1 - dy
    boxes.append((x0, y0_, x1, y1, label))
    if x0 < -0.5 or y0_ < -0.5 or x1 > Wpx + 0.5 or y1 > Hpx + 0.5:
        oob += 1


for ax in axes + [cax]:
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        if lbl.get_text():
            push(lbl.get_window_extent(r), lbl.get_text())
for t in texts:
    push(t.get_window_extent(r), t.get_text()[:16])

ov = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i], boxes[j]
        x0, y0_ = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        if x0 < x1 and y0_ < y1:
            ov.append(((x1 - x0) * (y1 - y0_), a[4][:20], b[4][:20]))
ov.sort(reverse=True)

print("canvas %.1f x %.1f pt (%.0f x %.0f mm) | in-figure %.2f pt = printed size"
      % (W, H, W_MM, H_MM, FS * PRINT_SCALE))
print("panel %.1f pt tall -> row pitch %.2f pt" % (ph, ph / NC))
print("texts %d | outside %d | overlapping pairs %d" % (len(boxes), oob, len(ov)))
for a_, t1, t2 in ov[:8]:
    print("   OV %7.1f px2  %r <-> %r" % (a_, t1, t2))
assert oob == 0, "text outside the canvas"
assert not ov, "overlapping text"
assert FS * PRINT_SCALE >= PRINT_FLOOR - 0.01, "printed type below the floor"

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGDIR, "FigS4.%s" % ext), dpi=300, facecolor="white")
print("wrote figures/FigS4.pdf/.png")
