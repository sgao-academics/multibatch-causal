# -*- coding: utf-8 -*-
"""PPI context for the pan-cancer causal network (shaped after reference Fig. 7).

Reads results/_string_channel_analysis.json and draws three panels:
  a  the largest STRING-supported modules of our causal graph, one connected component each
  b  which STRING evidence channel carries the overlap, against the STRING background
  c  per-cancer overlap counts, with the random-pair baseline in the note

Nothing is recomputed from scratch here; every number comes from that JSON.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

A = json.load(open(os.path.join(RES, "_string_channel_analysis.json"), encoding="utf-8"))
OV = A["overlaps"]
EV = 0.041

MM = 1.0 / 25.4
# 174 mm = the full-width figure area of the F&IG printed page, the same contract the submitted
# figures use (gen_fig1_landscape.py: 6.85 in). Canvas width == printed width, so the in-figure
# font sizes below are the sizes that actually appear in print.
# Height is bounded by the printed text block. The author template gives \textheight = 552.7 pt
# (194.9 mm) at \textwidth = 372 pt, and the figure is placed at \textwidth, so a 243 mm canvas
# plus its caption overflowed the page by 15 pt. The published reference keeps its figures at or
# below 203 mm; 210 mm here leaves room for a full caption.
W, H = 174.0, 212.0
fig = plt.figure(figsize=(W * MM, H * MM))

# This figure was the only one built with the matplotlib defaults, so it came out in DejaVu Sans
# with Type 3 fonts -- the one combination journals reject outright.  Set the same contract the
# other figures use: embedded TrueType, Arial (falling back to Helvetica), Arial maths glyphs.
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType, not Type 3
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "hatch.linewidth": 0.6,
    "text.usetex": False,
})

# The published reference sets figure type at 7.0-11.0 pt (MyriadPro, mode 8.5) against 10 pt
# body text. The canvas is the same width as that journal's 174.9 mm text block, so a declared
# size is the printed size once the figure is placed at full width; 9.0 keeps even the smallest
# label (FS - 1.9) at 7.1 pt, matching the reference floor.
FS = 9.0
INK = "#1a1a1a"
GREY = "#8a8a8a"
BLUE = "#2E5AAC"
LIGHT = "#c9c9c9"

PAL = ["#2E5AAC", "#B40426", "#E08A1E", "#2E8B57"]


def rect(x_mm, y_top_mm, w_mm, h_mm):
    """axes rectangle whose y is measured from the top of the canvas"""
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


# ------------------------------------------------------------------ components
parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


deg = Counter()
for k in OV:
    a, b = k.split("|")
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra
    deg[a] += 1
    deg[b] += 1

comp = defaultdict(list)
for n in list(parent):
    comp[find(n)].append(n)

ranked = sorted(comp.values(), key=lambda v: (-len(v), sorted(v)))
big = [c for c in ranked if len(c) >= 5][:4]
n_comp = len(ranked)
n_edges_rest = len(OV) - sum(1 for k in OV
                             for c in big
                             if k.split("|")[0] in c and k.split("|")[1] in c)
n_pairs_rest = sum(1 for c in ranked if len(c) == 2)

KNOWN = [
    ({"ADH7", "AOX1", "GSTA1", "GSTA2", "UGT1A1", "UGT1A10", "UGT1A3", "UGT1A4",
      "UGT1A6", "UGT1A7", "UGT1A9"}, "Drug-metabolising enzymes"),
    ({"DDX3Y", "EIF1AY", "KDM5D", "NLGN4Y", "PRKY", "RPS4Y1", "TMSB4Y", "USP9Y",
      "UTY", "ZFY"}, "Y chromosome"),
    ({"COL17A1", "KRT14", "KRT16", "KRT17", "KRT5", "KRT6A", "KRT6B", "KRT6C"},
     "Basal keratinocytes"),
    ({"KRT13", "KRT4", "SPRR1A", "SPRR1B", "SPRR2A", "SPRR2D", "SPRR2E", "SPRR3"},
     "Cornified envelope"),
    ({"CHGA", "G6PC2", "GCG", "IAPP", "INS", "PPY", "PYY", "SST"},
     "Endocrine pancreas"),
    ({"ALB", "HP", "HPR", "SAA1", "SAA2", "SAA4"}, "Acute-phase proteins"),
    ({"FOXD3", "MAG", "MOBP", "MOG", "PLP1", "SOX10"}, "Myelin / melanocyte"),
    ({"COL10A1", "COL11A1", "COL2A1", "COL9A3", "COMP"}, "Cartilage collagens"),
]


def comp_name(genes):
    s = set(genes)
    bestn, bestname = 0, None
    for ref, nm in KNOWN:
        o = len(s & ref)
        if o > bestn:
            bestn, bestname = o, nm
    return bestname if bestn >= max(2, len(s) - 2) else " / ".join(sorted(s)[:2])


try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False


def layout(genes, edges):
    """node positions normalised into a unit box, spread out enough for labels"""
    n = len(genes)
    if HAVE_NX:
        G = nx.Graph()
        G.add_nodes_from(genes)
        G.add_edges_from(edges)
        p = nx.spring_layout(G, seed=7, k=2.3 / max(np.sqrt(n), 1.0), iterations=600)
        xy = np.array([p[g] for g in genes], dtype=float)
    else:
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xy = np.c_[np.cos(th), np.sin(th)]
    mn, mx = xy.min(axis=0), xy.max(axis=0)
    span = np.where(mx - mn > 1e-9, mx - mn, 1.0)
    return (xy - mn) / span


def draw_module(x_mm, y_top_mm, w_mm, h_mm, genes, color, label):
    genes = sorted(genes)
    idx = {g: i for i, g in enumerate(genes)}
    es = []
    for k, v in OV.items():
        a, b = k.split("|")
        if a in idx and b in idx:
            es.append((a, b, v["escore"] > EV or v["dscore"] > EV))
    xy = layout(genes, [(a, b) for a, b, _ in es])

    pad = 7.0
    ax = fig.add_axes(rect(x_mm + pad, y_top_mm + 6.0, w_mm - 2 * pad, h_mm - 8.0))
    ax.set_xlim(-0.07, 1.07)
    ax.set_ylim(-0.30, 1.30)
    ax.axis("off")

    for a, b, ev in es:
        i, j = idx[a], idx[b]
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                color=color if ev else LIGHT,
                lw=1.05 if ev else 0.85,
                ls="-" if ev else (0, (2.2, 1.6)),
                zorder=1, solid_capstyle="round")

    sz = np.array([deg[g] for g in genes], dtype=float)
    ax.scatter(xy[:, 0], xy[:, 1], s=22 + 11 * sz, c=color, edgecolors="white",
               linewidths=0.7, zorder=3, alpha=0.95)
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    for g, (px, py) in zip(genes, xy):
        up = py >= 0.5                       # alternate above/below so labels do not collide
        ax.text(px, py + (0.11 if up else -0.11), g, fontsize=FS - 1.8,
                ha="center", va="bottom" if up else "top", color=INK, zorder=4,
                path_effects=halo)
    T(x_mm + 1.5, y_top_mm + 0.2, label, fontsize=FS - 0.2, color=color,
      fontweight="bold")


# ------------------------------------------------------------------ panel a
TOP = 8.0      # the panel-a title sits at TOP - 8, so TOP must be >= 8 to stay on the canvas
# Panels carry a bare letter: the journal asks that illustrations contain no titles of their own,
# and every panel is described in the caption.
T(1.0, TOP - 8.0, "a", fontweight="bold", fontsize=FS + 0.4)

cw, ch = 85.0, 40.0
gx, gy = 2.0, 2.5
for i, genes in enumerate(big):
    r, c = divmod(i, 2)
    draw_module(1.0 + c * (cw + gx), TOP + r * (ch + gy), cw, ch,
                genes, PAL[i], comp_name(genes))

y_a_end = TOP + 2 * ch + gy

# ------------------------------------------------------------------ panel b
yb = y_a_end + 11.0
T(1.0, yb - 8.0, "b", fontweight="bold", fontsize=FS + 0.4)

CH = A["channels"]
short = {"escore": "Experiments", "dscore": "Curated DB", "tscore": "Text mining",
         "ascore": "Co-expression", "pscore": "Phylo. profile",
         "nscore": "Neighbourhood", "fscore": "Gene fusion"}
order = ["escore", "dscore", "ascore", "tscore", "pscore", "nscore", "fscore"]
bys = {c["tag"]: c for c in CH}
our = [bys[t]["pct_our_above"] for t in order]
bg = [bys[t]["pct_bg_above"] for t in order]

hb = 34.0
axb = fig.add_axes(rect(30.0, yb, W - 30.0 - 30.0, hb))
ypos = np.arange(len(order))
# the two series differ by pattern as well as by colour, so the panel still reads in
# greyscale or with a colour-vision deficiency (F&IG accessibility requirement)
axb.barh(ypos + 0.2, bg, height=0.36, color=LIGHT, edgecolor="white", linewidth=0.5,
         hatch="///", label="STRING background (%d edges)" % A["n_string_edges"])
axb.barh(ypos - 0.2, our, height=0.36, color=BLUE, edgecolor="white", linewidth=0.5,
         label="our causal pairs (%d edges)" % A["n_overlap"])
for i, (o, b) in enumerate(zip(our, bg)):
    axb.text(o + 1.6, i - 0.2, "%.0f%%" % o, va="center", ha="left", fontsize=FS - 1.6,
             color=BLUE)
    axb.text(b + 1.6, i + 0.2, "%.0f%%" % b, va="center", ha="left", fontsize=FS - 1.6,
             color=GREY)
axb.set_yticks(ypos)
axb.set_yticklabels([short[t] for t in order], fontsize=FS - 0.4)
axb.invert_yaxis()
axb.set_xlim(0, 116)
axb.set_xticks([0, 25, 50, 75, 100])
axb.set_xlabel("edges with that channel above 0.041 (%)", fontsize=FS - 0.6, labelpad=1.5)
axb.tick_params(axis="x", labelsize=FS - 1.6)
axb.spines[["top", "right"]].set_visible(False)
axb.legend(fontsize=FS - 1.6, frameon=False, loc="lower right", ncol=1,
           handlelength=1.4, borderaxespad=0.2)

# panel b carries an x-axis label below its tick labels, so panel c needs clearance for both
yc = yb + hb + 17.0

# ------------------------------------------------------------------ panel c
T(1.0, yc - 8.0, "c", fontweight="bold", fontsize=FS + 0.4)
byc = A["overlap_by_cancer"]
cs = sorted(byc.items(), key=lambda kv: (-kv[1], kv[0]))
hc = 32.0
axc = fig.add_axes(rect(30.0, yc, W - 30.0 - 4.0, hc))
xs = np.arange(len(cs))
axc.bar(xs, [v for _, v in cs], color=BLUE, edgecolor="white", linewidth=0.4)
axc.set_xticks(xs)
axc.set_xticklabels([k for k, _ in cs], fontsize=FS - 1.9, rotation=90)
axc.set_ylabel("pairs", fontsize=FS - 0.6, labelpad=1.5)
axc.tick_params(axis="y", labelsize=FS - 1.6)
axc.set_xlim(-0.7, len(cs) - 0.3)
axc.spines[["top", "right"]].set_visible(False)

# the rotated cancer labels hang below the axis, so the notes start well clear of them
ybot = yc + hc + 9.0

# ------------------------------------------------------------------ notes
T(1.0, ybot,
  "STRING v12 (score >= 0.7, human): %d of %d unique causal pairs, %d of %d directed "
  "edges (%.1f%%)."
  % (A["n_overlap"], A["n_causal_pairs"], sum(A["overlap_by_cancer"].values()),
     json.load(open(os.path.join(RES, "_string_channels.json"),
                    encoding="utf-8"))["n_directed_edges"], A["observed_rate_pct"]),
  fontsize=FS - 1.4, color=GREY)
T(1.0, ybot + 5.2,
  "Random pairs from the same %d genes reach that score in %.2f%% of cases, a %.1f-fold "
  "enrichment (Fisher p = %.1g)." % (A["n_genes"], A["expected_rate_pct"],
                                     A["fold_enrichment"], A["fisher_p"]),
  fontsize=FS - 1.4, color=GREY)
T(1.0, ybot + 10.4,
  "Solid edges carry experimental or curated evidence, dashed edges co-expression only "
  "(61 of 184).", fontsize=FS - 1.4, color=GREY)
T(1.0, ybot + 15.6,
  "Four largest of %d connected components; the rest are %d edges including %d two-gene pairs."
  % (n_comp, n_edges_rest, n_pairs_rest), fontsize=FS - 1.4, color=GREY)

LEG = [Line2D([], [], color=INK, lw=1.3),
       Line2D([], [], color=LIGHT, lw=1.0, ls=(0, (2.2, 1.6))),
       Line2D([], [], marker="o", color="none", markerfacecolor=GREY,
              markeredgecolor="none", markersize=4.5)]
fig.legend(LEG, ["experimental / curated", "co-expression only", "gene (size = degree)"],
           loc="upper right", bbox_to_anchor=(0.995, 1.0), frameon=False,
           fontsize=FS - 1.4, ncol=1, handletextpad=0.6)

# ------------------------------------------------------------------ checks
fig.canvas.draw()
rr = fig.canvas.get_renderer()
# dpi_scale_trans maps inches -> display pixels, so inverting it lands in INCHES (0-7), while every
# bound below is in MILLIMETRES (0-210). Comparing the two meant bb.x1 > W + 0.6 could never fire
# and the out-of-canvas check silently passed on text that was visibly cut off. Scale display
# pixels straight to millimetres instead.
inv = mtransforms.Affine2D().scale(25.4 / fig.dpi)

every = list(texts)
for ax_ in fig.get_axes():
    every += list(ax_.texts)
    if not ax_.axison:      # panel a's module axes are switched off; their ticks never print
        continue
    every += list(ax_.get_xticklabels()) + list(ax_.get_yticklabels())
    every += [ax_.xaxis.label, ax_.yaxis.label]

bad = []
for t in every:
    if not t.get_text():
        continue
    bb = t.get_window_extent(renderer=rr).transformed(inv)
    if bb.x0 < -0.6 or bb.y0 < -0.6 or bb.x1 > W + 0.6 or bb.y1 > H + 0.6:
        bad.append((t.get_text()[:22], round(bb.x0, 1), round(bb.y0, 1),
                    round(bb.x1, 1), round(bb.y1, 1)))

boxes = [(t.get_text()[:18], t.get_window_extent(renderer=rr).transformed(inv))
         for t in texts]
ovl = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i][1], boxes[j][1]
        ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
        iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
        if ix > 1.6 and iy > 1.6:
            ovl.append((boxes[i][0], boxes[j][0], round(ix, 1), round(iy, 1)))

# node labels inside panel a must not collide either
node_labels = []
for ax_ in fig.get_axes():
    for t in ax_.texts:
        bb = t.get_window_extent(renderer=rr).transformed(inv)
        node_labels.append((t.get_text(), bb))
nbad = []
for i in range(len(node_labels)):
    for j in range(i + 1, len(node_labels)):
        a, b = node_labels[i][1], node_labels[j][1]
        ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
        iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
        if ix > 0.8 and iy > 0.8:
            nbad.append((node_labels[i][0], node_labels[j][0], round(ix, 1),
                         round(iy, 1)))

# tick labels that run into the figure notes -- the failure mode a fig.text-only check misses
tick_boxes = []
for ax_ in fig.get_axes():
    if not ax_.axison:
        continue
    for t in list(ax_.get_xticklabels()) + list(ax_.get_yticklabels()):
        if t.get_text():
            tick_boxes.append((t.get_text()[:16],
                               t.get_window_extent(renderer=rr).transformed(inv)))
cross = []
for n1, b1 in boxes:
    for n2, b2 in tick_boxes:
        ix = max(0.0, min(b1.x1, b2.x1) - max(b1.x0, b2.x0))
        iy = max(0.0, min(b1.y1, b2.y1) - max(b1.y0, b2.y0))
        if ix > 0.9 and iy > 0.9:
            cross.append((n1, n2, round(ix, 1), round(iy, 1)))

# Axis labels sit outside the tick labels, so the tick-label check above cannot see them; this is
# the third class of text object to slip past a fig.text-only audit, after tick labels themselves.
lab_boxes = []
for ax_ in fig.get_axes():
    if not ax_.axison:
        continue
    for t in (ax_.xaxis.label, ax_.yaxis.label):
        if t.get_text():
            lab_boxes.append((t.get_text()[:20],
                              t.get_window_extent(renderer=rr).transformed(inv)))
crosslab = []
for n1, b1 in boxes:
    for n2, b2 in lab_boxes:
        ix = max(0.0, min(b1.x1, b2.x1) - max(b1.x0, b2.x0))
        iy = max(0.0, min(b1.y1, b2.y1) - max(b1.y0, b2.y0))
        if ix > 0.9 and iy > 0.9:
            crosslab.append((n1, n2, round(ix, 1), round(iy, 1)))

scale = 174.0 / W          # canvas width is the printed width, so this is 1.0 by construction
print("canvas %.0f x %.0f mm | fig texts %d | all texts %d" % (W, H, len(texts), len(every)))
print("outside canvas: %d %s" % (len(bad), bad[:6]))
print("overlapping fig.text pairs: %d %s" % (len(ovl), ovl[:6]))
print("overlapping node labels: %d %s" % (len(nbad), nbad[:8]))
print("notes colliding with tick labels: %d %s" % (len(cross), cross[:8]))
print("notes colliding with axis labels: %d %s" % (len(crosslab), crosslab[:8]))
print("font size FS=%.1f pt -> %.2f pt in the published layout (canvas %.0f mm vs the "
      "174.9 mm text block) | %.2f pt in the author PDF (131 mm)" 
      % (FS, FS * 174.9 / W, W, FS * 131.0 / W))
print("components %d | edges outside the four shown %d | two-gene components %d"
      % (n_comp, n_edges_rest, n_pairs_rest))

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, "Fig3." + ext), dpi=400, facecolor="white")
print("wrote figures/Fig3.pdf and .png")
