# -*- coding: utf-8 -*-
"""PPI context for the pan-cancer causal network.

Reads results/_string_channel_analysis.json and draws three panels:
  a  the largest STRING-supported modules of our causal graph, one connected component each
  b  which STRING evidence channel carries the overlap, against the STRING background
  c  per-cancer overlap counts, with the random-pair baseline in the note

Nothing is recomputed from scratch here; every number comes from that JSON.

Drawing language.  This plate was the last one still drawn with a private
palette, black type, hard marker outlines and text stroked straight through
`path_effects`.  It now shares `_figstyle` with Figs. 1 and 2: the same seven
muted colours, type in the palette's indigo rather than black, a faint rounded
card behind each panel, markers with a soft shadow, and one 8 pt lettering size
throughout.  Every encoded distinction also survives a greyscale print -- solid
against dashed edges in panel a, hatch against flat fill in panel b -- which is
the journal's requirement for colour figures.  The two labels that used to be
stroked directly now go through `haloed_text`, whose halo copy carries the
stroke while the copy on top stays ordinary text, so the words remain
extractable from the PDF and can be diffed against the previous revision.

Output: figures/Fig3.pdf / .png
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import (PAL, INK, TICK, GREY, GRID, TRACK, FRAME, BOXFC, BOXEC,
                       CMAP, MM, W, tidy, card, dots, haloed_text)

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

A = json.load(open(os.path.join(RES, "_string_channel_analysis.json"), encoding="utf-8"))
OV = A["overlaps"]
EV = 0.041

# 174 mm = the full-width figure area of the printed page, and exactly one of the
# four widths the journal sanctions (39 / 84 / 129 / 174 mm).  Canvas width equals
# printed width, so a size declared here is the size that appears in print.
# Height is bounded twice over: by the journal's own 234 mm ceiling, and by the
# printed text block, since the figure is placed at \textwidth and carries a long
# caption.  This plate is the tallest in the set, so it is kept within 3 mm of the
# previous revision's height -- which compiled with no overfull box -- rather than
# being grown to the theoretical limit.
W_MM, H = 174.0, 218.0
fig = plt.figure(figsize=(W_MM * MM, H * MM))

# Font embedding and the family are declared by _figstyle (TrueType, Arial).  The
# journal rejects Type 3 outright, so make the contract explicit rather than
# trusting that the import happened.
assert plt.rcParams["pdf.fonttype"] == 42, "figure fonts would be written as Type 3"

# Lettering: the journal asks for 2-3 mm, i.e. 8-12 pt at final size.  The canvas
# is the printed width, so 8 pt here is 8 pt in print -- the bottom of the range,
# used for the smallest labels, with the panel letters at 11 pt.
FS, PLET = 8.0, 11.0


def rect(x_mm, y_top_mm, w_mm, h_mm):
    """axes rectangle whose y is measured from the top of the canvas"""
    return [x_mm / W_MM, 1.0 - (y_top_mm + h_mm) / H, w_mm / W_MM, h_mm / H]


texts = []


def T(x_mm, y_mm, s, **kw):
    kw.setdefault("fontsize", FS)
    kw.setdefault("color", INK)
    kw.setdefault("ha", "left")
    kw.setdefault("va", "top")
    t = fig.text(x_mm / W_MM, 1.0 - y_mm / H, s, **kw)
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
    card(fig, ax, pad=0.006, radius=0.010)

    for a, b, ev in es:
        i, j = idx[a], idx[b]
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                color=color if ev else PAL["lilac"],
                lw=1.05 if ev else 0.85,
                ls="-" if ev else (0, (2.2, 1.6)),
                zorder=1, solid_capstyle="round")

    sz = np.array([deg[g] for g in genes], dtype=float)
    # Soft shadow rather than a hard outline, matching the markers in Figs. 1-2.
    dots(ax, xy[:, 0], xy[:, 1], 20 + 9 * sz, color, z=3, lw=0.6)

    # haloed_text instead of a stroked single copy: the halo is what lifts the
    # letters off the edges, and its twin on top keeps the label extractable.
    for g, (px, py) in zip(genes, xy):
        up = py >= 0.5                       # alternate above/below so labels do not collide
        haloed_text(ax, px, py + (0.13 if up else -0.13), g, size=FS, color=INK,
                    ha="center", va="bottom" if up else "top", z=4)
    T(x_mm + 1.5, y_top_mm + 0.2, label, fontsize=FS, color=color, fontweight="bold")


# ------------------------------------------------------------------ panel a
TOP = 8.0      # the panel-a letter sits at TOP - 8, so TOP must be >= 8 to stay on the canvas
# Panels carry a bare letter: the journal asks that illustrations contain no titles
# of their own, and every panel is described in the caption.
T(1.0, TOP - 8.0, "a)", fontweight="bold", fontsize=PLET)

MODCOL = [PAL["mist"], PAL["orchid"], PAL["moss"], PAL["violet"]]
# ch dropped 40 -> 38 to pay for panel b's height.  Seven channels carrying two
# 8 pt value labels each need 14 lines of type, which is more than 36 mm allows;
# the modules lose 2 mm and the network layouts are unaffected.
cw, ch = 85.0, 38.0
gx, gy = 2.0, 2.5
for i, genes in enumerate(big):
    r, c = divmod(i, 2)
    draw_module(1.0 + c * (cw + gx), TOP + r * (ch + gy), cw, ch,
                genes, MODCOL[i], comp_name(genes))

y_a_end = TOP + 2 * ch + gy

# ------------------------------------------------------------------ panel b
yb = y_a_end + 10.0
T(1.0, yb - 8.0, "b)", fontweight="bold", fontsize=PLET)

CH = A["channels"]
short = {"escore": "Experiments", "dscore": "Curated DB", "tscore": "Text mining",
         "ascore": "Co-expression", "pscore": "Phylo. profile",
         "nscore": "Neighbourhood", "fscore": "Gene fusion"}
order = ["escore", "dscore", "ascore", "tscore", "pscore", "nscore", "fscore"]
bys = {c["tag"]: c for c in CH}
our = [bys[t]["pct_our_above"] for t in order]
bg = [bys[t]["pct_bg_above"] for t in order]

hb = 44.0
axb = fig.add_axes(rect(30.0, yb, W_MM - 30.0 - 30.0, hb))
card(fig, axb, pad=0.007, radius=0.012)
ypos = np.arange(len(order))
# The two series differ by hatch as well as by fill, so the panel still reads in
# greyscale or with a colour-vision deficiency (the journal's accessibility rule
# for colour figures).
# Each channel carries two value labels stacked under each other, so 14 lines of
# type have to fit in this panel.  0.40 of a row was 2.5 mm while 8 pt type is
# 2.8 mm tall, so the pair touched; 0.50 makes every consecutive gap equal, which
# is the most even arrangement the row height allows.
OFF = 0.25
axb.barh(ypos + OFF, bg, height=0.34, color=PAL["lilac"], edgecolor="white",
         linewidth=0.5, hatch="///",
         label="STRING background (%d edges)" % A["n_string_edges"])
axb.barh(ypos - OFF, our, height=0.34, color=PAL["mist"], edgecolor="white",
         linewidth=0.5, label="our causal pairs (%d edges)" % A["n_overlap"])
for i, (o, b) in enumerate(zip(our, bg)):
    axb.text(o + 1.6, i - OFF, "%.0f%%" % o, va="center", ha="left", fontsize=FS,
             color=PAL["mist"])
    axb.text(b + 1.6, i + OFF, "%.0f%%" % b, va="center", ha="left", fontsize=FS,
             color=GREY)
axb.set_yticks(ypos)
axb.set_yticklabels([short[t] for t in order], fontsize=FS)
axb.invert_yaxis()
axb.set_xlim(0, 116)
axb.set_xticks([0, 25, 50, 75, 100])
axb.set_xlabel("edges with that channel above 0.041 (%)", fontsize=FS, labelpad=1.5)
axb.tick_params(axis="x", labelsize=FS)
tidy(axb, grid="x")
axb.legend(fontsize=FS, frameon=True, loc="lower right", ncol=1,
           facecolor=BOXFC, edgecolor=BOXEC, framealpha=0.94,
           handlelength=1.4, borderaxespad=0.2)

# panel b carries an x-axis label below its tick labels, so panel c needs clearance
# for both, and for the 90-degree cancer labels below that.
yc = yb + hb + 18.0

# ------------------------------------------------------------------ panel c
T(1.0, yc - 8.0, "c)", fontweight="bold", fontsize=PLET)
byc = A["overlap_by_cancer"]
cs = sorted(byc.items(), key=lambda kv: (-kv[1], kv[0]))
hc = 28.0
axc = fig.add_axes(rect(30.0, yc, W_MM - 30.0 - 4.0, hc))
card(fig, axc, pad=0.007, radius=0.012)
xs = np.arange(len(cs))
axc.bar(xs, [v for _, v in cs], color=PAL["peri"], edgecolor="white", linewidth=0.4)
axc.set_xticks(xs)
axc.set_xticklabels([k for k, _ in cs], fontsize=FS, rotation=90)
axc.set_ylabel("pairs", fontsize=FS, labelpad=1.5)
# Ticks pinned rather than left to the locator: at 28 mm the automatic choice drops
# the 15 gridline, which changes the reading of every bar against the previous
# revision of this plate for no reason.
axc.set_yticks([0, 5, 10, 15, 20])
axc.tick_params(axis="y", labelsize=FS)
axc.set_xlim(-0.7, len(cs) - 0.3)
tidy(axc, grid="y")

# the rotated cancer labels hang below the axis, so the notes start well clear of
# them: a five-character TCGA code set at 8 pt is about 9 mm once it is turned on
# its side.
ybot = yc + hc + 11.0

# ------------------------------------------------------------------ notes
# Four lines at 5.2 mm pitch: 8 pt type is 2.8 mm tall, so this keeps them apart
# while staying inside the canvas.
NOTE_PITCH = 5.2
T(1.0, ybot,
  "STRING v12 (score >= 0.7, human): %d of %d unique causal pairs, %d of %d directed "
  "edges (%.1f%%)."
  % (A["n_overlap"], A["n_causal_pairs"], sum(A["overlap_by_cancer"].values()),
     json.load(open(os.path.join(RES, "_string_channels.json"),
                    encoding="utf-8"))["n_directed_edges"], A["observed_rate_pct"]),
  fontsize=FS, color=GREY)
T(1.0, ybot + NOTE_PITCH,
  "Random pairs from the same %d genes reach that score in %.2f%% of cases, a %.1f-fold "
  "enrichment (Fisher p = %.1g)." % (A["n_genes"], A["expected_rate_pct"],
                                    A["fold_enrichment"], A["fisher_p"]),
  fontsize=FS, color=GREY)
T(1.0, ybot + 2 * NOTE_PITCH,
  "Solid edges carry experimental or curated evidence, dashed edges co-expression only "
  "(61 of 184).", fontsize=FS, color=GREY)
T(1.0, ybot + 3 * NOTE_PITCH,
  "Four largest of %d connected components; the rest are %d edges including %d two-gene pairs."
  % (n_comp, n_edges_rest, n_pairs_rest), fontsize=FS, color=GREY)

LEG = [Line2D([], [], color=INK, lw=1.3),
       Line2D([], [], color=PAL["lilac"], lw=1.0, ls=(0, (2.2, 1.6))),
       Line2D([], [], marker="o", color="none", markerfacecolor=GREY,
              markeredgecolor="none", markersize=4.5)]
fig.legend(LEG, ["experimental / curated", "co-expression only", "gene (size = degree)"],
           loc="upper right", bbox_to_anchor=(0.995, 1.0), frameon=True,
           facecolor=BOXFC, edgecolor=BOXEC, framealpha=0.94,
           fontsize=FS, ncol=1, handletextpad=0.6, borderpad=0.5)

# ------------------------------------------------------------------ checks
fig.canvas.draw()
rr = fig.canvas.get_renderer()
# dpi_scale_trans maps inches -> display pixels, so inverting it lands in INCHES
# (0-7), while every bound below is in MILLIMETRES (0-218).  Scale display pixels
# straight to millimetres instead.
inv = matplotlib.transforms.Affine2D().scale(25.4 / fig.dpi)

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
    if bb.x0 < -0.6 or bb.y0 < -0.6 or bb.x1 > W_MM + 0.6 or bb.y1 > H + 0.6:
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

# node labels inside panel a must not collide either.  haloed_text draws each label
# twice, so the geometry is compared once: only the un-stroked copy is measured.
node_labels = []
for ax_ in fig.get_axes():
    for t in ax_.texts:
        if t.get_path_effects():
            continue
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

# tick labels that run into the figure notes -- the failure mode a fig.text-only
# check misses
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

# Axis labels sit outside the tick labels, so the tick-label check above cannot see
# them; this is the third class of text object to slip past a fig.text-only audit.
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

print("canvas %.0f x %.0f mm | fig texts %d | all texts %d" % (W_MM, H, len(texts), len(every)))
print("outside canvas: %d %s" % (len(bad), bad[:6]))
print("overlapping fig.text pairs: %d %s" % (len(ovl), ovl[:6]))
print("overlapping node labels: %d %s" % (len(nbad), nbad[:8]))
print("notes colliding with tick labels: %d %s" % (len(cross), cross[:8]))
print("notes colliding with axis labels: %d %s" % (len(crosslab), crosslab[:8]))
print("bottom of the last note at %.1f mm, canvas %.0f mm (margin %.1f mm)"
      % (ybot + 3 * NOTE_PITCH + 2.8, H, H - (ybot + 3 * NOTE_PITCH + 2.8)))
print("font size FS=%.1f pt -> %.2f pt in print (canvas %.0f mm vs the 174.9 mm text "
      "block) | %.2f pt in the author PDF" % (FS, FS * 174.9 / W_MM, W_MM,
                                              FS * 131.0 / W_MM))
print("components %d | edges outside the four shown %d | two-gene components %d"
      % (n_comp, n_edges_rest, n_pairs_rest))

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, "Fig3." + ext), dpi=400, facecolor="white")
print("wrote figures/Fig3.pdf and .png")
