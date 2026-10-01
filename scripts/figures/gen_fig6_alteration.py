# -*- coding: utf-8 -*-
"""Figure 6: somatic alteration burden of the per-cohort hub genes.

The plate asks one question -- are the genes that occupy the inferred causal
networks also the genes that drive cancer genomically -- and answers it no, three
times over.  Rebuilding it was not a matter of taste: all three panels used to
draw that negative answer as an absence, and an absence is invisible.

* (a) was a bar of near-zero height for each hub against an open circle for the
      driver.  The bars are invisible, so the panel reads as "driver values",
      not as "driver values against a hub that is flat on the floor".
* (b) was the same, with amplification up and deletion down.
* (c) was a heat map in which 47% of the cells are exactly zero and therefore
      white; a white cell reads as missing artwork, not as a measured zero.

So each panel now marks the zero instead of omitting it.  (a) becomes a dumbbell
whose lower head is drawn whether or not its value is zero, so the flat row of
hub values at the bottom is the picture; (b) draws points rather than bars, so a
zero is a point sitting on the axis; (c) draws only the non-zero cells as scaled
dots on a faint grid, which turns the sparsity itself into the message (990
cells, 47% empty, 14 above 10%).

The three panels share one x axis -- the 33 cohorts, ordered by the driver value
-- and the cohort names are printed once, under panel (c), so that a cohort can
be read straight down the figure.

Drawing language
----------------
Same tokens as Figures 1-6, imported from `_figstyle`: the seven muted colours,
type in the palette's indigo, a faint rounded card behind each panel, the same
single 8 pt lettering size, panel letters at 11 pt bold.

Journal requirements applied here
---------------------------------
* 174 mm wide -- the printed text width, so the point sizes are the sizes that
  appear in print.  `bbox_inches='tight'` is deliberately NOT used.
* Nothing below 8 pt; the page is the canvas and the canvas check is a hard gate.
* Sans-serif Arial throughout, including the maths text; Type 42 embedding.
* Amplification and deletion differ by direction as well as by colour, and a
  filled head differs from an open one, so the encodings survive greyscale.

Output: figures/Fig6.pdf / .png
"""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import (PAL, INK, GREY, TICK, FRAME, TRACK, SHADOW, MM, CMAP,
                       tidy, card, dots, haloed_text_pt, panel_letters)
from _gene_symbols import canonical

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

d = json.load(open(os.path.join(RES, "_hub_variant_landscape.json"), encoding="utf-8"))
HUBS, COH = d["hubs"], d["cohorts"]
DRIVERS = ("TP53", "PIK3CA", "KRAS")

# ---------------------------------------------------------------- data
recs = []
for c in sorted(COH):
    x = COH[c]
    ns, nc, h = max(1, x["n_sequenced"]), max(1, x["n_cna"]), x["hub"]
    g = lambda k, gg, n: 100.0 * x[k].get(gg, 0) / n
    recs.append(dict(c=c, h=h, ns=x["n_sequenced"], nc=x["n_cna"],
                     hm=g("mut_nonsyn", h, ns), ha=g("amp", h, nc),
                     hd=g("homdel", h, nc),
                     drv=max(g("mut_nonsyn", q, ns) for q in DRIVERS)))
recs.sort(key=lambda r: -r["drv"])
N = len(recs)
labels = [r["c"] for r in recs]
xs = np.arange(N)
hm = np.array([r["hm"] for r in recs])
dr = np.array([r["drv"] for r in recs])
ha = np.array([r["ha"] for r in recs])
hd = np.array([r["hd"] for r in recs])

# The JSON keeps the spelling the source matrix carries (C9ORF84, MGC29506), and
# mut_nonsyn is keyed by that same spelling: the lookup key has to stay as loaded,
# so only the printed row labels go through canonical().
HUBS_RAW = sorted(set(HUBS.values()))
GENES = [canonical(g) for g in HUBS_RAW]
assert len(set(GENES)) == len(GENES), "canonical() collapsed two hub genes onto one label"
M = np.zeros((len(HUBS_RAW), N))
for j, r in enumerate(recs):
    x, ns = COH[r["c"]], max(1, COH[r["c"]]["n_sequenced"])
    for i, gg in enumerate(HUBS_RAW):
        M[i, j] = 100.0 * x["mut_nonsyn"].get(gg, 0) / ns
ORDER = np.argsort(-M.max(axis=1))

print("cohorts %d | hub genes %d | cells %d" % (N, len(GENES), M.size))
print("renamed for print: %s" % ", ".join("%s->%s" % (r, canonical(r))
                                          for r in HUBS_RAW if canonical(r) != r))

# ---- assertions: the caption's numbers must hold in the data ----
if int((hm == 0).sum()) != 15 or abs(np.median(hm) - 0.34) > 0.01:
    print("ABORT: hub mutation statistics moved (zeros %d, median %.3f)"
          % (int((hm == 0).sum()), np.median(hm)))
    sys.exit(1)
if abs(np.median(dr) - 31.23) > 0.05:
    print("ABORT: driver median moved (%.3f)" % np.median(dr))
    sys.exit(1)
if abs(100.0 * (M == 0).sum() / M.size - 47.0) > 1.5 or int((M > 10).sum()) != 14:
    print("ABORT: matrix sparsity moved (%.0f%% zero, %d above 10)"
          % (100.0 * (M == 0).sum() / M.size, int((M > 10).sum())))
    sys.exit(1)
print("claim checks passed: hub zeros 15/33, hub median %.2f%%, driver median %.2f%%,"
      " matrix %.0f%% zero, %d cells above 10%%"
      % (np.median(hm), np.median(dr), 100.0 * (M == 0).sum() / M.size,
         int((M > 10).sum())))
# how often the own-cohort cell is the row maximum -- the caption has to say this
_ownmax = sum(1 for i, gg in enumerate(HUBS_RAW)
              if (o := [j for j, r in enumerate(recs) if r["h"] == gg])
              and M[i, o[0]] == M[i].max() and M[i].max() > 0)
print("own-cohort cell is the row maximum in %d of %d rows" % (_ownmax, len(GENES)))
_rev = [(r["c"], r["hm"], r["drv"]) for r in recs if r["drv"] - r["hm"] <= 0]
print("cohorts where the hub value reaches or exceeds its driver: %d%s"
      % (len(_rev), " -> " + ", ".join("%s %.1f vs %.1f" % t for t in _rev) if _rev else ""))
for i, gg in enumerate(HUBS_RAW):
    o = [j for j, r in enumerate(recs) if r["h"] == gg]
    print("   %-9s own %-5s at x=%2d  value %5.1f%%  row max %5.1f%%"
          % (GENES[i], ",".join(recs[j]["c"] for j in o), o[0], M[i, o[0]], M[i].max()))

FS = 8.0
W_MM, H_MM = 174.0, 224.0
fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
axa = fig.add_axes([0.128, 0.765, 0.700, 0.205])
axb = fig.add_axes([0.128, 0.585, 0.700, 0.135])
axc = fig.add_axes([0.128, 0.075, 0.700, 0.470])

CO_HUB = PAL["peri"]        # the hub value: the low side of every pair
CO_DRV = PAL["indigo"]      # the canonical driver: the high side
CO_AMP = PAL["orchid"]
CO_DEL = PAL["mist"]

# ===================================================================== a) dumbbell
for i in range(N):
    axa.plot([i, i], [hm[i], dr[i]], color="#DCE3EE", lw=1.1, zorder=1,
             solid_capstyle="round")
# filled against open, not only coloured: the two heads of the dumbbell have to
# stay apart in greyscale (Springer asks that no distinction rest on colour alone)
dots(axa, xs, hm, 30, "white", z=5, ec=CO_HUB, lw=1.2, shadow=False)
dots(axa, xs, dr, 26, CO_DRV, z=5, ec="white", lw=0.7, shadow=False)
axa.axhline(np.median(dr), color=CO_DRV, lw=0.6, ls=(0, (3, 2.4)), zorder=2, alpha=0.8)
axa.axhline(np.median(hm), color=CO_HUB, lw=0.6, ls=(0, (3, 2.4)), zorder=2, alpha=0.8)
# The two medians are labelled at the right-hand end of their own rule, where the
# driver curve has already fallen to the floor and the label sits in free space.
axa.text(N - 0.9, np.median(dr) + 3.2, "31.2%", ha="right", va="bottom",
         fontsize=FS, color=CO_DRV)
axa.text(N - 0.9, np.median(hm) + 3.2, "0.34%", ha="right", va="bottom",
         fontsize=FS, color=CO_HUB)
# legend in the free upper-right corner: the driver value falls away left-to-right
dots(axa, [22.6], [87.0], 26, CO_DRV, z=6, ec="white", lw=0.7, shadow=False)
axa.text(23.4, 87.0, "canonical driver", fontsize=FS, color=INK, va="center")
dots(axa, [22.6], [74.0], 30, "white", z=6, ec=CO_HUB, lw=1.2, shadow=False)
axa.text(23.4, 74.0, "cohort hub gene", fontsize=FS, color=INK, va="center")
axa.set_ylabel("non-synonymous\nmutation frequency (%)", fontsize=FS, linespacing=1.3)
axa.set_ylim(-4, 100)
axa.set_yticks([0, 25, 50, 75, 100])
axa.set_xlim(-0.7, N - 0.3)
axa.set_xticks(xs)
axa.set_xticklabels([])
axa.tick_params(axis="x", length=1.5)

# ===================================================================== b) CNA points
axb.axhline(0, color=FRAME, lw=0.7, zorder=1)
for i in range(N):
    if ha[i] > 0:
        axb.plot([i, i], [0, ha[i]], color="#DCE3EE", lw=1.1, zorder=1)
    if hd[i] > 0:
        axb.plot([i, i], [0, -hd[i]], color="#DCE3EE", lw=1.1, zorder=1)
dots(axb, xs, ha, 26, CO_AMP, z=5, ec="white", lw=0.7, shadow=False)
dots(axb, xs, -hd, 26, CO_DEL, z=5, ec="white", lw=0.7, shadow=False)
axb.set_ylabel("copy-number\nalteration (%)", fontsize=FS, linespacing=1.3)
axb.set_ylim(-11.5, 11.5)
axb.set_yticks([-10, -5, 0, 5, 10])
axb.set_yticklabels(["10", "5", "0", "5", "10"])
axb.set_xlim(-0.7, N - 0.3)
axb.set_xticks(xs)
axb.set_xticklabels([])
axb.tick_params(axis="x", length=1.5)
axb.text(0.995, 0.96, "amplification", transform=axb.transAxes, ha="right", va="top",
         fontsize=FS, color=CO_AMP)
axb.text(0.995, 0.04, "deep deletion", transform=axb.transAxes, ha="right", va="bottom",
         fontsize=FS, color=CO_DEL)

# ===================================================================== c) dot matrix
ramp = LinearSegmentedColormap.from_list(
    "hl", ["#C6D3E5", "#7A9AC3", "#4D6EAF", "#9A5A9F", "#9E2B33"])
for i_row, i in enumerate(ORDER):
    for j in range(N):
        v = M[i, j]
        if v <= 0:
            continue
        axc.scatter(j, i_row, s=3.0 + 30.0 * np.clip(v / 20.0, 0, 1),
                    c=[ramp(np.clip(v / 20.0, 0, 1))], edgecolors="white",
                    linewidths=0.35, zorder=3)
    for j in [j for j, r in enumerate(recs) if r["h"] == HUBS_RAW[i]]:
        axc.add_patch(plt.Rectangle((j - 0.5, i_row - 0.5), 1, 1, fill=False,
                                    edgecolor=INK, lw=0.6, zorder=6))
axc.set_yticks(range(len(ORDER)))
axc.set_yticklabels([GENES[i] for i in ORDER], fontsize=FS)
axc.set_xticks(xs)
axc.set_xticklabels(labels, rotation=90, fontsize=FS)
axc.set_xlim(-0.7, N - 0.3)
axc.set_ylim(len(ORDER) - 0.5, -0.5)
axc.tick_params(axis="y", length=0, pad=2.5)
axc.tick_params(axis="x", pad=1.5)
# a faint grid, so an empty cell still reads as a measured zero and not as a gap
axc.set_xticks(np.arange(-0.5, N, 1), minor=True)
axc.set_yticks(np.arange(-0.5, len(ORDER), 1), minor=True)
axc.grid(which="minor", color=TRACK, lw=0.4, zorder=0)
axc.tick_params(which="minor", length=0)

cax = fig.add_axes([0.845, 0.075, 0.011, 0.090])
_sm = plt.cm.ScalarMappable(cmap=ramp, norm=plt.Normalize(0, 20))
cb = fig.colorbar(_sm, cax=cax)
cb.set_label("mutation frequency (%)", fontsize=FS, labelpad=2)
cb.ax.tick_params(length=1.5, labelsize=FS)
cb.set_ticks([0, 10, 20])
cb.outline.set_linewidth(0.5)

for _a in (axa, axb):
    for _s in ("top", "right"):
        _a.spines[_s].set_visible(False)
    tidy(_a)
for _s in ("top", "right", "left", "bottom"):
    axc.spines[_s].set_linewidth(0.5)
card(fig, axa, pad=0.010, radius=0.014)
card(fig, axb, pad=0.010, radius=0.014)
panel_letters(fig, [(axa, "a"), (axb, "b"), (axc, "c")], dx=-0.030, dy=0.006, size=11)

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
    _bb = _t.get_window_extent(renderer=_rend)
    if _bb.x0 < -12 or _bb.y0 < -12 or _bb.x1 > _W + 12 or _bb.y1 > _H + 12:
        _bad.append((_t.get_text()[:34].replace("\n", " "), round(_bb.x0, 1), round(_bb.x1, 1)))
if _bad:
    print("ABORT: %d text elements leave the canvas" % len(_bad))
    for _b in _bad[:12]:
        print("   ", _b)
    sys.exit(1)
print("canvas check passed: all text inside %.2f x %.2f mm" % (W_MM, H_MM))

fig.savefig(os.path.join(FIG, "Fig6.pdf"), facecolor="white")
fig.savefig(os.path.join(FIG, "Fig6.png"), facecolor="white")
plt.close(fig)

print("Fig6 done: %d KB (pdf) / %d KB (png)"
      % (os.path.getsize(os.path.join(FIG, "Fig6.pdf")) // 1024,
         os.path.getsize(os.path.join(FIG, "Fig6.png")) // 1024))
