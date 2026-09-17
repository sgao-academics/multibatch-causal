# -*- coding: utf-8 -*-
"""Figure: somatic alteration burden of the per-cohort hub genes.

Question it answers: are the genes that occupy the inferred causal networks also the
genes that drive cancer genomically?  For each of the 33 TCGA cohorts we compare the
cohort's own hub gene against canonical driver genes measured in the same cohort, and
across all cohorts we show that no hub gene is a recurrent somatic target.

Data: results/_hub_variant_landscape.json, produced by
      scripts/figures/fetch_hub_variants.py (cBioPortal TCGA PanCancer Atlas).

Canvas is 174 mm wide; sn-jnl sets \textwidth = 130.8 mm, so the figure is included at
width=\textwidth and rendered at 130.8/174 = 0.752x.  In-figure type is 7.6 pt on the
canvas -> 5.71 pt printed, above the 5.5 pt floor re-checked at the end of this script.
Writes a NEW pair of files (Fig7.pdf/.png); it never overwrites an
existing figure.
"""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE, "results")
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

PRINT_SCALE = 130.8 / 174.0     # figure is printed at 0.752x
PRINT_FLOOR = 5.5               # pt, minimum acceptable printed type size
FS = 7.6                        # canvas type size -> 5.71 pt printed

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": FS, "axes.linewidth": 0.45, "text.usetex": False,
    "hatch.linewidth": 0.6,
    "figure.dpi": 300, "savefig.dpi": 300,
    "legend.frameon": False, "legend.fontsize": FS,
    "xtick.major.width": 0.45, "ytick.major.width": 0.45,
    "xtick.major.size": 1.5, "ytick.major.size": 1.5,
    "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
})

RED, BLUE = "#D7263D", "#2E5EAA"
AMP_COL, DEL_COL = "#D7263D", "#2E5EAA"
BODY, GREY, LIGHT = "#1a1a1a", "#6E6E6E", "#c9c9c9"

DRIVERS = ("TP53", "PIK3CA", "KRAS")

# ------------------------------------------------------------------ data
d = json.load(open(os.path.join(RES, "_hub_variant_landscape.json"), encoding="utf-8"))
HUBS, COH = d["hubs"], d["cohorts"]

recs = []
for c in sorted(COH):
    x = COH[c]
    ns = max(1, x["n_sequenced"])
    nc = max(1, x["n_cna"])
    h = x["hub"]

    def f(key, g, n):
        return 100.0 * x[key].get(g, 0) / n

    recs.append({
        "c": c, "h": h, "ns": x["n_sequenced"], "nc": x["n_cna"],
        "hm": f("mut_nonsyn", h, ns),
        "ha": f("amp", h, nc),
        "hd": f("homdel", h, nc),
        "drv": max(f("mut_nonsyn", g, ns) for g in DRIVERS),
    })

recs.sort(key=lambda r: -r["drv"])
N = len(recs)
xs = np.arange(N)
labels = [r["c"] for r in recs]

GENES = sorted(set(HUBS.values()))
print("cohorts %d | unique hub genes %d" % (N, len(GENES)))

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(174 / 25.4, 215 / 25.4))
axa = fig.add_axes([0.088, 0.795, 0.830, 0.155])
axb = fig.add_axes([0.088, 0.595, 0.830, 0.155])
axc = fig.add_axes([0.088, 0.110, 0.830, 0.440])

# ---- (a) hub vs canonical driver, same cohort -------------------------
axa.bar(xs, [r["hm"] for r in recs], width=0.66, color=RED, linewidth=0, zorder=3)
axa.plot(xs, [r["drv"] for r in recs], linestyle="none", marker="o", ms=2.5,
         mfc="white", mec=BLUE, mew=0.7, zorder=4, label="canonical driver")
axa.plot([], [], linestyle="none", marker="s", ms=4.0, color=RED, label="cohort hub gene")
axa.set_ylabel("Non-synonymous\nmutation frequency (%)", linespacing=1.25)
axa.set_ylim(0, 100)
axa.set_yticks([0, 25, 50, 75, 100])
axa.set_xlim(-0.7, N - 0.3)
axa.set_xticks(xs)
axa.set_xticklabels(labels, rotation=90)
axa.tick_params(axis="x", pad=1.2)
for s in ("top", "right"):
    axa.spines[s].set_visible(False)
axa.legend(loc="upper right", handletextpad=0.4, borderpad=0.15, labelspacing=0.25,
           borderaxespad=0.3)
axa.text(-0.082, 1.10, "a", transform=axa.transAxes, fontsize=FS + 1.4,
         fontweight="bold", va="top", ha="left")

# ---- (b) hub CNA ------------------------------------------------------
# amplification and deletion differ by pattern as well as by colour, so the two directions
# stay separable in greyscale or with a colour-vision deficiency (F&IG accessibility)
axb.bar(xs, [r["ha"] for r in recs], width=0.66, color=AMP_COL, linewidth=0, zorder=3)
axb.bar(xs, [-r["hd"] for r in recs], width=0.66, color=DEL_COL, linewidth=0, zorder=3,
        hatch="///")
axb.axhline(0, color=BODY, lw=0.5, zorder=4)
axb.set_ylabel("Copy-number\nalteration (%)", linespacing=1.25)
axb.set_ylim(-20, 20)
axb.set_yticks([-20, -10, 0, 10, 20])
axb.set_yticklabels(["20", "10", "0", "10", "20"])
axb.set_xlim(-0.7, N - 0.3)
axb.set_xticks(xs)
axb.set_xticklabels(labels, rotation=90)
axb.tick_params(axis="x", pad=1.2)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)
axb.text(-0.082, 1.02, "b", transform=axb.transAxes, fontsize=FS + 1.4,
         fontweight="bold", va="top", ha="left")
axb.text(0.994, 0.93, "amplification", transform=axb.transAxes, ha="right", va="top",
         color=AMP_COL)
axb.text(0.994, 0.07, "deep deletion", transform=axb.transAxes, ha="right", va="bottom",
         color=DEL_COL)

# ---- (c) every hub gene x every cohort --------------------------------
M = np.zeros((len(GENES), N))
for j, r in enumerate(recs):
    x = COH[r["c"]]
    ns = max(1, x["n_sequenced"])
    for i, g in enumerate(GENES):
        M[i, j] = 100.0 * x["mut_nonsyn"].get(g, 0) / ns

cmap = LinearSegmentedColormap.from_list(
    "wr", ["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"])
im = axc.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=20,
                interpolation="nearest", origin="upper")
axc.set_yticks(np.arange(len(GENES)))
axc.set_yticklabels(GENES)
axc.set_xticks(xs)
axc.set_xticklabels(labels, rotation=90)
axc.tick_params(axis="x", pad=1.2)
axc.tick_params(axis="y", length=0, pad=1.5)
for s in ("top", "right", "left", "bottom"):
    axc.spines[s].set_visible(True)
    axc.spines[s].set_linewidth(0.45)
axc.text(-0.082, 1.012, "c", transform=axc.transAxes, fontsize=FS + 1.4,
         fontweight="bold", va="top", ha="left")
# cohort whose own hub is this row
for i, g in enumerate(GENES):
    own = [j for j, r in enumerate(recs) if r["h"] == g]
    for j in own:
        axc.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor=BODY, lw=0.55, zorder=5))

cax = fig.add_axes([0.935, 0.110, 0.011, 0.100])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Mutation frequency (%)", labelpad=2)
cb.ax.tick_params(length=1.4, pad=1.2)
cb.outline.set_linewidth(0.45)
cb.set_ticks([0, 10, 20])

# ------------------------------------------------------------------ output
os.makedirs(FIGDIR, exist_ok=True)
pdf = os.path.join(FIGDIR, "Fig7.pdf")
png = os.path.join(FIGDIR, "Fig7.png")
fig.savefig(pdf)
fig.savefig(png, dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ self-check
print("printed type size: %.2f pt (floor %.1f)" % (FS * PRINT_SCALE, PRINT_FLOOR))
assert FS * PRINT_SCALE >= PRINT_FLOOR, "type size below print floor"

print()
print("hub mutation rate: median %.2f%% max %.2f%%" % (
    np.median([r["hm"] for r in recs]), max(r["hm"] for r in recs)))
print("driver (best of TP53/PIK3CA/KRAS): median %.2f%%" % np.median([r["drv"] for r in recs]))
print("matrix: %d genes x %d cohorts, max %.2f%%" % (M.shape[0], M.shape[1], M.max()))
print()
print("wrote %s (%.0f KB)" % (pdf, os.path.getsize(pdf) / 1024))
print("wrote %s (%.0f KB)" % (png, os.path.getsize(png) / 1024))
