# -*- coding: utf-8 -*-
"""Figure S1: parameter sensitivity of the NOTEARS pipeline, and what the
recovered edge count does to per-cohort reuse.

Rebuilt on the same drawing language as Figures 1-7 (`_figstyle`): the muted
palette, indigo type, a rounded card behind each panel, markers carrying a soft
shadow rather than a hard outline, panel letters as bold ``a)``, and one 8 pt
lettering size.  The earlier version was drawn with the matplotlib defaults --
three saturated series colours, a 7.6 pt tick size and bare ``a``-``d`` titles --
which made the supplementary plates read as a different paper from the main text.

Panels
------
a  lambda   l1 regularization against recovered edge count, three cohorts.
b  tau      edge threshold against recovered edge count, three cohorts.
c  spread   the distribution of per-cohort edge counts across all 33 cohorts.
d  reuse    edge count against that cohort's reuse rate.  Kept as a panel because
            it is the mechanism behind the headline dispersion: a cohort whose
            recovery is inflated also shares less of it.

Two layout facts are forced by the data rather than chosen:
*  The reference lines (lambda = 0.01, tau = 0.3) are dashed rules whose meaning
  is carried by the legend, not by an in-artwork title -- the journal asks that
  an illustration carry no title of its own.
*  Panel c draws all 33 cohorts as points behind the box.  A box alone hides how
  few cohorts the whiskers stand for, and the distribution is far too skewed for
  a summary to be read off safely.

Output: figures/FigS1.pdf / .png   (canvas 174 x 152 mm = the printed measure)
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _figstyle import (PAL, INK, GREY, TICK, TRACK, GRID, FRAME, MM,
                       CARD_FC, CARD_EC, BOXFC, BOXEC, SERIES, tidy, card, dots)

ROOT = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

FS = 8.0        # every label, tick and annotation in the artwork
FL = 11.0       # panel letters, as in Figures 1-7
W_MM, H_MM = 174.0, 152.0
W, H = W_MM * MM, H_MM * MM

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
d = json.load(open(os.path.join(RES, "_figure_data.json"), encoding="utf-8"))
sens = d["sensitivity"]
lam, tau = sens["lam"], sens["tau"]
CANCERS = ["BRCA", "LUAD", "COAD"]

nt = json.load(open(os.path.join(RES, "_pipeline_notears.json"), encoding="utf-8"))
cancers_raw = sorted([k for k in nt if isinstance(nt[k], dict) and "W" in nt[k]])
pair_count = Counter()
for c in cancers_raw:
    nd = nt[c]
    Wm, genes = np.array(nd["W"]), nd["genes"]
    for i in range(100):
        for j in range(100):
            if i != j and abs(Wm[i, j]) > 0.3:
                pair_count[(genes[i], genes[j])] += 1
edges_raw, reuse_all = [], []
for c in cancers_raw:
    nd = nt[c]
    Wm, genes = np.array(nd["W"]), nd["genes"]
    ec = set()
    for i in range(100):
        for j in range(100):
            if i != j and abs(Wm[i, j]) > 0.3:
                ec.add((genes[i], genes[j]))
    edges_raw.append(int(nd["edges"]) if "edges" in nd else len(ec))
    reuse_all.append(100.0 * sum(1 for p in ec if pair_count.get(p, 0) >= 2)
                     / max(len(ec), 1))

ec_arr = np.asarray(d["edge_counts"], dtype=float)
MEAN, SD = ec_arr.mean(), ec_arr.std(ddof=1)
# The scatter's x and the box's data have to be the same 33 numbers, or the two
# panels would be describing different quantities.
assert sorted(edges_raw) == sorted(int(v) for v in ec_arr), \
    "the scatter's edge counts and the boxplot's are not the same data"
r_re, p_re = spearmanr(edges_raw, reuse_all)
r_re, p_re = float(r_re), float(p_re)

# --------------------------------------------------------------------------
# Layout, in mm from the top-left
# --------------------------------------------------------------------------
TOP, H1, GAPY, H2, BOT = 6.5, 57.0, 16.5, 57.0, 15.0
LEFT, CW, CGAP, RIGHT = 12.0, 72.0, 15.0, 3.0
R1Y, R2Y = TOP, TOP + H1 + GAPY


def ax_rect(x0, y0, w, h):
    return [x0 / W_MM, 1.0 - (y0 + h) / H_MM, w / W_MM, h / H_MM]


fig = plt.figure(figsize=(W, H))
fig.patch.set_facecolor("white")

axa = fig.add_axes(ax_rect(LEFT, R1Y, CW, H1))
axb = fig.add_axes(ax_rect(LEFT + CW + CGAP, R1Y, CW, H1))
axc = fig.add_axes(ax_rect(LEFT, R2Y, CW, H2))
axd = fig.add_axes(ax_rect(LEFT + CW + CGAP, R2Y, CW, H2))

# --------------------------------------------------------------------------
# a  lambda sensitivity
# --------------------------------------------------------------------------
lam_x = [0.001, 0.005, 0.01, 0.02]
for (cname, mk), c in zip(SERIES, CANCERS):
    y = lam[c]
    axa.plot(lam_x, y, "-", color=PAL[cname], lw=1.15, zorder=3,
             solid_capstyle="round")
    dots(axa, lam_x, y, 26, PAL[cname], z=4, marker=mk, lw=0.7)
axa.axvline(0.01, color=FRAME, lw=1.0, ls=(0, (3.0, 2.4)), zorder=2)
axa.set_xscale("log")
axa.set_xticks(lam_x)
axa.set_xticklabels(["0.001", "0.005", "0.01", "0.02"])
axa.set_xlim(0.00075, 0.030)
axa.set_ylim(0, 196)
axa.set_yticks([0, 50, 100, 150])
axa.set_xlabel(r"$\lambda_1$ (regularization)")
axa.set_ylabel("Causal edges")
axa.xaxis.set_minor_locator(mticker.NullLocator())
tidy(axa, grid="y")

# --------------------------------------------------------------------------
# b  tau sensitivity
# --------------------------------------------------------------------------
tau_x = [0.1, 0.2, 0.3, 0.4, 0.5]
for (cname, mk), c in zip(SERIES, CANCERS):
    y = tau[c]
    axb.plot(tau_x, y, "-", color=PAL[cname], lw=1.15, zorder=3,
             solid_capstyle="round")
    dots(axb, tau_x, y, 26, PAL[cname], z=4, marker=mk, lw=0.7)
axb.axvline(0.3, color=FRAME, lw=1.0, ls=(0, (3.0, 2.4)), zorder=2)
axb.set_xticks(tau_x)
axb.set_xlim(0.05, 0.55)
axb.set_ylim(0, 618)
axb.set_yticks([0, 100, 200, 300, 400, 500])
axb.set_xlabel(r"$\tau$ (edge threshold)")
axb.set_ylabel("Causal edges")
tidy(axb, grid="y")

# The reference rule is explained in the legend instead of by a title inside the
# artwork, so neither panel needs a text label sitting on the data.
_ref = Line2D([], [], color=FRAME, lw=1.0, ls=(0, (3.0, 2.4)))
for ax in (axa, axb):
    hs = [Line2D([], [], color=PAL[cn], marker=mk, lw=1.15, ms=3.6,
                 mec="white", mew=0.7) for cn, mk in SERIES]
    ax.legend(hs + [_ref], CANCERS + ["selected value"], loc="upper right",
              frameon=False, fontsize=FS, handlelength=1.9, handletextpad=0.5,
              borderpad=0.0, borderaxespad=0.35, labelspacing=0.28)

# --------------------------------------------------------------------------
# c  edge-count distribution, all 33 cohorts
# --------------------------------------------------------------------------
q1, q3 = np.percentile(ec_arr, [25, 75])
iqr = q3 - q1
lo = ec_arr[ec_arr >= q1 - 1.5 * iqr].min()
hi = ec_arr[ec_arr <= q3 + 1.5 * iqr].max()
stats = dict(med=float(np.median(ec_arr)), q1=float(q1), q3=float(q3),
             whislo=float(lo), whishi=float(hi), fliers=[])
axc.bxp([stats], positions=[1.0], widths=0.36, patch_artist=True,
        showfliers=False, zorder=3,
        boxprops=dict(facecolor="#E7EDF6", edgecolor=PAL["peri"], lw=0.9),
        medianprops=dict(color=PAL["mist"], lw=1.6),
        whiskerprops=dict(color=PAL["peri"], lw=0.9),
        capprops=dict(color=PAL["peri"], lw=0.9))
rng = np.random.default_rng(42)
jit = rng.uniform(-0.175, 0.175, size=ec_arr.size)
dots(axc, 1.0 + jit, ec_arr, 17, PAL["mist"], z=5, alpha=0.72, lw=0.5)
axc.set_xlim(0.55, 1.58)
axc.set_ylim(0, 268)
axc.set_xticks([1.0])
axc.set_xticklabels(["33 cancers"])
axc.set_yticks([0, 50, 100, 150, 200])
axc.set_ylabel("Causal edges")
tidy(axc, grid="y")
axc.annotate("mean %.1f\nSD %.1f" % (MEAN, SD), xy=(1.55, 262),
             ha="right", va="top", fontsize=FS, color=INK, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.38", fc=BOXFC, ec=BOXEC, lw=0.7),
             zorder=8)

# --------------------------------------------------------------------------
# d  edge count against reuse rate
# --------------------------------------------------------------------------
dots(axd, edges_raw, reuse_all, 26, PAL["mist"], z=5, lw=0.6)
axd.set_xlim(5, 245)
axd.set_ylim(-4, 74)
axd.set_xticks([50, 100, 150, 200])
axd.set_yticks([0, 20, 40, 60])
axd.set_xlabel("Edge count")
axd.set_ylabel("Reuse rate (%)")
tidy(axd, grid="y")
# Both numbers are formatted from the recomputed statistics, so the artwork
# cannot drift away from the caption.
_p_txt = "$p < 0.001$" if p_re < 0.001 else "$p = %.3f$" % p_re
axd.annotate("Spearman $r = %.2f$\n%s" % (r_re, _p_txt), xy=(238, 70),
             ha="right", va="top", fontsize=FS, color=INK, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.38", fc=BOXFC, ec=BOXEC, lw=0.7),
             zorder=8)

# --------------------------------------------------------------------------
# Cards and panel letters
# --------------------------------------------------------------------------
for ax in (axa, axb, axc, axd):
    card(fig, ax, pad=0.012)
for ax, s in zip((axa, axb, axc, axd), "abcd"):
    bb = ax.get_position()
    fig.text(bb.x0 - 0.020, bb.y1 + 0.012, s + ")", fontsize=FL,
             fontweight="bold", ha="left", va="bottom", color=INK)

# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------
fig.canvas.draw()
rr = fig.canvas.get_renderer()
import matplotlib.transforms as mtr
inv = mtr.Affine2D().scale(25.4 / fig.dpi)

every = []
for ax in (axa, axb, axc, axd):
    every += [t for t in ax.texts if t.get_text()]
    every += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
    every += [ax.xaxis.label, ax.yaxis.label]
    if ax.get_legend() is not None:
        every += list(ax.get_legend().get_texts())
for t in fig.texts:
    every.append(t)

bad = []
boxes = []
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
        if ix > 1.4 and iy > 1.2:
            ovl.append((boxes[i][0], boxes[j][0], round(ix, 1), round(iy, 1)))

print("canvas %.0f x %.0f mm | panels 4 | FS %.1f pt | letters %.1f pt"
      % (W_MM, H_MM, FS, FL))
print("edge counts: mean %.1f  SD %.1f  median %.1f  min %d  max %d"
      % (MEAN, SD, np.median(ec_arr), ec_arr.min(), ec_arr.max()))
print("Spearman(edges, reuse) recomputed: r=%.2f" % r_re)
print("tick size floor: %.2f pt printed" % (FS * 174.9 / W_MM))
print("outside canvas: %d %s" % (len(bad), bad[:6]))
print("overlapping text pairs: %d %s" % (len(ovl), ovl[:6]))

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, "FigS1." + ext), dpi=400, facecolor="white")
print("wrote figures/FigS1.pdf and .png  (%.1f KB / %.1f KB)"
      % (os.path.getsize(os.path.join(FIG, "FigS1.pdf")) / 1024.0,
         os.path.getsize(os.path.join(FIG, "FigS1.png")) / 1024.0))
