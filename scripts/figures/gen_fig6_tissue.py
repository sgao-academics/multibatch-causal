# -*- coding: utf-8 -*-
"""Figure 6: tissue specificity of the per-cancer network hub genes.

Three panels, replacing the nine the plate used to carry.

Why the seven box panels had to go
----------------------------------
Panels c-i used to be the same box plot drawn seven times, once per hub gene:
seven panels, seven sets of axes, between them one message.  What actually
separates those genes is the *shape* of where they sit across the 33 cohorts --
SLC22A6 and GABRA1 are pinned to the floor in 24-28 of them and then spike,
while CD14 sits in a single high band everywhere -- and a box plot cannot show a
shape.  One ridgeline carries all seven, and the shape difference is the panel.

The planned alternative was a gene x cohort dot matrix on expression z-scores.
It was dropped on the numbers, not on taste: only 3 of the 7 rows have the gene's
own cancer as the brightest cell of its row, CD14's own cancer (LAML) is the
*dimmest* cell of its row, and once SLC22A6's 28 zero cohorts are z-scored the
whole matrix collapses to one flat tint -- 231 dots carrying no readable pattern.
The own-cohort reading survives as the right-hand column instead.

Drawing language
----------------
Same tokens as Figures 1-5, imported from `_figstyle`: the seven muted colours,
type in the palette's indigo, a faint rounded card behind each panel, the same
one 8 pt lettering size, panel letters at 11 pt bold.

Journal requirements applied here
---------------------------------
* 174 mm wide -- the printed text width, so the point sizes are the sizes that
  appear in print.  `bbox_inches='tight'` is deliberately NOT used.
* Nothing below 8 pt; the page is the canvas and the canvas check is a hard gate.
* Sans-serif Arial throughout, including the maths text; Type 42 embedding.
* Every encoded distinction survives greyscale: the hub dots are open circles on
  a filled violin, the null is a dashed line against a solid one.

Output: figures/Fig6.pdf / .png
"""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from scipy.stats import gaussian_kde, mannwhitneyu

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import (PAL, INK, GREY, TICK, FRAME, MM, tidy, card, dots,
                       haloed_text_pt, panel_letters)

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

d = json.load(open(os.path.join(RES, "_tau_specificity.json"), encoding="utf-8"))
p1 = json.load(open(os.path.join(RES, "_hub_expression.json"), encoding="utf-8"))
cancers = d["cancers"]
hub_of = d["hub_of_cancer"]
raw = d["hub_raw"]
stats = p1["stats"]
hub_tau = np.asarray(d["hub_tau"], dtype=float)
nonhub_tau = np.asarray(d["nonhub_tau"], dtype=float)
rank_mean = d["hub_rank_mean"]
perm_p = d["hub_rank_perm_p"]

mw_p = float(mannwhitneyu(hub_tau, nonhub_tau, alternative="two-sided")[1])
hub_lower = bool(hub_tau.mean() < nonhub_tau.mean())

# The seven panels this figure used to carry, kept as the panel-c rows.
SHOW = ["SLC22A6", "GABRA1", "LTF", "NAPSA", "CXCL5", "ADH1B", "CD14"]
COLS = [PAL["mist"], PAL["orchid"], PAL["moss"], PAL["violet"],
        PAL["peri"], PAL["indigo"], PAL["lilac"]]
own_of = {g: next((c for c in cancers if hub_of.get(c) == g), None) for g in SHOW}

# ---- assertions: the caption's claims must hold in the data ----
if [c for c in SHOW if own_of[c] is None]:
    print("ABORT: a drawn gene is not the hub of any cohort")
    sys.exit(1)
_med = np.full((len(SHOW), len(cancers)), np.nan)
for _i, _g in enumerate(SHOW):
    for _j, _c in enumerate(cancers):
        _v = raw.get(_g, {}).get(_c)
        if _v:
            _med[_i, _j] = float(np.median(_v))
if np.isnan(_med).any():
    print("ABORT: the 7 x 33 median matrix has holes")
    sys.exit(1)
if not (0.60 < hub_tau.mean() < 0.63 and 0.69 < nonhub_tau.mean() < 0.71):
    print("ABORT: tau means moved (hub %.3f, non-hub %.3f)" % (hub_tau.mean(), nonhub_tau.mean()))
    sys.exit(1)
if not (abs(mw_p - 0.047) < 0.002 and abs(rank_mean - 11.7) < 0.1 and abs(perm_p - 0.001) < 0.001):
    print("ABORT: the quoted statistics moved (MW %.4f, rank %.2f, perm %.4f)"
          % (mw_p, rank_mean, perm_p))
    sys.exit(1)
print("claim checks passed: MW p=%.4f, rank mean %.2f vs 17.0, perm p=%.4f"
      % (mw_p, rank_mean, perm_p))

_DIAG = sum(1 for _i in range(len(SHOW))
            if np.nanargmax(_med[_i]) == cancers.index(own_of[SHOW[_i]]))
print("own cancer is the brightest cell in %d of the %d rows -- the reason the planned "
      "dot matrix was dropped" % (_DIAG, len(SHOW)))
for _i, _g in enumerate(SHOW):
    _o = np.argsort(-_med[_i])
    _rk = list(_o).index(cancers.index(own_of[_g])) + 1
    print("   %-8s own %-5s rank %2d/33  zeros in %2d of 33 cohorts  peak %.1f"
          % (_g, own_of[_g], _rk, int((_med[_i] == 0).sum()), np.nanmax(_med[_i])))

FS = 8.0                                     # the single lettering size of this plate
W_MM, H_MM = 174.0, 216.0

fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
axa = fig.add_axes([0.083, 0.638, 0.352, 0.300])
axb = fig.add_axes([0.600, 0.638, 0.385, 0.300])
axc = fig.add_axes([0.082, 0.052, 0.712, 0.512])

# ===================================================================== a) tau
parts = axa.violinplot([nonhub_tau, hub_tau], positions=[1, 2], widths=0.74,
                       showextrema=False, showmedians=False)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(PAL["lilac"] if i == 0 else PAL["mist"])
    pc.set_alpha(0.58)
    pc.set_edgecolor("white")
    pc.set_linewidth(0.4)
bp = axa.boxplot([nonhub_tau, hub_tau], positions=[1, 2], widths=0.20, showfliers=False,
                 patch_artist=True, medianprops=dict(color="white", linewidth=1.0))
for i, b in enumerate(bp["boxes"]):
    b.set_facecolor(PAL["violet"] if i == 0 else PAL["mist"])
    b.set_edgecolor("white")
    b.set_linewidth(0.4)

# The 1,745 non-hubs can only be shown as a distribution; the 30 hubs fit as
# individuals, so they are drawn as individuals.  The asymmetry is the honesty.
_rng = np.random.default_rng(0)
_jit = _rng.uniform(-0.21, 0.21, len(hub_tau))
dots(axa, 2.0 + _jit, hub_tau, 15, "white", z=6, shadow=False,
     ec=PAL["indigo"], lw=0.7, alpha=0.95)
axa.set_xticks([1, 2])
axa.set_xticklabels(["non-hub\n$n$ = 1,745", "hub\n$n$ = 30"], fontsize=FS, linespacing=1.6)
axa.set_ylabel(r"tissue-specificity index $\tau$", fontsize=FS)
axa.set_ylim(-0.02, 1.02)
axa.set_xlim(0.40, 2.60)
axa.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
haloed_text_pt(axa, 1.5, 0.985, "$p$ = %.3f  (hubs %s)"
               % (mw_p, "lower" if hub_lower else "higher"), dx=0, dy=0, size=FS,
               color=INK, ha="center", va="top")
axa.tick_params(axis="x", length=0)

# ===================================================================== b) rank
_ranks = np.array(sorted(v["rank"] for v in stats.values()))
axb.plot(_ranks, np.arange(1, len(_ranks) + 1) / len(_ranks), marker="o", ms=2.8,
         color=PAL["mist"], lw=1.1, label="observed (%d hubs)" % len(_ranks))
axb.plot([1, 33], [1 / 33, 1.0], ls="--", lw=0.9, color=PAL["orchid"],
         label="uniform null")
axb.set_xlabel("rank of own cancer", fontsize=FS)
axb.set_ylabel("cumulative fraction of hubs", fontsize=FS)
# Linear axis, as the panel was drawn before the redesign: rank is an ordinal
# count of cohorts, and a log axis would re-space the very comparison the null
# line exists to make.
axb.set_xlim(0.5, 33.5)
axb.set_ylim(0, 1.03)
axb.set_xticks([1, 5, 10, 15, 20, 25, 30, 33])
axb.legend(fontsize=FS, frameon=False, loc="lower right", handlelength=1.6)
haloed_text_pt(axb, 0.62, 0.955, "mean rank %.1f (null 17.0)\npermutation $p$ = %.3f"
               % (rank_mean, perm_p), dx=0, dy=0, size=FS, color=INK,
               ha="left", va="top")

# ===================================================================== c) ridgeline
_grid = np.linspace(-0.6, 16.6, 600)
_tr = blended_transform_factory(axc.transAxes, axc.transData)
for i, g in enumerate(SHOW):
    base = len(SHOW) - i
    axc.plot([-0.6, 16.6], [base, base], color=FRAME, lw=0.5, zorder=1)
    _kde = gaussian_kde(_med[i], bw_method=0.30)
    _dens = _kde(_grid)
    # 0.82 of the row pitch, not 0.92: ADH1B's peak and the baseline one row above
    # it were closing to 0.08 of a row, which reads as a collision.
    _dens = _dens / _dens.max() * 0.82
    axc.fill_between(_grid, base, base + _dens, color=COLS[i], alpha=0.50, lw=0,
                     zorder=2 + i)
    axc.plot(_grid, base + _dens, color="white", lw=0.9, zorder=2 + i)
    _yv = _med[i, cancers.index(own_of[g])]
    _ytop = base + float(np.interp(_yv, _grid, _dens))
    axc.plot([_yv, _yv], [base, _ytop], color=INK, lw=0.7, ls=(0, (2.0, 1.6)), zorder=12)
    dots(axc, [_yv], [base], 26, PAL["violet"], z=13, ec="white", lw=0.8)
    _rk = list(np.argsort(-_med[i])).index(cancers.index(own_of[g])) + 1
    axc.text(1.014, base + 0.02, own_of[g], transform=_tr, ha="left", va="center",
             fontsize=FS, color=INK)
    axc.text(1.014, base - 0.30, "rank %d" % _rk, transform=_tr, ha="left",
             va="center", fontsize=FS, color=GREY)
axc.text(1.014, len(SHOW) + 1.02, "own cohort", transform=_tr, ha="left", va="bottom",
         fontsize=FS, color=GREY)
axc.set_yticks(range(1, len(SHOW) + 1))
axc.set_yticklabels([r"$\it{%s}$" % g for g in SHOW[::-1]], fontsize=FS)
axc.set_xlabel("log$_2$(TPM+1), density of the 33 cohort medians", fontsize=FS)
axc.set_xlim(-0.6, 16.6)
axc.set_ylim(0.42, len(SHOW) + 1.05)
axc.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
axc.tick_params(axis="y", length=0)

for _a in (axa, axb, axc):
    for _s in ("top", "right"):
        _a.spines[_s].set_visible(False)
    tidy(_a)
axc.spines["left"].set_visible(False)
card(fig, axa, pad=0.011, radius=0.014)
card(fig, axb, pad=0.010, radius=0.014)
card(fig, axc, pad=0.008, radius=0.012)
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
    _b = _t.get_window_extent(renderer=_rend)
    if _b.x0 < -12 or _b.y0 < -12 or _b.x1 > _W + 12 or _b.y1 > _H + 12:
        _bad.append((_t.get_text()[:34].replace("\n", " "), round(_b.x0, 1), round(_b.x1, 1)))
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
