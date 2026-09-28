# -*- coding: utf-8 -*-
"""Figure 5: overall-survival scan of the per-cohort network hubs.

  a) forest plot of every cohort scanned (33), ordered by significance
  b) Kaplan-Meier curves for the six that pass the log-rank threshold

Why the forest plot was added
-----------------------------
The plate used to draw only the six passing cohorts.  Its own docstring said the
opposite of what the artwork showed: "the panel reports an exhaustive scan, not a
selected subset -- the count of scanned cohorts is part of the message".  Nothing
in the picture carried that count.  Every cohort does carry a hazard ratio with a
95% interval, and the six intervals that exclude 1 are exactly the six that pass,
so one forest plot states both facts at once and the KM curves stay as the
detail behind it.

The two panels deliberately do not repeat each other: the forest plot owns all of
the statistics (HR, interval, direction, sample size), so the KM plates carry only
the curves, their censor ticks and the split they are based on.

The gene tested in each cohort is the highest-degree gene of that cohort's
inferred network -- fixed by the network rather than picked on the outcome -- so
the six that separate are an exhaustive-scan result, not a selection.

Journal requirements applied here
---------------------------------
* 174 mm wide (the printed text width) and 225 mm tall, inside the 234 mm ceiling.
* Nothing below 8 pt anywhere, axis labels and legends included.
* Sans-serif Arial throughout, including the maths text.
* Both encoded distinctions survive greyscale: in the forest plot filled against
  open markers, in the KM plates solid against dashed curves.

Output: figures/Fig5.pdf / .png
"""
import os, sys, json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory as blend

from lifelines import KaplanMeierFitter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import PAL, INK, GREY, TICK, MM, tidy, card
from _gene_symbols import canonical, labels

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
_LOCAL = os.path.join(ROOT, "data")
DATA = os.environ.get("MULTIBATCH_DATA") or _LOCAL
OSDIR = os.path.join(DATA, "validation", "pancan_os")

ALPHA = 0.05
FS = 8.0                      # the single lettering size of this plate
PLET = 11.0                   # panel letters, matching the other figures
LOW_COL, HIGH_COL = PAL["mist"], PAL["orchid"]


def event(s):
    return 1 if (isinstance(s, str) and s.startswith("1")) else 0


def sample_to_patient(s):
    p = str(s).split("-")
    return "-".join(p[:3]) if len(p) >= 3 else str(s)


def dataset(cohort, gene):
    """Median split of one gene in one cohort, joined to that cohort's survival."""
    expr = pd.read_csv(os.path.join(DATA, "TCGA_%s_HiSeqV2.tsv" % cohort),
                       sep="\t", index_col=0)
    # not a bare .upper(): the hub of ACC prints as SHOC1 while Xena labels the row
    # C9orf84, so the lookup has to resolve both the rename and the case.
    idx = labels(expr.index)
    key = idx.get(canonical(gene).upper())
    if key is None:
        return None, None
    ser = expr.loc[key]
    agg = {}
    for s, v in ser.items():
        agg.setdefault(sample_to_patient(s), []).append(v)
    agg = {k: float(np.mean(v)) for k, v in agg.items()}
    os_rows = json.load(open(os.path.join(OSDIR, "%s_os.json" % cohort)))
    m = pd.DataFrame([{"patient": r["patient"], "os_months": float(r["os_months"]),
                       "event": event(r["os_status"])} for r in os_rows])
    m["gexpr"] = m["patient"].map(agg)
    m = m.dropna(subset=["gexpr", "os_months"])
    med = m["gexpr"].median()
    return key, (m[m["gexpr"] >= med], m[m["gexpr"] < med])


# ------------------------------------------------------------------ the scan
scan = json.load(open(os.path.join(RES, "_km_pancan.json"), encoding="utf-8"))
rows = []
for c, v in scan.items():
    h = v.get("hub")
    st = v["genes"].get(h) if h else None
    if st:
        rows.append((c, h, st))
rows.sort(key=lambda t: t[2]["logrank_p"])
passers = [r for r in rows if r[2]["logrank_p"] < ALPHA]
print("scanned %d cohorts, %d pass log-rank p < %.2f" % (len(rows), len(passers), ALPHA))
for c, h, st in rows:
    if st["logrank_p"] < ALPHA:
        print("   %-5s %-10s p=%.3g HR=%.2f [%.2f-%.2f]  n=%d/%d"
              % (c, h, st["logrank_p"], st["hr"], st["hr_lo"], st["hr_hi"],
                 st["n_low"], st["n_high"]))
_excl = [r for r in rows if not (r[2]["hr_lo"] <= 1.0 <= r[2]["hr_hi"])]
print("intervals excluding 1: %d;  identical to the passers: %s"
      % (len(_excl), set(r[0] for r in _excl) == set(r[0] for r in passers)))
assert len(passers) <= 6, "panel set does not fit the grid"
NMAX = max(r[2]["n_low"] + r[2]["n_high"] for r in rows)

W_MM, H_MM = 174.0, 225.0
fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))

# ================================================================= a) forest
axF = fig.add_axes([0.150, 0.425, 0.695, 0.533])
TX = blend(axF.transAxes, axF.transData)     # x in axes units, y in row units

XLO, XHI = 0.02, 45.0
axF.set_xscale("log")
axF.set_xlim(XLO, XHI)
axF.set_ylim(-1.0, len(rows) - 0.2)
axF.set_yticks([])

for i, (c, h, st) in enumerate(rows):
    if st["logrank_p"] < ALPHA:              # very light band under the six
        axF.axhspan(len(rows) - 1 - i - 0.44, len(rows) - 1 - i + 0.44,
                    color="#F1F4F8", zorder=0)
axF.axvline(1.0, color="#9AA3B0", lw=0.8, ls=(0, (3, 2.4)), zorder=1)

for i, (c, h, st) in enumerate(rows):
    y = len(rows) - 1 - i
    sig = st["logrank_p"] < ALPHA
    col = (LOW_COL if st["hr"] < 1 else HIGH_COL) if sig else "#B9C0CB"
    axF.plot([st["hr_lo"], st["hr_hi"]], [y, y], color=col,
             lw=1.5 if sig else 0.8, solid_capstyle="round",
             alpha=1.0 if sig else 0.75, zorder=3)
    s = 12 + 44.0 * ((st["n_low"] + st["n_high"]) / float(NMAX))
    axF.scatter([st["hr"]], [y], s=s, facecolor=col if sig else "white",
                edgecolor="white" if sig else col, linewidths=0.7,
                zorder=5)
    axF.text(-0.013, y, c, transform=TX, ha="right", va="center", fontsize=FS,
             color=INK if sig else TICK,
             fontweight="bold" if sig else "normal", zorder=6)
    axF.text(1.013, y, canonical(h), transform=TX, ha="left", va="center",
             fontsize=FS, color=INK if sig else GREY, fontstyle="italic", zorder=6)

axF.set_xticks([0.03, 0.1, 0.3, 1, 3, 10, 30])
axF.set_xticklabels(["0.03", "0.1", "0.3", "1", "3", "10", "30"])
axF.set_xlabel("hazard ratio, high vs low expression   (95% CI, log scale)",
               labelpad=4)
for _s in ("top", "right", "left"):
    axF.spines[_s].set_visible(False)
tidy(axF)
axF.tick_params(length=2.4, pad=2)
card(fig, axF, pad=0.007, radius=0.010)

# The strip above the forest plot: the letter sits clear of the first row label,
# which otherwise shares its bounding box with the top row of the table.
fig.text(0.150, 0.976, "%d of %d cohorts pass log-rank $p < 0.05$"
         % (len(passers), len(rows)), ha="left", va="top", fontsize=FS, color=INK)
fig.text(0.845, 0.976, "shaded rows = pass", ha="right", va="top",
         fontsize=FS, color=GREY)
fig.text(0.117, 0.980, "a)", ha="left", va="top", fontsize=PLET,
         fontweight="bold", color=INK)

# ================================================================= b) curves
fig.text(0.072, 0.400, "b)", ha="left", va="top", fontsize=PLET,
         fontweight="bold", color=INK)
gs = fig.add_gridspec(2, 3, left=0.076, right=0.988, bottom=0.050, top=0.372,
                      wspace=0.34, hspace=0.50)
nrow, ncol = 2, 3
for i, (cohort, hub, st) in enumerate(passers):
    ax = fig.add_subplot(gs[i // ncol, i % ncol])
    shown = canonical(hub)
    _, (hi, lo) = dataset(cohort, hub)
    kmf = KaplanMeierFitter()
    for name, sub, col in (("Low", lo, LOW_COL), ("High", hi, HIGH_COL)):
        kmf.fit(sub["os_months"], sub["event"],
                label="%s ($n$ = %d)" % (name, len(sub)))
        kmf.plot_survival_function(ax=ax, color=col, lw=1.15, ci_show=True,
                                   ci_alpha=0.16, show_censors=True,
                                   censor_styles={"marker": "|", "ms": 3.2,
                                                  "mew": 0.65})
    # the two groups differ by line style as well as by colour, so the curves stay
    # separable in greyscale; the censor ticks carry a marker, so they are skipped
    for ln in ax.get_lines():
        if ln.get_marker() in ("None", "") and ln.get_color() == HIGH_COL:
            ln.set_linestyle((0, (3.5, 1.6)))
    ax.set_title("%s \u00b7 %s" % (cohort, shown), fontsize=FS,
                 fontweight="bold", color=INK, pad=3.0)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([0, 50, 100, 150])
    tidy(ax)
    ax.tick_params(length=2.0, pad=1.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if i % ncol == 0:
        ax.set_ylabel("Overall survival", fontsize=FS, labelpad=1.8)
    # lifelines writes its own "timeline" label on every call, so the axis has to be
    # set -- or blanked -- afterwards, or the top row reads "timeline".
    ax.set_xlabel("Months" if i // ncol == nrow - 1 else "", fontsize=FS, labelpad=1.8)
    ax.legend(fontsize=FS, frameon=False, loc="lower left",
              handlelength=1.25, handletextpad=0.35, labelspacing=0.30,
              borderpad=0.1, bbox_to_anchor=(-0.015, -0.022))
    ax.set_xticklabels(["0", "50", "100", "150"], fontsize=FS)
    ax.set_yticklabels(["0", "0.5", "1.0"], fontsize=FS)

# ---- nothing may leave the canvas ----
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_W, _H = fig.get_size_inches() * fig.dpi
_bad = []
for _t in fig.findobj(matplotlib.text.Text):
    if _t.axes is None and _t.get_figure() is not fig:
        continue
    if not _t.get_visible() or not _t.get_text().strip():
        continue
    _b = _t.get_window_extent(renderer=_rend)
    if _b.x0 < -12 or _b.y0 < -12 or _b.x1 > _W + 12 or _b.y1 > _H + 12:
        _bad.append((_t.get_text()[:30].replace("\n", " "), round(_b.x0, 1),
                     round(_b.x1, 1)))
if _bad:
    print("ABORT: %d text elements leave the canvas" % len(_bad))
    for b in _bad[:12]:
        print("   ", b)
    sys.exit(1)
print("canvas check passed: all text inside %.2f x %.2f in"
      % tuple(fig.get_size_inches()))

fig.savefig(os.path.join(FIG, "Fig5.pdf"), facecolor="white")
fig.savefig(os.path.join(FIG, "Fig5.png"), facecolor="white")
plt.close(fig)
print("Fig5 done: %d KB (pdf) / %d KB (png)"
      % (os.path.getsize(os.path.join(FIG, "Fig5.pdf")) // 1024,
         os.path.getsize(os.path.join(FIG, "Fig5.png")) // 1024))
