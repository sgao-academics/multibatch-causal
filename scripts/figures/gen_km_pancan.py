"""Pan-cancer Kaplan-Meier panel for the per-cohort network hubs.

Every TCGA cohort is scanned: the cohort's own hub gene (the highest-degree gene in that
cohort's inferred causal network) is split at its median expression and tested for overall
survival. Six of the 33 cohorts pass a 5% log-rank threshold. Because the gene for each
cohort is fixed by the network rather than chosen on the outcome, the panel reports an
exhaustive scan, not a selected subset -- the count of scanned cohorts is part of the message.

The canvas is the journal text width (130.8 mm, sn-jnl sn-basic), so the figure is included
at width=\textwidth with no rescaling and the type sizes below are the printed sizes.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gene_symbols import canonical, labels

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE, 'results')
FIG = os.path.join(BASE, 'figures')
# TCGA matrices: ./data/ inside the package, or the folder named by MULTIBATCH_DATA
_LOCAL = os.path.join(BASE, 'data')
DATA = os.environ.get('MULTIBATCH_DATA') or _LOCAL
OSDIR = os.path.join(DATA, 'validation', 'pancan_os')

ALPHA = 0.05
W_MM, H_MM = 130.8, 150.0
LOW_COL, HIGH_COL = '#2C6FB5', '#C0392B'

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    # Arial for the maths glyphs as well: the DejaVu default was supplying the hazard-ratio
    # times sign, the minus sign and the italic gene symbols as a second family
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'text.usetex': False, 'axes.linewidth': 0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.labelsize': 5.6, 'ytick.labelsize': 5.6,
})


def event(s):
    return 1 if (isinstance(s, str) and s.startswith('1')) else 0


def sample_to_patient(s):
    p = str(s).split('-')
    return '-'.join(p[:3]) if len(p) >= 3 else str(s)


def dataset(cohort, gene):
    """Median split of one gene in one cohort, joined to that cohort's overall survival."""
    expr = pd.read_csv(os.path.join(DATA, 'TCGA_%s_HiSeqV2.tsv' % cohort),
                       sep='\t', index_col=0)
    # not a bare .upper(): the hub of ACC is printed as SHOC1 while Xena labels the row C9orf84,
    # so the lookup has to resolve both the rename and the case. See _gene_symbols.py.
    idx = labels(expr.index)
    key = idx.get(canonical(gene).upper())
    if key is None:
        return None, None
    ser = expr.loc[key]
    agg = {}
    for s, v in ser.items():
        agg.setdefault(sample_to_patient(s), []).append(v)
    agg = {k: float(np.mean(v)) for k, v in agg.items()}
    os_rows = json.load(open(os.path.join(OSDIR, '%s_os.json' % cohort)))
    m = pd.DataFrame([{'patient': r['patient'], 'os_months': float(r['os_months']),
                       'event': event(r['os_status'])} for r in os_rows])
    m['gexpr'] = m['patient'].map(agg)
    m = m.dropna(subset=['gexpr', 'os_months'])
    med = m['gexpr'].median()
    return key, (m[m['gexpr'] >= med], m[m['gexpr'] < med])


def pfmt(p):
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        return '$p = %.1f\\times10^{%d}$' % (p / 10.0 ** e, e)
    return '$p = %.3f$' % p


# ------------------------------------------------------------------ select the panel set
scan = json.load(open(os.path.join(RES, '_km_pancan.json')))
panels = []
for c, v in sorted(scan.items()):
    h = v.get('hub')
    st = v['genes'].get(h) if h else None
    if st and st['logrank_p'] < ALPHA:
        panels.append((c, h, st))
panels.sort(key=lambda t: t[2]['logrank_p'])
print('scanned %d cohorts, %d pass log-rank p < %.2f' % (len(scan), len(panels), ALPHA))
for c, h, st in panels:
    print('   %-5s %-10s p=%.3g HR=%.2f [%.2f-%.2f]  n=%d/%d'
          % (c, h, st['logrank_p'], st['hr'], st['hr_lo'], st['hr_hi'], st['n_low'], st['n_high']))

nrow, ncol = 3, 2
assert len(panels) <= nrow * ncol, 'panel set does not fit the grid'
fig, axes = plt.subplots(nrow, ncol, figsize=(W_MM / 25.4, H_MM / 25.4))
fig.subplots_adjust(left=0.088, right=0.985, top=0.958, bottom=0.078,
                    hspace=0.46, wspace=0.30)

for i, (cohort, hub, st) in enumerate(panels):
    ax = axes[i // ncol][i % ncol]
    # the row label may be a legacy spelling; the panel prints the approved symbol
    shown = canonical(hub)
    _, (hi, lo) = dataset(cohort, hub)
    kmf = KaplanMeierFitter()
    for name, sub, col in (('Low', lo, LOW_COL), ('High', hi, HIGH_COL)):
        kmf.fit(sub['os_months'], sub['event'], label='%s ($n$ = %d)' % (name, len(sub)))
        kmf.plot_survival_function(ax=ax, color=col, lw=1.15,
                                   ci_show=True, ci_alpha=0.16,
                                   show_censors=True,
                                   censor_styles={'marker': '|', 'ms': 3.2, 'mew': 0.65})
    # The low and high groups differ by line style as well as by colour, so the two curves stay
    # separable in greyscale or with a colour-vision deficiency (F&IG accessibility). The censor
    # ticks carry a marker, so they are skipped: only the step curves themselves are dashed.
    for ln in ax.get_lines():
        if ln.get_marker() in ('None', '') and ln.get_color() == HIGH_COL:
            ln.set_linestyle((0, (3.5, 1.6)))
    # middle dot as a plain Arial glyph: $\cdot$ has no Arial symbol and was rendered from
    # Cmsy10, putting a second family into an otherwise single-family figure
    ax.set_title('%s \u00b7 %s' % (cohort, shown), fontsize=6.9, fontweight='bold',
                 color='#1A1A1A', pad=3.0)
    ax.text(-0.02, 1.075, '%s' % 'abcdef'[i], transform=ax.transAxes,
            fontsize=7.4, fontweight='bold', color='#1A1A1A', ha='left', va='bottom')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([0, 50, 100, 150])
    ax.tick_params(length=2.0, pad=1.2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    if i % ncol == 0:
        ax.set_ylabel('Overall survival', fontsize=6.2, labelpad=1.6)
    # lifelines labels the axis "timeline" on every call, so set it (or blank it) afterwards
    ax.set_xlabel('Months' if i // ncol == nrow - 1 else '', fontsize=6.2, labelpad=1.4)

    # legend in the lower left: the empty corner for every survival curve
    ax.legend(fontsize=5.6, frameon=False, loc='lower left',
              handlelength=1.15, handletextpad=0.35, labelspacing=0.28,
              bbox_to_anchor=(-0.012, -0.018))

    # The curves always leave the lower left free, and that is where the legend sits. For the
    # statistics block, walk candidate slots down the right-hand side and stop at the first
    # one that no survival curve passes through, so the numbers never land on a step.
    x0, x1 = ax.get_xlim()
    curves_axes = []
    for sub in (hi, lo):
        kf = KaplanMeierFitter().fit(sub['os_months'], sub['event'])
        tt = np.linspace(x0, x1, 240)
        ss = np.interp(tt, kf.survival_function_.index.values,
                       kf.survival_function_.iloc[:, 0].values, left=1.0)
        curves_axes.append(((tt - x0) / (x1 - x0), ss))

    bw, bh = 0.50, 0.20                      # statistics block, in axes fractions
    best, best_hits = 0.975, None
    for ytop in (0.975, 0.775, 0.575):
        hits = sum(int(np.sum((xs > 1.0 - bw) & (ys > ytop - bh) & (ys < ytop)))
                   for xs, ys in curves_axes)
        if best_hits is None or hits < best_hits:
            best, best_hits = ytop, hits
        if hits == 0:
            break
    ax.text(0.985, best, '%s\nHR = %.2f (%.2f--%.2f)' % (pfmt(st['logrank_p']), st['hr'],
                                                         st['hr_lo'], st['hr_hi']),
            transform=ax.transAxes, ha='right', va='top', fontsize=5.6, linespacing=1.45)
    print('   %-5s stats block at y=%.3f  curve hits=%d' % (cohort, best, best_hits))

for j in range(len(panels), nrow * ncol):
    axes[j // ncol][j % ncol].axis('off')

png = os.path.join(FIG, 'Fig5.png')
pdf = os.path.join(FIG, 'Fig5.pdf')
fig.savefig(png, dpi=400)
fig.savefig(pdf)
plt.close(fig)
print('\nsaved: %s (%.0f KB) / %s (%.0f KB)  |  canvas %.1f x %.1f mm'
      % (png, os.path.getsize(png) / 1024, pdf, os.path.getsize(pdf) / 1024, W_MM, H_MM))
