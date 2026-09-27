# -*- coding: utf-8 -*-
"""Figure 2 of the manuscript -- robustness to a gene-disjoint second panel.

(a) recurring pairs of both panels as paired bars, coloured by axis
(b) sample size against edge count for both panels, with the n >= 200 subset
(c) driver genes against panel size, restated with the two-panel context

Canvas and type follow the convention used by every other figure in this
package: 6.85 in (174 mm) wide, 8 pt base type, panel parts marked by a
lower-case letter.  The three panels are laid out as one tall left column (the
twenty-row bar panel needs the height) and two stacked panels on the right.
No title is drawn inside the artwork -- the manuscript carries the caption.

Output: figures/Fig2_panels.pdf / .png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA = os.path.join(_ROOT, 'data')
RES = os.path.join(_ROOT, 'results')
PANELS = os.path.join(DATA, 'panels')
FIG = os.path.join(_ROOT, 'figures')

OUT_PNG = os.path.join(FIG, 'Fig2_panels.png')
OUT_PDF = os.path.join(FIG, 'Fig2_panels.pdf')
TAU = 0.3

MM = 1.0 / 25.4
W = 174.0 * MM

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'axes.linewidth': 0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'figure.dpi': 400, 'savefig.dpi': 400, 'text.usetex': False,
})

p2 = json.load(open(os.path.join(RES, '_panel_B_robustness.json'), encoding='utf-8'))
attr = json.load(open(os.path.join(RES, '_axis_attribution.json'), encoding='utf-8'))

C = {'immune/lymphoid': '#1f77b4', 'chemokine': '#ff7f0e', 'erythrocyte': '#d62728',
     'stromal/complement': '#8c564b', 'IEG/activation': '#9467bd',
     'myeloid/DC': '#2ca02c', 'other': '#7f7f7f'}


def cls1(g, e):
    s = {g, e}
    if s & {'CXCL9', 'CXCL10', 'CXCL11', 'CXCL1', 'IL8'}:
        return 'chemokine'
    if s & {'HBA1', 'HBB'}:
        return 'erythrocyte'
    if s & {'COL10A1', 'COL11A1', 'C7', 'PLCXD3'}:
        return 'stromal/complement'
    if s & {'CCL18', 'CHIT1', 'IL6', 'FOSB'}:
        return 'myeloid/DC'
    return 'immune/lymphoid'


def cls2(g, e):
    s = {g, e}
    if s & {'NR4A3', 'EGR3'}:
        return 'IEG/activation'
    if s & {'CD1A'}:
        return 'myeloid/DC'
    return 'immune/lymphoid'


fig = plt.figure(figsize=(W, 174.0 * MM * 0.82))
outer = gridspec.GridSpec(1, 2, width_ratios=[1.18, 1.0], wspace=0.46,
                          left=0.075, right=0.985, top=0.945, bottom=0.105)


def letter(ax, s, x=-0.20, y=1.12):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')


# ---- (a) two stacked blocks in one barh, separated by a gap ----
axa = fig.add_subplot(outer[0])
rows = [('GAP', None, None)]
for e in sorted(p2['panel1']['shared_edges'], key=lambda x: -x['n']):
    rows.append((f"{e['src']}\u2192{e['dst']}", e['n'], cls1(e['src'], e['dst'])))
rows.append(('GAP', None, None))
for e in sorted(p2['panel2']['shared_edges'], key=lambda x: -x['n']):
    rows.append((f"{e['src']}\u2192{e['dst']}", e['n'], cls2(e['src'], e['dst'])))

ylab, ypos = [], []
y = 0.0
for lab, val, c in rows:
    if val is None:
        y -= 1.6
        continue
    axa.barh(y, val, color=C[c], height=0.70, edgecolor='k', linewidth=0.35)
    axa.text(val + 0.5, y, lab, va='center', fontsize=7.0)
    ylab.append(lab)
    ypos.append(y)
    y -= 1.0
axa.axvline(10, ls='--', c='k', lw=1.0)
axa.set_yticks([])
axa.set_xlim(0, 36)
axa.set_ylim(min(ypos) - 2.2, max(ypos) + 2.0)
axa.set_xticks([0, 10, 20, 30])
axa.set_xlabel('cohorts (of 33) in which the edge recurs', fontsize=8)
letter(axa, 'a', x=-0.14, y=1.06)
n1 = len(p2['panel1']['shared_edges'])
mid1 = float(np.mean(ypos[:n1]))
mid2 = float(np.mean(ypos[n1:]))
axa.text(35.4, mid1, 'Panel 1\n(14 edges)', fontsize=6.6, ha='right', va='center',
         linespacing=1.25, bbox=dict(fc='#eef3fb', ec='#1f77b4', alpha=0.95,
                                     pad=1.4, lw=0.5))
axa.text(35.4, mid2, 'Panel 2\n(6 edges)', fontsize=6.6, ha='right', va='center',
         linespacing=1.25, bbox=dict(fc='#eefaf1', ec='#2ca02c', alpha=0.95,
                                     pad=1.4, lw=0.5))
axa.text(0.970, 0.995, 'gene overlap of the two panels: 0',
         transform=axa.transAxes, fontsize=6.6, style='italic', va='top', ha='right')
axa.grid(alpha=0.18, axis='x', lw=0.4)

right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.85)

# ---- (b) n against edges for both panels ----
axb = fig.add_subplot(right[0])
for tag, path, col, mk in [('panel 1', os.path.join(RES, '_shared_panel_notears.json'), '#1f77b4', 'o'),
                           ('panel 2', os.path.join(RES, '_panel2_notears.json'), '#2ca02c', '^')]:
    d = json.load(open(path, encoding='utf-8'))
    cs = sorted(k for k, v in d.items() if isinstance(v, dict) and 'W' in v)
    ns = np.array([d[c]['n'] for c in cs], float)
    es = np.array([np.sum(np.abs(np.array(d[c]['W'], dtype=float)) > TAU) for c in cs], float)
    axb.scatter(ns, es, s=16, c=col, marker=mk, edgecolors='k', linewidths=0.3, zorder=4,
                label='%s: $r=%.3f$' % (tag, p2['panel1' if tag == 'panel 1' else 'panel2']['r_all']))
    m = ns >= 200
    axb.scatter(ns[m], es[m], s=34, facecolors='none', edgecolors=col, linewidths=1.1, zorder=5)
axb.axvline(100, ls=':', c='k', lw=0.8)
axb.set_xscale('log')
axb.set_ylim(-52, 250)
axb.set_yticks([0, 50, 100, 150, 200, 250])
axb.set_xlabel('cohort sample size $n$ (log scale)', fontsize=8)
axb.set_ylabel('edges by NOTEARS', fontsize=8)
letter(axb, 'b', x=-0.30, y=1.18)
axb.text(0.985, 0.97,
         'all 33\n  $r=%.3f$ / $%.3f$\n$n\\geq200$ (open)\n  $r=%.3f$ / $%+.3f$'
         % (p2['panel1']['r_all'], p2['panel2']['r_all'],
            p2['panel1']['r_n200'], p2['panel2']['r_n200']),
         transform=axb.transAxes, va='top', ha='right', fontsize=6.4, linespacing=1.3,
         bbox=dict(fc='white', ec='0.7', alpha=0.92, pad=1.4, lw=0.4))
axb.legend(fontsize=6.6, loc='lower left', framealpha=0.93, handletextpad=0.4,
           borderpad=0.3, labelspacing=0.25)
axb.grid(alpha=0.22, lw=0.4)

# ---- (c) driver curve, restated with the two-panel context ----
axc = fig.add_subplot(right[1])
dcur = attr['driver_threshold_curve']
tk = dcur['thresholds']
dh = [dcur['driver_hits'][str(t)] if str(t) in dcur['driver_hits'] else dcur['driver_hits'][t] for t in tk]
axc.plot(range(len(tk)), dh, 'o-', c='#2ca02c', ms=3.2, lw=1.2)
for i, v in enumerate(dh):
    axc.annotate(str(v), (i, v), textcoords='offset points', xytext=(0, 4.5),
                 ha='center', fontsize=6.4)
axc.axvspan(-0.3, 0.3, color='#d62728', alpha=0.13)
axc.text(0.35, 200, 'both panels\nuse $k=100$', fontsize=6.4, ha='left',
         linespacing=1.25, color='#8c1a1a')
axc.set_xticks(range(len(tk)))
axc.set_xticklabels([str(t) for t in tk], fontsize=6.4, rotation=30, ha='right')
axc.set_xlim(-0.6, len(tk) - 0.4)
axc.set_ylim(-12, 285)
axc.set_yticks([0, 100, 200])
axc.set_xlabel('MAD top-$k$ panel size', fontsize=8)
axc.set_ylabel('driver-gene hits', fontsize=8)
letter(axc, 'c', x=-0.30, y=1.18)
axc.grid(alpha=0.22, lw=0.4)

fig.savefig(OUT_PNG)
fig.savefig(OUT_PDF)
print('wrote', OUT_PNG)
print('wrote', OUT_PDF)
