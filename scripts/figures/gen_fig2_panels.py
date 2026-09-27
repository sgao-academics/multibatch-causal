# -*- coding: utf-8 -*-
"""W2-N: Figure S1 -- the two panels side by side (zero gene overlap, same axes).

(a) recurring edges of both panels as paired bars, coloured by axis
(b) n against edges reproduced on both panels, with the n >= 200 subset of each
(c) driver genes against panel size (same content as the right half of Figure 1d)

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

E = RES  # scratch alias
OUT_PNG = os.path.join(FIG, 'Fig2_panels.png')
OUT_PDF = os.path.join(FIG, 'Fig2_panels.pdf')
TAU = 0.3

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


fig = plt.figure(figsize=(15.5, 5.6))
gs = gridspec.GridSpec(1, 3, wspace=0.34, left=0.055, right=0.985, top=0.86, bottom=0.15)

# ---- (a) driven by two stacked sub-panels, done as one barh with a gap ----
axa = fig.add_subplot(gs[0, 0])
rows = []
rows.append(('GAP', None, None))
for e in sorted(p2['panel1']['shared_edges'], key=lambda x: -x['n']):
    rows.append((f"{e['src']}\u2192{e['dst']}", e['n'], cls1(e['src'], e['dst'])))
rows.append(('GAP', None, None))
for e in sorted(p2['panel2']['shared_edges'], key=lambda x: -x['n']):
    rows.append((f"{e['src']}\u2192{e['dst']}", e['n'], cls2(e['src'], e['dst'])))

ylab, ypos = [], []
y = 0.0
for lab, val, c in rows:
    if val is None:
        y -= 1.5
        continue
    axa.barh(y, val, color=C[c], height=0.72, edgecolor='k', linewidth=0.4)
    axa.text(val + 0.4, y, lab, va='center', fontsize=8.8)
    ylab.append(lab)
    ypos.append(y)
    y -= 1.0
axa.axvline(10, ls='--', c='k', lw=1.0)
axa.set_yticks([])
axa.set_xlim(0, 34)
axa.set_ylim(min(ypos) - 2.4, max(ypos) + 3.2)
axa.set_xlabel('cohorts (of 33) in which the edge recurs', fontsize=10)
axa.set_title('(a)  Two gene-disjoint panels,\nthe same kind of axis', fontsize=12,
              fontweight='bold', loc='left')
n1 = len(p2['panel1']['shared_edges'])
mid1 = float(np.mean(ypos[:n1]))
mid2 = float(np.mean(ypos[n1:]))
axa.text(33.2, mid1, 'Panel 1\n(14 edges)', fontsize=8.6, ha='right', va='center',
         bbox=dict(fc='#eef3fb', ec='#1f77b4', alpha=0.95))
axa.text(33.2, mid2, 'Panel 2\n(6 edges)', fontsize=8.6, ha='right', va='center',
         bbox=dict(fc='#eefaf1', ec='#2ca02c', alpha=0.95))
axa.text(0.02, 0.982, 'gene overlap of the two panels: 0', transform=axa.transAxes,
         fontsize=9, style='italic', va='top')

# ---- (b) n vs edges for both panels ----
axb = fig.add_subplot(gs[0, 1])
rowsB = []
for tag, path, col, mk in [('panel 1', os.path.join(RES, '_shared_panel_notears.json'), '#1f77b4', 'o'),
                           ('panel 2', os.path.join(RES, '_panel2_notears.json'), '#2ca02c', '^')]:
    d = json.load(open(path, encoding='utf-8'))
    cs = sorted(k for k, v in d.items() if isinstance(v, dict) and 'W' in v)
    ns = np.array([d[c]['n'] for c in cs], float)
    es = np.array([np.sum(np.abs(np.array(d[c]['W'], dtype=float)) > TAU) for c in cs], float)
    axb.scatter(ns, es, s=46, c=col, marker=mk, edgecolors='k', linewidths=0.4,
                zorder=4, label='%s: $r=%.3f$' % (tag, p2['panel1' if tag == 'panel 1' else 'panel2']['r_all']))
    m = ns >= 200
    r_hi = p2['panel1' if tag == 'panel 1' else 'panel2']['r_n200']
    axb.scatter(ns[m], es[m], s=95, facecolors='none', edgecolors=col, linewidths=1.5, zorder=5)
    rowsB.append((tag, r_hi))
axb.axvline(100, ls=':', c='k', lw=1.0)
axb.set_xscale('log')
axb.set_ylim(-10, 240)
axb.set_xlabel('cohort sample size $n$ (log scale)', fontsize=10)
axb.set_ylabel('edges inferred by NOTEARS', fontsize=10)
axb.set_title('(b)  Sample-size dependence\nreproduces across panels', fontsize=12,
              fontweight='bold', loc='left')
axb.text(0.98, 0.97,
         'all 33 cohorts:\n  panel 1  $r=%.3f$\n  panel 2  $r=%.3f$\n'
         '$n\\geq200$ (open circles):\n  panel 1  $r=%.3f$ ($p=0.41$)\n  panel 2  $r=%+.3f$ ($p=0.88$)'
         % (p2['panel1']['r_all'], p2['panel2']['r_all'],
            p2['panel1']['r_n200'], p2['panel2']['r_n200']),
         transform=axb.transAxes, va='top', ha='right', fontsize=8.5,
         bbox=dict(fc='white', ec='0.7', alpha=0.93))
axb.legend(fontsize=9, loc='lower left', framealpha=0.93)
axb.grid(alpha=0.22)

# ---- (c) driver curve, restated with the two-panel context ----
axc = fig.add_subplot(gs[0, 2])
dcur = attr['driver_threshold_curve']
tk = dcur['thresholds']
dh = [dcur['driver_hits'][str(t)] if str(t) in dcur['driver_hits'] else dcur['driver_hits'][t] for t in tk]
axc.plot(range(len(tk)), dh, 'o-', c='#2ca02c', ms=6, lw=1.6)
for i, v in enumerate(dh):
    axc.annotate(str(v), (i, v), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8.5)
axc.axvspan(-0.3, 0.3, color='#d62728', alpha=0.13)
axc.text(0.0, 235, 'both panels\nuse $k=100$', fontsize=8.5, ha='left', color='#8c1a1a')
axc.set_xticks(range(len(tk)))
axc.set_xticklabels([str(t) for t in tk], fontsize=8.5)
axc.set_ylim(-12, 265)
axc.set_xlabel('MAD top-$k$ panel size', fontsize=10)
axc.set_ylabel('driver-gene hits (33 cohorts)', fontsize=10)
axc.set_title('(c)  Both panels sit where\ndrivers are absent', fontsize=12,
              fontweight='bold', loc='left')
axc.grid(alpha=0.22)

fig.suptitle('Supplementary: robustness of the finding to a gene-disjoint panel',
             fontsize=13.5, fontweight='bold', y=0.985)
fig.savefig(OUT_PNG, dpi=220, bbox_inches='tight', facecolor='white')
fig.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
print('wrote', OUT_PNG)
print('wrote', OUT_PDF)
