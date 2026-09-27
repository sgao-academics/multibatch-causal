# -*- coding: utf-8 -*-
"""W2-K: Figure 1, four panels (publication quality, all labels in English).

(a) median-matrix heatmaps, aligned against misaligned, with the shared row/column band
(b) identifiability scatter with the shuffled control
(c) the 14 recurring edges, coloured by non-malignant axis
(d) left: correlation of PC1 with external stromal and immune markers; right: driver genes
    against panel size

Output: figures/Fig1_main.pdf / .png
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
OUT_PNG = os.path.join(FIG, 'Fig1_main.png')
OUT_PDF = os.path.join(FIG, 'Fig1_main.pdf')
TAU = 0.3

full = json.load(open(os.path.join(RES, '_recurrence_analysis.json'), encoding='utf-8'))
attr = json.load(open(os.path.join(RES, '_axis_attribution.json'), encoding='utf-8'))
neg = json.load(open(os.path.join(RES, '_shuffle_control.json'), encoding='utf-8'))
panel = json.load(open(os.path.join(PANELS, 'panel_A_100genes.json'), encoding='utf-8'))['panel']

AXIS = {
    'IFN-\u03b3 / CXCR3 chemokine': (['CXCL9', 'CXCL10', 'CXCL11'], '#1f77b4'),
    'neutrophil / CXCR2 chemokine': (['CXCL1', 'IL8'], '#ff7f0e'),
    'erythrocyte content': (['HBA1', 'HBB'], '#d62728'),
    'plasma-cell / Ig locus': (['ADAM6', 'IGJ', 'MGC29506'], '#9467bd'),
    'stromal / complement': (['COL10A1', 'COL11A1', 'C7', 'PLCXD3'], '#8c564b'),
    'myeloid / cytokine': (['CCL18', 'CHIT1', 'IL6', 'FOSB'], '#2ca02c'),
}


def axis_of(g):
    for k, (gs_, c) in AXIS.items():
        if g in gs_:
            return k, c
    return 'other', '#7f7f7f'


def load(p):
    d = json.load(open(p, encoding='utf-8'))
    cs = sorted(k for k, v in d.items() if isinstance(v, dict) and 'W' in v)
    return d, cs


def medmat(store, cs):
    M = np.stack([np.abs(np.array(store[c]['W'], dtype=float)) for c in cs], 0)
    m = np.median(M, 0)
    np.fill_diagonal(m, 0)
    return m


sh, cs = load(os.path.join(RES, '_shared_panel_notears.json'))
old, cs2 = load(os.path.join(RES, '_pipeline_notears.json'))
meda = medmat(sh, cs)
medb = medmat(old, [c for c in cs2 if np.array(old[c]['W']).shape[0] == 100])

fig = plt.figure(figsize=(16.5, 12.5))
gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.26,
                       left=0.06, right=0.975, top=0.90, bottom=0.07)

# ================= (a) =================
axa = fig.add_subplot(gs[0, 0])
axa.axis('off')
axa.set_title('(a)  Panel alignment is a precondition', fontsize=13.5,
              fontweight='bold', loc='left', pad=12)
sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 0],
                                       height_ratios=[22, 1], hspace=0.20, wspace=0.26)
cbar_ax = fig.add_subplot(sub[1, :])
for k, (M, ttl) in enumerate([(meda, 'aligned panel (this work)'),
                              (medb, 'misaligned panel (original pipeline)')]):
    a = fig.add_subplot(sub[0, k])
    im = a.imshow(M, cmap='magma', vmin=0, vmax=0.8, interpolation='nearest')
    a.set_title(ttl, fontsize=11, pad=6)
    a.set_xticks([]); a.set_yticks([])
    a.set_xlabel('max $|$median$|$ = %.3f   |   edges $>\\tau$ = %d'
                 % (M.max(), int(np.sum(M > TAU))), fontsize=10)
cb = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
cb.set_label('$|$median $W_{ij}|$ across 33 cohorts', fontsize=9.5)
cb.ax.tick_params(labelsize=8.5)

# ================= (b) =================
axb = fig.add_subplot(gs[0, 1])
ns = np.array([sh[c]['n'] for c in cs], float)
es = np.array([np.sum(np.abs(np.array(sh[c]['W'], dtype=float)) > TAU) for c in cs], float)
lo = ns < 200
axb.scatter(ns[lo], es[lo], s=50, c='#d62728', marker='o', zorder=4,
            edgecolors='k', linewidths=0.4, label='real data, $n<200$')
axb.scatter(ns[~lo], es[~lo], s=50, c='#1f77b4', marker='o', zorder=4,
            edgecolors='k', linewidths=0.4, label='real data, $n\\geq200$')
pc = neg['per_cohort']
axb.scatter([x['n'] for x in pc], [x['shuffled'] for x in pc], s=36, marker='s',
            facecolors='none', edgecolors='#555555', linewidths=1.2, zorder=3,
            label='shuffled control')
axb.axvline(100, ls=':', c='k', lw=1.0)
axb.set_xscale('log')
axb.set_ylim(-10, max(es.max(), 160) * 1.16)
axb.text(104, axb.get_ylim()[1] * 0.97, '$n=d=100$', fontsize=9, va='top')
axb.annotate('empirical control boundary\n$n^{*}\\approx172$ ($\\approx1.7d$)',
             xy=(172, 4), xytext=(215, 78), fontsize=9,
             arrowprops=dict(arrowstyle='->', lw=1.0, color='#555555'))
sl, ic = np.polyfit(np.log(ns), es, 1)
xx = np.linspace(ns.min(), ns.max(), 120)
axb.plot(xx, sl * np.log(xx) + ic, '--', c='#d62728', lw=1.3, alpha=0.65)
m = ns >= 200
sl2, ic2 = np.polyfit(np.log(ns[m]), es[m], 1)
x2 = np.linspace(205, ns.max(), 60)
axb.plot(x2, sl2 * np.log(x2) + ic2, '--', c='#1f77b4', lw=1.3, alpha=0.65)
axb.set_xlabel('cohort sample size $n$ (log scale)', fontsize=11)
axb.set_ylabel('edges inferred by NOTEARS', fontsize=11)
axb.set_title('(b)  Spurious edges are a rank-deficiency artefact', fontsize=13.5,
              fontweight='bold', loc='left', pad=12)
axb.text(0.985, 0.97,
         'real, all 33:  $r=%.3f$, $p=%.1e$\nreal, $n\\geq200$ (18):  $r=%.3f$, $p=%.2f$\n'
         'shuffled, all 33:  $r=%.3f$\nshuffled, $n\\geq200$:  all zero'
         % (full['panel_identifiability_all']['r'], full['panel_identifiability_all']['p'],
            full['panel_identifiability_n200']['r'], full['panel_identifiability_n200']['p'],
            neg['shuffled']['r_all']),
         transform=axb.transAxes, va='top', ha='right', fontsize=9,
         bbox=dict(fc='white', ec='0.7', alpha=0.92))
axb.legend(fontsize=9, loc='lower left', framealpha=0.92)
axb.grid(alpha=0.22)

# ================= (c) =================
axc = fig.add_subplot(gs[1, 0])
edges = sorted(full['aligned']['shared_edges'], key=lambda e: -e['n_cohorts'])
ys = np.arange(len(edges))[::-1]
for y, e in zip(ys, edges):
    k, c = axis_of(e['src'])
    axc.barh(y, e['n_cohorts'], color=c, height=0.70, edgecolor='k', linewidth=0.4)
    axc.text(e['n_cohorts'] + 0.5, y, '%s$\\rightarrow$%s' % (e['src'], e['dst']),
             va='center', fontsize=9.2)
axc.axvline(10, ls='--', c='k', lw=1.1)
axc.set_yticks([])
axc.set_xlim(0, 43)
axc.set_ylim(-1.9, len(edges) - 0.3)
axc.set_xlabel('number of cohorts (of 33) in which the edge recurs', fontsize=11)
axc.set_title('(c)  The reproducible backbone is entirely non-malignant',
              fontsize=13.5, fontweight='bold', loc='left', pad=12)
hs = [plt.Rectangle((0, 0), 1, 1, fc=c) for k, (g, c) in AXIS.items()]
axc.legend(hs, list(AXIS), fontsize=8.4, loc='lower right', framealpha=0.96,
           borderpad=0.5)
axc.text(0.955, 0.48, 'recurrence threshold\n($\\geq$10/33 cohorts)',
         transform=axc.transAxes, fontsize=8.6, ha='right', va='center',
         color='0.25')
axc.text(0.400, 0.022, '14 edges  |  6 non-malignant axes  |  0 canonical drivers',
         transform=axc.transAxes, fontsize=9.8, style='italic', ha='center')

# ================= (d) =================
subd = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.45)
axd1 = fig.add_subplot(subd[0])
r_nm = np.array([x['r_PC1_nonmal'] for x in attr['purity_attribution']['per_cohort']])
axd1.hist(r_nm, bins=np.arange(0.35, 1.00, 0.05), color='#1f77b4', alpha=0.88,
          edgecolor='k', linewidth=0.5)
axd1.axvline(r_nm.mean(), ls='--', c='#d62728', lw=1.5)
axd1.set_xlim(0.35, 0.98)
axd1.set_xlabel('per-cohort $r$\n(panel PC1 vs external\nstroma/immune markers)', fontsize=9.5)
axd1.set_ylabel('number of cohorts', fontsize=10.5)
axd1.set_title('(d)  Dominant axis is microenvironmental content',
               fontsize=13.5, fontweight='bold', loc='left', pad=12)
axd1.text(0.04, 0.97, 'mean $r=%.3f$\n33/33 cohorts positive\n'
                      'MP edge $\\lambda_+=%.2f$\n$\\lambda_1=%.1f$ ($%.1f\\times$)'
          % (r_nm.mean(), attr['mp']['mp_upper'], attr['mp']['top_eigenvalues'][0],
             attr['mp']['lam1_over_edge']),
          transform=axd1.transAxes, va='top', fontsize=8.8,
          bbox=dict(fc='white', ec='0.7', alpha=0.92))
axd1.grid(alpha=0.22, axis='y')

axd2 = fig.add_subplot(subd[1])
dcur = attr['driver_threshold_curve']
tk = dcur['thresholds']
dh = [dcur['driver_hits'][str(t)] if str(t) in dcur['driver_hits'] else dcur['driver_hits'][t] for t in tk]
axd2.plot(range(len(tk)), dh, 'o-', c='#2ca02c', ms=6, lw=1.6)
for i, v in enumerate(dh):
    axd2.annotate(str(v), (i, v), textcoords='offset points', xytext=(0, 7),
                  ha='center', fontsize=8.5)
axd2.set_xticks(range(len(tk)))
axd2.set_xticklabels([str(t) for t in tk], fontsize=8.5)
axd2.set_ylim(-12, max(dh) * 1.25)
axd2.set_xlabel('MAD top-$k$ panel size', fontsize=9.5)
axd2.set_ylabel('driver-gene hits', fontsize=10.5)
axd2.text(0.06, 0.94, 'drivers enter only\nat $k\\gtrsim1000$', transform=axd2.transAxes,
          fontsize=9, ha='left', va='top', color='0.25')
axd2.grid(alpha=0.22)

fig.suptitle('Multi-cohort causal discovery on TCGA: the reproducible structure is microenvironmental, '
             'not tumour-cell-intrinsic', fontsize=15, fontweight='bold', y=0.962)
fig.savefig(OUT_PNG, dpi=220, bbox_inches='tight', facecolor='white')
fig.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
print('wrote', OUT_PNG)
print('wrote', OUT_PDF)
