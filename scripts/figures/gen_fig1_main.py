# -*- coding: utf-8 -*-
"""Figure 1 of the manuscript, four panels (all labels in English).

(a) element-wise median adjacency matrix, aligned against misaligned
(b) edges per cohort against sample size, with the shuffled control
(c) the 14 recurring pairs, coloured by non-malignant axis
(d) left: per-cohort correlation of PC1 with external stromal/immune markers;
    right: driver-gene entry against panel size

Canvas and type follow the convention used by every other figure in this
package: 6.85 in (174 mm) wide, 8 pt base type, panel parts marked by a
lower-case letter.  No title is drawn inside the artwork -- the manuscript
carries the caption (Springer: "Do not include titles or captions within
your illustrations"), and the printed size keeps the declared point size.

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

OUT_PNG = os.path.join(FIG, 'Fig1_main.png')
OUT_PDF = os.path.join(FIG, 'Fig1_main.pdf')
TAU = 0.3

MM = 1.0 / 25.4  # mm -> inch
W = 174.0 * MM   # canvas width, matching every other figure in the package

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

fig = plt.figure(figsize=(W, 174.0 * MM * 0.86))
gs = gridspec.GridSpec(2, 2, hspace=0.78, wspace=0.36,
                       left=0.085, right=0.985, top=0.965, bottom=0.085)


def letter(ax, s, x=-0.22, y=1.16):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')


# ================= (a) =================
axa = fig.add_subplot(gs[0, 0])
axa.axis('off')
letter(axa, 'a', x=-0.26, y=1.20)
sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 0],
                                       height_ratios=[22, 1], hspace=0.34, wspace=0.30)
cbar_ax = fig.add_subplot(sub[1, :])
for k, (M, ttl) in enumerate([(meda, 'aligned panel (this work)'),
                              (medb, 'misaligned panel (original)')]):
    a = fig.add_subplot(sub[0, k])
    im = a.imshow(M, cmap='magma', vmin=0, vmax=0.8, interpolation='nearest')
    a.set_title(ttl, fontsize=7.2, pad=2.5)
    a.set_xticks([]); a.set_yticks([])
    a.set_xlabel('max $|$med$|$ = %.3f\nedges $>\\tau$ = %d'
                 % (M.max(), int(np.sum(M > TAU))), fontsize=7.2, labelpad=1.5)
cb = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
cb.set_label('$|$median $W_{ij}|$ across 33 cohorts', fontsize=7.0, labelpad=2)
cb.ax.tick_params(labelsize=6.5, length=1.8, width=0.4)

# ================= (b) =================
axb = fig.add_subplot(gs[0, 1])
ns = np.array([sh[c]['n'] for c in cs], float)
es = np.array([np.sum(np.abs(np.array(sh[c]['W'], dtype=float)) > TAU) for c in cs], float)
lo = ns < 200
axb.scatter(ns[lo], es[lo], s=16, c='#d62728', marker='o', zorder=4,
            edgecolors='k', linewidths=0.3, label='real data, $n<200$')
axb.scatter(ns[~lo], es[~lo], s=16, c='#1f77b4', marker='o', zorder=4,
            edgecolors='k', linewidths=0.3, label='real data, $n\\geq200$')
pc = neg['per_cohort']
axb.scatter([x['n'] for x in pc], [x['shuffled'] for x in pc], s=12, marker='s',
            facecolors='none', edgecolors='#555555', linewidths=0.8, zorder=3,
            label='shuffled control')
axb.axvline(100, ls=':', c='k', lw=0.8)
axb.axvline(172, ls=(0, (4, 2)), c='#555555', lw=0.8)
axb.set_xscale('log')
axb.set_ylim(-10, max(es.max(), 160) * 1.18)
axb.text(94, axb.get_ylim()[1] * 0.975, '$n=d=100$', fontsize=7, va='top', ha='right')
axb.text(176, axb.get_ylim()[1] * 0.985, 'empirical boundary $n^{*}\\approx172$',
         fontsize=6.2, rotation=90, va='top', ha='left', color='0.35')
sl, ic = np.polyfit(np.log(ns), es, 1)
xx = np.linspace(ns.min(), ns.max(), 120)
axb.plot(xx, sl * np.log(xx) + ic, '--', c='#d62728', lw=1.0, alpha=0.65)
m = ns >= 200
sl2, ic2 = np.polyfit(np.log(ns[m]), es[m], 1)
x2 = np.linspace(205, ns.max(), 60)
axb.plot(x2, sl2 * np.log(x2) + ic2, '--', c='#1f77b4', lw=1.0, alpha=0.65)
axb.set_xlabel('cohort sample size $n$ (log scale)', fontsize=8)
axb.set_ylabel('edges inferred by NOTEARS', fontsize=8)
letter(axb, 'b')
axb.text(0.985, 0.975,
         'real, all 33\n  $r=%.3f$, $p=%.0e$\nreal, $n\\geq200$\n  $r=%.3f$, $p=%.2f$\n'
         'shuffled\n  all 33: $r=%.3f$\n  $n\\geq200$: zero'
         % (full['panel_identifiability_all']['r'], full['panel_identifiability_all']['p'],
            full['panel_identifiability_n200']['r'], full['panel_identifiability_n200']['p'],
            neg['shuffled']['r_all']),
         transform=axb.transAxes, va='top', ha='right', fontsize=6.4, linespacing=1.32,
         bbox=dict(fc='white', ec='0.7', alpha=0.92, pad=1.5, lw=0.4))
axb.legend(fontsize=6.6, loc='lower left', framealpha=0.92, handletextpad=0.4,
           borderpad=0.3, labelspacing=0.25)
axb.grid(alpha=0.22, lw=0.4)

# ================= (c) =================
axc = fig.add_subplot(gs[1, 0])
edges = sorted(full['aligned']['shared_edges'], key=lambda e: -e['n_cohorts'])
ys = np.arange(len(edges))[::-1]
for y, e in zip(ys, edges):
    k, c = axis_of(e['src'])
    axc.barh(y, e['n_cohorts'], color=c, height=0.70, edgecolor='k', linewidth=0.35)
    axc.text(e['n_cohorts'] + 0.6, y, '%s$\\rightarrow$%s' % (e['src'], e['dst']),
             va='center', fontsize=6.8)
axc.axvline(10, ls='--', c='k', lw=1.0)
axc.set_yticks([])
axc.set_xlim(0, 44)
axc.set_ylim(-2.6, len(edges) - 0.3)
axc.set_xticks([0, 10, 20, 30, 40])
axc.set_xlabel('cohorts (of 33) in which the edge recurs', fontsize=8)
letter(axc, 'c')
hs = [plt.Rectangle((0, 0), 1, 1, fc=c) for k, (g, c) in AXIS.items()]
axc.legend(hs, list(AXIS), fontsize=6.2, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, 1.02), frameon=False, handlelength=1.0,
           handletextpad=0.35, columnspacing=1.1, borderpad=0.2, labelspacing=0.28)
axc.text(0.985, 0.55, 'recurrence threshold\n($\\geq$10/33 cohorts)',
         transform=axc.transAxes, fontsize=6.4, ha='right', va='center',
         color='0.25')
axc.text(0.55, 0.02, '14 edges  |  6 non-malignant axes  |  0 canonical drivers',
         transform=axc.transAxes, fontsize=6.8, style='italic', ha='center')
axc.grid(alpha=0.18, axis='x', lw=0.4)

# ================= (d) =================
subd = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.60)
axd1 = fig.add_subplot(subd[0])
r_nm = np.array([x['r_PC1_nonmal'] for x in attr['purity_attribution']['per_cohort']])
axd1.hist(r_nm, bins=np.arange(0.35, 1.00, 0.05), color='#1f77b4', alpha=0.88,
          edgecolor='k', linewidth=0.4)
axd1.axvline(r_nm.mean(), ls='--', c='#d62728', lw=1.2)
axd1.set_xlim(0.35, 0.98)
axd1.set_xticks([0.4, 0.6, 0.8])
axd1.set_xlabel('per-cohort $r$\n(PC1 vs stroma/\nimmune markers)', fontsize=7.2,
                linespacing=1.25)
axd1.set_ylabel('cohorts', fontsize=7.5)
letter(axd1, 'd', x=-0.42, y=1.16)
axd1.text(0.03, 0.975, 'mean $r=%.3f$\n33/33 positive\n$\\lambda_1=%.1f$\n($%.1f\\times$ MP)'
         % (r_nm.mean(), attr['mp']['top_eigenvalues'][0],
            attr['mp']['lam1_over_edge']),
         transform=axd1.transAxes, va='top', fontsize=6.2, linespacing=1.3,
         bbox=dict(fc='white', ec='0.7', alpha=0.92, pad=1.2, lw=0.4))
axd1.grid(alpha=0.22, axis='y', lw=0.4)

axd2 = fig.add_subplot(subd[1])
dcur = attr['driver_threshold_curve']
tk = dcur['thresholds']
dh = [dcur['driver_hits'][str(t)] if str(t) in dcur['driver_hits'] else dcur['driver_hits'][t] for t in tk]
axd2.plot(range(len(tk)), dh, 'o-', c='#2ca02c', ms=3.2, lw=1.2)
for i, v in enumerate(dh):
    axd2.annotate(str(v), (i, v), textcoords='offset points', xytext=(0, 4.5),
                  ha='center', fontsize=6.4)
axd2.set_xticks(range(len(tk)))
axd2.set_xticklabels([str(t) for t in tk], fontsize=6.4)
axd2.set_ylim(-12, max(dh) * 1.32)
axd2.set_yticks([0, 100, 200])
axd2.set_xlabel('MAD top-$k$ panel size', fontsize=7.2)
axd2.set_ylabel('driver-gene hits', fontsize=7.5)
axd2.text(0.05, 0.95, 'drivers enter only\nat $k\\gtrsim1000$', transform=axd2.transAxes,
          fontsize=6.4, ha='left', va='top', color='0.25', linespacing=1.3)
axd2.grid(alpha=0.22, lw=0.4)

# No suptitle: the manuscript carries the caption, and the canvas must stay
# exactly W wide so the declared point sizes survive placement.
fig.savefig(OUT_PNG)
fig.savefig(OUT_PDF)
print('wrote', OUT_PNG)
print('wrote', OUT_PDF)
