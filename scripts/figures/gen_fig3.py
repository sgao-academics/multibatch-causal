"""Figure 3: baseline comparisons.

Three panels, one for each check the section runs:

  a  the cross-cancer sharing rate of the two estimators, under both counting
     conventions, so the comparison between methods is made on like for like
  b  the overlap between the two edge sets: how much they agree on and how
     much each keeps to itself
  c  the pooled fit against the per-cancer panel

Every number is read from results/_baseline_stats.json, which
scripts/figures/gen_baseline_stats.py writes from the stored checkpoints, so
the panels cannot drift away from the text.

Designed at 174 mm, the full-width figure area of the printed page, and
embedded at \\textwidth in the manuscript, so the font sizes set here are the
sizes that appear in print.

A per-cancer edge-count panel used to sit in this figure. It plotted the same
series twice: GENIE3 was truncated to the per-cancer NOTEARS edge count, so
both bar groups were identical by construction (33 of 33 cohorts) and the
panel carried no information beyond the fact that the truncation was applied.
It has been removed rather than redrawn.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, 'results')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

with open(os.path.join(RESULTS, '_baseline_stats.json'), encoding='utf-8') as fh:
    S = json.load(fh)

NT, G3, OV, PL = S['notears'], S['genie3'], S['overlap'], S['pooled']

BLUE, BLUE_L = '#2166AC', '#67A9CF'
RED, RED_L = '#B2182B', '#EF8A62'
PURPLE, ORANGE = '#762A83', '#D6604D'

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,      # embed TrueType, not Type 3
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    # Arial for the maths text as well: the default mathtext set is DejaVu Sans, and its glyphs
    # have no business being in an otherwise Arial figure
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'hatch.linewidth': 0.6,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.5, 'text.usetex': False,
})

fig = plt.figure(figsize=(6.85, 2.7))


def tidy(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=2.5, width=0.5)


def comma(n):
    return '{:,}'.format(int(n))


# ---------------------------------------------------------------- a) sharing
ax1 = fig.add_subplot(1, 3, 1)

groups = [('NOTEAR\n', 0.0, 1.0, BLUE, BLUE_L, NT), ('GENIE3', 2.3, 3.3, RED, RED_L, G3)]
legend_handles = []
for label, xd, xu, cd, cu, stat in groups:
    pct_d = stat['directed']['pct_ge3']
    pct_u = stat['undirected']['pct_ge3']
    # hatch as well as colour, so the two counting conventions stay apart for colour-blind readers
    hd = ax1.bar(xd, pct_d, 0.8, color=cd, edgecolor='white', linewidth=0.4, hatch='///')
    hu = ax1.bar(xu, pct_u, 0.8, color=cu, edgecolor='white', linewidth=0.4, hatch='...')
    a1 = ax1.text(xd, pct_d + 0.07, '%.1f' % pct_d, ha='center', va='bottom',
                  fontsize=7, fontweight='bold', color=cd)
    a2 = ax1.text(xu, pct_u + 0.07, '%.1f' % pct_u, ha='center', va='bottom',
                  fontsize=7, fontweight='bold', color=cd)
    a1.set_clip_on(True)
    a2.set_clip_on(True)
    legend_handles.append(hd)
    legend_handles.append(hu)

ax1.set_xlim(-0.75, 4.05)
ax1.set_ylim(0, 3.15)
ax1.set_yticks([0, 1, 2, 3])
ax1.set_xticks([0.5, 2.8])
ax1.set_xticklabels(['NOTEARS', 'GENIE3'], fontsize=7.5)
ax1.set_ylabel('Pairs in $\\geq$3 cancers (%)')
ax1.set_title('a)', loc='left', fontweight='bold', fontsize=9, pad=3)
ax1.legend([legend_handles[0], legend_handles[1]],
           ['directed', 'undirected'],
           fontsize=7, loc='upper center', frameon=False,
           handlelength=1.1, handleheight=0.8, borderpad=0.1, labelspacing=0.25)
tidy(ax1)

# ----------------------------------------------------------------- b) overlap
ax2 = fig.add_subplot(1, 3, 2)

ov_u = OV['undirected']
rows = [('NOTEARS', NT['undirected']['unique_pairs'], BLUE),
        ('GENIE3', G3['undirected']['unique_pairs'], RED)]
shared = ov_u['shared_pairs']
for i, (label, total, colour) in enumerate(rows):
    private = total - shared
    ax2.barh(i, shared, 0.42, color=PURPLE, edgecolor='white', linewidth=0.4, hatch='///')
    ax2.barh(i, private, 0.42, left=shared, color=colour, edgecolor='white', linewidth=0.4,
             hatch='...')
    ax2.text(shared / 2, i, comma(shared), ha='center', va='center',
             fontsize=7, color='white', fontweight='bold')
    ax2.text(shared + private / 2, i, comma(private), ha='center', va='center',
             fontsize=7, color='white', fontweight='bold')

ax2.set_yticks([0, 1])
ax2.set_yticklabels([rows[0][0], rows[1][0]], fontsize=7.5)
ax2.set_ylim(-0.6, 1.75)
ax2.set_xlim(0, max(r[1] for r in rows) * 1.06)
ax2.set_xlabel('Gene-pairs')
ax2.set_title('b)', loc='left', fontweight='bold', fontsize=9, pad=3)
ax2.legend([plt.Rectangle((0, 0), 1, 1, color=PURPLE),
            plt.Rectangle((0, 0), 1, 1, color=BLUE),
            plt.Rectangle((0, 0), 1, 1, color=RED)],
           ['shared', 'NOTEARS only', 'GENIE3 only'],
           fontsize=7, loc='upper right', frameon=False, ncol=1,
           handlelength=1.1, handleheight=0.8, borderpad=0.1, labelspacing=0.25)
tidy(ax2)

# ------------------------------------------------------------------ c) pooled
ax3 = fig.add_subplot(1, 3, 3)

tot, pooled = NT['total_edges'], PL['edges']
fold = PL['fold_vs_per_cancer_total']
ax3.bar([0, 1], [tot, pooled], 0.5, color=[BLUE, ORANGE],
        edgecolor='white', linewidth=0.4, hatch=['///', '\\\\\\'])
ax3.set_yscale('log')
ax3.set_ylim(10, 10000)
ax3.set_yticks([10, 100, 1000])
ax3.set_yticklabels(['10', '100', '1,000'])
ax3.set_xlim(-0.6, 1.6)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Per-cancer\n(all cohorts)', 'Pooled'], fontsize=7.5)
ax3.set_ylabel('Edges (log scale)')
ax3.set_title('c)', loc='left', fontweight='bold', fontsize=9, pad=3)
ax3.text(0, tot * 1.25, comma(tot), ha='center', va='bottom', fontsize=7,
         fontweight='bold', color=BLUE)
ax3.text(1, pooled * 1.25, comma(pooled), ha='center', va='bottom', fontsize=7,
         fontweight='bold', color=ORANGE)
ax3.annotate('', xy=(1, pooled * 2.1), xytext=(1, tot * 0.62),
             arrowprops=dict(arrowstyle='<->', color='#555555', linewidth=0.6))
ax3.text(1.07, np.sqrt(tot * pooled) * 1.05, '%.0f$\\times$' % round(fold),
         ha='left', va='center', fontsize=7, color='#555555')
tidy(ax3)

plt.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.20, wspace=0.42)

for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIGDIR, 'Fig3.%s' % ext), dpi=300)
plt.close(fig)

print('Fig3: sharing NOTEARS %.1f/%.1f%% GENIE3 %.1f/%.1f%% | overlap undirected %d | pooled %d vs %d (%.0fx)'
      % (NT['directed']['pct_ge3'], NT['undirected']['pct_ge3'],
         G3['directed']['pct_ge3'], G3['undirected']['pct_ge3'],
         shared, pooled, tot, round(fold)))
print('Fig3 done: %d KB' % (os.path.getsize(os.path.join(FIGDIR, 'Fig3.pdf')) // 1024))
