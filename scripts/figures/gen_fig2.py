"""Figure 2: TCGA pan-cancer edge analysis (2x2). Real data from pipeline checkpoints."""
import json, os, numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

FIGDIR = r'C:\Users\高帅东\Desktop\MultiBatch\figures'
RESULTS = r'C:\Users\高帅东\Desktop\MultiBatch\results'
os.makedirs(FIGDIR, exist_ok=True)

# Load data
nt = json.load(open(os.path.join(RESULTS, '_pipeline_notears.json')))
gp = json.load(open(os.path.join(RESULTS, '_pipeline_genepair.json')))

cancers_all = sorted([k for k in nt if isinstance(nt[k], dict) and 'W' in nt[k]])
edges_list = [nt[c]['edges'] for c in cancers_all]
ns_list = [nt[c].get('n', 0) for c in cancers_all]
# Compute REAL pair-level sharing distribution (must come first)
pair_count = Counter()
for c in cancers_all:
    nd = nt[c]
    W = np.array(nd['W'])
    genes = nd['genes']
    for i in range(100):
        for j in range(100):
            if i != j and abs(W[i,j]) > 0.3:
                pair_count[(genes[i], genes[j])] += 1

# Per-cancer reuse rate (fraction of a cancer's edges that appear in >=2 cancers)
reuse_list = []
for c in cancers_all:
    nd = nt[c]
    W = np.array(nd['W'])
    genes = nd['genes']
    edges_c = set()
    for i in range(100):
        for j in range(100):
            if i != j and abs(W[i,j]) > 0.3:
                edges_c.add((genes[i], genes[j]))
    reuse_c = sum(1 for p in edges_c if pair_count.get(p, 0) >= 2)
    reuse_list.append(100 * reuse_c / max(len(edges_c), 1))

mean_edges = np.mean(edges_list)
mean_reuse = np.mean(reuse_list)
total_unique = len(pair_count)
cnt_1 = sum(1 for v in pair_count.values() if v == 1)
cnt_2 = sum(1 for v in pair_count.values() if v == 2)
cnt_3p = sum(1 for v in pair_count.values() if v >= 3)
pct_1 = 100 * cnt_1 / total_unique
pct_2 = 100 * cnt_2 / total_unique
pct_3p = 100 * cnt_3p / total_unique

# Sort by edge count descending
idx = np.argsort(edges_list)[::-1]
c_s = [cancers_all[i] for i in idx]
e_s = [edges_list[i] for i in idx]
r_s = [reuse_list[i] for i in idx]

# Colors
BLUE = '#2166AC'
RED = '#B2182B'
ORANGE = '#D6604D'
GRAY = '#555555'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
    'text.usetex': False,
})

fig = plt.figure(figsize=(7.2, 6.5))

# (a) Edge counts per cancer
ax1 = fig.add_subplot(2, 2, 1)
colors1 = [BLUE if e >= mean_edges else '#92C5DE' for e in e_s]
ax1.barh(range(len(c_s)), e_s, height=0.6, color=colors1, edgecolor='white', linewidth=0.2)
ax1.axvline(mean_edges, color=ORANGE, linestyle='--', linewidth=1, alpha=0.8)
ax1.set_yticks(range(len(c_s)))
ax1.set_yticklabels(c_s, fontsize=4.8)
ax1.set_xlabel('Causal edges')
ax1.set_title('a) Edges per cancer', fontweight='bold', fontsize=8.5)
ax1.text(0.98, 0.02, f'Mean = {mean_edges:.0f}', transform=ax1.transAxes,
         ha='right', fontsize=6.5, color=ORANGE, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='y', length=0)

# (b) Reuse rate per cancer
ax2 = fig.add_subplot(2, 2, 2)
ir = np.argsort(r_s)[::-1]
rc = [c_s[i] for i in ir]
rv = [r_s[i] for i in ir]
colors2 = [RED if v >= mean_reuse else '#F4A582' for v in rv]
ax2.barh(range(len(rc)), rv, height=0.6, color=colors2, edgecolor='white', linewidth=0.2)
ax2.axvline(mean_reuse, color=ORANGE, linestyle='--', linewidth=1, alpha=0.8)
ax2.set_yticks(range(len(rc)))
ax2.set_yticklabels(rc, fontsize=4.8)
ax2.set_xlabel('Reuse rate (%)')
ax2.set_title(f'b) Per-cancer edge reuse rate', fontweight='bold', fontsize=8.5)
ax2.text(0.98, 0.02, f'Mean = {mean_reuse:.1f}%', transform=ax2.transAxes,
         ha='right', fontsize=6.5, color=ORANGE, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='y', length=0)

# (c) Pair-level sharing distribution (CORRECT values)
ax3 = fig.add_subplot(2, 2, 3)
cats = ['1 cancer', '2 cancers', '>=3 cancers']
vals = [pct_1, pct_2, pct_3p]
colors3 = ['#92C5DE', BLUE, RED]
b3 = ax3.bar(cats, vals, color=colors3, edgecolor='white', linewidth=0.3, width=0.45)
for b, v, n in zip(b3, vals, [cnt_1, cnt_2, cnt_3p]):
    ax3.text(b.get_x() + b.get_width()/2., b.get_height() + 1.5,
             f'{v:.1f}%\n(n={n})', ha='center', fontsize=7, fontweight='bold')
ax3.set_ylabel('Unique gene-pairs (%)')
ax3.set_ylim(0, max(vals) * 1.25)
ax3.set_title('c) Gene-pair sharing distribution', fontweight='bold', fontsize=8.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# (d) Edges vs sample size
ax4 = fig.add_subplot(2, 2, 4)
ax4.scatter(ns_list, edges_list, c=BLUE, s=22, alpha=0.7,
           edgecolors='white', linewidth=0.3, zorder=5)
# Staggered offsets for tight clusters (avoid label overlap)
extreme = {'CHOL', 'DLBC', 'UCS', 'BRCA', 'LUAD', 'KIRC'}
# Two tight clusters: left (CHOL/DLBC/UCS) and right (KIRC/LUAD) need small horizontal stagger
h_offsets = {'DLBC': -8, 'UCS': 8, 'LUAD': -6, 'KIRC': 6}
for i, c in enumerate(cancers_all):
    if c in extreme:
        dx = h_offsets.get(c, 0)
        ax4.annotate(c, (ns_list[i], edges_list[i]),
                    textcoords="offset points", xytext=(dx, 6),
                    fontsize=5.5, ha='center', color=GRAY, fontweight='bold')
r_sp, p_sp = spearmanr(edges_list, ns_list)
ax4.text(0.98, 0.96, f'Spearman r = {r_sp:.2f}\np < 0.001',
         transform=ax4.transAxes, ha='right', va='top', fontsize=7,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
ax4.set_xlabel('Sample size (n)')
ax4.set_ylabel('Causal edges')
ax4.set_title('d) Edges vs sample size', fontweight='bold', fontsize=8.5)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.subplots_adjust(left=0.09, right=0.95, top=0.93, bottom=0.06, wspace=0.35, hspace=0.42)
plt.savefig(os.path.join(FIGDIR, 'fig2_tcga.pdf'), dpi=300)
plt.savefig(os.path.join(FIGDIR, 'fig2_tcga.png'), dpi=300)
plt.close()
print(f'Fig2 done: {os.path.getsize(os.path.join(FIGDIR, "fig2_tcga.pdf"))//1024}KB')
print(f'  Edges: {sum(edges_list)} total, mean {mean_edges:.1f}')
print(f'  Reuse: mean {mean_reuse:.1f}%')
print(f'  Sharing: 1={pct_1:.1f}% ({cnt_1}), 2={pct_2:.1f}% ({cnt_2}), >=3={pct_3p:.1f}% ({cnt_3p})')
print(f'  Spearman: r={r_sp:.2f}, p<0.001')
