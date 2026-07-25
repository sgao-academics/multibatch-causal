"""Figure 4: Parameter sensitivity and edge statistics (2x2)."""
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
d = json.load(open(os.path.join(RESULTS, '_figure_data.json')))
sens = d['sensitivity']
lam = sens['lam']
tau = sens['tau']

# Recompute reuse rates from raw data
nt = json.load(open(os.path.join(RESULTS, '_pipeline_notears.json')))
cancers_raw = sorted([k for k in nt if isinstance(nt[k], dict) and 'W' in nt[k]])
pair_count = Counter()
edges_list_raw = []
for c in cancers_raw:
    nd = nt[c]
    W = np.array(nd['W'])
    genes = nd['genes']
    edges_list_raw.append(nd['edges'])
    for i in range(100):
        for j in range(100):
            if i != j and abs(W[i,j]) > 0.3:
                pair_count[(genes[i], genes[j])] += 1
reuse_all = []
for c in cancers_raw:
    nd = nt[c]
    W = np.array(nd['W'])
    genes = nd['genes']
    edges_c = set()
    for i in range(100):
        for j in range(100):
            if i != j and abs(W[i,j]) > 0.3:
                edges_c.add((genes[i], genes[j]))
    reuse_c = sum(1 for p in edges_c if pair_count.get(p, 0) >= 2)
    reuse_all.append(100 * reuse_c / max(len(edges_c), 1))

# Colorblind-safe palette
BLUE = '#2166AC'
ORANGE = '#D6604D'
PURPLE = '#762A83'
GRAY = '#555555'
CANCER_COLORS = [BLUE, ORANGE, PURPLE]

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
    'text.usetex': False,
})

fig = plt.figure(figsize=(7.2, 6.2))
cancers_3 = ['BRCA', 'LUAD', 'COAD']

# (a) Lambda sensitivity
ax1 = fig.add_subplot(2, 2, 1)
lam_x = [0.001, 0.005, 0.01, 0.02]
for ci, c in enumerate(cancers_3):
    ax1.plot(lam_x, lam[c], 'o-', color=CANCER_COLORS[ci], label=c,
             markersize=5, linewidth=1.2)
ax1.axvline(0.01, color=ORANGE, linestyle='--', linewidth=0.8, alpha=0.5)
ax1.set_xlabel('lambda_1')
ax1.set_ylabel('Causal edges')
ax1.set_title('a) L1 regularization sensitivity', fontweight='bold', fontsize=8.5)
ax1.set_xscale('log')
ax1.legend(fontsize=6.5, frameon=False, loc='upper right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, alpha=0.25, linewidth=0.3)

# (b) Tau sensitivity
ax2 = fig.add_subplot(2, 2, 2)
tau_x = [0.1, 0.2, 0.3, 0.4, 0.5]
for ci, c in enumerate(cancers_3):
    ax2.plot(tau_x, tau[c], 's-', color=CANCER_COLORS[ci], label=c,
             markersize=5, linewidth=1.2)
ax2.axvline(0.3, color=ORANGE, linestyle='--', linewidth=0.8, alpha=0.5)
ax2.set_xlabel('tau (edge threshold)')
ax2.set_ylabel('Causal edges')
ax2.set_title('b) Edge threshold sensitivity', fontweight='bold', fontsize=8.5)
ax2.legend(fontsize=6.5, frameon=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.25, linewidth=0.3)

# (c) Edge count distribution
ax3 = fig.add_subplot(2, 2, 3)
e = d['edge_counts']
ax3.boxplot(e, vert=True, patch_artist=True, widths=0.35,
            boxprops=dict(facecolor='#92C5DE', alpha=0.8, linewidth=0.6),
            medianprops=dict(color=BLUE, linewidth=1.2),
            whiskerprops=dict(linewidth=0.6),
            capprops=dict(linewidth=0.6),
            flierprops=dict(marker='o', markersize=3, markerfacecolor=PURPLE))
mean_e = np.mean(e)
sd_e = np.std(e, ddof=1)
ax3.set_xticklabels(['33 cancers'])
ax3.set_ylabel('Causal edges')
ax3.set_title(f'c) Edge distribution (mean={mean_e:.0f}, SD={sd_e:.0f})',
             fontweight='bold', fontsize=8.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# (d) Edge count vs reuse rate (NOT in Fig 2)
ax4 = fig.add_subplot(2, 2, 4)
ax4.scatter(edges_list_raw, reuse_all, c=BLUE, s=22, alpha=0.7,
           edgecolors='white', linewidth=0.3, zorder=5)
r_re, p_re = spearmanr(edges_list_raw, reuse_all)
ax4.text(0.98, 0.96, f'Spearman r = {r_re:.2f}\np = {p_re:.3f}',
         transform=ax4.transAxes, ha='right', va='top', fontsize=7,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
ax4.set_xlabel('Edge count')
ax4.set_ylabel('Reuse rate (%)')
ax4.set_title('d) Edge count vs reuse rate', fontweight='bold', fontsize=8.5)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06,
                    wspace=0.35, hspace=0.42)
plt.savefig(os.path.join(FIGDIR, 'fig4_sensitivity.pdf'), dpi=300)
plt.savefig(os.path.join(FIGDIR, 'fig4_sensitivity.png'), dpi=300)
plt.close()
print(f'Fig4 done: {os.path.getsize(os.path.join(FIGDIR, "fig4_sensitivity.pdf"))//1024}KB')
print(f'  Spearman(edges, reuse): r={r_re:.2f}, p={p_re:.3f}')
