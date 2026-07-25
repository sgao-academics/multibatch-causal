"""Figure 3: Baseline comparison and external validation (2x2)."""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGDIR = r'C:\Users\高帅东\Desktop\MultiBatch\figures'
RESULTS = r'C:\Users\高帅东\Desktop\MultiBatch\results'
os.makedirs(FIGDIR, exist_ok=True)

d = json.load(open(os.path.join(RESULTS, '_figure_data.json')))
cancers = d['cancers']
nt_edges = d['edge_counts']
g3_edges = d['genie3_edges']
g3o = d['genie3_overlap']
gp = d['genepair']
total = d['total_edges']
pooled_edges = d['pooled']['edges']

BLUE, RED, PURPLE, ORANGE, GRAY = '#2166AC', '#B2182B', '#762A83', '#D6604D', '#555555'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.5, 'text.usetex': False,
})

fig = plt.figure(figsize=(7.2, 6.2))

# (a) NOTEARS vs GENIE3 per-cancer
ax1 = fig.add_subplot(2, 2, 1)
x = np.arange(33)
w = 0.38
ax1.bar(x - w/2, nt_edges, w, color=BLUE, label='NOTEARS',
        edgecolor='white', linewidth=0.1)
ax1.bar(x + w/2, g3_edges, w, color=RED, alpha=0.85, label='GENIE3',
        edgecolor='white', linewidth=0.1)
ax1.set_xticks(x[::4])
ax1.set_xticklabels([cancers[i] for i in range(0, 33, 4)], fontsize=5)
ax1.set_ylabel('Edges')
ax1.set_title('a) Per-cancer edge counts', fontweight='bold', fontsize=8.5)
ax1.legend(fontsize=6.5, loc='upper right', frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# (b) Cross-cancer sharing
ax2 = fig.add_subplot(2, 2, 2)
cats = ['NOTEARS', 'GENIE3']
vals = [gp['shared_pct'], g3o['genie3_shared_pct']]
b2 = ax2.bar(cats, vals, color=[BLUE, RED], edgecolor='white',
             linewidth=0.3, width=0.4)
for b, v in zip(b2, vals):
    ax2.text(b.get_x() + b.get_width()/2., b.get_height() + 0.25,
             f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax2.set_ylabel('Shared in >=3 cancers (%)')
ax2.set_ylim(0, max(vals) * 1.35)
ax2.set_title('b) Cross-cancer edge sharing', fontweight='bold', fontsize=8.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# (c) Pooled vs per-cancer
ax3 = fig.add_subplot(2, 2, 3)
ax3.bar(['Per-cancer\n(total)', 'Pooled'], [total, pooled_edges],
        color=[BLUE, ORANGE], edgecolor='white', linewidth=0.3, width=0.4)
ax3.text(0, total + 100, f'{total:,}', ha='center', fontsize=10,
         fontweight='bold', color=BLUE)
ax3.text(1, pooled_edges + 5, f'{pooled_edges}', ha='center', fontsize=10,
         fontweight='bold', color=ORANGE)
fold = total // pooled_edges
ax3.set_ylabel('Total edges')
ax3.set_title(f'c) Pooled: {fold}x reduction', fontweight='bold', fontsize=8.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# (d) Overlap + STRING/TRRUST
ax4 = fig.add_subplot(2, 2, 4)
nt_only = total - g3o['overlap_edges']
ov = g3o['overlap_edges']
ax4.barh(['Gene-pair overlap'], [nt_only], color=BLUE, edgecolor='white',
        height=0.3, label='NOTEARS only')
ax4.barh(['Gene-pair overlap'], [ov], color=PURPLE, edgecolor='white',
        height=0.3, left=[nt_only], label='Shared')
ax4.legend(fontsize=6.5, loc='lower left', frameon=False)
ax4.set_xlabel('Gene-pairs')
ax4.set_title('d) NOTEARS-GENIE3 overlap', fontweight='bold', fontsize=8.5)
stats = (
    f'Shared: {ov} pairs\n'
    f'  {g3o["overlap_pct_notears"]:.1f}% of NOTEARS\n'
    f'  {g3o["overlap_pct_genie3"]:.1f}% of GENIE3\n'
    f'STRING: 22/2445 (0.8%)\n'
    f'TRRUST: 0/2445'
)
ax4.text(0.98, 0.97, stats, transform=ax4.transAxes, ha='right', va='top',
         fontsize=6, color=GRAY, fontstyle='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.06,
                    wspace=0.35, hspace=0.42)
plt.savefig(os.path.join(FIGDIR, 'fig3_baselines.pdf'), dpi=300)
plt.savefig(os.path.join(FIGDIR, 'fig3_baselines.png'), dpi=300)
plt.close()
print(f'Fig3 done: {os.path.getsize(os.path.join(FIGDIR, "fig3_baselines.pdf"))//1024}KB')
