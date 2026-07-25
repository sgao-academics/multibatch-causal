"""Figure 1: Synthetic validation (2x2). Real data from V6 MultiBatchCausalV6 output."""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, 'results')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Ground truth from synth_ckpt
synth = json.load(open(os.path.join(RESULTS, 'synth_ckpt.json')))
d = synth['d']
W0_true = np.array(synth['W0_true'])
W_trues = [np.array(W) for W in synth['W_trues']]
W2_true = W_trues[1]  # Batch 2: rewired + new

# V6 actual output
v6 = json.load(open(os.path.join(RESULTS, '_v6_original_output.json')))
W0_rec = np.array(v6['W0'])
Delta2 = np.array(v6['Deltas'][2])
h0 = v6['h_W0']
total_ok, total_gt = v6['total_ok'], v6['total_gt']

TAU_DISP = 0.12  # annotate edges above this

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
    'text.usetex': False,
})

def plot_matrix(ax, W, title, show_all=False):
    im = ax.imshow(W, cmap='RdBu_r', aspect='equal', vmin=-1, vmax=1)
    ax.set_xticks(range(d)); ax.set_yticks(range(d))
    ax.set_xticklabels([str(i) for i in range(d)], fontsize=5.5)
    ax.set_yticklabels([str(i) for i in range(d)], fontsize=5.5)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=3)
    for i in range(d):
        for j in range(d):
            val = W[i,j]
            if show_all:
                if abs(val) < 0.01: continue
            elif abs(val) < TAU_DISP:
                continue
            c = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=4.5, color=c, fontweight='bold')

# Build figure
fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.5))

# a) Ground truth W0
plot_matrix(axes[0,0], W0_true, 'a) Ground truth shared backbone W0', show_all=True)

# b) Ground truth Batch 2
plot_matrix(axes[0,1], W2_true, 'b) Ground truth Batch 2 (rewired 2,5 + new 0,7)', show_all=True)

# c) Recovered W0 (real V6 output - shows estimation noise)
plot_matrix(axes[1,0], W0_rec, 'c) Recovered W0 (10/11 shared edges, h=5.2e-6)', show_all=False)

# d) Delta_2 deviation
plot_matrix(axes[1,1], Delta2, 'd) Deviation D2 (rewired 2,5=-0.85, new 0,7=+0.76)', show_all=False)

# Colorbar
cbar_ax = fig.add_axes([0.15, 0.02, 0.70, 0.012])
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(-1, 1))
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cb.set_label('Edge weight', fontsize=7)
cb.ax.tick_params(labelsize=5.5)

fig.text(0.5, 0.055,
         f'Recovery: {total_ok}/{total_gt} ({synth["recovery_pct"]:.0f}%). '
         f'DAG constraint h(W0) = {h0:.2e} (Stage 2 augmented Lagrangian projection).',
         ha='center', fontsize=6.5, style='italic', color='#555555')

plt.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.09, wspace=0.32, hspace=0.38)
plt.savefig(os.path.join(FIGDIR, 'fig1_synthetic.pdf'), dpi=300)
plt.savefig(os.path.join(FIGDIR, 'fig1_synthetic.png'), dpi=300)
plt.close()
print(f'Fig1 done: {os.path.getsize(os.path.join(FIGDIR, "fig1_synthetic.pdf"))//1024}KB')
