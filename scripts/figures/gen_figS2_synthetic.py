"""Figure S2: Synthetic validation (2x2). Real data from V6 MultiBatchCausalV6 output."""
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
W2_true = W_trues[1]  # Batch 1: rewired + new

# V6 actual output
v6 = json.load(open(os.path.join(RESULTS, '_v6_original_output.json')))
W0_rec = np.array(v6['W0'])
D1 = np.array(v6['Deltas'][2])  # deviation of the rewired batch (referred to as Batch 1)
h0 = v6['h_W0']
total_ok, total_gt = v6['total_ok'], v6['total_gt']

TAU_DISP = 0.12  # annotate edges above this

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed TrueType, not Type 3
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
    ax.set_xticklabels([str(i) for i in range(d)], fontsize=6.6)
    ax.set_yticklabels([str(i) for i in range(d)], fontsize=6.6)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=3)
    for i in range(d):
        for j in range(d):
            val = W[i,j]
            if show_all:
                if abs(val) < 0.01: continue
            elif abs(val) < TAU_DISP:
                continue
            c = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6.6, color=c, fontweight='bold')

# Build figure
fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.5))

# Panels carry a bare letter: the journal asks that illustrations contain no titles of their own,
# and the supplementary caption describes all four panels.
plot_matrix(axes[0,0], W0_true, 'a', show_all=True)
plot_matrix(axes[0,1], W2_true, 'b', show_all=True)
plot_matrix(axes[1,0], W0_rec, 'c', show_all=False)
plot_matrix(axes[1,1], D1, 'd', show_all=False)

# Colorbar
cbar_ax = fig.add_axes([0.15, 0.02, 0.70, 0.012])
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(-1, 1))
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cb.set_label('Edge weight', fontsize=7.6)
cb.ax.tick_params(labelsize=6.6)

# The recovery figures used to be printed inside the artwork as an italic note.  The journal asks
# for no title or caption inside an illustration, and the supplementary caption carries both
# numbers, so the note is gone; the same labels are also enlarged to clear the 7 pt print floor
# (this panel is 160 mm wide on the canvas and prints at 1.09x).

plt.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.09, wspace=0.32, hspace=0.38)
plt.savefig(os.path.join(FIGDIR, 'FigS2.pdf'), dpi=300)
plt.savefig(os.path.join(FIGDIR, 'FigS2.png'), dpi=300)
plt.close()
print(f'FigS2 done: {os.path.getsize(os.path.join(FIGDIR, "FigS2.pdf"))//1024}KB')
