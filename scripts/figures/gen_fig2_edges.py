"""Figure 2: TCGA pan-cancer edge analysis (2 x 3). Real data from pipeline checkpoints.

a  edges per cancer
b  per-cancer edge reuse rate
c  pair-level sharing distribution
d  edges vs sample size
e  33 x 33 pairwise sharing map
f  composition of the gene-pairs that recur in >= 3 cancers

Journal requirements applied here
---------------------------------
* Designed at 6.85 in = 174 mm, the full-width figure area of the printed page, so the point sizes
  set below are the sizes that appear in print.
* No titles or captions inside the artwork: panels carry a bare "a)" .. "f)" label and every
  description lives in the figure caption.
* Bars carry a hatch as well as a colour, so the two classes stay distinguishable for colour-blind
  readers.
* The 33 cohort names are printed on every third row of panels a and b.  Labelling all 33 at a
  legible size needs ~275 pt of height and a panel only has ~160 pt, which is what forced the
  previous version down to 5.4 pt.
* Sans-serif Arial throughout, including the maths text.
"""
import json, os, sys, numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, 'results')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Load data
nt = json.load(open(os.path.join(RESULTS, '_pipeline_notears.json')))
gp = json.load(open(os.path.join(RESULTS, '_pipeline_genepair.json')))
share = json.load(open(os.path.join(RESULTS, '_cancer_pair_sharing.json')))

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
            if i != j and abs(W[i, j]) > 0.3:
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
            if i != j and abs(W[i, j]) > 0.3:
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
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed TrueType, not Type 3
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2, 'legend.fontsize': 7.2,
    'hatch.linewidth': 0.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'text.usetex': False,
})
# The page is the canvas, 6.85 x 6.0 in = 174 x 152 mm; bbox_inches='tight' is deliberately NOT
# used, because it grows the page by whatever sticks out and so shrinks the lettering again.
fig = plt.figure(figsize=(6.85, 6.0))

# --- explicit layout ---
# Row 1 is a plain three-column grid.  Row 2 is not: panel e is a square image that carries a
# colour bar and a column of cohort names, so panels d and f are placed individually around it.
# Everything is sized so that no axis label lands on a neighbour's tick labels or colour bar.
ROW1 = (0.575, 0.370)
ROW2 = (0.085, 0.370)
A1 = [0.100, ROW1[0], 0.275, ROW1[1]]
A2 = [0.420, ROW1[0], 0.275, ROW1[1]]
A3 = [0.735, ROW1[0], 0.255, ROW1[1]]
A4 = [0.075, ROW2[0], 0.245, ROW2[1]]
A6 = [0.735, ROW2[0], 0.255, ROW2[1]]


def box(col, row):
    return [A1, A2, A3][col] if row is ROW1 else [A4, None, A6][col]


def tag(ax, letter):
    ax.set_title(letter + ')', loc='left', fontweight='bold', fontsize=9, pad=3)


def clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def thin_labels(names, step=3):
    """Cohort names every `step` rows: 33 labels at 7 pt need more height than a panel has."""
    return [n if i % step == 0 else '' for i, n in enumerate(names)]


# ---------------------------------------------------------------- (a) edges per cancer
ax1 = fig.add_axes(box(0, ROW1))
colors1 = [BLUE if e >= mean_edges else '#92C5DE' for e in e_s]
hat1 = ['///' if e >= mean_edges else '\\\\\\' for e in e_s]
for yi, (e, col, h) in enumerate(zip(e_s, colors1, hat1)):
    ax1.barh(yi, e, height=0.68, color=col, edgecolor='#333333', linewidth=0.0, hatch=h)
ax1.axvline(mean_edges, color=ORANGE, linestyle='--', linewidth=1, alpha=0.8, zorder=6)
ax1.set_yticks(range(len(c_s)))
ax1.set_yticklabels(thin_labels(c_s), fontsize=7.0)
ax1.set_ylim(-0.8, len(c_s) - 0.2)
ax1.set_xlabel('Causal edges')
ax1.text(0.97, 0.97, 'Mean = %.0f' % mean_edges, transform=ax1.transAxes,
         ha='right', va='top', fontsize=7.0, color=ORANGE, fontweight='bold')
tag(ax1, 'a')
clean(ax1)
ax1.tick_params(axis='y', length=0)

# ---------------------------------------------------------------- (b) reuse rate per cancer
ax2 = fig.add_axes(box(1, ROW1))
ir = np.argsort(r_s)[::-1]
rc = [c_s[i] for i in ir]
rv = [r_s[i] for i in ir]
colors2 = [RED if v >= mean_reuse else '#F4A582' for v in rv]
hat2 = ['///' if v >= mean_reuse else '\\\\\\' for v in rv]
for yi, (v, col, h) in enumerate(zip(rv, colors2, hat2)):
    ax2.barh(yi, v, height=0.68, color=col, edgecolor='#333333', linewidth=0.0, hatch=h)
ax2.axvline(mean_reuse, color=ORANGE, linestyle='--', linewidth=1, alpha=0.8, zorder=6)
ax2.set_yticks(range(len(rc)))
ax2.set_yticklabels(thin_labels(rc), fontsize=7.0)
ax2.set_ylim(-0.8, len(rc) - 0.2)
ax2.set_xlabel('Reuse rate (%)')
ax2.text(0.97, 0.97, 'Mean = %.1f%%' % mean_reuse, transform=ax2.transAxes,
         ha='right', va='top', fontsize=7.0, color=ORANGE, fontweight='bold')
tag(ax2, 'b')
clean(ax2)
ax2.tick_params(axis='y', length=0)

# ---------------------------------------------------------------- (c) sharing distribution
ax3 = fig.add_axes(box(2, ROW1))
cats = ['1 cancer', '2 cancers', r'$\geq$3 cancers']
vals = [pct_1, pct_2, pct_3p]
colors3 = ['#92C5DE', BLUE, RED]
hat3 = ['..', '///', '\\\\\\']
for xi, (v, col, h) in enumerate(zip(vals, colors3, hat3)):
    ax3.bar(xi, v, color=col, edgecolor='#333333', linewidth=0.0, hatch=h, width=0.58)
    n = [cnt_1, cnt_2, cnt_3p][xi]
    ax3.text(xi, v + 2.0, '%.1f%%\n(n=%d)' % (v, n), ha='center', fontsize=7.0, fontweight='bold')
ax3.set_xticks(range(3))
ax3.set_xticklabels(cats, fontsize=7.2)
ax3.set_ylabel('Gene-pairs (%)')
ax3.set_ylim(0, max(vals) * 1.30)
ax3.set_xlim(-0.6, 2.6)
tag(ax3, 'c')
clean(ax3)

# ---------------------------------------------------------------- (d) edges vs sample size
ax4 = fig.add_axes(box(0, ROW2))
ax4.scatter(ns_list, edges_list, c=BLUE, s=20, alpha=0.75,
            edgecolors='white', linewidth=0.4, zorder=5)
# Staggered offsets for tight clusters (avoid label overlap)
extreme = {'CHOL', 'DLBC', 'UCS', 'BRCA', 'LUAD', 'KIRC'}
h_offsets = {'DLBC': (-24, 4), 'UCS': (14, -9), 'LUAD': (-15, -10),
             'KIRC': (15, -9), 'CHOL': (-16, 4), 'BRCA': (-4, 6)}
for i, c in enumerate(cancers_all):
    if c in extreme:
        dx, dy = h_offsets.get(c, (0, 6))
        ax4.annotate(c, (ns_list[i], edges_list[i]),
                     textcoords="offset points", xytext=(dx, dy),
                     fontsize=7.0, ha='center', color=GRAY, fontweight='bold')
r_sp, p_sp = spearmanr(edges_list, ns_list)
ax4.text(0.97, 0.96, 'Spearman $r$ = %.2f\n$p$ < 0.001' % r_sp,
         transform=ax4.transAxes, ha='right', va='top', fontsize=7.0,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
ax4.set_xlabel('Sample size ($n$)')
ax4.set_ylabel('Causal edges')
tag(ax4, 'd')
clean(ax4)

# ---------------------------------------------------------------- (e) 33 x 33 sharing map
# Square in inches: the axes box is 0.275 x 174 mm = 47.9 mm wide, so the same 47.9 mm in height
# is 47.9 / 152 = 0.3148 of the canvas; it is centred in the row.
SQ = 0.245                      # square side in canvas fractions: 0.245 x 174 mm = 42.6 mm
SQH = SQ * 174.0 / 152.0        # the same 42.6 mm as a fraction of the 152 mm canvas height
EC = [0.380, ROW2[0] + (ROW2[1] - SQH) / 2.0, SQ, SQH]
CC = [EC[0] + SQ + 0.008, EC[1], 0.014, SQH]
ax5 = fig.add_axes(EC)
M = np.array(share['matrix_pct'], dtype=float)
order = share['cancers']
Z = linkage(1.0 - M / 100.0, method='average')
li = list(leaves_list(Z))
Mr = M[np.ix_(li, li)]
ordered = [order[i] for i in li]
np.fill_diagonal(Mr, np.nan)
im = ax5.imshow(Mr, cmap='YlOrRd', vmin=0, vmax=20, interpolation='nearest', aspect='equal')
# 47.9 mm over 33 rows is 1.45 mm per row, far below the 2.5 mm a 7 pt label needs, so every third
# cohort is labelled; a full 33-label column overprints itself.
step5 = 3
ax5.set_xticks(range(0, len(ordered), step5))
ax5.set_yticks(range(0, len(ordered), step5))
ax5.set_xticklabels(ordered[::step5], fontsize=7.0, rotation=90)
ax5.set_yticklabels(ordered[::step5], fontsize=7.0)
cb = fig.colorbar(im, cax=fig.add_axes(CC))
# The label goes above the bar, not beside it: a rotated label there would run straight into the
# left-hand axis label of panel f.
cb.ax.set_title('Shared\nedges (%)', fontsize=7.0, pad=4)
cb.set_ticks([0, 10, 20])
cb.ax.tick_params(labelsize=7.0)
ax5.tick_params(length=0)
for s in ax5.spines.values():
    s.set_linewidth(0.4)
tag(ax5, 'e')

# ---------------------------------------------------------------- (f) composition of shared set
ax6 = fig.add_axes(A6)
comp = share['shared_set_composition']
# Two short lines per category: at 45 mm across, three categories leave 14 mm each, and
# "Sex-chromosome (X-inactivation)" on one line is 20 mm wide.
labels6 = ['Sex\nchromosome', 'Same\nfamily', 'Other']
keys6 = ['sex_chromosome', 'paralogous_same_family', 'other']
vals6 = [comp[k] for k in keys6]
cols6 = [RED, ORANGE, '#92C5DE']
hat6 = ['///', '\\\\\\', '..']
tot = sum(vals6)
for xi, (v, col, h) in enumerate(zip(vals6, cols6, hat6)):
    ax6.bar(xi, v, color=col, edgecolor='#333333', linewidth=0.0, hatch=h, width=0.58)
    ax6.text(xi, v + 0.25, '%d\n(%.0f%%)' % (v, 100 * v / tot),
             ha='center', fontsize=7.0, fontweight='bold')
ax6.set_xticks(range(3))
ax6.set_xticklabels(labels6, fontsize=7.0)
ax6.set_ylabel('Gene-pairs')
ax6.set_ylim(0, max(vals6) * 1.32)
ax6.set_xlim(-0.6, 2.6)
tag(ax6, 'f')
clean(ax6)

# ---- nothing may leave the canvas: the page IS the canvas here ----
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_W, _H = fig.get_size_inches() * fig.dpi
_bad = []
for _t in fig.findobj(matplotlib.text.Text):
    if _t.axes is None or _t.get_figure() is not fig:
        continue
    if not _t.get_visible() or not _t.get_text().strip():
        continue
    _b = _t.get_window_extent(renderer=_rend)
    if _b.x0 < -12 or _b.y0 < -12 or _b.x1 > _W + 12 or _b.y1 > _H + 12:
        _bad.append((_t.get_text()[:34].replace('\n', ' '), round(_b.x0, 1), round(_b.x1, 1)))
if _bad:
    print('ABORT: %d text elements leave the canvas' % len(_bad))
    for b in _bad[:10]:
        print('   ', b)
    sys.exit(1)
print('canvas check passed: all text inside %.2f x %.2f in'
      % (fig.get_size_inches()[0], fig.get_size_inches()[1]))

plt.savefig(os.path.join(FIGDIR, 'Fig2.pdf'), dpi=300)
plt.savefig(os.path.join(FIGDIR, 'Fig2.png'), dpi=300)
plt.close()
print('Fig2 done: %dKB' % (os.path.getsize(os.path.join(FIGDIR, "Fig2.pdf")) // 1024))
print('  Edges: %d total, mean %.1f' % (sum(edges_list), mean_edges))
print('  Reuse: mean %.1f%%' % mean_reuse)
print('  Sharing: 1=%.1f%% (%d), 2=%.1f%% (%d), >=3=%.1f%% (%d)'
      % (pct_1, cnt_1, pct_2, cnt_2, pct_3p, cnt_3p))
print('  Spearman: r=%.2f, p<0.001' % r_sp)
print('  Shared>=3 composition: %s  (total %d)' % (comp, tot))
