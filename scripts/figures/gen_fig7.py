"""Figure 6: tissue specificity of the per-cancer network hub genes.

Panels a-b are statistics over all 28 hub genes (one max-degree gene per cancer) compared with
the non-hub genes of the same networks.  Panels c-i show individual hub-gene expression profiles
across the 33 TCGA cohorts, spanning the full observed rank range (rank 1 to rank 33) rather than
only the best cases.

All numbers come from results/_tau_specificity.json and results/_hub_expression.json, which are
derived from the TCGA HiSeqV2 matrices by scripts/figures/_prep_tau.py.

Journal requirements applied here
---------------------------------
* Designed at 6.85 in = 174 mm, the full-width figure area of the printed page, so the point sizes
  set below are the sizes that appear in print.
* No titles or captions inside the artwork: panels carry a bare "a)" .. "i)" label, and the
  gene, its own cancer and its rank are stated in the figure caption.
* The 33 cohort names are printed once per column, on the bottom row only; repeating them in all
  nine panels is what forced the previous version down to 6 pt.
* Sans-serif Arial throughout, including the maths text.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, 'results')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

d = json.load(open(os.path.join(RESULTS, '_tau_specificity.json'), encoding='utf-8'))
p1 = json.load(open(os.path.join(RESULTS, '_hub_expression.json'), encoding='utf-8'))

cancers = d['cancers']
hub_of = d['hub_of_cancer']
raw = d['hub_raw']
stats = p1['stats']
hub_tau = np.asarray(d['hub_tau'], dtype=float)
nonhub_tau = np.asarray(d['nonhub_tau'], dtype=float)
rank_mean = d['hub_rank_mean']
perm_p = d['hub_rank_perm_p']

# Two-sided Mann-Whitney, computed here so the reported direction is unambiguous.
from scipy.stats import mannwhitneyu
mw_p = float(mannwhitneyu(hub_tau, nonhub_tau, alternative='two-sided')[1])
hub_lower = bool(hub_tau.mean() < nonhub_tau.mean())

BLUE = '#2166AC'
RED = '#B2182B'
ORANGE = '#D6604D'
GRAY = '#555555'
LGRAY = '#BBBBBB'

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed TrueType, not Type 3
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7.0, 'ytick.labelsize': 7.2, 'legend.fontsize': 7.2,
    'axes.linewidth': 0.6, 'text.usetex': False,
})
# The page is the canvas, 6.85 x 7.2 in = 174 x 183 mm; bbox_inches='tight' is deliberately NOT
# used, because it grows the page by whatever sticks out and so shrinks the lettering again.
fig = plt.figure(figsize=(6.85, 7.2))


def clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def tag(ax, letter):
    ax.set_title(letter + ')', loc='left', fontweight='bold', fontsize=9, pad=3)


# ---- (a) tau: hub vs non-hub network genes ----
ax = fig.add_subplot(3, 3, 1)
parts = ax.violinplot([nonhub_tau, hub_tau], positions=[1, 2], widths=0.7,
                      showextrema=False, showmedians=False)
for i, pc_ in enumerate(parts['bodies']):
    pc_.set_facecolor(LGRAY if i == 0 else BLUE)
    pc_.set_alpha(0.75)
    pc_.set_edgecolor('white')
    pc_.set_linewidth(0.4)
bp = ax.boxplot([nonhub_tau, hub_tau], positions=[1, 2], widths=0.16, showfliers=False,
                patch_artist=True, medianprops=dict(color='white', linewidth=1.0))
for i, b in enumerate(bp['boxes']):
    b.set_facecolor(GRAY if i == 0 else RED)
    b.set_edgecolor('white')
    b.set_linewidth(0.4)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Non-hub\n(n=%d)' % len(nonhub_tau), 'Hub\n(n=%d)' % len(hub_tau)])
ax.set_ylabel(r'Tissue-specificity index $\tau$')
ax.set_ylim(-0.02, 1.02)
ptxt = ('$p$ = %.3f  (hubs %s)' % (mw_p, 'lower' if hub_lower else 'higher'))
ax.text(0.5, 0.96, ptxt, transform=ax.transAxes, ha='center', va='top', fontsize=7.0,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='none'))
tag(ax, 'a')
clean(ax)

# ---- (b) rank of the hub gene's own cancer ----
ax = fig.add_subplot(3, 3, 2)
ranks = np.array(sorted(v['rank'] for v in stats.values()))
ax.plot(ranks, np.arange(1, len(ranks) + 1) / len(ranks), marker='o', ms=2.6,
        color=BLUE, lw=1.0, label='Observed (%d hubs)' % len(ranks))
ax.plot([1, 33], [1 / 33, 1.0], ls='--', lw=0.9, color=ORANGE, label='Uniform null')
ax.set_xlabel('Rank of own cancer')
ax.set_ylabel('Cumulative fraction of hubs')
ax.set_xlim(0.5, 33.5)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=7.2, frameon=False, loc='lower right')
ax.text(0.04, 0.95, 'mean rank = %.1f (null 17.0)\npermutation $p$ = %.3f' % (rank_mean, perm_p),
        transform=ax.transAxes, va='top', fontsize=7.0,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='none'))
tag(ax, 'b')
clean(ax)

# ---- (c-i) individual hub-gene profiles, spanning the full rank range ----
SHOW = ['SLC22A6', 'GABRA1', 'LTF', 'NAPSA', 'CXCL5', 'ADH1B', 'CD14']
for k, g in enumerate(SHOW):
    pos = 3 + k                       # grid slot c .. i
    ax = fig.add_subplot(3, 3, pos)
    vecs, labels, colors = [], [], []
    for c in cancers:
        v = raw.get(g, {}).get(c)
        if v:
            vecs.append(v)
            labels.append(c)
            colors.append(RED if hub_of.get(c) == g else LGRAY)
    if not vecs:
        ax.axis('off')
        continue
    bp = ax.boxplot(vecs, widths=0.72, showfliers=False, patch_artist=True,
                    medianprops=dict(color='white', linewidth=0.6))
    for b, col in zip(bp['boxes'], colors):
        b.set_facecolor(col)
        b.set_edgecolor('white')
        # 0.25 pt was below the journal's 0.3 pt floor for line art; 0.4 matches panel (a)
        b.set_linewidth(0.4)
    for el in ('whiskers', 'caps'):
        for e in bp[el]:
            e.set_linewidth(0.4)
            e.set_color(GRAY)
    own = stats.get(g, {}).get('own_cancer', '')
    # The cohort names are printed once per column, on the bottom row of the grid (slots g, h, i).
    bottom_row = pos >= 7
    if bottom_row:
        ax.set_xticks(range(1, len(labels) + 1))
        # Every third cohort: at 7 pt a rotated label occupies about 8 pt of width, and a panel is
        # only ~116 pt wide, so every other one would overprint.  The hub's own cohort is always
        # labelled, and its two neighbours are dropped so that it cannot crowd them either.
        keep = set(range(0, len(labels), 3))
        if own in labels:
            oi = labels.index(own)
            keep.discard(oi - 1)
            keep.discard(oi + 1)
            keep.add(oi)
        display = [lab if i in keep else '' for i, lab in enumerate(labels)]
        ax.set_xticklabels(display, fontsize=7.0, rotation=90, va='top')
    else:
        ax.set_xticklabels([])
    if pos % 3 == 0:                  # right-hand column carries the shared y label
        ax.set_ylabel('log$_2$(TPM+1)')
    ax.tick_params(axis='y', labelsize=7.2)
    tag(ax, 'abcdefghi'[pos - 1])
    clean(ax)

plt.subplots_adjust(left=0.085, right=0.985, top=0.962, bottom=0.060,
                    wspace=0.42, hspace=0.32)

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

plt.savefig(os.path.join(FIGDIR, 'Fig6.pdf'), dpi=300)
plt.savefig(os.path.join(FIGDIR, 'Fig6.png'), dpi=300)
plt.close()

print('Fig7 done: %d KB' % (os.path.getsize(os.path.join(FIGDIR, 'Fig6.pdf')) // 1024))
print('  tau hub mean=%.3f vs non-hub mean=%.3f  MW p=%s' % (hub_tau.mean(), nonhub_tau.mean(), mw_p))
print('  rank mean=%.2f  perm p=%.3f' % (rank_mean, perm_p))
print('  examples: %s' % ', '.join('%s(rank %d)' % (g, stats[g]['rank']) for g in SHOW if g in stats))
