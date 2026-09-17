"""Figure 9: cross-platform expression concordance on the DepMap cell-line panel.

For each STRING-confirmed network edge that maps unambiguously onto DepMap, the expression of the
two genes is correlated across 1,684 cell lines (OmicsExpression, log2(TPM+1)).  Panels c and d
show the strongest and the weakest case, so the full spread is visible rather than the best case
only.

As an internal consistency check the Spearman coefficients are recomputed here from the raw
cell-line values and compared with results/_depmap_validation.json; a mismatch aborts the figure.

Journal requirements applied here
---------------------------------
* The figure is designed at 6.85 in = 174 mm, the full-width figure area of the printed page, so
  the point sizes set below are the sizes that appear in print.
* No titles or captions inside the artwork: panels carry a bare "a)" .. "d)" label and every
  description lives in the figure caption.
* Bars carry a hatch pattern as well as a colour, so the two classes stay distinguishable for
  colour-blind readers.
* Sans-serif Arial throughout, including the maths text, so no DejaVu fallback glyphs appear.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, 'results')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

sc = json.load(open(os.path.join(RESULTS, '_depmap_scatter.json'), encoding='utf-8'))
pairs = sc['pairs']
expr = sc['expr']
models = sc['models']

# ---- consistency check: recompute r from raw values ----
mismatch = []
for r in pairs:
    a, b = r['A'], r['B']
    if a not in expr or b not in expr:
        mismatch.append((a, b, 'missing column'))
        continue
    x = np.asarray(expr[a], dtype=float)
    y = np.asarray(expr[b], dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        mismatch.append((a, b, 'too few'))
        continue
    rho, _ = spearmanr(x[m], y[m])
    if abs(rho - r['expr_corr']) > 0.005:
        mismatch.append((a, b, 'rho %.4f vs cached %.4f' % (rho, r['expr_corr'])))
if mismatch:
    print('ABORT: cached correlations do not reproduce')
    for m in mismatch:
        print('   ', m)
    sys.exit(1)
print('consistency check passed: all %d edges reproduce from raw cell-line values' % len(pairs))

order = sorted(range(len(pairs)), key=lambda i: pairs[i]['expr_corr'])
pl = [pairs[i] for i in order]

BLUE = '#2166AC'
RED = '#B2182B'
ORANGE = '#D6604D'
GRAY = '#555555'

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed TrueType, not Type 3
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2, 'legend.fontsize': 7.2,
    'hatch.linewidth': 0.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.linewidth': 0.6, 'text.usetex': False,
})
# bbox_inches='tight' is deliberately NOT used: it grows the page by whatever sticks out past
# the canvas, which is exactly what shrank the lettering of the previous version.  The page is
# the canvas, 6.85 x 5.2 in = 174 x 132 mm, and the margins below are sized so that nothing
# leaves it.

# designed at 174 mm = 493 pt, the full-width figure area of the F&IG printed page
fig = plt.figure(figsize=(6.85, 5.2))


def clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def tag(ax, letter):
    ax.set_title(letter + ')', loc='left', fontweight='bold', fontsize=9, pad=3)


# index used to cross-reference the bars of panel a with the points of panel b
n = len(pl)
idx_of = {id(p): i + 1 for i, p in enumerate(pl)}

# ---- (a) expression concordance per edge ----
ax = fig.add_subplot(2, 2, 1)
lbl = ['%d  %s \u2192 %s' % (i + 1, p['A'], p['B']) for i, p in enumerate(pl)]
rv = [p['expr_corr'] for p in pl]
cols = [RED if v >= 0.5 else BLUE for v in rv]
hat = ['///' if v >= 0.5 else '\\\\\\' for v in rv]
y = np.arange(len(pl))
for yi, (v, c, h) in enumerate(zip(rv, cols, hat)):
    ax.barh(yi, v, color=c, height=0.62, edgecolor='#333333', linewidth=0.0, hatch=h, alpha=0.9)
    ax.text(v + 0.015, yi, '%.2f' % v, va='center', fontsize=7.0, color=GRAY, fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(lbl, fontsize=7.0)
ax.set_xlim(0, 1.12)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel('Spearman $r$ across 1,684 cell lines')
tag(ax, 'a')
clean(ax)

# ---- (b) transcriptomic vs functional co-dependency ----
ax = fig.add_subplot(2, 2, 2)
xv = np.array([p['expr_corr'] for p in pairs], dtype=float)
yv = np.array([p['crispr_corr'] if p['crispr_corr'] == p['crispr_corr'] else np.nan
               for p in pairs], dtype=float)
ok = np.isfinite(yv)
ax.axhline(0, color=GRAY, lw=0.6, ls=':', zorder=1)
ax.scatter(xv[ok], yv[ok], s=34, c=BLUE, alpha=0.85, edgecolors='white', linewidth=0.5, zorder=5)
# The seven edges are too close together in x for their gene-pair names: the names are wide
# enough to overprint each other and to spill outside the axes.  Each point is therefore
# labelled with the bar number it carries in panel a.
idx_ok = np.where(ok)[0]
for i in idx_ok:
    ax.annotate(str(idx_of[id(pairs[i])]), (xv[i], yv[i]), textcoords='offset points',
                xytext=(0, 9), fontsize=7.4, fontweight='bold',
                ha='center', va='center', color=GRAY, zorder=6)
rho_rp = np.nan
if ok.sum() >= 5:
    rho_rp, p_rp = spearmanr(xv[ok], yv[ok])
ax.set_xlabel('Expression concordance, $r$')
ax.set_ylabel('CRISPR co-dependency, $r$')
ax.set_xlim(0.10, 0.98)
ax.set_ylim(-0.02, 0.36)
sub = 'n = %d edges with CRISPR data' % int(ok.sum())
if rho_rp == rho_rp:
    sub += '\nrank correlation between panels: $\\rho$ = %.2f' % rho_rp
ax.text(0.97, 0.97, sub, transform=ax.transAxes, ha='right', va='top', fontsize=7.0, color=GRAY)
tag(ax, 'b')
clean(ax)

# ---- (c)(d) strongest and weakest edge ----
for k, idx in enumerate([int(np.argmax(rv)), int(np.argmin(rv))]):
    ax = fig.add_subplot(2, 2, 3 + k)
    p = pairs[order[idx]]
    a, b = p['A'], p['B']
    x = np.asarray(expr[a], dtype=float)
    yv2 = np.asarray(expr[b], dtype=float)
    m = np.isfinite(x) & np.isfinite(yv2)
    ax.scatter(x[m], yv2[m], s=5, c=BLUE, alpha=0.35, edgecolors='none', zorder=5)
    if m.sum() > 40:
        xv2, yv3 = x[m], yv2[m]
        edges = np.quantile(xv2, np.linspace(0, 1, 9))
        bx, by = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (xv2 >= lo) & (xv2 <= hi)
            if sel.sum() >= 10:          # only bins with enough cell lines
                bx.append(float(np.median(xv2[sel])))
                by.append(float(np.median(yv3[sel])))
        if len(bx) >= 3:
            ax.plot(bx, by, color=RED, lw=1.1, marker='o', ms=2.4, zorder=6)
    ax.set_xlabel('%s, log$_2$(TPM+1)' % a)
    ax.set_ylabel('%s, log$_2$(TPM+1)' % b)
    pstr = '$p$ < 0.001' if p['expr_p'] < 1e-3 else ('$p$ = %.2g' % p['expr_p'])
    tag(ax, 'c' if k == 0 else 'd')
    ax.text(0.97, 0.05, '$r$ = %.2f, %s' % (p['expr_corr'], pstr),
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7.4,
            color=GRAY, fontweight='bold')
    clean(ax)

plt.subplots_adjust(left=0.197, right=0.985, top=0.940, bottom=0.095, wspace=0.715, hspace=0.36)
plt.savefig(os.path.join(FIGDIR, 'Fig9.pdf'), dpi=300)
plt.savefig(os.path.join(FIGDIR, 'Fig9.png'), dpi=300)
plt.close()

print('Fig8 done: %d KB' % (os.path.getsize(os.path.join(FIGDIR, 'Fig9.pdf')) // 1024))
print('  edges plotted = %d ; r range = %.2f - %.2f' % (len(pairs), min(rv), max(rv)))
print('  strongest = %s->%s r=%.2f ; weakest = %s->%s r=%.2f'
      % (pairs[order[-1]]['A'], pairs[order[-1]]['B'], max(rv),
         pairs[order[0]]['A'], pairs[order[0]]['B'], min(rv)))
