"""Figure 4: pathway enrichment of the inferred per-cancer causal networks.

  a) LUAD    b) BRCA    c) CHOL (no set reaches significance; shown as a null case)
  d) pan-cancer pooled network

The Kaplan-Meier curves live in their own figure and the DepMap concordance in Figure 8, so neither
is repeated here.  Every number is read from results/_enrichment_percancer.json and
results/_enrichment_global.json, and three claims made in the manuscript text are asserted below;
a failure aborts the figure.

Journal requirements applied here
---------------------------------
* Designed at 6.85 in = 174 mm, the full-width figure area of the printed page, so the point sizes
  set below are the sizes that appear in print.
* No titles or captions inside the artwork: panels carry a bare "a)" .. "d)" label, and the set
  counts that used to sit on top of the bars are folded into the axis label.
* Bars carry a hatch as well as a colour, so the significant and non-significant sets stay
  distinguishable for colour-blind readers.
* Sans-serif Arial throughout, including the maths text.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE, 'results')
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

per = json.load(open(os.path.join(RES, '_enrichment_percancer.json'), encoding='utf-8'))['per_cancer']
glo = json.load(open(os.path.join(RES, '_enrichment_global.json'), encoding='utf-8'))

# ---- assertions: the manuscript's textual claims must hold in the data ----
def find(cancer, name):
    return next((e for e in per[cancer]['enrichment'] if e['set'] == name), None)

checks = [
    ('LUAD', 'REACTOME_SURFACTANT_METABOLISM'),
    ('LUAD', 'REACTOME_TOLL_LIKE_RECEPTOR_CASCADES'),
    ('BRCA', 'REACTOME_DEVELOPMENTAL_BIOLOGY'),
    ('BRCA', 'ONDER_CDH1_TARGETS_1_DN'),
    ('BRCA', 'ONDER_CDH1_TARGETS_2_DN'),
]
for c, n in checks:
    if find(c, n) is None:
        print('ABORT: missing %s / %s' % (c, n)); sys.exit(1)
chol_sig = [e for e in per['CHOL']['enrichment'] if e['p'] < 0.05]
if chol_sig:
    print('ABORT: CHOL unexpectedly has %d significant sets' % len(chol_sig)); sys.exit(1)
gnames = {e['set'] for e in glo['global_enrichment']}
for n in ['HSIAO_LIVER_SPECIFIC_GENES', 'SWEET_LUNG_CANCER_KRAS_UP']:
    if n not in gnames:
        print('ABORT: global set %s missing' % n); sys.exit(1)
print('claim checks passed: LUAD/BRCA named sets present, CHOL has 0 significant sets,')
print('                     global panel contains the two sets named in the text')

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed TrueType, not Type 3
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7.2, 'ytick.labelsize': 7.0, 'legend.fontsize': 7.2,
    'hatch.linewidth': 0.5,
    'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'text.usetex': False,
})
BLUE = '#1B5E8C'; RED = '#C0392B'; ORANGE = '#D6604D'; GRAY = '#777777'

# The page is the canvas, 6.85 x 4.3 in = 174 x 109 mm; bbox_inches='tight' is deliberately NOT
# used, because it grows the page by whatever sticks out and so shrinks the lettering again.
fig = plt.figure(figsize=(6.85, 4.3))


def short(name):
    """MSigDB set names run to 40+ characters.  30 characters is the longest that keeps a 7 pt
    label inside the left margin of a two-column layout at 174 mm, and the elision has to start
    that late: the two Toll-like receptor sets first differ in character 29, so a shorter cut
    would render them as the same string."""
    s = name.replace('_', ' ')
    return s if len(s) <= 30 else s[:29].rstrip() + '\u2026'


def panel(ax, entries, letter, topn=6, color=BLUE):
    sel = sorted(entries, key=lambda e: -float(e['enrichment']))[:topn]
    sel = sel[::-1]
    n_sig = sum(1 for e in entries if float(e['p']) < 0.05)
    for i, e in enumerate(sel):
        fold = float(e['enrichment']); p = float(e['p'])
        sig = p < 0.05
        ax.barh(i, fold, color=color if sig else GRAY, alpha=0.9, height=0.66,
                edgecolor='#333333', linewidth=0.0,
                hatch='///' if sig else 'xxx')
        ax.text(fold + max(0.06, 0.03 * fold), i, '%.2f' % fold,
                va='center', fontsize=7.0, color='#444444')
    labs = [short(e['set']) for e in sel]
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels(labs, fontsize=7.0)
    ax.set_xlabel('Enrichment fold\n(%d of %d sets at $p$ < 0.05)' % (n_sig, len(entries)))
    ax.set_xlim(0, max(float(e['enrichment']) for e in sel) * 1.30)
    ax.set_title(letter + ')', loc='left', pad=3, fontsize=9, fontweight='bold')
    ax.grid(axis='x', alpha=0.15)
    ax.tick_params(axis='y', length=0)
    return sel


panel(fig.add_subplot(2, 2, 1), per['LUAD']['enrichment'], 'a', color=BLUE)
panel(fig.add_subplot(2, 2, 2), per['BRCA']['enrichment'], 'b', color=RED)
panel(fig.add_subplot(2, 2, 3), per['CHOL']['enrichment'], 'c', color=ORANGE)
panel(fig.add_subplot(2, 2, 4), glo['global_enrichment'], 'd', color=BLUE)

plt.subplots_adjust(left=0.288, right=0.975, top=0.935, bottom=0.155,
                    wspace=1.47, hspace=0.45)

# ---- nothing may leave the canvas: the page IS the canvas here ----
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_W, _H = fig.get_size_inches() * fig.dpi
_bad = []
for _t in fig.findobj(matplotlib.text.Text):
    # only text that actually belongs to an axes of this figure; matplotlib keeps detached
    # tick-label instances alive, and those are not drawn
    if _t.axes is None or _t.get_figure() is not fig:
        continue
    if not _t.get_visible() or not _t.get_text().strip():
        continue
    _b = _t.get_window_extent(renderer=_rend)
    # 12 units = 3 pt of slack: matplotlib's layout box is a little tighter than the glyph
    # extents PyMuPDF reports, so this is an early warning, not the final gate (see _mb_one)
    if _b.x0 < -12 or _b.y0 < -12 or _b.x1 > _W + 12 or _b.y1 > _H + 12:
        _ax = _t.axes.get_position().bounds if _t.axes is not None else None
        _bad.append((_t.get_text()[:34].replace('\n', ' '), round(_b.x0, 1), round(_b.x1, 1), _ax))
if _bad:
    with open(os.path.join(FIG, '_canvas_check.txt'), 'w', encoding='utf-8') as fh:
        fh.write('ABORT: %d text elements leave the canvas\n' % len(_bad))
        for b in _bad[:40]:
            fh.write('   %r\n' % (b,))
    print('ABORT: %d text elements leave the canvas' % len(_bad))
    for b in _bad[:10]:
        print('   ', b)
    sys.exit(1)
print('canvas check passed: all text inside %.2f x %.2f in' % (fig.get_size_inches()[0], fig.get_size_inches()[1]))

fig.savefig(os.path.join(FIG, 'Fig4.pdf'), dpi=300)
fig.savefig(os.path.join(FIG, 'Fig4.png'), dpi=300)
plt.close(fig)
print('FigBio done: %d KB (pdf) / %d KB (png)'
      % (os.path.getsize(os.path.join(FIG, 'Fig4.pdf')) // 1024,
         os.path.getsize(os.path.join(FIG, 'Fig4.png')) // 1024))
print('  LUAD sets=%d  BRCA sets=%d  CHOL sets=%d  global sets=%d'
      % (len(per['LUAD']['enrichment']), len(per['BRCA']['enrichment']),
         len(per['CHOL']['enrichment']), len(glo['global_enrichment'])))
