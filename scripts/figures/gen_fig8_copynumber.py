# -*- coding: utf-8 -*-
"""Figure 8: the recurring pairs carry no copy-number coupling beyond proximity.

The plate answers the one objection the rest of the paper leaves open.  Every other
analysis places the 14 recurring edges on the expression axis; none of them rules
out the possibility that the two genes of a pair simply share a copy-number state,
which produces correlated expression with no regulatory relation at all.  The
aligned-panel diagnostics did use the GISTIC 2 matrix, but only as a cross-platform
check on the dominant principal component -- never on the pairs themselves.  So the
pairs are measured directly here.

Panel a repeats the recurrence picture -- where the 14 loci sit, and which of them
emit chords -- so that the reader meets the same 14 objects in both panels.

Panel b is the new evidence.  Thresholded copy-number states are correlated between
the two genes of each recurring pair, and those correlations are read against 2,000
random gene pairs drawn in five genomic-distance strata.  The null is not optional:
copy number is spatially organised, so neighbouring genes co-vary at rho = 0.98 by
construction and a bare "rho is high" would be meaningless.  Against that null the
set splits in two.  The four cis pairs sit inside the neighbouring-gene cloud and
are therefore indistinguishable from co-amplification of one locus -- CXCL9/10/11
lie 13-18 kb apart in the 4q21.1 chemokine cluster, a classic amplicon.  The nine
trans pairs join genes on different chromosomes, co-vary no more than the
cross-chromosome background, and so cannot be copy-number artefacts.

Drawing language
----------------
Same tokens as every other figure in this package, imported from `_figstyle`: the
seven muted colours, type in the palette's indigo, and the shared canvas width.  The
ideogram is greyscale on purpose, so that hue is spent only on the cis/trans
distinction and the plate survives greyscale printing; the greys in STAIN are the
cytoband stains and are not palette colours.

Journal requirements applied here
---------------------------------
* 174 mm wide -- the printed text width, so the point sizes are the sizes that
  appear in print.  `bbox_inches='tight'` is deliberately NOT used.
* Sans-serif Arial throughout, Type 42 embedding, no Type 3.
* Shape (square / circle) as well as hue encodes cis versus trans.

Output: figures/Fig8.pdf / .png
"""
import gzip
import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Wedge, PathPatch, Circle, FancyBboxPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import PAL, INK, GREY, TICK, FRAME, BOXFC, BOXEC, MM
from _gene_symbols import canonical

# _figstyle has already fixed the house rcParams (Arial, Type 42 embedding, 8 pt
# base type).  Only the one token it does not set is added here.
matplotlib.rcParams['axes.unicode_minus'] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(ROOT, 'results')
DAT = os.path.join(ROOT, 'data')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

W_MM, H_MM = 174.0, 213.0

# ============================================================ data
coords = json.load(open(os.path.join(DAT, 'fig8_gene_coords.json'), encoding='utf-8'))
DATA = json.load(open(os.path.join(RES, '_fig8_edges.json'), encoding='utf-8'))
EDGES = DATA['edges']
ND = json.load(open(os.path.join(RES, '_fig8_null_dist.json'), encoding='utf-8'))
CNA = json.load(open(os.path.join(RES, '_fig8_cna_results.json'), encoding='utf-8'))
SC, EROW = ND['scatter'], ND['edges']

CYTO, CHRLEN = {}, {}
with gzip.open(os.path.join(DAT, 'hg19_cytoBand.txt.gz'), 'rt',
               encoding='utf-8', errors='ignore') as f:
    for line in f:
        c, s, e, band, stain = line.rstrip('\n').split('\t')
        CYTO.setdefault(c, []).append((int(s), int(e), band, stain))
        CHRLEN[c] = max(CHRLEN.get(c, 0), int(e))

CHROMS = ['chr%d' % i for i in range(1, 23)] + ['chrX', 'chrY']
TOTAL = sum(CHRLEN[c] for c in CHROMS)

GAP = 0.85
SPAN = 360.0 - GAP * len(CHROMS)
ANG, cur = {}, 90.0
for c in CHROMS:
    sp = CHRLEN[c] / TOTAL * SPAN
    ANG[c] = (cur, cur - sp)
    cur -= sp + GAP


def ang_of(chrom, bp):
    a0, a1 = ANG[chrom]
    return a0 + (a1 - a0) * (bp / CHRLEN[chrom])


def pol(a, r):
    t = math.radians(a)
    return r * math.cos(t), r * math.sin(t)


R_IN, R_OUT = 0.90, 1.00
R_BEAD = 0.866
R_LINK = 0.843
LAYERS = [1.070, 1.200, 1.330]

STAIN = {'gneg': '#FFFFFF', 'gpos25': '#DCDCDC', 'gpos50': '#B4B4B4',
         'gpos75': '#7E7E7E', 'gpos100': '#3C3C3C',
         'acen': PAL['orchid'], 'gvar': '#EDEDED', 'stalk': '#C8C8C8'}
CIS, TR = PAL['moss'], PAL['mist']
# The two markers carry the palette's moss and mist; the label type needs a darker
# reading of the same two hues so that 8 pt text keeps its contrast, and those two
# readings are fixed here rather than re-derived per run.
CIS_DK, TR_DK, TR_LT = '#5E8A47', '#7A88A0', '#A8B2C2'
HALO = [pe.withStroke(linewidth=2.4, foreground='white')]

LOCUS_GAP = 1_500_000
by_chr = {}
for g, c in coords.items():
    by_chr.setdefault(c['chrom'], []).append(g)

loci = []
for chrom, gs in by_chr.items():
    gs = sorted(gs, key=lambda g: coords[g]['start'])
    run = []
    for g in gs:
        if run and coords[g]['start'] - coords[run[-1]]['end'] < LOCUS_GAP:
            run.append(g)
        else:
            if run:
                loci.append(run)
            run = [g]
    if run:
        loci.append(run)

trans_genes = set()
for e in EDGES:
    if not e['cis']:
        trans_genes.add(e['src'])
        trans_genes.add(e['dst'])


def locus_label(gs):
    # Print the HGNC-approved symbol while the co-ordinates and the copy-number table keep the
    # spelling Xena carries (IL8 for CXCL8, MGC29506 for MZB1), so the plate agrees with Table 4
    # and with the rest of the paper: see _gene_symbols.
    disp = [canonical(g) for g in gs]
    if len(disp) == 1:
        return disp[0]
    if set(gs) == {'CXCL9', 'CXCL10', 'CXCL11'}:
        return 'CXCL9-11'
    if set(gs) == {'CXCL1', 'IL8'}:
        return 'CXCL1/CXCL8'
    if set(gs) == {'C7', 'PLCXD3'}:
        return 'C7/PLCXD3'
    return '/'.join(disp)


records = []
for gs in loci:
    chrom = coords[gs[0]]['chrom']
    mid = sum((coords[g]['start'] + coords[g]['end']) / 2 for g in gs) / len(gs)
    records.append({'a': ang_of(chrom, mid), 'genes': gs, 'mid': mid,
                    'chrom': chrom, 'label': locus_label(gs),
                    'band': coords[gs[0]]['band'],
                    'cis': all(g not in trans_genes for g in gs), 'layer': 0})
records.sort(key=lambda r: r['a'])

# ============================================================ claim checks
# The caption states these; the plate must not be able to drift from them.
_fail = []


def _check(ok, msg):
    if not ok:
        _fail.append(msg)


STRAT = ['same band (<1 Mb)', 'same arm (1-10 Mb)', 'same chr (10-100 Mb)',
         'same chr (>100 Mb)', 'different chr']
MED = [ND['strata'][s]['median'] for s in STRAT]
P95 = [ND['strata'][s]['p95'] for s in STRAT]
EXPECT_MED = [0.980, 0.902, 0.606, 0.104, 0.042]


def stratum_of(dist_bp):
    """The stratum a pair falls in, using the same cut-offs the null was drawn with."""
    if not dist_bp or dist_bp <= 0:
        return STRAT[4]
    for name, lo, hi in ((STRAT[0], 0, 1e6), (STRAT[1], 1e6, 1e7),
                         (STRAT[2], 1e7, 1e8), (STRAT[3], 1e8, 3e8)):
        if lo <= dist_bp < hi:
            return name
    return STRAT[4]


_check(DATA['n_cis'] == 5 and DATA['n_trans'] == 9,
       'the edge counts moved (cis %s, trans %s)' % (DATA['n_cis'], DATA['n_trans']))
_check(len(EROW) == 14 and len(EDGES) == 14, 'the recurring set is no longer 14 edges')
_check(ND['n_samples'] == 9415, 'the aligned-panel sample count moved (%s)' % ND['n_samples'])
_check(CNA['n_samples'] == 9415 and CNA['n_cohorts'] == 33,
       'the copy-number matrix no longer covers the aligned panel')
for i, (got, want) in enumerate(zip(MED, EXPECT_MED)):
    _check(abs(got - want) < 0.003,
           'stratum %s median moved: %.3f vs %.3f' % (STRAT[i], got, want))
bad = [r for r in EROW if r['rho'] > ND['strata'][stratum_of(r['dist_bp'])]['p95']]
_check(not bad, 'a recurring pair exceeds its stratum 95th percentile: %s'
       % [(r['src'], r['dst'], r['rho']) for r in bad])
n_cis_loci = sum(1 for r in records if r['cis'])
_check(n_cis_loci == 3, 'the number of all-cis loci moved (%d)' % n_cis_loci)

if _fail:
    print('ABORT: the data no longer support the caption')
    for m in _fail:
        print('  - ' + m)
    sys.exit(1)

print('claim checks passed: cis %d / trans %d edges, %d all-cis loci, '
      '%d samples, medians %s'
      % (DATA['n_cis'], DATA['n_trans'], n_cis_loci, ND['n_samples'],
         ' '.join('%.3f' % m for m in MED)))

# ============================================================ canvas
fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
# Panel a is a tight square: the data range hugs the ink (ring + outward gene
# labels) so the circos fills its box instead of floating in it.  The circos is
# round, so the ink stops well above the square's lower edge -- panel b is pushed
# up into that slack, which is why it may overlap the square without touching it
# (the square is axis-off and carries nothing down there).  Panel b's right edge
# lines up with panel a's ink; its left edge is pulled in for the rotated y-label.
BLK, A_LEFT, A_BOT = 164.0, 5.0, 47.0
B_LEFT, B_BOT, B_W, B_H = 18.0, 10.6, 148.0, 54.0
ax = fig.add_axes([A_LEFT / W_MM, A_BOT / H_MM, BLK / W_MM, BLK / H_MM])
ax.set_xlim(-1.50, 1.50)
ax.set_ylim(-1.50, 1.50)
ax.set_aspect('equal')
ax.axis('off')
axb = fig.add_axes([B_LEFT / W_MM, B_BOT / H_MM, B_W / W_MM, B_H / H_MM])

# ============================================================ panel a : circos
for c in CHROMS:
    for s, e, band, stain in CYTO[c]:
        a0, a1 = ang_of(c, s), ang_of(c, e)
        if abs(a0 - a1) < 1e-6:
            continue
        ax.add_patch(Wedge((0, 0), R_OUT, min(a0, a1), max(a0, a1),
                           width=R_OUT - R_IN,
                           facecolor=STAIN.get(stain, '#FFFFFF'),
                           edgecolor='white', linewidth=0.22, zorder=3))
for rr in (R_IN, R_OUT):
    ax.add_patch(Circle((0, 0), rr, fc='none', ec=FRAME, lw=0.6, zorder=3.2))

CHORD = []          # sampled ribbon geometry -- the key's guard rail checks against it
seen = {}
for e in EDGES:
    if e['cis']:
        continue
    a, b = e['src'], e['dst']
    key = tuple(sorted([a, b]))
    off = seen.get(key, 0) * 0.85
    seen[key] = seen.get(key, 0) + 1
    a1 = ang_of(coords[a]['chrom'], (coords[a]['start'] + coords[a]['end']) / 2) + off
    a2 = ang_of(coords[b]['chrom'], (coords[b]['start'] + coords[b]['end']) / 2) + off
    d = (a2 - a1 + 180) % 360 - 180
    am = a1 + d / 2
    x1, y1 = pol(a1, R_LINK)
    x2, y2 = pol(a2, R_LINK)
    cx, cy = pol(am, R_LINK * 0.55)
    path = Path([(x1, y1), (cx, cy), (x2, y2)],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    lw = 1.1 + (e['n'] - 10) / 18.0 * 3.9
    ax.add_patch(PathPatch(path, fc='none', ec=TR, lw=lw, alpha=0.44,
                           capstyle='round', zorder=2))
    for i in range(121):
        t = i / 120.0
        u = 1 - t
        CHORD.append((u * u * x1 + 2 * u * t * cx + t * t * x2,
                      u * u * y1 + 2 * u * t * cy + t * t * y2))

cis_pairs = {}
for e in EDGES:
    if e['cis']:
        k = tuple(sorted([e['src'], e['dst']]))
        cis_pairs[k] = max(cis_pairs.get(k, 0), e['n'])
for (g1, g2), n in cis_pairs.items():
    mid = ((coords[g1]['start'] + coords[g1]['end']) / 2 +
           (coords[g2]['start'] + coords[g2]['end']) / 2) / 2
    a = ang_of(coords[g1]['chrom'], mid)
    ax.add_patch(Wedge((0, 0), R_OUT + 0.011, a - 1.15, a + 1.15, width=0.024,
                       facecolor=CIS, edgecolor='none', alpha=0.98, zorder=5))

for r in records:
    x, y = pol(r['a'], R_BEAD)
    ax.plot([x], [y], marker='s' if r['cis'] else 'o',
            markersize=5.8 if r['cis'] else 5.2,
            markerfacecolor=CIS if r['cis'] else TR,
            markeredgecolor='white', markeredgewidth=0.85,
            linestyle='none', zorder=6)

USED_CHR = {coords[g]['chrom'] for g in coords}
for c in CHROMS:
    a0, a1 = ANG[c]
    am = (a0 + a1) / 2
    x, y = pol(am, R_OUT + 0.013)
    rot = am - 90
    if 90 < am % 360 < 270:
        rot += 180
    hot = c in USED_CHR
    ax.text(x, y, c[3:], rotation=rot, rotation_mode='anchor', ha='center',
            va='center', fontsize=6.5 if hot else 5.9,
            color=INK if hot else '#BDC5D3',
            fontweight='bold' if hot else 'normal', zorder=5)

label_objs = []


def clear_labels():
    for t in label_objs:
        t.remove()
    del label_objs[:]


def draw_label(r):
    a = r['a']
    R = LAYERS[r['layer']]
    rot = a - 90
    ha = 'left'
    if 90 < a % 360 < 270:
        rot += 180
        ha = 'right'
    xa, ya = pol(a, R)
    col = CIS_DK if r['cis'] else INK
    t1 = ax.text(xa, ya, r['label'], rotation=rot, rotation_mode='anchor',
                 ha=ha, va='bottom', fontsize=8.2, color=col,
                 fontweight='bold', zorder=8)
    t1.set_path_effects(HALO)
    t2 = ax.text(xa, ya, ' ' + r['band'], rotation=rot, rotation_mode='anchor',
                 ha=ha, va='top', fontsize=6.8, color=GREY, zorder=8)
    t2.set_path_effects(HALO)
    label_objs.extend([t1, t2])
    return t1, t2


def data_bbox(t):
    bb = t.get_window_extent(fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    pts = [inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y0)),
           inv.transform((bb.x1, bb.y1)), inv.transform((bb.x0, bb.y1))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def box_hit(b, o):
    return not (b[2] < o[0] or o[2] < b[0] or b[3] < o[1] or o[3] < b[1])


objs = {}
for r in records:
    r['layer'] = 0
    objs[id(r)] = draw_label(r)
fig.canvas.draw()
for r in records:
    r['_bb'] = [data_bbox(t) for t in objs[id(r)]]

rings = [[] for _ in LAYERS]
for r in records:
    px, py = pol(r['a'], LAYERS[0])
    for L in range(len(LAYERS)):
        qx, qy = pol(r['a'], LAYERS[L])
        dx, dy = qx - px, qy - py
        boxes = [(b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy) for b in r['_bb']]
        if all(not box_hit(b, o) for b in boxes for o in rings[L]):
            r['layer'] = L
            rings[L].extend(boxes)
            break
    else:
        r['layer'] = len(LAYERS) - 1

clear_labels()
for r in records:
    draw_label(r)

for r in records:
    a = r['a']
    R = LAYERS[r['layer']]
    x0, y0 = pol(a, R_OUT + 0.004)
    x1, y1 = pol(a, R - 0.008)
    ax.add_line(Line2D([x0, x1], [y0, y1], color=FRAME, lw=0.75, zorder=4))

# The panel-a key is drawn further down, once the gene labels are on the canvas
# and its final resting place can be checked against them.

# panel letter, top-left corner of the square as journals set them
ax.text(-1.475, 1.475, 'a', fontsize=11, fontweight='bold', color=INK,
        ha='left', va='top', zorder=12)

# ============================================================ panel b
XLBL = ['\u2264 1 Mb', '1\u201310 Mb', '10\u2013100 Mb', '> 100 Mb', 'diff.\nchromosome']

rgx = random.Random(3)
for i, s in enumerate(STRAT):
    sel = [j for j, v in enumerate(SC['stratum']) if v == s]
    jx = [i + rgx.uniform(-0.30, 0.30) for _ in sel]
    jy = [SC['rho'][j] for j in sel]
    axb.scatter(jx, jy, s=5.5, c='#D3DAE5', linewidths=0, zorder=1)

# median curve + the 95th percentile of the random pairs, which is the line the
# recurring edges must stay under to be unremarkable
axb.plot(range(5), P95, ':', color='#C2CBDA', lw=1.4, zorder=2)
axb.plot(range(5), MED, '-', color='#8E9BB0', lw=1.5, zorder=3)
axb.plot(range(5), MED, 'o', color='#8E9BB0', ms=4.2, zorder=5,
         markeredgecolor='white', markeredgewidth=0.8)

# separation before the cross-chromosome column
axb.axvline(3.5, color='#DCE2EA', lw=1.0, zorder=0)

# the 14 edges
cis_e = [r for r in EROW if r['cls'] == 'cis']
tr_e = [r for r in EROW if r['cls'] == 'trans']
for grp, xi, col, mk in ((cis_e, 0, CIS, 's'), (tr_e, 4, TR, 'o')):
    for r in grp:
        dx = rgx.uniform(-0.26, 0.26)
        axb.plot([xi + dx], [r['rho']], marker=mk, ms=6.2,
                 markerfacecolor=col, markeredgecolor='white',
                 markeredgewidth=1.0, linestyle='none', zorder=6)

axb.set_xlim(-0.65, 4.75)
axb.set_ylim(-0.22, 1.14)
axb.set_xticks(range(5))
axb.set_xticklabels(XLBL, fontsize=7.0, color=TICK)
for lbl, col in zip(axb.get_xticklabels(), [CIS_DK, TICK, TICK, TICK, TR_DK]):
    lbl.set_color(col)
    if col != TICK:
        lbl.set_fontweight('bold')
axb.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axb.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.00'], fontsize=6.8, color=TICK)
axb.set_ylabel('copy-number co-variation\n(Spearman \u03c1)', fontsize=7.2,
               color=INK, labelpad=3.5)
axb.set_xlabel('genomic distance between the two genes', fontsize=7.2,
               color=INK, labelpad=3.0)
for sp in ('top', 'right'):
    axb.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axb.spines[sp].set_color(FRAME)
    axb.spines[sp].set_linewidth(0.8)
axb.tick_params(length=2.6, width=0.7, color=FRAME, pad=2)
axb.grid(axis='y', color='#EDF0F5', lw=0.7, zorder=0)
axb.set_axisbelow(True)

# group captions
axb.text(0.0, -0.155, 'cis edges', fontsize=7.0, color=CIS_DK, ha='center',
         va='center', fontweight='bold', zorder=7)
axb.text(4.0, -0.155, 'trans edges', fontsize=7.0, color=TR_DK, ha='center',
         va='center', fontweight='bold', zorder=7)
axb.text(2.0, -0.155, 'random gene pairs (2,000)', fontsize=6.6, color=TR_LT,
         ha='center', va='center', zorder=7)

# annotate the two decisive pairs
axb.annotate('CXCL9\u201311\n' + '\u03c1 = 1.00', xy=(0.06, 1.00),
             xytext=(0.62, 1.03), fontsize=6.6, color=CIS_DK, ha='left',
             va='center', zorder=8,
             arrowprops=dict(arrowstyle='-', color=CIS_DK, lw=0.7,
                             shrinkA=0, shrinkB=2))
axb.annotate('plasma-Ig axis\n' + '\u03c1 = 0.07', xy=(4.0, 0.069),
             xytext=(3.55, 0.31), fontsize=6.6, color=TR_DK, ha='left',
             va='center', zorder=8,
             arrowprops=dict(arrowstyle='-', color=TR_DK, lw=0.7,
                             shrinkA=0, shrinkB=3))
axb.annotate('C7/PLCXD3\n' + '\u03c1 = 0.98', xy=(-0.06, 0.984),
             xytext=(-0.58, 0.80), fontsize=6.6, color=CIS_DK, ha='left',
             va='center', zorder=8,
             arrowprops=dict(arrowstyle='-', color=CIS_DK, lw=0.7,
                             shrinkA=0, shrinkB=2))
axb.text(2.15, 1.055, 'dotted: 95th percentile of the random pairs',
         fontsize=6.5, color=TR_LT, ha='left', va='center', zorder=7)
# panel letter, in the strip's left margin so it lines up with panel a
axb.text(-0.092, 1.075, 'b', transform=axb.transAxes, fontsize=11,
         fontweight='bold', color=INK, ha='left', va='top', zorder=12,
         clip_on=False)

# ---- layout bookkeeping -- where the label ink actually lands, in mm ----------
# The circos is round, so no label points straight down; the real floor of the ink
# decides how close panel b may sit.  Printed rather than guessed.
fig.canvas.draw()
_box = [data_bbox(t) for t in label_objs]
V = [A_LEFT + (b[0] + 1.50) * BLK / 3.0 for b in _box] + \
    [A_LEFT + (b[2] + 1.50) * BLK / 3.0 for b in _box]
W_ = [A_BOT + (b[1] + 1.50) * BLK / 3.0 for b in _box] + \
     [A_BOT + (b[3] + 1.50) * BLK / 3.0 for b in _box]
print('label ink: x %.1f .. %.1f mm | y %.1f .. %.1f mm' %
      (min(V), max(V), min(W_), max(W_)))
print('panel b top %.1f mm  ->  gap under the ink %.1f mm'
      % (B_BOT + B_H, min(W_) - (B_BOT + B_H)))

# ============================================================ panel a key
# Measured, not guessed.  Every rectangle big enough for the key that sits in the
# top-left of the disc lands on a ribbon -- the chords leaving HBA1 / HBB / ADAM6
# fan across exactly that corner.  The only sizeable empty patch is the one just
# above the centre, which is where the key goes; parked there it covers no ribbon
# at all, whereas a 0.94-wide box at dead centre hides 252 sampled ribbon points.
# The report below is the guard rail: a label growing into the key, the key
# straddling the ring, or the key covering a chord all fail loudly.
kx, ky = -0.06, 0.26
kw, kh = 0.72, 0.28
ax.add_patch(FancyBboxPatch((kx - kw / 2, ky - kh / 2), kw, kh,
                            boxstyle='round,pad=0.012,rounding_size=0.032',
                            facecolor=BOXFC, edgecolor=BOXEC, linewidth=0.9,
                            alpha=0.97, zorder=9))
X0 = kx - 0.300


def keyblock(y, color, name, count, marker='o'):
    ax.plot([X0 + 0.016, X0 + 0.158], [y, y], color=color, lw=3.4, alpha=0.85,
            solid_capstyle='butt', zorder=10)
    ax.plot([X0 + 0.016], [y], marker=marker, markersize=5.8,
            markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.85,
            linestyle='none', zorder=11)
    xt = X0 + 0.192
    ax.text(xt, y, name, fontsize=9.0, color=INK, fontweight='bold',
            va='center', zorder=10)
    ax.text(xt + 0.235, y, count, fontsize=7.6, color=TICK, va='center', zorder=10)


keyblock(ky + 0.075, TR, 'trans', '%d edges' % DATA['n_trans'], marker='o')
keyblock(ky - 0.075, CIS, 'cis', '%d edges' % DATA['n_cis'], marker='s')

# ---- guard rail -------------------------------------------------------------
_kbox = (kx - kw / 2, ky - kh / 2, kx + kw / 2, ky + kh / 2)
_clash = [t.get_text().strip() for t in label_objs if box_hit(data_bbox(t), _kbox)]
_cov = sum(1 for x, y in CHORD
           if _kbox[0] <= x <= _kbox[2] and _kbox[1] <= y <= _kbox[3])
_rc = [math.hypot(kx + sx * kw / 2, ky + sy * kh / 2)
       for sx in (-1, 1) for sy in (-1, 1)]
_place = ('inside the ring' if max(_rc) < R_IN else
          'outside the ring' if min(_rc) > R_OUT else 'STRADDLING THE RING')
_edge = min(kx - kw / 2 + 1.50, 1.50 - (kx + kw / 2))
print('key box: %s | covers %d of %d ribbon samples | %s | canvas margin %.1f mm'
      % ('CLASH with ' + ', '.join(_clash) if _clash
         else 'clear of all %d gene labels' % len(label_objs),
         _cov, len(CHORD), _place, _edge * BLK / 3.0))
if _clash or _cov or _place != 'inside the ring':
    print('ABORT: the key no longer sits clear inside the ring')
    sys.exit(1)

# ============================================================ save
fig.savefig(os.path.join(FIG, 'Fig8.png'), dpi=400, facecolor='white')
fig.savefig(os.path.join(FIG, 'Fig8.pdf'), facecolor='white')
print('loci=%d  cis markers=%d  trans ribbons=%d' %
      (len(records), len(cis_pairs), DATA['n_trans']))
print('median rho by stratum: %s' % ['%.3f' % m for m in MED])
print('ring histogram:', [sum(1 for r in records if r['layer'] == i)
                          for i in range(len(LAYERS))])
print('figure %.0f x %.0f mm' % (W_MM, H_MM))
