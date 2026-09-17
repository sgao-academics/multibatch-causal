"""Figure 1: pan-cancer expression landscape of the inferred regulatory networks.

Every number plotted here is derived from the analysis outputs:
  * _fig1_landscape.npz        per-cancer TCGA expression sub-matrices (_prep_fig1_landscape.py)
  * _fig1_manifest.json        per-cancer sample-type annotation
  * _per_cancer_network.json   hub gene of each inferred network
  * _cancer_pair_sharing.json  cross-cancer sharing and composition of the shared set
  * _pipeline_notears.json     per-cancer weight matrices (locates each shared gene-pair)

The canvas is 174 mm x 194 mm (6.85 in x 7.637 in). The submission template (sn-jnl, sn-basic)
sets \textwidth = 130.8 mm, so the figure is included at width=\textwidth and every element is
rendered at 0.752x. In-figure type is therefore sized so that the *printed* size stays at or
above 5.5 pt (7.6 pt on the canvas x 0.752 = 5.71 pt printed). PRINT_SCALE below re-checks this
on every run. Panel letters only; the descriptions live in the LaTeX caption.
"""
import os, sys, json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Wedge, Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.text import Text as MplText

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gene_symbols import canonical, index

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE, 'results')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

TAU = 0.3
MIN_NORMAL = 5
JITTER_CAP = 220

# sn-jnl (sn-basic) sets \textwidth = 370.7 pt = 130.8 mm, while the canvas is 174 mm wide, so
# the figure is scaled by 130.8/174 on the page. Every type size is chosen so that its printed
# size after this scaling is at least PRINT_FLOOR pt.
PRINT_SCALE = 130.8 / 174.0
PRINT_FLOOR = 5.5

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    # The mathtext default is DejaVu Sans, which would drop the only non-Arial glyphs
    # (the minus sign, the >= comparison, the p-value exponents) into a second family.
    # Point the maths glyphs at Arial so the whole figure is one embedded family.
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'font.size': 7.6, 'axes.linewidth': 0.45, 'text.usetex': False,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'legend.frameon': False, 'legend.fontsize': 7.6,
})

RED, ORANGE = '#D7263D', '#E08214'
COL_TUM, COL_NOR, COL_MET = '#E45756', '#4C78A8', '#54A24B'
BODY, MUTED, GREY = '#1a1a1a', '#9a9a9a', '#6E6E6E'

# ------------------------------------------------------------------ data
z = np.load(os.path.join(RES, '_fig1_landscape.npz'), allow_pickle=True)
GENES = list(z['genes'])
# The matrix rows keep the network table's spelling, while the hub names below come from
# _per_cancer_network.json and may carry the HGNC-approved symbol. Map the canonical spelling to
# the row position, so a panel is never dropped for a renamed or differently-cased gene.
# See _gene_symbols.py.
GIDX = {canonical(g).upper(): i for i, g in enumerate(GENES)}
CANCERS = sorted(z['cancers'])
MAN = json.load(open(os.path.join(RES, '_fig1_manifest.json'), encoding='utf-8'))
NET = json.load(open(os.path.join(RES, '_per_cancer_network.json'), encoding='utf-8'))
SHARE = json.load(open(os.path.join(RES, '_cancer_pair_sharing.json'), encoding='utf-8'))
NT = json.load(open(os.path.join(RES, '_pipeline_notears.json'), encoding='utf-8'))


def expr(cancer, gene, grp):
    v = z['X_' + cancer][GIDX[canonical(gene).upper()]]
    g = np.array(MAN[cancer]['groups'])
    return v[np.isfinite(v) & (g == grp)]


recur = {}
for c in CANCERS:
    for g in set(NET[c]['genes']):
        recur[g] = recur.get(g, 0) + 1
TOP_GENE, TOP_REC = max(recur.items(), key=lambda kv: kv[1])

PAIR_CANCERS = {}
for c in CANCERS:
    W = np.array(NT[c]['W'], dtype=float)
    genes = NT[c]['genes']
    A = np.abs(W) > TAU
    np.fill_diagonal(A, False)
    for i, j in zip(*np.where(A)):
        PAIR_CANCERS.setdefault((genes[i], genes[j]), set()).add(c)
print('most recurrent network gene: %s (%d/33)' % (TOP_GENE, TOP_REC))

# ------------------------------------------------------------------ canvas
FW, FH = 6.85, 7.637                    # inches = 174 mm x 194 mm
fig = plt.figure(figsize=(FW, FH))


def panel_letter(letter, x, y):
    fig.text(x, y, letter, fontsize=11.0, fontweight='bold', color=BODY,
             ha='left', va='top')


# ================================================================== a)
AXA = [0.082, 0.770, 0.902, 0.186]
axa = fig.add_axes(AXA)
xs = np.arange(len(CANCERS))
star_x, star_y, star_t = [], [], []
n_tested = n_sig = 0
lab_colors = []
halfw = 0.46
for i, c in enumerate(CANCERS):
    t = expr(c, TOP_GENE, 'Tumor')
    n = expr(c, TOP_GENE, 'Normal')
    has_n = len(n) >= MIN_NORMAL
    # tumour is the left half when a normal group is available, otherwise a full violin
    groups = [(-1, t, COL_TUM), (1, n, COL_NOR)] if has_n else [(0, t, COL_TUM)]
    for side, data, col in groups:
        if data is None or len(data) < 3:
            continue
        vp = axa.violinplot([data], positions=[i], widths=2 * halfw,
                            showextrema=False, showmedians=False, bw_method=0.30,
                            points=120)
        verts = vp['bodies'][0].get_paths()[0].vertices
        if side < 0:
            verts[:, 0] = np.clip(verts[:, 0], i - halfw, i)
        elif side > 0:
            verts[:, 0] = np.clip(verts[:, 0], i, i + halfw)
        vp['bodies'][0].set_facecolor(col)
        vp['bodies'][0].set_edgecolor('none')
        vp['bodies'][0].set_alpha(0.92)
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        w = halfw * 0.62
        x1 = i - w if side < 0 else i
        x2 = i + w if side > 0 else i
        if side == 0:
            x1, x2 = i - w, i + w
        axa.add_patch(Rectangle((min(x1, x2), q1), abs(x2 - x1), max(q3 - q1, 1e-6),
                                facecolor='white', edgecolor='none', alpha=0.72, zorder=4))
        axa.plot([x1, x2], [med, med], color='white', lw=1.0, zorder=5,
                 solid_capstyle='butt')
    lab_colors.append(RED if TOP_GENE in set(NET[c]['genes']) else MUTED)
    if has_n:
        p = stats.mannwhitneyu(t, n, alternative='two-sided').pvalue
        n_tested += 1
        star = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'
        n_sig += int(p < 0.05)
        star_x.append(i); star_t.append(star)

top = max(np.percentile(expr(c, TOP_GENE, 'Tumor'), 99.5) for c in CANCERS)
axa.set_ylim(0, top * 1.20)
for i, s in zip(star_x, star_t):
    axa.text(i, top * 1.10, s, ha='center', va='bottom', fontsize=7.6,
             color=BODY if s != 'ns' else MUTED, fontweight='bold' if s != 'ns' else 'normal')
axa.set_xlim(-0.85, len(CANCERS) - 0.15)
axa.set_xticks(xs)
axa.set_xticklabels(CANCERS, rotation=90, fontsize=7.6)
for lbl, col in zip(axa.get_xticklabels(), lab_colors):
    lbl.set_color(col)
axa.set_ylabel('%s expression\nlog$_2$(TPM+1)' % TOP_GENE, fontsize=8.4, labelpad=1.5)
axa.tick_params(axis='x', length=1.8, pad=1.2)
axa.tick_params(axis='y', length=1.8, pad=1.2, labelsize=7.6)
for sp in ('top', 'right'):
    axa.spines[sp].set_visible(False)
axa.legend(handles=[Patch(facecolor=COL_TUM, label='Tumour'),
                    Patch(facecolor=COL_NOR, label='Adjacent normal'),
                    Patch(facecolor=RED, label='red cohort label = %s in that network' % TOP_GENE)],
           loc='upper left', bbox_to_anchor=(0.0, 1.16), ncol=3,
           handlelength=1.0, handletextpad=0.35, columnspacing=1.1, fontsize=7.6)
panel_letter('a', 0.050, 0.976)

# ================================================================== b)
HUB_ROWS = [c for c in CANCERS if MAN[c]['n_normal'] >= MIN_NORMAL]
tiles = HUB_ROWS[:18]
PB_BOT, PB_TOP = 0.248, 0.686
nrow, ncol = 6, 3
pitch = (PB_TOP - PB_BOT) / nrow
AXH = pitch * 0.720
TW = 0.902 / (ncol + (ncol - 1) * 0.34)
tile_stats = []
for k, c in enumerate(tiles):
    r, cc = k // ncol, k % ncol
    x0 = 0.082 + cc * TW * 1.34
    y0 = PB_TOP - (r + 1) * pitch + (pitch - AXH)
    ax = fig.add_axes([x0, y0, TW, AXH])
    hub = NET[c]['hub']
    series = [('T', COL_TUM, expr(c, hub, 'Tumor')),
              ('N', COL_NOR, expr(c, hub, 'Normal'))]
    if MAN[c]['n_metastatic'] >= 3:
        series.append(('M', COL_MET, expr(c, hub, 'Metastatic')))
    rng = np.random.default_rng(20260915)
    vals = [s[2] for s in series]
    for si, (lab, col, v) in enumerate(series):
        if not len(v):
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        lo, hi = np.min(v), np.max(v)
        iqr = q3 - q1
        wlo = max(lo, q1 - 1.5 * iqr)
        whi = min(hi, q3 + 1.5 * iqr)
        ax.add_patch(Rectangle((si - 0.26, q1), 0.52, max(q3 - q1, 1e-6),
                               facecolor=col, edgecolor='white', lw=0.35, zorder=3))
        ax.plot([si, si], [whi, q3], color=col, lw=0.5, zorder=3)
        ax.plot([si, si], [q1, wlo], color=col, lw=0.5, zorder=3)
        ax.plot([si - 0.12, si + 0.12], [whi, whi], color=col, lw=0.5, zorder=3)
        ax.plot([si - 0.12, si + 0.12], [wlo, wlo], color=col, lw=0.5, zorder=3)
        ax.plot([si - 0.26, si + 0.26], [med, med], color='white', lw=0.9, zorder=4)
        show = v if len(v) <= JITTER_CAP else rng.choice(v, JITTER_CAP, replace=False)
        ax.scatter(si + rng.uniform(-0.21, 0.21, len(show)), show, s=0.35,
                   color='#2b2b2b', alpha=0.28, linewidths=0, zorder=5)
    groups = [v for v in vals if len(v) >= 2]
    if len(groups) >= 3:
        p = stats.kruskal(*groups).pvalue; test = 'KW'
    else:
        p = stats.mannwhitneyu(vals[0], vals[1], alternative='two-sided').pvalue; test = 'MW'
    fc = float(np.median(vals[0]) - np.median(vals[1]))
    star = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'
    tile_stats.append((c, hub, p, fc, star, test, [len(v) for v in vals]))
    ax.set_xlim(-0.62, len(series) - 0.38)
    ax.set_xticks([])
    lo = min(np.min(v) for v in vals if len(v)); hi = max(np.max(v) for v in vals if len(v))
    pad = 0.18 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad * 3.8)
    ax.set_title('%s \u00b7 %s' % (c, hub), fontsize=8.6, fontweight='bold', loc='left', pad=1.8)
    box = dict(facecolor='white', edgecolor='none', alpha=0.78, pad=0.7)
    if p == 0:
        ptxt = '$p$ < $10^{-300}$'
    elif p < 1e-2:
        ptxt = '$p$ = %.3g' % p
    else:
        ptxt = '$p$ = %.3f' % p
    ax.text(1.0, 0.99, '%s  %s' % (star, ptxt), transform=ax.transAxes, ha='right',
            va='top', fontsize=7.6, color=BODY, bbox=box,
            fontweight='bold' if star != 'ns' else 'normal')
    ax.text(0.0, 0.99, 'log$_2$FC %+.2f' % fc, transform=ax.transAxes, ha='left', va='top',
            fontsize=7.6, color=BODY, bbox=box)
    if cc == 0:
        ax.set_ylabel('log$_2$(TPM+1)', fontsize=7.6, labelpad=1.0)
    # the tiles are only ~31 pt tall, so three tick labels would collide: keep two
    ax.yaxis.set_major_locator(MaxNLocator(2, prune='both'))
    ax.tick_params(axis='y', labelsize=7.6, length=1.4, pad=0.8)
    ax.tick_params(axis='x', length=1.4, pad=0.8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
panel_letter('b', 0.050, 0.722)
fig.legend(handles=[Patch(facecolor=COL_TUM, label='T  tumour'),
                    Patch(facecolor=COL_NOR, label='N  normal'),
                    Patch(facecolor=COL_MET, label='M  metastasis')],
           loc='upper right', bbox_to_anchor=(0.984, 0.726), ncol=3,
           handlelength=0.9, handletextpad=0.35, columnspacing=1.2, fontsize=7.6)
n_hub_sig = sum(1 for t in tile_stats if t[4] != 'ns')

# ================================================================== c)
# frame height is bounded: the sector labels stick out 15 pt (top, "ACC") and 21 pt (bottom,
# "LUSC"/"MESO") beyond the ring, and the whole stack must fit between the canvas floor and the
# bottom of the last panel-b tile row. 115 pt of frame height is the largest safe value.
axc = fig.add_axes([0.100, 0.0345, 0.82, 0.209])
axc.set_aspect('equal'); axc.axis('off')
comp = SHARE['shared_ge3_pairs']
chords = []
for kind, col in [('sex_chromosome', RED), ('same_family', ORANGE)]:
    for a, b in comp[kind]:
        cs = sorted(PAIR_CANCERS.get((a, b), PAIR_CANCERS.get((b, a), set())))
        for ii in range(len(cs)):
            for jj in range(ii + 1, len(cs)):
                chords.append((CANCERS.index(cs[ii]), CANCERS.index(cs[jj]), col))
ang = {c: np.pi / 2 - 2 * np.pi * i / len(CANCERS) for i, c in enumerate(CANCERS)}
R_OUT = 1.0
sector = 2 * np.pi / len(CANCERS)
for c in CANCERS:
    t = ang[c]
    axc.add_patch(Wedge((0, 0), R_OUT, np.degrees(t - sector * 0.42), np.degrees(t + sector * 0.42),
                        width=0.075, facecolor='#C4C4C4', edgecolor='white', lw=0.35))
for i, j, col in chords:
    t1, t2 = ang[CANCERS[i]], ang[CANCERS[j]]
    p1 = (R_OUT * 0.94 * np.cos(t1), R_OUT * 0.94 * np.sin(t1))
    p2 = (R_OUT * 0.94 * np.cos(t2), R_OUT * 0.94 * np.sin(t2))
    path = Path([p1, (0.10 * np.cos(t1), 0.10 * np.sin(t1)),
                 (0.10 * np.cos(t2), 0.10 * np.sin(t2)), p2],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    axc.add_patch(PathPatch(path, facecolor='none', edgecolor=col, lw=0.55, alpha=0.55))
for c in CANCERS:
    t = ang[c]
    deg = np.degrees(t)
    rr = R_OUT * 1.075
    axc.text(rr * np.cos(t), rr * np.sin(t), c, fontsize=7.6,
             rotation=deg if -90 <= deg <= 90 else deg + 180, rotation_mode='anchor',
             ha='left' if -90 <= deg <= 90 else 'right', va='center', color='#666666')
# Every chord of the bundle crosses the middle of the ring, so a label at the centre sits on top
# of the connections. The headline count therefore goes into the blank band to the right of the
# ring, level with its centre: the sector labels reach x = 0.661, so 0.661-0.05 is free.
fig.text(0.8055, 0.139, '%d of %s pairs\nrecur in $\\geq$3 cancers'
         % (SHARE['n_shared_ge3'], format(SHARE['total_unique_pairs'], ',')),
         ha='center', va='center', fontsize=8.6, fontweight='bold', color=BODY, linespacing=1.5)
# fig.legend, not axc.legend: set_aspect('equal') shrinks the axes box, so axes fractions
# would put the legend on top of the chords.
fig.legend(handles=[Patch(color=RED, label='sex-chromosome / XCI (%d)'
                          % len(comp['sex_chromosome'])),
                    Patch(color=ORANGE, label='same gene family (%d)'
                          % len(comp['same_family'])),
                    Patch(color='#C4C4C4', label='one sector = one cohort')],
           loc='center left', bbox_to_anchor=(0.104, 0.139), ncol=1,
           handlelength=0.9, handletextpad=0.4, labelspacing=0.45, fontsize=7.6)
axc.set_xlim(-1.40, 1.40); axc.set_ylim(-1.16, 1.16)
panel_letter('c', 0.050, 0.244)

# ------------------------------------------------------------------ checks
_minfs = min(t.get_fontsize() for t in fig.findobj(MplText) if t.get_text().strip())
assert _minfs * PRINT_SCALE >= PRINT_FLOOR - 1e-6, \
    'smallest in-figure type is %.1f pt, which prints at %.2f pt (floor %.1f)' \
    % (_minfs, _minfs * PRINT_SCALE, PRINT_FLOOR)
assert len(chords) > 0, 'no chords drawn'
assert n_tested >= 15, 'too few tumour/normal tests: %d' % n_tested
assert n_hub_sig >= 8, 'too few significant hub tiles: %d' % n_hub_sig
assert SHARE['n_shared_ge3'] == 19 and len(comp['sex_chromosome']) == 12
assert len(tiles) == 18, 'expected 18 tiles, got %d' % len(tiles)

pdf = os.path.join(FIGDIR, 'Fig1.pdf')
fig.savefig(pdf); fig.savefig(os.path.join(FIGDIR, 'Fig1.png')); plt.close(fig)
print('panel a: %s in %d/33 networks | tested=%d significant=%d'
      % (TOP_GENE, TOP_REC, n_tested, n_sig))
print('panel b: %d tiles | significant=%d' % (len(tiles), n_hub_sig))
for t in tile_stats:
    print('   %-5s %-12s %-3s p=%.3g  log2FC=%+.2f  n=%s' % (t[0], t[1], t[4], t[2], t[3], t[6]))
print('panel c: %d chords from %d shared pairs' % (len(chords), SHARE['n_shared_ge3']))
print('smallest in-figure type %.1f pt -> %.2f pt printed | canvas %.0f x %.0f mm'
      % (_minfs, _minfs * PRINT_SCALE, FW * 25.4, FH * 25.4))
print('saved %s (%.0f KB)' % (pdf, os.path.getsize(pdf) / 1024))
