# -*- coding: utf-8 -*-
"""Figure 1 of the manuscript, four panels, on one muted cool palette.

(a) element-wise median adjacency matrix, aligned against misaligned (one shared colourbar)
(b) edges per cohort against sample size, with the shuffled control
(c) the 14 recurring pairs, coloured by non-malignant axis
(d) left: per-cohort PC1-vs-stroma/immune correlation, cohort by cohort
    right: driver-gene entry against panel size

The layout language, rather than the palette, is what changed in this revision.
Every panel now sits on a faint rounded card; bars are capsules resting on a very
light track; markers carry a soft drop shadow instead of a hard outline; the
annotation blocks are rounded containers; and all type is set in the palette's
indigo rather than black.  The canvas stays 174 mm wide with an 8 pt base size,
panels are marked with lower-case letters, and no title is drawn inside the
artwork (the manuscript carries the caption).

Every number the first version printed is still printed here; the panel-c legend
follows the axis wording of Table 2 exactly.

Output: figures/Fig1_main.pdf / .png
"""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.transforms import offset_copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gene_symbols import canonical  # legacy spelling -> the HGNC symbol the paper prints

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA = os.path.join(_ROOT, 'data')
RES = os.path.join(_ROOT, 'results')
PANELS = os.path.join(DATA, 'panels')
FIG = os.path.join(_ROOT, 'figures')

OUT_PNG = os.path.join(FIG, 'Fig1_main.png')
OUT_PDF = os.path.join(FIG, 'Fig1_main.pdf')
TAU = 0.3

MM = 1.0 / 25.4  # mm -> inch
W = 174.0 * MM   # canvas width, matching every other figure in the package
H = 174.0 * MM * 0.88

# --------------------------------------------------------------------------
# One palette for the whole figure, taken from the reference swatch strip.
# --------------------------------------------------------------------------
PAL = {
    'mist':   '#4D6EAF',   # pale mist blue
    'peri':   '#7A9AC3',   # periwinkle blue
    'lilac':  '#B0B9CB',   # lilac lavender
    'orchid': '#D0A0B6',   # orchid pink
    'moss':   '#8FB578',   # soft moss green
    'violet': '#9A5A9F',   # muted violet purple
    'indigo': '#3F3A5B',   # indigo plum
}
# monotone cool ramp for the matrices; the floor is a faint tint rather than pure
# white, so an all-zero matrix still reads as a matrix and not as missing artwork
CMAP = LinearSegmentedColormap.from_list(
    'mist_blue', ['#E9EEF5', '#C6D3E5', '#A9BFD9', '#7A9AC3', '#5A6A94', '#3F3A5B'])

# --- design tokens ---------------------------------------------------------
# Type is indigo plum, not black; cards and containers are a few percent away
# from the paper white; the grid is barely there.  This is what makes the sheet
# read as one designed figure rather than four plots drawn with the defaults.
INK = '#3F3A5B'      # all text, from the palette's deepest colour
TICK = '#5B6478'     # tick labels, one step lighter than the text
GREY = '#8A93A6'     # de-emphasised labels
GRID = '#EDF1F7'     # grid
TRACK = '#EFF3F8'    # capsule track behind the panel-c bars
FRAME = '#C9D2E0'    # spines
BOXFC = '#FBFCFE'    # rounded annotation container fill
BOXEC = '#DDE4EF'    # rounded annotation container edge
CARD_FC = '#FAFBFE'  # faint rounded card behind each panel
CARD_EC = '#E7EDF6'
SHADOW = '#AEB8CC'   # drop shadow under markers

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    # Type 42 = embed as TrueType.  Left at the default, matplotlib writes Type 3,
    # which the journal's production chain rejects outright.
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    # the one symbol Arial lacks is the \gtrsim in the driver-entry note; taking it
    # from Computer Modern makes the artwork agree with the manuscript's own glyph
    'mathtext.fallback': 'cm',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.linewidth': 0.6,
    'axes.edgecolor': FRAME, 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': TICK, 'ytick.color': TICK,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.4, 'ytick.major.size': 2.4,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 400, 'savefig.dpi': 400, 'text.usetex': False,
})


def tidy(ax, grid=None):
    """One axis language for every panel: open frame, thin ticks, faint grid."""
    ax.tick_params(direction='out', length=2.4, width=0.6)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def card(fig, ax, pad=0.011, radius=0.016, z=-4):
    """A faint rounded container behind a panel -- the reference artwork's habit."""
    bb = ax.get_position()
    fig.add_artist(FancyBboxPatch(
        (bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad, bb.height + 2 * pad,
        boxstyle='round,pad=0,rounding_size=%.4f' % radius,
        transform=fig.transFigure, fc=CARD_FC, ec=CARD_EC, lw=0.7, zorder=z))


def px_per_data(ax):
    """Display pixels per data unit, in x and y, after the figure has been drawn."""
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((1.0, 1.0))
    return (p1[0] - p0[0]), (p1[1] - p0[1])


def capsule(ax, x0, x1, y, h, color, z=3):
    """A horizontal capsule (rounded-end bar) in data coordinates.

    The cap radius is chosen so the ends stay circular on screen regardless of
    how the x and y axes are scaled -- a rounded rectangle drawn straight in
    data units would come out elliptical here.
    """
    sx, sy = px_per_data(ax)
    d_px = abs(h) * sy
    rad = (d_px / 2.0) / sx
    w = x1 - x0
    if w <= 2.0 * rad:
        rad = w / 2.0
    ax.add_patch(Rectangle((x0 + rad, y - h / 2.0), max(w - 2.0 * rad, 1e-9), h,
                           fc=color, ec='none', zorder=z))
    if w > 1e-9:
        d_pt = d_px * 72.0 / ax.figure.dpi
        ax.scatter([x0 + rad, x1 - rad], [y, y], s=d_pt * d_pt,
                   c=color, edgecolors='none', zorder=z)


def dots(ax, x, y, s, color, z=4, shadow=True, ec='white', lw=0.6,
         marker='o', alpha=1.0):
    """Markers with a soft drop shadow instead of a heavy outline."""
    if shadow:
        tr = offset_copy(ax.transData, fig=ax.figure, x=0.9, y=-0.9, units='points')
        ax.scatter(x, y, s=s * 1.04, c=SHADOW, alpha=0.32, edgecolors='none',
                   marker=marker, zorder=z - 0.5, transform=tr, clip_on=True)
    return ax.scatter(x, y, s=s, c=color, edgecolors=ec, linewidths=lw,
                      marker=marker, zorder=z, alpha=alpha)


def haloed_text(ax, x, y, s, size=8, color=INK, halo='white', hw=2.2, z=5, **kw):
    """Text with a soft halo, drawn as two stacked objects.

    Path effects on text rasterise the glyphs to outlines, which would break any
    later text extraction from the PDF, so only the underlying halo copy is
    stroked (that copy is what stops the capsule track showing through the
    letters); the copy on top is ordinary text and stays extractable.
    """
    ax.text(x, y, s, fontsize=size, color=halo, zorder=z,
            path_effects=[pe.withStroke(linewidth=hw, foreground=halo)], **kw)
    return ax.text(x, y, s, fontsize=size, color=color, zorder=z + 0.1, **kw)


full = json.load(open(os.path.join(RES, '_recurrence_analysis.json'), encoding='utf-8'))
attr = json.load(open(os.path.join(RES, '_axis_attribution.json'), encoding='utf-8'))
neg = json.load(open(os.path.join(RES, '_shuffle_control.json'), encoding='utf-8'))

# Keys are matched against the spellings carried by the result matrices, so the legacy form
# IL8 has to stay here; only what is printed goes through canonical().  The third field is
# the legend label, worded exactly as the axis column of Table 2.
AXIS = {
    'IFN-\u03b3 / CXCR3 chemokine': (['CXCL9', 'CXCL10', 'CXCL11'], PAL['mist'],
                                     'Interferon-$\\gamma$ / CXCR3'),
    'neutrophil / CXCR2 chemokine': (['CXCL1', 'IL8'], PAL['orchid'], 'Neutrophil / CXCR2'),
    'erythrocyte content': (['HBA1', 'HBB'], PAL['lilac'], 'Erythrocyte content'),
    'plasma-cell / Ig locus': (['ADAM6', 'IGJ', 'MGC29506'], PAL['violet'],
                               'Plasma-cell / Ig locus'),
    'stromal / complement': (['COL10A1', 'COL11A1', 'C7', 'PLCXD3'], PAL['moss'],
                             'Stroma / complement'),
    'myeloid / cytokine': (['CCL18', 'CHIT1', 'IL6', 'FOSB'], PAL['peri'], 'Myeloid / cytokine'),
}


def axis_of(g):
    for k, (gs_, c, lab) in AXIS.items():
        if g in gs_:
            return k, c, lab
    return 'other', '#9A9A9A', 'other'


def load(p):
    d = json.load(open(p, encoding='utf-8'))
    cs = sorted(k for k, v in d.items() if isinstance(v, dict) and 'W' in v)
    return d, cs


def medmat(store, cs):
    M = np.stack([np.abs(np.array(store[c]['W'], dtype=float)) for c in cs], 0)
    m = np.median(M, 0)
    np.fill_diagonal(m, 0)
    return m


sh, cs = load(os.path.join(RES, '_shared_panel_notears.json'))
old, cs2 = load(os.path.join(RES, '_pipeline_notears.json'))
meda = medmat(sh, cs)
medb = medmat(old, [c for c in cs2 if np.array(old[c]['W']).shape[0] == 100])

fig = plt.figure(figsize=(W, H))
gs = gridspec.GridSpec(2, 2, hspace=0.62, wspace=0.30,
                       left=0.100, right=0.985, top=0.930, bottom=0.090)

# ---------------------------------------------------------------- axes
axa = fig.add_subplot(gs[0, 0])
axa.axis('off')
# (a) is two stacked readings of the SAME 9,900 numbers.  On top, the two
# adjacency matrices on one shared colour scale; underneath, those same medians
# sorted and ranked.  The matrices alone are nearly uninformative -- the median
# |coefficient| sits at ~0.02 while the maximum is 0.73, so against a scale that
# has to reach 0.8 the misaligned panel is a flat field and the aligned one is a
# flat field with six specks.  The ranked curve is what makes the difference
# legible: the two designs stay an order of magnitude apart over most of the range.
# hspace carries the colourbar's own tick labels.  At 0.30 the band was 3.9 mm --
# thinner than one line of type -- and lifting the label floor to 8 pt pushed the
# ranked curve's caption straight into the "0" and "0.4" ticks above it.
sub = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[0, 0],
                                       height_ratios=[21, 1.0, 17], hspace=0.58, wspace=0.20)
mat_axes = [fig.add_subplot(sub[0, k]) for k in (0, 1)]
cbar_ax = fig.add_subplot(sub[1, :])
axdist = fig.add_subplot(sub[2, :])
axb = fig.add_subplot(gs[0, 1])
axc = fig.add_subplot(gs[1, 0])
subd = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.56,
                                        width_ratios=[1.10, 1.0])
axd1 = fig.add_subplot(subd[0])
axd2 = fig.add_subplot(subd[1])

# Rounded cards go in first, behind everything; the layout has to be final
# before the capsule radii can be worked out, so draw once here.
for _cax in (axa, axb, axc, axd1, axd2):
    card(fig, _cax)
fig.canvas.draw()

# ================= (a) =================
for k, (M, ttl) in enumerate([(meda, 'aligned panel (this work)'),
                              (medb, 'misaligned panel (original)')]):
    a = mat_axes[k]
    im = a.imshow(M, cmap=CMAP, vmin=0, vmax=0.8, interpolation='nearest')
    for s_ in a.spines.values():
        s_.set_visible(True); s_.set_color(FRAME); s_.set_linewidth(0.6)
    a.set_xticks([]); a.set_yticks([])
    a.set_title(ttl, fontsize=8, pad=3.6, color=INK)

cb = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
# Both matrices are drawn on THIS scale.  Ticks are kept to three and the caption
# is one short line, because the strip has to stay thin: every millimetre it takes
# here comes straight out of the ranked curve below.
cb.set_ticks([0.0, 0.4, 0.8])
cb.set_ticklabels(['0', '0.4', '0.8'])
# Label above the bar: below it there is only the hspace before the ranked curve,
# which is where a horizontal colourbar would otherwise drop its caption.
cb.ax.xaxis.set_label_position('top')
cb.ax.xaxis.set_ticks_position('bottom')
cb.set_label('median$_c$ $|W_{ij}|$   $\\cdot$   shared scale for both',
             fontsize=8, labelpad=1.8, color=INK)
cb.ax.tick_params(labelsize=8, length=1.8, width=0.5)
cb.outline.set_linewidth(0.5)
cb.outline.set_edgecolor(FRAME)

# --- the ranked reading of the very same numbers --------------------------
# Ranked on a log x, value on a log y: the horizontal distance between the two
# curves is the effect size, and the threshold line shows how many entries the
# aligned design actually lifts above tau.
occ = ~np.eye(meda.shape[0], dtype=bool)
rank = np.arange(1, int(occ.sum()) + 1)
srt_a = np.sort(meda[occ])[::-1]
for M, col, lab in [(meda, PAL['mist'], 'aligned panel'),
                    (medb, PAL['lilac'], 'misaligned panel')]:
    s = np.sort(M[occ])[::-1]
    axdist.plot(rank, np.maximum(s, 1e-9), '-', c=col, lw=1.4, zorder=3,
                solid_capstyle='round', label=lab)
axdist.axhline(TAU, ls=(0, (4, 2)), c=PAL['indigo'], lw=0.9, alpha=0.5, zorder=2)
n_above = int((srt_a > TAU).sum())
axdist.plot([n_above], [srt_a[n_above - 1]], 'o', ms=3.4, mfc=PAL['mist'],
            mec='white', mew=0.7, zorder=6)
# The count goes at the RIGHT end of the threshold line, not next to the crossing
# dot: near the dot it ran straight into the curve's own caption (measured 45 x 6 pt
# of overlap).  Out here the region is empty -- both curves are below 0.02.
haloed_text(axdist, rank.size * 0.97, TAU * 1.33,
            '%d pairs above $\\tau=0.3$' % n_above, size=8,
            ha='right', va='bottom', color=INK, z=7)
axdist.set_xscale('log'); axdist.set_yscale('log')
axdist.set_xlim(1, rank.size)
# Headroom to 2.0 is deliberate: the curves top out at 0.731, and the legend needs
# a band above them.  At a 1.35 ceiling the legend's lower edge cut into the
# aligned curve between rank 6 and rank 60.
axdist.set_ylim(3.2e-3, 2.0)
axdist.set_xticks([1, 10, 100, 1000])
axdist.set_xticklabels(['1', '10', '100', '1,000'])
axdist.set_yticks([0.01, 0.1, 1])
axdist.set_yticklabels(['0.01', '0.1', '1'])
axdist.set_xlabel('off-diagonal pair, ranked (9,900 in all)', labelpad=3.0)
axdist.set_ylabel('median$_c$ $|W_{ij}|$', labelpad=2.2)
tidy(axdist, grid='both')
# Direct labels instead of a legend box.  A box big enough for two 8 pt entries
# is 7 mm tall, and this axes is only ~16 mm: wherever it went it either cut into
# a curve or hid the threshold line.  Naming each curve where it is already clear
# costs no ground at all -- the top strip is above both curves across the whole
# panel, and the bottom strip is below the misaligned curve out to rank ~200.
# Kept close to the curve's own ceiling (0.731): lifted much higher the caption
# ran into the colourbar's tick labels sitting in the gap above this axes.
haloed_text(axdist, 1.06, meda.max() * 1.06, 'aligned panel (max %.3f)' % meda.max(),
            size=8, ha='left', va='bottom', color=PAL['mist'], z=7)
haloed_text(axdist, 1.06, medb.max() * 0.54, 'misaligned panel (max %.3f)' % medb.max(),
            size=8, ha='left', va='top', color=GREY, z=7)

# ================= (b) =================
ns = np.array([sh[c]['n'] for c in cs], float)
es = np.array([np.sum(np.abs(np.array(sh[c]['W'], dtype=float)) > TAU) for c in cs], float)
lo = ns < 200

sl, ic = np.polyfit(np.log(ns), es, 1)
xx = np.geomspace(ns.min(), ns.max(), 160)
sd = (es - (sl * np.log(ns) + ic)).std()
axb.fill_between(xx, sl * np.log(xx) + ic - 1.96 * sd, sl * np.log(xx) + ic + 1.96 * sd,
                 color=PAL['orchid'], alpha=0.13, lw=0, zorder=1)
axb.plot(xx, sl * np.log(xx) + ic, '--', c=PAL['orchid'], lw=1.2, alpha=0.95, zorder=2)
m = ns >= 200
sl2, ic2 = np.polyfit(np.log(ns[m]), es[m], 1)
x2 = np.geomspace(205, ns.max(), 100)
sd2 = (es[m] - (sl2 * np.log(ns[m]) + ic2)).std()
axb.fill_between(x2, sl2 * np.log(x2) + ic2 - 1.96 * sd2, sl2 * np.log(x2) + ic2 + 1.96 * sd2,
                 color=PAL['peri'], alpha=0.15, lw=0, zorder=1)
axb.plot(x2, sl2 * np.log(x2) + ic2, '--', c=PAL['peri'], lw=1.2, alpha=0.95, zorder=2)

pc = neg['per_cohort']
sc_lt = dots(axb, ns[lo], es[lo], 26, PAL['orchid'], z=5)
sc_ge = dots(axb, ns[~lo], es[~lo], 26, PAL['peri'], z=5)
sc_sh = dots(axb, [x['n'] for x in pc], [x['shuffled'] for x in pc], 20, 'none', z=4,
             ec=PAL['indigo'], lw=0.8, marker='s')

axb.axvline(100, ls=(0, (2, 2)), c=GREY, lw=0.8, zorder=3)
axb.axvline(172, ls=(0, (5, 2)), c=PAL['indigo'], lw=1.0, zorder=3)
axb.set_xscale('log')
axb.set_xlim(30, 3000)   # explicit, so the annotation blocks can be placed on known ground
# The floor is set well below zero so the two reference-line captions have a band
# of their own.  At a -12 floor they sat within a marker's height of the shuffled
# control squares lying on y = 0, and "n*" disappeared behind them.
axb.set_ylim(-30, max(es.max(), 160) * 1.45)
axb.set_yticks([0, 50, 100, 150, 200])
# The two reference lines are only 0.24 decades apart on this axis, so their labels
# have to grow outward: the $n=d$ caption sits to the left of its line and the
# boundary caption to the right of its own, otherwise the pair runs together.
axb.text(96, -18, '$n=d=100$', fontsize=8, va='center', ha='right', color=GREY)
axb.text(195, -18, '$n^{*}=172$', fontsize=8, va='center', ha='left',
         color=PAL['indigo'], fontweight='bold')
axb.set_xlabel('cohort sample size $n$ (log scale)')
axb.set_ylabel('edges inferred by NOTEARS')
tidy(axb, grid='both')
# The third statistic is wrapped onto a second line so the block stays narrow
# enough to sit entirely in the empty upper-right corner: the wide single-line
# version reached down into the salmon cloud and clipped one of its markers.
axb.text(0.985, 0.975,
         'all 33:  $r=%.3f$ ($p=%.0e$)\n$n\\geq200$:  $r=%.3f$ ($p=%.2f$)\n'
         'shuffled:  $r=%.3f$,\nnone above $n^{*}=172$'
         % (full['panel_identifiability_all']['r'], full['panel_identifiability_all']['p'],
            full['panel_identifiability_n200']['r'], full['panel_identifiability_n200']['p'],
            neg['shuffled']['r_all']),
         transform=axb.transAxes, va='top', ha='right', fontsize=8, linespacing=1.55,
         color=INK, zorder=7,
         bbox=dict(boxstyle='round,pad=0.42', fc=BOXFC, ec=BOXEC, lw=0.6, alpha=0.92))
# Centre-right of this panel is empty ground (checked against the marker
# positions); the lower-right corner was not, and the legend was sitting on the
# row of shuffled-control squares there.
axb.legend([sc_lt, sc_ge, sc_sh],
           ['real data, $n<200$', 'real data, $n\\geq200$', 'shuffled control'],
           fontsize=8, loc='center right', bbox_to_anchor=(0.995, 0.46),
           frameon=True, framealpha=0.94, edgecolor=BOXEC, facecolor=BOXFC,
           handletextpad=0.5, borderpad=0.5, labelspacing=0.38)

# ================= (c) =================
edges = sorted(full['aligned']['shared_edges'], key=lambda e: -e['n_cohorts'])
ys = np.arange(len(edges))[::-1]
# Limits first: the capsule radius is derived from the pixels-per-data-unit, so
# the axes have to be on their final scale before anything is drawn onto them.
axc.set_yticks([])
axc.set_xlim(0, 44)
# The threshold caption used to sit at the foot of the dashed line, where it ran
# into the italic summary below; it now sits on top of the line, which needs a
# little headroom above the first bar.
axc.set_ylim(-2.8, len(edges) + 0.95)
axc.set_xticks([0, 10, 20, 30, 40])
axc.set_xlabel('cohorts (of 33) in which the edge recurs')
tidy(axc, grid=None)
for y, e in zip(ys, edges):
    k, c, lab = axis_of(e['src'])
    capsule(axc, 0.0, 44.0, y, 0.66, TRACK, z=2)               # the light track
    capsule(axc, 0.0, float(e['n_cohorts']), y, 0.66, c, z=3)  # the value
    haloed_text(axc, e['n_cohorts'] + 0.7, y,
                '%s$\\rightarrow$%s' % (canonical(e['src']), canonical(e['dst'])),
                va='center', z=5)
axc.axvline(10, ls=(0, (4, 2)), c=PAL['indigo'], lw=0.9, alpha=0.55, zorder=4)
hs = [Rectangle((0, 0), 1, 1, fc=c, ec='none') for k, (g, c, lab) in AXIS.items()]
leg = axc.legend(hs, [lab for k, (g, c, lab) in AXIS.items()], fontsize=8,
                 loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.045), frameon=True,
                 facecolor=BOXFC, edgecolor=BOXEC, framealpha=0.94,
                 handlelength=0.62, handletextpad=0.4, columnspacing=1.0,
                 borderpad=0.45, labelspacing=0.42)
haloed_text(axc, 10.35, len(edges) + 0.24, 'threshold 10/33', color=PAL['indigo'],
            ha='left', va='center', z=6)
axc.text(0.985, 0.015, '14 edges  |  6 non-malignant axes  |  0 canonical drivers',
         transform=axc.transAxes, fontsize=8, style='italic', ha='right', va='bottom',
         color=GREY)

# ================= (d) =================
r_nm = np.array([x['r_PC1_nonmal'] for x in attr['purity_attribution']['per_cohort']])
rng = np.random.default_rng(7)
yj = rng.uniform(0.02, 0.16, size=r_nm.size)
mu = r_nm.mean()
sd_r = r_nm.std()
axd1.axvspan(mu - sd_r, mu + sd_r, color=PAL['peri'], alpha=0.10, lw=0, zorder=0)
axd1.axhline(0, color='#C4C4C4', lw=0.7, zorder=1)
axd1.plot([mu, mu], [0, 0.20], color=PAL['orchid'], lw=1.2, ls=(0, (4, 2)), zorder=2)
dots(axd1, r_nm, yj, 22, PAL['peri'], z=3)
# Anchored on the left of the mean line: on the right it overran the axes and
# collided with the neighbouring panel's y-axis label.
haloed_text(axd1, mu - 0.015, 0.205, 'mean', va='bottom', ha='right', z=6)

# A strip of 33 points shows where the cohorts land but not how tightly; a flat
# box above it carries the median and the quartiles, which is the reading the
# panel is actually about (33 of 33 positive, spread ~0.76 -- 0.86).
_bq1, _bmed, _bq3 = np.percentile(r_nm, [25, 50, 75])
_yb, _hh = 0.315, 0.036
axd1.plot([r_nm.min(), r_nm.max()], [_yb, _yb], color=PAL['indigo'], lw=0.7, zorder=3)
for _xv in (r_nm.min(), r_nm.max()):
    axd1.plot([_xv, _xv], [_yb - 0.024, _yb + 0.024], color=PAL['indigo'], lw=0.7, zorder=3)
axd1.add_patch(Rectangle((_bq1, _yb - _hh), _bq3 - _bq1, 2 * _hh,
                         fc=PAL['peri'], ec=PAL['indigo'], lw=0.6, alpha=0.8, zorder=4))
axd1.plot([_bmed, _bmed], [_yb - _hh, _yb + _hh], color='white', lw=1.1, zorder=5)

axd1.set_xlim(-0.02, 1.0)
axd1.set_ylim(-0.04, 0.52)
axd1.set_yticks([])
axd1.set_xticks([0, 0.5, 1.0])
axd1.set_xlabel('per-cohort $r$\n(PC1 vs stroma/immune markers)', fontsize=8,
                linespacing=1.35)
tidy(axd1, grid=None)
axd1.text(0.03, 0.97, '33/33 positive\nmean $r=%.3f$\n$\\lambda_1=%.1f$ ($%.1f\\times$ MP)'
          % (mu, attr['mp']['top_eigenvalues'][0], attr['mp']['lam1_over_edge']),
          transform=axd1.transAxes, va='top', ha='left', fontsize=8, linespacing=1.5,
          color=INK, zorder=6,
          bbox=dict(boxstyle='round,pad=0.42', fc=BOXFC, ec=BOXEC, lw=0.6, alpha=0.92))

dcur = attr['driver_threshold_curve']
tk = dcur['thresholds']
dh = [dcur['driver_hits'][str(t)] if str(t) in dcur['driver_hits'] else dcur['driver_hits'][t]
      for t in tk]
xs = np.arange(len(tk))
tv = [float(t) for t in tk]
idx = next((i for i, t in enumerate(tv) if t >= 1000), max(0, len(tk) // 2))
axd2.axvspan(idx - 0.5, len(tk) - 0.5, color=PAL['moss'], alpha=0.14, lw=0, zorder=0)
axd2.plot(xs, dh, '-', c=PAL['moss'], lw=1.4, zorder=3)
dots(axd2, xs, dh, 24, PAL['moss'], z=4)
for i, v in enumerate(dh):
    axd2.annotate(str(v), (i, v), textcoords='offset points', xytext=(0, 6.5),
                  ha='center', fontsize=8, color=INK, fontweight='bold', zorder=6)
axd2.set_xlim(-0.5, len(tk) - 0.5)
axd2.set_xticks(xs)
axd2.set_xticklabels([str(t) for t in tk], fontsize=8, rotation=45, ha='right',
                     rotation_mode='anchor')
axd2.set_ylim(-12, max(dh) * 1.48)
axd2.set_yticks([0, 100, 200])
axd2.set_xlabel('MAD top-$k$ panel size', fontsize=8, labelpad=1.5)
axd2.set_ylabel('driver-gene hits', fontsize=8)
axd2.text(0.04, 0.97, 'drivers enter\nat $k\\gtrsim1000$', transform=axd2.transAxes,
          fontsize=8, ha='left', va='top', color=INK, linespacing=1.4, zorder=6,
          bbox=dict(boxstyle='round,pad=0.42', fc=BOXFC, ec=BOXEC, lw=0.6, alpha=0.92))
tidy(axd2, grid='y')

# Panel letters last, in figure coordinates, so that the left-hand column is not
# clipped by the canvas edge (axes-fraction offsets push 'a' and 'c' off the page).
for _ax, _s in ((axa, 'a'), (axb, 'b'), (axc, 'c'), (axd1, 'd')):
    _bb = _ax.get_position()
    fig.text(_bb.x0 - 0.027, _bb.y1 + 0.007, _s + ')', fontsize=11, fontweight='bold',
             ha='left', va='bottom', color=INK)

# No suptitle: the manuscript carries the caption, and the canvas must stay
# exactly W wide so the declared point sizes survive placement.
fig.savefig(OUT_PNG)
fig.savefig(OUT_PDF)
print('wrote', OUT_PNG)
print('wrote', OUT_PDF)
