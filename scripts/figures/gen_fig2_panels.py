# -*- coding: utf-8 -*-
"""Figure 2 of the manuscript -- robustness to a gene-disjoint second panel.

(a) recurrence of the directed pairs for both panels; the primary panel is
    restated in ghost so the eye goes to what the substitution actually changed,
    the second panel is drawn solid
(b) sample-size dependence of edge count for the two panels, colour = panel and
    fill = whether the cohort clears n >= 200

Two decisions in this revision are structural rather than cosmetic.  The plate
now carries exactly two panels, because the manuscript caption describes two:
the earlier revision drew a third, a driver-entry curve that was also a redraw
of Fig. 1d-right from the same JSON, and no caption ever mentioned it.  And
panel (a) no longer uses filled capsules on a track -- that is Fig. 1c's
encoding, and reusing it here made two different comparisons read as one
picture.

Output: figures/Fig2_panels.pdf / .png
"""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gene_symbols import canonical  # legacy spelling -> the HGNC symbol the paper prints
from _figstyle import (PAL, INK, TICK, GREY, GRID, BOXFC, BOXEC, MM, W,
                       tidy, card, dots, haloed_text, panel_letters)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(_ROOT, 'results')
FIG = os.path.join(_ROOT, 'figures')

OUT_PNG = os.path.join(FIG, 'Fig2_panels.png')
OUT_PDF = os.path.join(FIG, 'Fig2_panels.pdf')
TAU = 0.3

p2 = json.load(open(os.path.join(RES, '_panel_B_robustness.json'), encoding='utf-8'))

# Keys are matched against the spellings carried by the result matrices, so the
# legacy form IL8 has to stay here; only what is printed goes through canonical().
AXIS1 = {
    'IFN-\u03b3 / CXCR3 chemokine': (['CXCL9', 'CXCL10', 'CXCL11'], PAL['mist'],
                                     'Interferon-$\\gamma$ / CXCR3'),
    'neutrophil / CXCR2 chemokine': (['CXCL1', 'IL8'], PAL['orchid'],
                                     'Neutrophil / CXCR2'),
    'erythrocyte content': (['HBA1', 'HBB'], PAL['lilac'], 'Erythrocyte content'),
    'plasma-cell / Ig locus': (['ADAM6', 'IGJ', 'MGC29506'], PAL['violet'],
                               'Plasma-cell / Ig locus'),
    'stromal / complement': (['COL10A1', 'COL11A1', 'C7', 'PLCXD3'], PAL['moss'],
                             'Stroma / complement'),
    'myeloid / cytokine': (['CCL18', 'CHIT1', 'IL6', 'FOSB'], PAL['peri'],
                           'Myeloid / cytokine'),
}
AXIS2 = {
    'T-cell / NK cytotoxicity': (['SLAMF6', 'NKG7', 'LAG3'], PAL['moss'],
                                 'T-cell / NK cytotoxicity'),
    'T-cell / DC trafficking': (['LTB', 'SPIB', 'C8orf80', 'CD1A'], PAL['orchid'],
                                'T-cell / DC trafficking'),
    'IEG / activation': (['NR4A3', 'EGR3'], PAL['violet'], 'IEG / activation'),
}


def axis_of(src, dst, table):
    s = {src, dst}
    for k, (gs_, c, lab) in table.items():
        if s & set(gs_):
            return c, lab
    return GREY, 'other'


# ------------------------------------------------------------------ layout
H = 174.0 * MM * 0.66
fig = plt.figure(figsize=(W, H))
outer = gridspec.GridSpec(1, 2, width_ratios=[1.42, 1.0], wspace=0.34,
                          left=0.078, right=0.985, top=0.920, bottom=0.118)
axa = fig.add_subplot(outer[0])
axb = fig.add_subplot(outer[1])
for _cax in (axa, axb):
    card(fig, _cax)

# ================= (a) =================
rows1 = sorted(p2['panel1']['shared_edges'], key=lambda x: -x['n'])
rows2 = sorted(p2['panel2']['shared_edges'], key=lambda x: -x['n'])
ys1 = list(range(len(rows1)))[::-1]                  # 13 .. 0
# The separator band now carries three lines, so it needs more than the two-line
# 3.2 it was sized for: at that width the outer lines sat within a text height of
# the rows above and below them.
gap = 5.6
ys2 = [ys1[-1] - gap - i for i in range(len(rows2))]  # -5.6 .. -10.6

axa.set_yticks([])
# 40, not 44: the longest caption ("CXCL10->CXCL11" starting at 28.9) still clears
# the right edge, and the extra four units went back to the bars.
axa.set_xlim(0, 40)
axa.set_ylim(ys2[-1] - 1.4, ys1[0] + 1.0)
axa.set_xticks([0, 10, 20, 30, 40])
axa.set_xlabel('cohorts (of 33) in which the edge recurs')
tidy(axa, grid=None)


def lollipop(ax, x, y, c, rod_alpha, dot_alpha):
    """A thin stem to the value with a round head -- the light counterpart of
    the capsule bars used in Fig. 1c, so the two panels do not read as one."""
    ax.plot([0, x], [y, y], color=c, lw=1.25, alpha=rod_alpha,
            solid_capstyle='round', zorder=3)
    dots(ax, [x], [y], 34, c, z=4, alpha=dot_alpha)


for e, y in zip(rows1, ys1):
    c, _ = axis_of(e['src'], e['dst'], AXIS1)
    lollipop(axa, e['n'], y, c, 0.26, 0.50)
    # The primary rows are the reference the second panel is read against, so they
    # recede in the lettering too -- not just in the mark.  With both sets of names
    # at full strength the ghost group still competed for attention.
    haloed_text(axa, e['n'] + 0.9, y, '%s$\\rightarrow$%s' % (canonical(e['src']),
                                                              canonical(e['dst'])),
                va='center', ha='left', z=5, color=GREY)
for e, y in zip(rows2, ys2):
    c, _ = axis_of(e['src'], e['dst'], AXIS2)
    lollipop(axa, e['n'], y, c, 0.62, 1.0)
    haloed_text(axa, e['n'] + 0.9, y, '%s$\\rightarrow$%s' % (canonical(e['src']),
                                                              canonical(e['dst'])),
                va='center', ha='left', z=5)

axa.axvline(10, ls=(0, (4, 2)), c=PAL['indigo'], lw=0.9, alpha=0.55, zorder=2)

# The gap carries the plate's one assertion: the substitution changed nothing
# that matters and shared nothing at all.  All three lines are haloed so the
# threshold rule passing behind them does not show through the lettering.
mid = (ys1[-1] + ys2[0]) / 2.0
haloed_text(axa, 20.0, mid + 1.06,
            'primary panel: %d edges      second panel: %d edges'
            % (len(rows1), len(rows2)),
            ha='center', va='center', size=8, color=GREY, z=6)
haloed_text(axa, 20.0, mid,
            'no gene shared between the two panels', ha='center', va='center',
            size=8, color=INK, z=6)
haloed_text(axa, 20.0, mid - 1.06,
            'colour = non-malignant axis (as in Fig. 1c)',
            ha='center', va='center', size=8, color=GREY, z=6)

# ================= (b) =================
series = [('primary panel', os.path.join(RES, '_shared_panel_notears.json'),
           PAL['peri'], 'o'),
          ('second panel', os.path.join(RES, '_panel2_notears.json'),
           PAL['moss'], '^')]

axb.set_xscale('log')
axb.set_xlim(30, 3000)
axb.set_ylim(-25, 265)

# The floor: every cohort above this line contributes almost no edges, and both
# panels put it in the same place.
_f0 = (np.log10(200) - np.log10(30)) / (np.log10(3000) - np.log10(30))
axb.axhspan(-25, 40, xmin=_f0, xmax=1.0, color=PAL['lilac'], alpha=0.16,
            zorder=1, lw=0)

for tag, path, col, mk in series:
    d = json.load(open(path, encoding='utf-8'))
    cs = sorted(k for k, v in d.items() if isinstance(v, dict) and 'W' in v)
    ns = np.array([d[c]['n'] for c in cs], float)
    es = np.array([np.sum(np.abs(np.array(d[c]['W'], dtype=float)) > TAU) for c in cs], float)
    below = ns < 200
    dots(axb, ns[below], es[below], 26, col, z=4, marker=mk)
    # n >= 200 is shown by the absence of fill, so colour is free to encode panel
    axb.scatter(ns[~below], es[~below], s=42, facecolors='none', edgecolors=col,
                linewidths=1.05, marker=mk, zorder=6)

axb.axvline(100, ls=(0, (4, 2)), c=PAL['indigo'], lw=0.9, alpha=0.55, zorder=3)
axb.set_yticks([0, 50, 100, 150, 200, 250])
axb.set_xlabel('cohort sample size $n$ (log scale)')
# Wording kept identical to Fig. 1b -- the two panels measure the same quantity.
axb.set_ylabel('edges inferred by NOTEARS')
tidy(axb, grid='y')

# Both captions sit in the strip under y = 0, which is empty ground: no cohort in
# either panel returns fewer than six edges, and the shaded band stops at y = 40.
axb.text(93, -14, '$n = d = 100$', fontsize=8, va='center', ha='right',
         color=GREY, zorder=7)
axb.text(2400, -14, 'no recovery past the floor', fontsize=8, va='center',
         ha='right', color=INK, zorder=7)

axb.text(0.985, 0.978,
         '$r$, all 33:  $%.3f$  /  $%.3f$\n$r$, $n\\geq200$:  $%+.3f$  /  $%+.3f$'
         % (p2['panel1']['r_all'], p2['panel2']['r_all'],
            p2['panel1']['r_n200'], p2['panel2']['r_n200']),
         transform=axb.transAxes, va='top', ha='right', fontsize=8,
         linespacing=1.6, color=INK, zorder=7,
         bbox=dict(boxstyle='round,pad=0.42', fc=BOXFC, ec=BOXEC, lw=0.6, alpha=0.94))

_handles = [
    Line2D([], [], marker='o', ls='none', mfc=PAL['peri'], mec='white', mew=0.6,
           ms=4.6, label='primary panel'),
    Line2D([], [], marker='^', ls='none', mfc=PAL['moss'], mec='white', mew=0.6,
           ms=4.6, label='second panel'),
    Line2D([], [], marker='o', ls='none', mfc='none', mec=TICK, mew=1.0, ms=4.6,
           label='$n \\geq 200$ (open)'),
]
# Below the statistics block: upper-right of the cloud is the one region with no
# markers (every cohort past n ~ 600 sits under 50 edges), whereas the lower-left
# corner was carrying a handful of them.
axb.legend(handles=_handles, loc='upper right', bbox_to_anchor=(0.992, 0.665),
           frameon=True, framealpha=0.94, edgecolor=BOXEC, facecolor=BOXFC,
           handletextpad=0.5, borderpad=0.5, labelspacing=0.38)

panel_letters(fig, ((axa, 'a'), (axb, 'b')))

fig.savefig(OUT_PNG)
fig.savefig(OUT_PDF)
print('wrote', OUT_PNG)
print('wrote', OUT_PDF)
