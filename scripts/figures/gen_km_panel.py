"""Figure 5: Kaplan-Meier overall survival for the four representative BRCA network genes.

Patients are split at the median expression of each gene and compared with a log-rank test.
The four log-rank p-values reported in the manuscript are asserted here, so that a change in the
data pipeline cannot silently leave the figure out of step with the text.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE, 'results')
FIG = os.path.join(BASE, 'figures')
# TCGA matrices: ./data/ inside the package, or the folder named by MULTIBATCH_DATA
_LOCAL = os.path.join(BASE, 'data')
DATA = os.environ.get('MULTIBATCH_DATA') or _LOCAL
VD = os.path.join(DATA, 'validation')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed TrueType, not Type 3
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'text.usetex': False,
})
BLUE = '#1f77b4'
RED = '#d62728'

GENES = ['TFF1', 'CST1', 'NKAIN1', 'NAT1']
# values as reported in the manuscript caption
EXPECT = {'TFF1': 1.1e-3, 'CST1': 2.8e-3, 'NKAIN1': 3.3e-3, 'NAT1': 2.8e-2}

expr = pd.read_csv(os.path.join(DATA, 'TCGA_BRCA_HiSeqV2.tsv'), sep='\t', index_col=0)
surv = json.load(open(os.path.join(VD, 'brca_survival.json')))


def event(s):
    return 1 if (isinstance(s, str) and s.startswith('1')) else 0


sdf = pd.DataFrame([{'patient': r['patient'], 'os_months': float(r['os_months']),
                     'event': event(r.get('os_status'))} for r in surv if r.get('patient')])


def sample_to_patient(s):
    parts = s.split('-')
    return '-'.join(parts[:3]) if len(parts) >= 3 else s


def groups(gene):
    pat = {}
    for s, v in expr.loc[gene].items():
        pat.setdefault(sample_to_patient(s), []).append(v)
    pat = {k: float(np.mean(v)) for k, v in pat.items()}
    m = sdf.copy()
    m['gexpr'] = m['patient'].map(pat)
    m = m.dropna(subset=['gexpr', 'os_months'])
    med = m['gexpr'].median()
    return m[m['gexpr'] >= med], m[m['gexpr'] < med]


fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
bad = []
for idx, g in enumerate(GENES):
    ax = axes[idx // 2][idx % 2]
    hi, lo = groups(g)
    lr = logrank_test(hi['os_months'], lo['os_months'], hi['event'], lo['event'])
    p = lr.p_value
    print('  %-7s log-rank p = %.4g  (manuscript: %.1e)  high n=%d low n=%d'
          % (g, p, EXPECT[g], len(hi), len(lo)))
    if abs(p - EXPECT[g]) / EXPECT[g] > 0.15:
        bad.append((g, p, EXPECT[g]))
    kmf = KaplanMeierFitter()
    kmf.fit(lo['os_months'], lo['event'], label='Low (n=%d)' % len(lo))
    kmf.plot_survival_function(ax=ax, color=BLUE, lw=1.5)
    kmf.fit(hi['os_months'], hi['event'], label='High (n=%d)' % len(hi))
    kmf.plot_survival_function(ax=ax, color=RED, lw=1.5)
    ax.set_title('%s ($p$ = %.2g)' % (g, p), fontsize=8)
    ax.set_xlabel('OS (months)', fontsize=7)
    ax.set_ylabel('Survival prob.', fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=6, frameon=False, loc='lower left')

if bad:
    print('ABORT: %d gene(s) do not reproduce the manuscript p-values: %s' % (len(bad), bad))
    sys.exit(1)

plt.tight_layout()
plt.savefig(os.path.join(FIG, 'km_brca_panel.png'), dpi=220)
plt.savefig(os.path.join(FIG, 'km_brca_panel.pdf'), dpi=220)
plt.close()
print('Fig5 done: %d KB (png) / %d KB (pdf)'
      % (os.path.getsize(os.path.join(FIG, 'km_brca_panel.png')) // 1024,
         os.path.getsize(os.path.join(FIG, 'km_brca_panel.pdf')) // 1024))
