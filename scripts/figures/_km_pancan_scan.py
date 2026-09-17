"""Pan-cancer Kaplan-Meier scan of the per-cohort network hubs.

For every TCGA cohort: read the expression matrix, take the cohort's own hub gene
(the highest-degree gene in that cohort's inferred network), split patients at the median
expression, and report the log-rank p plus a univariate Cox hazard ratio with 95% CI.
GSTM1, the most recurrent network gene overall, is scanned alongside for comparison.

Nothing is selected on the basis of the result: the gene per cohort is fixed in advance by
the network, not by the survival data.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gene_symbols import canonical, labels

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE, 'results')
# TCGA matrices: ./data/ inside the package, or the folder named by MULTIBATCH_DATA
_LOCAL = os.path.join(BASE, 'data')
DATA = os.environ.get('MULTIBATCH_DATA') or _LOCAL
OSDIR = os.path.join(DATA, 'validation', 'pancan_os')
OUT = os.path.join(RES, '_km_pancan.json')

hub_expr = json.load(open(os.path.join(RES, '_hub_expression.json')))
HUB = hub_expr['hub_of_cancer']
COHORTS = sorted(HUB.keys())
print('cohorts: %d' % len(COHORTS))

ALSO = ['GSTM1']


def event(s):
    return 1 if (isinstance(s, str) and s.startswith('1')) else 0


def sample_to_patient(s):
    p = str(s).split('-')
    return '-'.join(p[:3]) if len(p) >= 3 else str(s)


def read_row(cohort, genes):
    """Return {gene: {patient: mean expression}} for the requested genes."""
    path = os.path.join(DATA, 'TCGA_%s_HiSeqV2.tsv' % cohort)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep='\t', index_col=0)
    # Resolve through the symbol table, not a bare .upper(): the hub of ACC is now printed as
    # SHOC1 while Xena still labels the row C9orf84. See _gene_symbols.py.
    idx = labels(df.index)
    out = {}
    for g in genes:
        key = idx.get(canonical(g).upper())
        if key is None:
            out[g] = None
            continue
        ser = df.loc[key]
        agg = {}
        for s, v in ser.items():
            agg.setdefault(sample_to_patient(s), []).append(v)
        out[g] = {k: float(np.mean(v)) for k, v in agg.items()}
    return out


def analyse(expr, os_rows):
    sdf = pd.DataFrame([{'patient': r['patient'], 'os_months': float(r['os_months']),
                         'event': event(r['os_status'])} for r in os_rows])
    m = sdf.copy()
    m['gexpr'] = m['patient'].map(expr)
    m = m.dropna(subset=['gexpr', 'os_months'])
    if len(m) < 20:
        return None
    med = m['gexpr'].median()
    m['grp'] = np.where(m['gexpr'] >= med, 1, 0)
    hi, lo = m[m['grp'] == 1], m[m['grp'] == 0]
    if len(hi) < 5 or len(lo) < 5:
        return None
    p = logrank_test(hi['os_months'], lo['os_months'], hi['event'], lo['event']).p_value
    try:
        cph = CoxPHFitter()
        cph.fit(m[['os_months', 'event', 'grp']], duration_col='os_months', event_col='event')
        hr = float(np.exp(cph.params_['grp']))
        lo_ci, hi_ci = [float(np.exp(x)) for x in cph.confidence_intervals_.loc['grp']]
    except Exception:
        hr, lo_ci, hi_ci = float('nan'), float('nan'), float('nan')
    return {'n_low': int(len(lo)), 'n_high': int(len(hi)), 'logrank_p': float(p),
            'hr': hr, 'hr_lo': lo_ci, 'hr_hi': hi_ci}


results = {}
for c in COHORTS:
    osp = os.path.join(OSDIR, '%s_os.json' % c)
    if not os.path.exists(osp):
        print('%-5s  no survival file' % c)
        continue
    os_rows = json.load(open(osp))
    genes = ([HUB[c]] if HUB.get(c) else []) + ALSO
    genes = list(dict.fromkeys(genes))
    rows = read_row(c, genes)
    if rows is None:
        print('%-5s  expression matrix missing' % c)
        continue
    entry = {}
    for g in genes:
        if not rows.get(g):
            continue
        r = analyse(rows[g], os_rows)
        if r:
            r['gene'] = g
            entry[g] = r
    results[c] = {'hub': HUB.get(c), 'genes': entry}
    hg = HUB.get(c)
    h = entry.get(hg)
    g = entry.get('GSTM1')
    def fmt(x):
        if not x:
            return '            --            '
        return 'p=%.3g HR=%.2f[%.2f-%.2f]' % (x['logrank_p'], x['hr'], x['hr_lo'], x['hr_hi'])
    print('%-5s hub=%-12s %s | GSTM1 %s' % (c, hg, fmt(h), fmt(g)))

json.dump(results, open(OUT, 'w'), indent=1)
print('\nwritten: %s' % OUT)

n_hub_sig = sum(1 for c, v in results.items()
                if v['genes'].get(v['hub']) and v['genes'][v['hub']]['logrank_p'] < 0.05)
n_gst_sig = sum(1 for c, v in results.items()
                if v['genes'].get('GSTM1') and v['genes']['GSTM1']['logrank_p'] < 0.05)
print('hub gene significant at 5%% in %d of %d cohorts' % (n_hub_sig, len(results)))
print('GSTM1     significant at 5%% in %d of %d cohorts' % (n_gst_sig, len(results)))
