"""Derive cached data for the two new figures (hub-gene tissue specificity, DepMap concordance)
and the extended Fig. 2 panels (pairwise cancer sharing matrix, shared-set composition).

Sources (all pre-existing, nothing is fabricated):
  * results/_pipeline_notears.json      per-cancer NOTEARS W matrices + gene lists
  * <data>/TCGA_<C>_HiSeqV2.tsv                       log2(TPM+1) expression, 33 cancers
  * <data>/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv

  where <data> is ./data/ inside the package, or the folder named by MULTIBATCH_DATA.
Outputs:
  * results/_hub_expression.json
  * results/_cancer_pair_sharing.json
  * results/_depmap_scatter.json
"""
import os, sys, json, csv
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gene_symbols import canonical, index

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, 'results')
# TCGA matrices: ./data/ inside the package, or the folder named by MULTIBATCH_DATA
_LOCAL = os.path.join(BASE, 'data')
TCGA_DIR = os.environ.get('MULTIBATCH_DATA') or _LOCAL
DEPMAP = os.path.join(TCGA_DIR, 'depmap')
TAU = 0.3

def log(*a):
    print(*a, flush=True)

# ---------------------------------------------------------------- 1. NOTEARS graphs
nt = json.load(open(os.path.join(RESULTS, '_pipeline_notears.json'), encoding='utf-8'))
cancers = sorted(k for k in nt if isinstance(nt[k], dict) and 'W' in nt[k])
log('cancers = %d' % len(cancers))

per_cancer = {}
edge_sets = {}
for c in cancers:
    W = np.array(nt[c]['W'], dtype=float)
    genes = list(nt[c]['genes'])
    d = W.shape[0]
    A = (np.abs(W) > TAU).astype(float)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=0) + A.sum(axis=1)          # undirected degree inside the network
    edges = []
    for i in range(d):
        for j in range(d):
            if i != j and A[i, j] > 0:
                edges.append((genes[i], genes[j], float(W[i, j])))
    edge_sets[c] = set((a, b) for a, b, _ in edges)
    per_cancer[c] = {
        'n': int(nt[c].get('n', 0)),
        'n_edges': len(edges),
        'genes': genes,
        'degree': {genes[k]: float(deg[k]) for k in range(d)},
        # The hub is the one gene this file hands to readers and to every downstream figure, so it
        # is stored under its HGNC-approved symbol; the gene list itself is left as NOTEARS emitted
        # it. See _gene_symbols.py.
        'hub': canonical(genes[int(np.argmax(deg))]),
        'hub_degree': float(np.max(deg)),
    }
    log('  %-5s edges=%4d  hub=%-12s deg=%.0f' % (c, len(edges), per_cancer[c]['hub'], per_cancer[c]['hub_degree']))

json.dump(per_cancer, open(os.path.join(RESULTS, '_per_cancer_network.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- 2. pairwise cancer sharing
share_mat = np.zeros((len(cancers), len(cancers)), dtype=float)
for a_i, ca in enumerate(cancers):
    for b_i, cb in enumerate(cancers):
        if a_i == b_i:
            share_mat[a_i, b_i] = 100.0
            continue
        sa, sb = edge_sets[ca], edge_sets[cb]
        denom = min(len(sa), len(sb))
        share_mat[a_i, b_i] = 100.0 * len(sa & sb) / denom if denom else 0.0

pair_count = Counter()
for c in cancers:
    for e in edge_sets[c]:
        pair_count[e] += 1

SEX = {'XIST', 'TSIX', 'RPS4Y1', 'RPS4Y2', 'DDX3Y', 'EIF1AY', 'KDM5D', 'USP9Y', 'UTY', 'NLGN4Y',
       'ZFY', 'TMSB4Y', 'PRKY', 'TXLNGY', 'TTTY15', 'CYORF15B', 'GSTM1', 'AMELY', 'PCDH11Y'}
shared3 = {p: n for p, n in pair_count.items() if n >= 3}
sex_hits = {p: n for p, n in shared3.items() if p[0] in SEX or p[1] in SEX}
paralog = {p: n for p, n in shared3.items() if p not in sex_hits and p[0][:4] == p[1][:4]}

sharing = {
    'cancers': cancers,
    'matrix_pct': share_mat.tolist(),
    'total_unique_pairs': len(pair_count),
    'n_shared_ge2': int(sum(1 for v in pair_count.values() if v >= 2)),
    'n_shared_ge3': int(len(shared3)),
    'sex_chromosome_shared': sorted(sex_hits.keys()),
    'n_sex_chromosome_shared': len(sex_hits),
    'n_paralog_same_prefix': len(paralog),
    'shared_set_composition': {
        'sex_chromosome': len(sex_hits),
        'paralogous_same_family': len(paralog),
        'other': len(shared3) - len(sex_hits) - len(paralog),
    },
    'top_shared_pairs': [[a, b, n] for (a, b), n in pair_count.most_common(20)],
}
json.dump(sharing, open(os.path.join(RESULTS, '_cancer_pair_sharing.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
log('\nsharing: unique=%d  >=2:%d  >=3:%d  sex=%d  paralog=%d  other=%d' % (
    sharing['total_unique_pairs'], sharing['n_shared_ge2'], sharing['n_shared_ge3'],
    len(sex_hits), len(paralog), sharing['shared_set_composition']['other']))

# ---------------------------------------------------------------- 3. hub-gene expression (TCGA)
hub_genes = sorted({per_cancer[c]['hub'] for c in cancers})
log('\nhub genes to profile (%d): %s' % (len(hub_genes), hub_genes))

expr = defaultdict(dict)          # expr[cancer][gene] = [values]
found = Counter()
# A Xena row name has to be tied back to a network gene both case-insensitively and across legacy
# aliases, or the row is dropped without a warning -- which is how two hub genes went missing.
# The index hands back the canonical spelling, which is what keys this dictionary.
hub_idx = index(hub_genes)
for c in cancers:
    p = os.path.join(TCGA_DIR, 'TCGA_%s_HiSeqV2.tsv' % c)
    if not os.path.exists(p):
        log('  MISSING %s' % p)
        continue
    with open(p, encoding='utf-8', errors='replace') as f:
        hdr = f.readline().rstrip('\n').split('\t')
        nsamp = len(hdr) - 1
        for line in f:
            if not line or line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            g = hub_idx.get(canonical(parts[0]).upper())
            if g is None:
                continue
            vals = []
            for v in parts[1:]:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
            if vals:
                expr[c][g] = vals
                found[g] += 1
    log('  %-5s n=%4d  matched hub genes=%d' % (c, nsamp, sum(1 for g in hub_genes if g in expr[c])))

# tissue-specificity statistic: rank of the "own" cancer for each hub gene
stats = {}
for c in cancers:
    g = per_cancer[c]['hub']
    if g not in expr.get(c, {}):
        continue
    med = {}
    for cc in cancers:
        if g in expr.get(cc, {}):
            med[cc] = float(np.median(expr[cc][g]))
    if len(med) < 20:
        continue
    own = med.get(c)
    if own is None:
        continue
    order = sorted(med.items(), key=lambda kv: -kv[1])
    rank = 1 + [k for k, _ in order].index(c)
    stats[g] = {
        'own_cancer': c, 'rank': int(rank), 'n_cancers_with_gene': len(med),
        'own_median': own, 'max_median': order[0][1], 'max_cancer': order[0][0],
        'top5': [k for k, _ in order[:5]],
    }
log('\nhub-gene tissue specificity: %d genes ranked' % len(stats))
ranks = [v['rank'] for v in stats.values()]
for g, v in sorted(stats.items(), key=lambda kv: kv[1]['rank']):
    log('  %-12s own=%-5s rank=%2d/%d  top5=%s' % (g, v['own_cancer'], v['rank'], v['n_cancers_with_gene'], v['top5']))

json.dump({'expr': {c: {g: v for g, v in d.items()} for c, d in expr.items()},
           'hub_of_cancer': {c: per_cancer[c]['hub'] for c in cancers},
           'stats': stats,
           'cancers': cancers},
          open(os.path.join(RESULTS, '_hub_expression.json'), 'w', encoding='utf-8'),
          ensure_ascii=False)
log('\nwrote _hub_expression.json')

# ---------------------------------------------------------------- 4. DepMap scatter columns
dep = json.load(open(os.path.join(RESULTS, '_depmap_validation.json'), encoding='utf-8'))
need = sorted({r['A'] for r in dep} | {r['B'] for r in dep})
log('\nDepMap genes needed: %s' % need)

omi = os.path.join(DEPMAP, 'OmicsExpressionProteinCodingGenesTPMLogp1.csv')
cols = {}
with open(omi, encoding='utf-8', errors='replace') as f:
    r = csv.reader(f)
    hdr = next(r)
    want = {}
    for idx, h in enumerate(hdr):
        g = h.split(' (')[0].strip()
        if g in need and g not in want:
            want[g] = idx
    log('  matched columns: %s' % {k: hdr[v] for k, v in want.items()})
    data = {g: [] for g in want}
    models = []
    for row in r:
        if len(row) < len(hdr):
            continue
        models.append(row[0])
        for g, idx in want.items():
            try:
                data[g].append(float(row[idx]))
            except (ValueError, IndexError):
                data[g].append(float('nan'))
            if len(models) % 400 == 0 and g == list(want)[0]:
                pass
log('  cell lines read = %d' % len(models))

json.dump({'models': models, 'expr': data,
           'pairs': dep},
          open(os.path.join(RESULTS, '_depmap_scatter.json'), 'w', encoding='utf-8'),
          ensure_ascii=False)
log('wrote _depmap_scatter.json')
log('DONE')
