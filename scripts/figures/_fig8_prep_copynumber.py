# -*- coding: utf-8 -*-
"""Recompute the four side files Figure 8 is drawn from.

`gen_fig8_copynumber.py` draws the published plate out of four JSON files in
``results/``.  This script is what produces those four, from the two copy-number
matrices ``_fig8_fetch_cna.py`` streams off UCSC Xena and from the recurrence
table the rest of the package already ships:

    results/_fig8_edges.json          the 14 recurring pairs, placed on hg19 and
                                      called cis (same locus, <= 1 Mb) or trans
    results/_fig8_cna_results.json    the raw copy-number coupling of each pair,
                                      pan-cancer and per cohort, plus the
                                      per-gene alteration frequencies
    results/_fig8_null_dist.json      rho against genomic distance for 2,000
                                      random pairs in five distance strata, with
                                      the stratum medians and 95th percentiles
    results/_fig8_null_results.json   the two-class null (neighbouring pairs,
                                      cross-chromosome pairs) and where each of
                                      the 14 edges sits inside its own class

Nothing here is a new analysis: every number was already published, and
re-running this script reproduces the shipped files.  It is in the package so
that the copy-number objection Figure 8 answers can be re-derived from source
rather than taken on trust.

Inputs
------
    data/TCGA_*_HiSeqV2.tsv     the per-cohort expression matrices.  Only their
                                headers are read, to recover the sample ->
                                cancer-type map; the measurements are not
                                redistributed (see README, Data Availability).
    data/fig8_cna_genes.npz     the 18 genes of the recurring edges
    data/fig8_cna_pool.npz      the seeded random gene pool for the null
    data/fig8_gene_coords.json  hg19 spans and bands of those 18 genes
    data/hg19_cytoBand.txt.gz   hg19 cytoband stains
    results/_recurrence_analysis.json   the recurrence table

The first three come from ``_fig8_fetch_cna.py``; the last three ship with the
package.  ``data/hg19_refGene.txt.gz`` is downloaded if absent -- the null is
stratified by genomic distance, so it needs positions for genes the panel does
not cover.

Usage
-----
    python scripts/figures/_fig8_fetch_cna.py        # once, needs the network
    python scripts/figures/_fig8_prep_copynumber.py  # then this, offline

Output: the four results/_fig8_*.json above.
"""
import glob
import gzip
import json
import os
import random
import sys
import time
import urllib.request

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
DAT = os.path.join(ROOT, 'data')
RES = os.path.join(ROOT, 'results')

GENES_NPZ = os.path.join(DAT, 'fig8_cna_genes.npz')
POOL_NPZ = os.path.join(DAT, 'fig8_cna_pool.npz')
COORDS_JSON = os.path.join(DAT, 'fig8_gene_coords.json')
CYTOBAND = os.path.join(DAT, 'hg19_cytoBand.txt.gz')
REFGENE = os.path.join(DAT, 'hg19_refGene.txt.gz')
REFGENE_URL = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz'
UA = {'User-Agent': 'multibatch-fig8/1.0'}

EDGES_JSON = os.path.join(RES, '_fig8_edges.json')
CNA_JSON = os.path.join(RES, '_fig8_cna_results.json')
NULLDIST_JSON = os.path.join(RES, '_fig8_null_dist.json')
NULLRES_JSON = os.path.join(RES, '_fig8_null_results.json')

CIS_WINDOW = 1_000_000          # 1 Mb -- the usual cis-regulatory window
MIN_COHORT_N = 50               # a cohort has to be this big to give an rho
PER_STRATUM = 400               # random pairs per distance stratum
N_PAIRS = 500                   # pairs per class in the two-class null

# the genes under test, in the symbols the copy-number table carries them under
TARGET = ['CXCL9', 'CXCL10', 'CXCL11', 'CXCL1', 'IL8', 'CXCL8', 'C7', 'PLCXD3',
          'HBB', 'HBA1', 'IL6', 'FOSB', 'ADAM6', 'IGJ', 'JCHAIN', 'MZB1',
          'COL10A1', 'COL11A1', 'CCL18', 'CHIT1']

STRATA = [('same band (<1 Mb)', 0, 1e6),
          ('same arm (1-10 Mb)', 1e6, 1e7),
          ('same chr (10-100 Mb)', 1e7, 1e8),
          ('same chr (>100 Mb)', 1e8, 3e8),
          ('different chr', None, None)]


# --------------------------------------------------------------------- inputs
def ensure_refgene():
    if os.path.exists(REFGENE):
        return REFGENE
    print('    downloading hg19 refGene ...')
    req = urllib.request.Request(REFGENE_URL, headers=UA)
    data = urllib.request.urlopen(req, timeout=300).read()
    with open(REFGENE, 'wb') as f:
        f.write(data)
    print('    %.1f MB -> %s' % (len(data) / 1048576, os.path.relpath(REFGENE, ROOT)))
    return REFGENE


def load_gene_coords(path):
    """(chromosome, midpoint) per symbol, major chromosomes only."""
    tmp = {}
    with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 13:
                continue
            chrom, s, e, sym = p[2], int(p[4]), int(p[5]), p[12]
            if not chrom.startswith('chr') or '_' in chrom or not sym:
                continue
            d = tmp.setdefault(sym, {'chrom': chrom, 'start': s, 'end': e, 'n': 0})
            if d['chrom'] == chrom:
                d['start'] = min(d['start'], s)
                d['end'] = max(d['end'], e)
                d['n'] += 1
    return {k: (v['chrom'], (v['start'] + v['end']) // 2)
            for k, v in tmp.items() if v['n'] >= 1}


def ensure_gene_coords():
    """The 18 panel genes' spans and bands.  Ships with the package; rebuilt
    from UCSC refGene + cytoband if it is missing."""
    if os.path.exists(COORDS_JSON):
        return json.load(open(COORDS_JSON, encoding='utf-8'))

    print('    rebuilding %s from refGene + cytoband'
          % os.path.relpath(COORDS_JSON, ROOT))
    alias = {'IL8': ['CXCL8'], 'IGJ': ['JCHAIN', 'IGJ'], 'MGC29506': ['MZB1']}
    panel = ['CXCL10', 'CXCL11', 'CXCL9', 'HBB', 'HBA1', 'IL6', 'FOSB', 'CXCL1',
             'IL8', 'C7', 'PLCXD3', 'ADAM6', 'IGJ', 'MGC29506', 'COL10A1',
             'COL11A1', 'CCL18', 'CHIT1']
    accept = {a: p for p in panel for a in alias.get(p, [p])}

    cyto = {}
    with gzip.open(CYTOBAND, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            c, s, e, band, _stain = line.rstrip('\n').split('\t')
            cyto.setdefault(c, []).append((int(s), int(e), band))

    def band_of(chrom, pos):
        for s, e, b in cyto.get(chrom, []):
            if s <= pos < e:
                return b
        return None

    spans = {}
    with gzip.open(ensure_refgene(), 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < 13:
                continue
            chrom, s, e, name2 = p[2], int(p[4]), int(p[5]), p[12]
            if name2 in accept and chrom.startswith('chr') and '_' not in chrom:
                cur = spans.get(accept[name2])
                if cur is None:
                    spans[accept[name2]] = {'official': name2, 'chrom': chrom,
                                            'start': s, 'end': e, 'n_tx': 1}
                else:
                    cur['start'] = min(cur['start'], s)
                    cur['end'] = max(cur['end'], e)
                    cur['n_tx'] += 1
    out = {}
    for p in panel:
        s = spans.get(p)
        if not s:
            continue
        s['band'] = band_of(s['chrom'], (s['start'] + s['end']) // 2)
        s['panel'] = p
        out[p] = s
    json.dump(out, open(COORDS_JSON, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    return out


def build_sample2cohort():
    """sample -> cancer type, read off the 33 per-cohort expression matrices.

    Only the header row of each file is touched.
    """
    files = sorted(glob.glob(os.path.join(DAT, 'TCGA_*_HiSeqV2.tsv')))
    if not files:
        raise SystemExit('no data/TCGA_*_HiSeqV2.tsv found.  The expression '
                         'matrices are not redistributed: download the 33 '
                         'per-cohort HiSeqV2 tables from UCSC Xena first '
                         '(see README, Data Availability).')
    s2c = {}
    for p in files:
        coh = os.path.basename(p).split('_')[1]
        with open(p, 'rt', encoding='utf-8', errors='ignore') as f:
            hdr = f.readline().rstrip('\n').split('\t')
        for s in (x.strip() for x in hdr[1:]):
            if s:
                s2c[s] = coh
    return s2c


def load_matrix(path, what):
    if not os.path.exists(path):
        raise SystemExit('%s is missing - run '
                         'scripts/figures/_fig8_fetch_cna.py first.' % path)
    z = np.load(path, allow_pickle=True)
    try:
        samples = [str(s) for s in z['samples']]
        genes = {k: z[k] for k in z.files if k != 'samples'}
    finally:
        z.close()
    print('    %-34s %d genes x %d samples' % (what, len(genes), len(samples)))
    return samples, genes


# ------------------------------------------------------------- the four files
def write_edges(coords, rec):
    """Where the 14 recurring pairs sit on hg19, and whether either end is close
    enough to the other to be co-amplified."""
    rows = []
    for e in rec['aligned']['shared_edges']:
        a, b = e['src'], e['dst']
        ca, cb = coords[a], coords[b]
        same_chr = ca['chrom'] == cb['chrom']
        dist = (abs((ca['start'] + ca['end']) / 2 - (cb['start'] + cb['end']) / 2)
                if same_chr else None)
        rows.append({'src': a, 'dst': b, 'n': e['n_cohorts'], 'frac': e['frac'],
                     'chr_a': ca['chrom'], 'band_a': ca['band'],
                     'chr_b': cb['chrom'], 'band_b': cb['band'],
                     'dist_bp': dist, 'same_chr': same_chr,
                     'same_band': ca['band'] == cb['band'],
                     'cis': bool(same_chr and dist <= CIS_WINDOW)})
    rows.sort(key=lambda r: -r['n'])

    n_cis = sum(1 for r in rows if r['cis'])
    out = {'cis_window': CIS_WINDOW, 'edges': rows,
           'n_cis': n_cis, 'n_trans': len(rows) - n_cis,
           'w_cis': sum(r['n'] for r in rows if r['cis']),
           'w_trans': sum(r['n'] for r in rows if not r['cis'])}
    json.dump(out, open(EDGES_JSON, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('    %s  cis %d / trans %d' % (os.path.relpath(EDGES_JSON, ROOT),
                                         out['n_cis'], out['n_trans']))
    return out


def pairstats(a, b):
    """Co-variation between two thresholded copy-number vectors."""
    n = len(a)
    if n < MIN_COHORT_N:
        return None
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        rho, p = 0.0, 1.0
    else:
        rho, p = stats.spearmanr(a, b)
    changed = (np.abs(a) >= 1) | (np.abs(b) >= 1)        # at least one altered
    both = (np.abs(a) >= 1) & (np.abs(b) >= 1)
    same = both & (np.sign(a) == np.sign(b))
    return dict(n=int(n), rho=float(rho), p=float(p),
                n_any=int(changed.sum()), n_both=int(both.sum()),
                n_same=int(same.sum()),
                concord=float(same.sum() / changed.sum()) if changed.sum() else float('nan'),
                jaccard=float(both.sum() / changed.sum()) if changed.sum() else float('nan'))


def write_cna_results(samples, genes, s2c, edges):
    """The copy-number coupling of each recurring pair, pan-cancer and per cohort."""
    idx = [i for i, s in enumerate(samples) if s in s2c]
    sub = {g: v[idx] for g, v in genes.items()}
    coh = np.array([s2c[samples[i]] for i in idx])
    print('    %d/%d samples fall inside the manuscript cohorts (%d cohorts)'
          % (len(idx), len(samples), len(set(coh))))

    gene_freq = {}
    for g in sorted(sub):
        a, l, any_ = (int((sub[g] >= 1).sum()), int((sub[g] <= -1).sum()),
                      int((np.abs(sub[g]) >= 1).sum()))
        gene_freq[g] = dict(gain=100.0 * a / len(idx), loss=100.0 * l / len(idx),
                            any=100.0 * any_ / len(idx))

    rows = []
    for e in edges['edges']:
        a, b = e['src'], e['dst']
        st = pairstats(sub[a], sub[b])
        rec = dict(src=a, dst=b, cls='cis' if e['cis'] else 'trans',
                   n_cohorts=e['n'], dist_bp=e['dist_bp'],
                   band_a=e['band_a'], band_b=e['band_b'], **st)
        per = []
        for c in sorted(set(coh)):
            m = coh == c
            if m.sum() < MIN_COHORT_N:
                continue
            st_c = pairstats(sub[a][m], sub[b][m])
            if st_c:
                per.append(st_c['rho'])
        rec['per_cohort_rho'] = per
        rec['n_coh_ge50'] = len(per)
        rows.append(rec)

    # every pair among the 18 genes, as the background the edges are read against
    order = sorted(sub)
    bg = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            st = pairstats(sub[order[i]], sub[order[j]])
            if st:
                bg.append(st['rho'])
    bg = np.array(bg)
    cis = [r['rho'] for r in rows if r['cls'] == 'cis']
    tr = [r['rho'] for r in rows if r['cls'] == 'trans']
    print('    cis rho median %+.3f | trans rho median %+.3f | background median %+.3f'
          % (np.median(cis), np.median(tr), np.median(bg)))

    json.dump(dict(n_samples=len(idx), n_cohorts=len(set(coh)),
                   gene_frequency=gene_freq, edges=rows,
                   background=dict(median=float(np.median(bg)),
                                   mean=float(bg.mean()), sd=float(bg.std())),
                   cis_median_rho=float(np.median(cis)),
                   trans_median_rho=float(np.median(tr))),
              open(CNA_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('    %s' % os.path.relpath(CNA_JSON, ROOT))
    return idx, coh


def rho_of(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(stats.spearmanr(a, b)[0])


def write_null_dist(samples, genes, s2c, coords, edges):
    """rho as a function of genomic distance, for random pairs.

    Copy number is spatially organised: two genes 13 kb apart co-vary because
    they share a locus, not because one regulates the other.  The five strata are
    the band each of the 14 edges has to be read against.
    """
    idx = [i for i, s in enumerate(samples) if s in s2c]
    D = {g: v[idx] for g, v in genes.items()}
    pool = [g for g in D if g in coords and g not in TARGET]
    print('    null gene pool: %d' % len(pool))

    rg = random.Random(11)
    buckets = {n: [] for n, _, _ in STRATA}
    tries = 0
    while any(len(v) < PER_STRATUM for v in buckets.values()) and tries < 3000000:
        tries += 1
        a, b = rg.sample(pool, 2)
        ca, ma = coords[a]
        cb, mb = coords[b]
        name = None
        if ca != cb:
            name = 'different chr'
        else:
            dd = abs(ma - mb)
            for n, lo, hi in STRATA[:4]:
                if lo <= dd < hi:
                    name = n
                    break
        if name and len(buckets[name]) < PER_STRATUM:
            buckets[name].append((a, b, ca, ma, cb, mb))

    out = {}
    for name, _lo, _hi in STRATA:
        rr = np.array([rho_of(D[a], D[b]) for a, b, *_ in buckets[name]])
        out[name] = dict(n=len(rr), median=float(np.median(rr)), mean=float(rr.mean()),
                         sd=float(rr.std()), p95=float(np.percentile(rr, 95)),
                         p99=float(np.percentile(rr, 99)), max=float(rr.max()),
                         _rho=rr.tolist())
        print('    %-22s n=%4d  median %+0.3f  p95 %+0.3f'
              % (name, len(rr), out[name]['median'], out[name]['p95']))

    sc = {'dist_bp': [], 'rho': [], 'stratum': []}
    for name, _lo, _hi in STRATA:
        for a, b, ca, ma, cb, mb in buckets[name]:
            sc['dist_bp'].append(-1.0 if ca != cb else float(abs(ma - mb)))
            sc['rho'].append(rho_of(D[a], D[b]))
            sc['stratum'].append(name)

    cna = json.load(open(CNA_JSON, encoding='utf-8'))
    rows = []
    for e in edges['edges']:
        a, b = e['src'], e['dst']
        rho = next(r['rho'] for r in cna['edges'] if r['src'] == a and r['dst'] == b)
        rows.append(dict(src=a, dst=b, cls='cis' if e['cis'] else 'trans',
                         n_cohorts=e['n'], dist_bp=e['dist_bp'], rho=rho))

    json.dump({'strata': out, 'scatter': sc, 'edges': rows, 'n_samples': len(idx)},
              open(NULLDIST_JSON, 'w', encoding='utf-8'), ensure_ascii=False)
    print('    %s  (%d scatter points)'
          % (os.path.relpath(NULLDIST_JSON, ROOT), len(sc['rho'])))


def write_null_results(samples, genes, s2c, coords):
    """Where each of the 14 edges sits inside its own null class."""
    idx = [i for i, s in enumerate(samples) if s in s2c]
    D = {g: v[idx] for g, v in genes.items()}
    pool = [g for g in D if g in coords and g not in TARGET]
    print('    null gene pool: %d' % len(pool))

    rg = random.Random(7)
    near, far = [], []
    i = 0
    while len(near) < N_PAIRS * 4 or len(far) < N_PAIRS * 4:
        i += 1
        if i > 4000000:
            break
        a, b = rg.sample(pool, 2)
        ca, ma = coords[a]
        cb, mb = coords[b]
        if ca == cb and abs(ma - mb) <= 1000000:
            if len(near) < N_PAIRS * 4:
                near.append((a, b))
        elif ca != cb:
            if len(far) < N_PAIRS * 4:
                far.append((a, b))
    near = rg.sample(near, min(N_PAIRS, len(near)))
    far = rg.sample(far, min(N_PAIRS, len(far)))

    res = {}
    for label, pairs in (('cis_like (<1 Mb, same chromosome)', near),
                         ('trans_like (cross-chromosome)', far)):
        t = time.time()
        rr = np.array([rho_of(D[a], D[b]) for a, b in pairs])
        res[label] = dict(n=len(rr), median=float(np.median(rr)), mean=float(rr.mean()),
                          sd=float(rr.std()), p05=float(np.percentile(rr, 5)),
                          p95=float(np.percentile(rr, 95)),
                          p99=float(np.percentile(rr, 99)), max=float(rr.max()))
        print('    %-38s n=%d  median %+0.3f  p95 %+0.3f  (%.0fs)'
              % (label, len(rr), res[label]['median'], res[label]['p95'],
                 time.time() - t))
        res[label]['_rho'] = rr.tolist()

    cna = json.load(open(CNA_JSON, encoding='utf-8'))
    for r in cna['edges']:
        band = np.array(res['cis_like (<1 Mb, same chromosome)' if r['cls'] == 'cis'
                            else 'trans_like (cross-chromosome)']['_rho'])
        r['null_percentile'] = float(100.0 * (band < r['rho']).mean())
    worse = [r for r in cna['edges'] if r['null_percentile'] > 95]
    print('    edges above their own class\'s 95th percentile: %d/%d'
          % (len(worse), len(cna['edges'])))

    json.dump({'null': res, 'edges': cna['edges'],
               'n_samples': len(idx), 'n_null_genes': len(pool)},
              open(NULLRES_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('    %s' % os.path.relpath(NULLRES_JSON, ROOT))


def main():
    n_cis_head = None
    for p in (GENES_NPZ, POOL_NPZ):
        if not os.path.exists(p):
            raise SystemExit('%s is missing - run '
                             'scripts/figures/_fig8_fetch_cna.py first.' % p)

    print('coords  :')
    coords = ensure_gene_coords()
    print('    %d/%d panel genes' % (len(coords), 18))
    rec = json.load(open(os.path.join(RES, '_recurrence_analysis.json'),
                         encoding='utf-8'))
    print('    %d recurring edges in the recurrence table'
          % len(rec['aligned']['shared_edges']))

    print('cohorts :')
    s2c = build_sample2cohort()
    print('    %d samples over %d cancer types' % (len(s2c), len(set(s2c.values()))))

    print('edges   :')
    edges = write_edges(coords, rec)

    print('roster  :')
    samples, genes = load_matrix(GENES_NPZ, 'panel matrix')
    missing = sorted(set(coords) - set(genes))
    if missing:
        print('    WARNING: panel genes absent from the matrix: %s' % missing)
    write_cna_results(samples, genes, s2c, edges)

    print('null    :')
    ref = ensure_refgene()
    allcoords = load_gene_coords(ref)
    print('    %d symbols with a position on a major chromosome' % len(allcoords))
    psamples, pgenes = load_matrix(POOL_NPZ, 'null pool matrix')
    if list(psamples) != list(samples):
        print('    WARNING: the two matrices list their samples in a different '
              'order; the null assumes they agree')
    write_null_dist(psamples, pgenes, s2c, allcoords, edges)
    write_null_results(psamples, pgenes, s2c, allcoords)

    print('')
    print('done: 4 files written under %s' % os.path.relpath(RES, ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
