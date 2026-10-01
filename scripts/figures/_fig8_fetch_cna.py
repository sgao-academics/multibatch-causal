# -*- coding: utf-8 -*-
"""Fetch the copy-number matrices Figure 8's correlations are computed from.

Figure 8 asks whether the 14 recurring gene pairs are simply two genes sharing a
copy-number state -- correlated expression with no regulatory relation behind
it.  Answering that needs the copy-number calls themselves, and the only public
source of them is the pan-cancer GISTIC 2 thresholded table that UCSC Xena
distributes: 84.7 MB gzipped, 10,845 samples x ~24,000 genes, one row per gene.

That file is never written to disk.  It is streamed once and only the rows the
two downstream steps read are kept:

* ``data/fig8_cna_genes.npz`` -- the 18 genes the 14 recurring edges touch, under
  the manuscript's own symbols.  Three of them are matched under a newer symbol
  as well (IL8/CXCL8, IGJ/JCHAIN, MGC29506/MZB1).  ``_fig8_prep_copynumber.py``
  turns these into ``results/_fig8_cna_results.json``, which Figure 8b draws on.

* ``data/fig8_cna_pool.npz`` -- a seeded random sample of genes on the major
  chromosomes.  This is the pool the distance-stratified null is drawn from;
  without it, "rho = 0.98 for a pair 13 kb apart" cannot be read against what a
  neighbouring *random* pair shows.  The draw advances once per gene row of the
  file, so the same rows come out on every run and the null is reproducible
  rather than merely random.

Both are TCGA-derived measurements.  They stay out of version control under the
``data/*`` exclusion in .gitignore and are rebuilt here; the two annotation
tables the package does carry (cytoband stains, gene spans) are public genome
annotation rather than measurements.

Usage
-----
    python scripts/figures/_fig8_fetch_cna.py                 # both matrices
    python scripts/figures/_fig8_fetch_cna.py --what genes    # just the 18
    python scripts/figures/_fig8_fetch_cna.py --what pool     # just the pool

Output: data/fig8_cna_genes.npz, data/fig8_cna_pool.npz
"""
import gzip
import os
import random
import sys
import time
import urllib.request

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
DAT = os.path.join(ROOT, 'data')
os.makedirs(DAT, exist_ok=True)

URL = ('https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/'
       'TCGA.PANCAN.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz')
REFGENE_URL = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz'
UA = {'User-Agent': 'multibatch-fig8/1.0'}

GENES_NPZ = os.path.join(DAT, 'fig8_cna_genes.npz')
POOL_NPZ = os.path.join(DAT, 'fig8_cna_pool.npz')
REFGENE = os.path.join(DAT, 'hg19_refGene.txt.gz')

# manuscript symbol -> the symbols the GISTIC table may carry it under
PANEL_LOOKUP = {
    'CXCL9': ['CXCL9'], 'CXCL10': ['CXCL10'], 'CXCL11': ['CXCL11'],
    'CXCL1': ['CXCL1'], 'IL8': ['IL8', 'CXCL8'], 'C7': ['C7'],
    'PLCXD3': ['PLCXD3'], 'HBB': ['HBB'], 'HBA1': ['HBA1'], 'IL6': ['IL6'],
    'FOSB': ['FOSB'], 'ADAM6': ['ADAM6'], 'IGJ': ['IGJ', 'JCHAIN'],
    'MGC29506': ['MZB1', 'MGC29506'], 'COL10A1': ['COL10A1'],
    'COL11A1': ['COL11A1'], 'CCL18': ['CCL18'], 'CHIT1': ['CHIT1'],
}
_PANEL = {a: canon for canon, alts in PANEL_LOOKUP.items() for a in alts}

# The genes under test never re-enter the null pool as "random" partners, and
# their rows do not advance the draw.
POOL_EXCLUDE = {'CXCL9', 'CXCL10', 'CXCL11', 'CXCL1', 'IL8', 'CXCL8', 'C7',
                'PLCXD3', 'HBB', 'HBA1', 'IL6', 'FOSB', 'ADAM6', 'IGJ',
                'JCHAIN', 'MZB1', 'COL10A1', 'COL11A1', 'CCL18', 'CHIT1'}

POOL_SEED = 42
POOL_RATE = 0.055
POOL_MAX = 1200


def ensure_refgene():
    """The hg19 refGene table, downloaded once and cached under data/.

    Needed before the draw, not after: the pool is restricted to genes that have
    a position on a major chromosome, so the coordinates have to be in hand while
    the copy-number rows stream past.
    """
    if os.path.exists(REFGENE):
        print('refGene  : using cached %s' % os.path.relpath(REFGENE, ROOT))
        return REFGENE
    print('refGene  : downloading %s' % REFGENE_URL)
    req = urllib.request.Request(REFGENE_URL, headers=UA)
    data = urllib.request.urlopen(req, timeout=300).read()
    with open(REFGENE, 'wb') as f:
        f.write(data)
    print('refGene  : %.1f MB -> %s' % (len(data) / 1048576,
                                        os.path.relpath(REFGENE, ROOT)))
    return REFGENE


def load_gene_coords(path):
    """(chromosome, midpoint) per symbol, major chromosomes only.

    Columns of refGene are bin, name, chrom, strand, txStart, txEnd, ... name2.
    A symbol may have many transcripts; the span is their union, and an entry is
    kept only when its transcripts agree on a chromosome.
    """
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


def split_row(payload):
    return np.array(payload.decode('utf-8', 'ignore').split('\t'), dtype=np.float32)


def main():
    what = 'both'
    if '--what' in sys.argv:
        what = sys.argv[sys.argv.index('--what') + 1]
    if what not in ('both', 'genes', 'pool'):
        raise SystemExit('--what must be one of: both, genes, pool')

    coords = load_gene_coords(ensure_refgene())
    print('refGene  : %d symbols on the major chromosomes' % len(coords))

    req = urllib.request.Request(URL, headers=UA)
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=1800)
    print('GISTIC   : HTTP %s  content-length %s bytes'
          % (resp.status, resp.headers.get('Content-Length')))
    g = gzip.GzipFile(fileobj=resp)
    samples = g.readline().decode('utf-8', 'ignore').rstrip('\n').split('\t')[1:]
    print('GISTIC   : %d samples in the matrix' % len(samples))

    rng = random.Random(POOL_SEED)
    genes, pool, n_rows, n_pool = {}, {}, 0, 0
    for line in g:
        n_rows += 1
        i = line.find(b'\t')
        name = line[:i].decode('utf-8', 'ignore').strip()
        if name in _PANEL and what in ('both', 'genes'):
            genes[_PANEL[name]] = split_row(line[i + 1:])
        if name in POOL_EXCLUDE:
            continue                     # tested genes do not advance the draw
        if name in coords and rng.random() < POOL_RATE and n_pool < POOL_MAX:
            if what in ('both', 'pool'):
                pool[name] = split_row(line[i + 1:])
            n_pool += 1

    print('GISTIC   : scanned %d gene rows in %.0f s' % (n_rows, time.time() - t0))

    rc = 0
    if what in ('both', 'genes'):
        missing = sorted(set(PANEL_LOOKUP) - set(genes))
        np.savez_compressed(GENES_NPZ, samples=np.array(samples), **genes)
        print('panel    : %d/%d genes -> %s (%.1f KB)'
              % (len(genes), len(PANEL_LOOKUP), os.path.relpath(GENES_NPZ, ROOT),
                 os.path.getsize(GENES_NPZ) / 1024.0))
        if missing:
            print('panel    : MISSING %s' % missing)
            rc = 1
    if what in ('both', 'pool'):
        np.savez_compressed(POOL_NPZ, samples=np.array(samples), **pool)
        print('pool     : %d genes -> %s (%.1f KB)'
              % (len(pool), os.path.relpath(POOL_NPZ, ROOT),
                 os.path.getsize(POOL_NPZ) / 1024.0))
    return rc


if __name__ == '__main__':
    sys.exit(main())
