"""Prepare the data underlying Figure 1 (pan-cancer expression landscape).

Reads the per-cancer TCGA HiSeqV2 matrices, extracts every gene that occurs in any of the
33 inferred networks, and stores the resulting sub-matrices together with a sample-type
annotation. All downstream panels of Figure 1 are derived from this one file, so no number
in the figure is transcribed by hand.

Output: results/_fig1_landscape.npz
"""
import os, sys, glob, json, gzip
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, 'results')
# TCGA matrices: ./data/ inside the package, or the folder named by MULTIBATCH_DATA
_LOCAL = os.path.join(BASE, 'data')
DATA = os.environ.get('MULTIBATCH_DATA') or _LOCAL
OUT = os.path.join(RESULTS, '_fig1_landscape.npz')

# ---- how we group the TCGA sample-type codes ----
TUMOR_CODES = {'01', '02', '03', '05'}
NORMAL_CODES = {'11'}
META_CODES = {'06', '07'}


def group_of(code):
    if code in TUMOR_CODES:
        return 'Tumor'
    if code in NORMAL_CODES:
        return 'Normal'
    if code in META_CODES:
        return 'Metastatic'
    return None


net = json.load(open(os.path.join(RESULTS, '_per_cancer_network.json'), encoding='utf-8'))
cancers = sorted(net.keys())
print('cancers in network file: %d' % len(cancers))

# gene universe = every gene appearing in any per-cancer network (includes all hubs)
genes = sorted({g for c in cancers for g in net[c]['genes']})
gidx = {g: i for i, g in enumerate(genes)}
# The Xena HiSeqV2 matrices spell the unnamed-reading-frame loci in mixed case ("C11orf86")
# while the network tables use upper case ("C11ORF86"). Match case-insensitively, otherwise
# those rows are silently dropped.
gidx_ci = {}
for i, g in enumerate(genes):
    gidx_ci.setdefault(g.upper(), i)
print('network gene universe: %d unique symbols' % len(genes))

store = {}
manifest = {}
for cancer in cancers:
    p = os.path.join(DATA, 'TCGA_%s_HiSeqV2.tsv' % cancer)
    if not os.path.exists(p):
        print('  MISSING %s' % p)
        continue
    op = gzip.open if p.endswith('.gz') else open
    n_rows_hit = 0
    with op(p, 'rt', encoding='utf-8', errors='ignore') as f:
        hdr = f.readline().rstrip('\n').split('\t')
        samples = hdr[1:]
        groups = []
        for s in samples:
            parts = s.split('-')
            code = parts[3][:2] if len(parts) >= 4 else '??'
            groups.append(group_of(code))
        M = np.full((len(genes), len(samples)), np.nan, dtype=np.float32)
        for line in f:
            i = line.find('\t')
            if i < 0:
                continue
            g = line[:i]
            j = gidx_ci.get(g.upper())
            if j is None:
                continue
            vals = line[i + 1:].rstrip('\n').split('\t')
            try:
                arr = np.fromiter((float(v) if v not in ('', 'NA', 'NaN') else np.nan
                                   for v in vals), dtype=np.float32, count=len(samples))
            except Exception:
                continue
            M[j] = arr
            n_rows_hit += 1
    store['X_' + cancer] = M
    manifest[cancer] = dict(
        n_samples=len(samples),
        n_tumor=int(sum(1 for g in groups if g == 'Tumor')),
        n_normal=int(sum(1 for g in groups if g == 'Normal')),
        n_metastatic=int(sum(1 for g in groups if g == 'Metastatic')),
        n_unclassified=int(sum(1 for g in groups if g is None)),
        n_genes_hit=n_rows_hit,
        barcodes=np.array(samples, dtype=object),
        groups=np.array([g if g else 'Other' for g in groups], dtype=object),
    )
    print('  %-6s %5d samples (T=%4d N=%3d M=%3d) genes_hit=%5d'
          % (cancer, len(samples), manifest[cancer]['n_tumor'],
             manifest[cancer]['n_normal'], manifest[cancer]['n_metastatic'], n_rows_hit))

np.savez_compressed(OUT, genes=np.array(genes, dtype=object),
                    cancers=np.array(cancers, dtype=object),
                    **store)
# sidecar with the per-cancer sample metadata (kept as JSON, small)
meta = {c: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in manifest[c].items()} for c in manifest}
json.dump(meta, open(os.path.join(RESULTS, '_fig1_manifest.json'), 'w', encoding='utf-8'), indent=1)

print()
print('saved -> %s  (%.1f MB)' % (OUT, os.path.getsize(OUT) / 1e6))
tot_t = sum(m['n_tumor'] for m in manifest.values())
tot_n = sum(m['n_normal'] for m in manifest.values())
tot_m = sum(m['n_metastatic'] for m in manifest.values())
print('TOTAL tumor=%d normal=%d metastatic=%d' % (tot_t, tot_n, tot_m))
print('cancers with >=5 normals: %d' % sum(1 for m in manifest.values() if m['n_normal'] >= 5))
