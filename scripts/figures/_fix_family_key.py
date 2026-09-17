"""Recompute the composition of the gene-pairs shared by >=3 cancers, using a documented
same-family rule: the first three alphanumeric characters of the two gene symbols must match
(e.g. KLK7/KLK8 -> KLK, TMPRSS11A/TMPRSS11D -> TMP).  Sex-chromosome / X-inactivation genes are
counted first and excluded from the family bucket.  Only results/_cancer_pair_sharing.json is
rewritten; the sharing matrix itself is preserved.
"""
import json, os, re, sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, 'results')

nt = json.load(open(os.path.join(RESULTS, '_pipeline_notears.json'), encoding='utf-8'))
sh = json.load(open(os.path.join(RESULTS, '_cancer_pair_sharing.json'), encoding='utf-8'))

SEX = {'XIST', 'TSIX', 'RPS4Y1', 'RPS4Y2', 'DDX3Y', 'EIF1AY', 'KDM5D', 'USP9Y', 'UTY', 'NLGN4Y',
       'ZFY', 'TMSB4Y', 'PRKY', 'TXLNGY', 'TTTY15', 'CYORF15A', 'CYORF15B', 'GSTM1', 'AMELY',
       'PCDH11Y'}

def key3(g):
    m = re.match(r'[A-Za-z0-9]{0,3}', g)
    return m.group(0).upper() if m else g

pair_count = Counter()
for c in sorted(k for k in nt if isinstance(nt[k], dict) and 'W' in nt[k]):
    W = nt[c]['W']; genes = nt[c]['genes']
    for i in range(100):
        wi = W[i]
        for j in range(100):
            if i != j and abs(wi[j]) > 0.3:
                pair_count[(genes[i], genes[j])] += 1

shared3 = sorted([p for p, n in pair_count.items() if n >= 3])
sex_hits, fam_hits, other = [], [], []
for a, b in shared3:
    if a in SEX or b in SEX:
        sex_hits.append([a, b])
    elif key3(a) == key3(b):
        fam_hits.append([a, b])
    else:
        other.append([a, b])

print('shared >= 3 : %d' % len(shared3))
print('  sex-chromosome / X-inactivation : %d  %s' % (len(sex_hits), sex_hits))
print('  same gene family (3-char prefix) : %d  %s' % (len(fam_hits), fam_hits))
print('  other : %d  %s' % (len(other), other))

sh['shared_set_composition'] = {
    'sex_chromosome': len(sex_hits),
    'paralogous_same_family': len(fam_hits),
    'other': len(other),
}
sh['shared_ge3_pairs'] = {'sex_chromosome': sex_hits,
                          'same_family': fam_hits,
                          'other': other}
sh['family_key_rule'] = 'first three alphanumeric characters of the gene symbol'
sh['top_shared_pairs'] = [[a, b, n] for (a, b), n in pair_count.most_common(20)]
json.dump(sh, open(os.path.join(RESULTS, '_cancer_pair_sharing.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('rewrote _cancer_pair_sharing.json')
