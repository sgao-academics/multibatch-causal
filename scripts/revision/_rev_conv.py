# -*- coding: utf-8 -*-
"""Which index order does the paper's edge convention use?  Cheap: cached W only.

recurrence() counts present[i,j] = #{c : |W_c[i,j]| > tau} and names the pair
genes[i] -> genes[j].  The bootstrap stage must sample the SAME matrix position, so
verify [i,j] against the published support counts before trusting either.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _rev_lib as L

panel = L.load_panel()['panel']
fits, cs = L.load_fits('_shared_panel_notears.json')
M, med = L.median_matrix(fits, cs, dref=len(panel))
present = L.recurrence(M, L.TAU)

rec = L.recurrence_list(present, panel, 10)
print('recurring pairs (>=10 cohorts) :', len(rec))
for r in rec:
    i, j = panel.index(r['src']), panel.index(r['dst'])
    fwd = int(sum(1 for c in cs if abs(float(np.array(fits[c]['W'])[i][j])) > L.TAU))
    rev = int(sum(1 for c in cs if abs(float(np.array(fits[c]['W'])[j][i])) > L.TAU))
    print('  %-9s -> %-9s  n_reported=%2d   |W[i,j]|=%2d   |W[j,i]|=%2d'
          % (r['src'], r['dst'], r['n'], fwd, rev))

print()
print('median matrix: sum(med>tau) =', int((med > L.TAU).sum()), '(paper says 6)')
