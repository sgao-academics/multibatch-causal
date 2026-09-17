"""Fetch overall-survival records for all 33 TCGA cohorts from cBioPortal.

Same endpoint that produced brca_survival.json (PanCanAtlas 2018 clinical data), so the
cohort definitions match the Xena expression matrices used everywhere else in the project.
Writes one JSON per cohort plus a combined file.
"""
import os
import sys
import json
import time

import requests

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOCAL = os.path.join(BASE, 'data')
# TCGA matrices: ./data/ inside the package, or the folder named by MULTIBATCH_DATA
OUT = os.path.join(os.environ.get('MULTIBATCH_DATA') or _LOCAL,
                   'validation', 'pancan_os')
os.makedirs(OUT, exist_ok=True)
API = "https://www.cbioportal.org/api/studies/%s/clinical-data"
HDR = {"User-Agent": "SSCAGate/1.0"}

# cohort -> cBioPortal study id.  READ is not a separate PanCanAtlas study; it lives in coadread.
COHORTS = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
           'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV',
           'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM',
           'UCEC', 'UCS', 'UVM']
STUDY_OVERRIDE = {'COAD': 'coadread', 'READ': 'coadread'}


def fetch(study):
    r = requests.get(API % ('%s_tcga_pan_can_atlas_2018' % study),
                     params={"clinicalDataType": "PATIENT", "projection": "DETAILED",
                             "pageSize": 100000},
                     timeout=120, headers=HDR)
    if r.status_code != 200:
        return None, 'HTTP %s' % r.status_code
    pat = {}
    for d in r.json():
        pat.setdefault(d.get('patientId', ''), {})[d.get('clinicalAttributeId', '')] = d.get('value', '')
    rows = []
    for pid, a in pat.items():
        m, s = a.get('OS_MONTHS'), a.get('OS_STATUS')
        if not m or not s:
            continue
        try:
            rows.append({'patient': pid, 'os_months': float(m), 'os_status': s})
        except ValueError:
            continue
    return rows, None


def save(name, rows):
    p = os.path.join(OUT, '%s_os.json' % name)
    with open(p, 'w') as f:
        json.dump(rows, f)
    return p


cache = {}
combined = {}
for c in COHORTS:
    if c in cache:
        rows = cache[c]
    else:
        study = STUDY_OVERRIDE.get(c, c.lower())
        t0 = time.time()
        rows, err = fetch(study)
        if err:
            print('%-5s FAILED  %s' % (c, err))
            continue
        cache[c] = rows
        print('%-5s %-40s patients=%-4d events=%-4d  %.1fs'
              % (c, '%s_tcga_pan_can_atlas_2018' % study, len(rows),
                 sum(1 for r in rows if str(r['os_status']).startswith('1')), time.time() - t0))
    combined[c] = rows

for c, rows in combined.items():
    # COAD and READ share the coadread study, so they legitimately hold identical records
    save(c, rows)
save('_ALL', combined)

print('\ncohorts written: %d  ->  %s' % (len(combined), OUT))
tot = sum(len(v) for k, v in combined.items() if k not in ('READ',))
print('total patients (COAD/READ counted once): %d' % tot)
