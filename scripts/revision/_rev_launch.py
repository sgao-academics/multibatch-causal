# -*- coding: utf-8 -*-
"""Detached launcher for the revision night batch (DETACHED_PROCESS -- V8 rule).

usage:  python _rev_launch.py <stages> <lognamestem> [REV_LIMIT] [REV_DIR]
example: python _rev_launch.py all night1
         python _rev_launch.py all smoke 2 _rev_smoke
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The interpreter that runs the launcher is the interpreter the batch needs; MULTIBATCH_PYTHON
# overrides it for a scheduled run that uses a different one.
PY = os.environ.get('MULTIBATCH_PYTHON') or sys.executable
script = os.path.join(HERE, '_rev_night.py')

stages = sys.argv[1] if len(sys.argv) > 1 else 'all'
stem = sys.argv[2] if len(sys.argv) > 2 else 'run'
env = dict(os.environ)
if len(sys.argv) > 3:
    env['REV_LIMIT'] = sys.argv[3]
if len(sys.argv) > 4:
    env['REV_DIR'] = sys.argv[4]
if len(sys.argv) > 5:
    env['REV_WORKERS'] = sys.argv[5]
env.setdefault('REV_WORKERS', '10')

# Pin BLAS to one thread *in the launching environment*, so the parent process and every
# spawned worker see exactly the same setting.  Setting it after numpy is imported (as
# L.burst does) only affects children -- which made the rotated fits non-deterministic
# between the serial and the parallel smoke run.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    env[_v] = '1'

logp = os.path.join(HERE, '_rev_%s.log' % stem)
lf = open(logp, 'w', encoding='utf-8')
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
p = subprocess.Popen([PY, script, stages], stdout=lf, stderr=subprocess.STDOUT,
                     cwd=HERE, env=env,
                     creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
print('launched pid=%d stages=%s log=%s REV_LIMIT=%s REV_DIR=%s'
      % (p.pid, stages, logp, env.get('REV_LIMIT'), env.get('REV_DIR', '_rev')))
