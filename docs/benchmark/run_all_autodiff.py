"""
Running a suite of autodiff benchmarks.
  run_all_autodiff.py --logs=autodiff_cpu.pkl 2> autodiff_cpu.stderr
"""

import argparse
import subprocess
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--logs', type=str)
args = parser.parse_args()

def run_single(case, n, d, r, R=None):
  cmd = ['python3', 'run_single_autodiff.py', '--case=%s' % case,
         '--n=%d' % n, '--d=%d' % d,
         '--tt_rank_vec=%d' % r, '--logs=%s' % args.logs]
  if R is not None:
    cmd.append('--tt_rank_mat=%d' % R)
    cmd.append('--m=%d' % n)
  try:
    print(' '.join(cmd))
    print(subprocess.check_output(cmd))
  except:
    print('Running subprocess failed.')
    pass


for n in [20, 100, 500]:
  for d in [10, 20, 40]:
    for r in [5, 10, 20]:
      run_single('completion', n, d, r)
      run_single('ExpMachines', n, d, r)
      for R in [5, 10, 20]:
        run_single('xAx', n, d, r, R)
        run_single('xABx', n, d, r, R)
        run_single('RayleighQuotient', n, d, r, R)
