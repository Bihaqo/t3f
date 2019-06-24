import subprocess
import utils

test_case = utils.Completion(3, 3, 4)
utils.test(test_case)

for n in [20, 100, 500]:
  for d in [10, 20, 30]:
    for r in [5, 10, 20]:
      print(subprocess.check_output(['python3', 'completion.py', '--n=%d' % n, '--d=%d' % d, '--r=%d' % r]))