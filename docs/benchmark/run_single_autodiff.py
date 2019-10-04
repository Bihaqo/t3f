import argparse
import utils
import pickle
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--logs', type=str)
parser.add_argument('--case', type=str)
parser.add_argument('--m', type=int)
parser.add_argument('--n', type=int)
parser.add_argument('--d', type=int)
parser.add_argument('--tt_rank_mat', type=int)
parser.add_argument('--tt_rank_vec', type=int)
args = parser.parse_args()

if args.case == 'completion':
  assert args.m is None and args.tt_rank_mat is None
  case = utils.Completion(args.n, args.d, args.tt_rank_vec)
elif args.case == 'xAx':
  case = utils.BilinearXAX(args.m, args.n, args.d, args.tt_rank_mat, args.tt_rank_vec)
elif args.case == 'xABx':
  case = utils.BilinearXABX(args.m, args.n, args.d, args.tt_rank_mat, args.tt_rank_vec)
elif args.case == 'ExpMachines':
  assert args.m is None and args.tt_rank_mat is None
  case = utils.ExpMachines(args.n, args.d, args.tt_rank_vec)
elif args.case == 'RayleighQuotient':
  case = utils.RayleighQuotient(args.m, args.n, args.d, args.tt_rank_mat, args.tt_rank_vec)
else:
  print('Dont know this case.')

print(args.case, case.settings)
utils.benchmark(args.case, case, args.logs)


