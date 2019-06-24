import argparse
import utils

# test_case = utils.BilinearXABX(3, 3, 3, 4, 5)
# utils.test(test_case)

parser = argparse.ArgumentParser()
parser.add_argument('--m', type=int)
parser.add_argument('--n', type=int)
parser.add_argument('--d', type=int)
parser.add_argument('--tt_rank_mat', type=int)
parser.add_argument('--tt_rank_vec', type=int)
args = parser.parse_args()

case = utils.BilinearXABX(args.m, args.n, args.d, args.tt_rank_mat, args.tt_rank_vec)

print(case.settings)
utils.benchmark(case)
