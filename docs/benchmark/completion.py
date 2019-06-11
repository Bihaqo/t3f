import argparse
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int)
parser.add_argument('--d', type=int)
parser.add_argument('--tt_rank', type=int)
args = parser.parse_args()

case = utils.Completion(args.n, args.d, args.tt_rank)

all_logs.append(utils.benchmark(case))
