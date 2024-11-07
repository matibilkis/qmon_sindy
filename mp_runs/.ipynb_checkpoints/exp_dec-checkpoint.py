import os
import multiprocessing as mp
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
global itraj
itraj = args.itraj

cores = 8
python = "/data/jupyter/software/envs/master/bin/python3.11"
def int_seed(seed):
    os.system("{} numerics/integration/external_forces/exp-dec.py --itraj {}".format(python,itraj+seed))
    os.system("{} numerics/NN/modes/exp_dec.py --itraj {}".format(python,itraj+seed))

with mp.Pool(cores) as p:
    p.map(int_seed, range(0,cores))
