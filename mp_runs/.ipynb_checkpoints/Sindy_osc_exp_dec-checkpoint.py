import os
import multiprocessing as mp
from datetime import datetime
import argparse
import numpy as np

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
global itraj
itraj = args.itraj

python = "/data/jupyter/software/envs/master/bin/python3.11"
alphas = list(np.linspace(0., 1., 8))
#alphas = [0.0, 1e-6, 0.5e-5, 1e-5, 0.5e-4, 1e-4, 0.5e-3,1e-3, 0.5e-2,1e-2, 0.5, 0.1]
def int_seed(alpha, seed=0):
    #os.system("{} numerics/integration/external_forces/osc-exp-dec.py --itraj {}".format(python,itraj+seed))
    #os.system("{} numerics/NN/modes/Sindy_osc_exp_dec.py --itraj {} --printing 0".format(python,itraj+seed))
    os.system("{} numerics/NN/modes/Sindy_osc_exp_dec.py --alpha {} --printing 0".format(python,alphas[itraj]))

# with mp.Pool(cores) as p:
#     p.map(int_seed, range(0,cores))
int_seed(itraj)
