import os
import multiprocessing as mp
from datetime import datetime
import argparse
import numpy as np
import socket
import getpass

user = getpass.getuser()
uu = socket.gethostname()
if uu in ["pop-os"]:
    python = "python3"
else:
    python = "/data/jupyter/software/envs/master/bin/python3.11"
cores=7


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
global itraj
itraj = args.itraj

alphas = np.logspace(-6,1,8)
def int_seed(alpha):
    # os.system("{} numerics/integration/external_forces/sin.py --itraj 1".format(python))
    os.system("{} numerics/NN/modes/sin.py --alpha {} --lr 0.0001".format(python,alphas[alpha-1]))

with mp.Pool(cores) as p:
    p.map(int_seed, range(0,cores))
