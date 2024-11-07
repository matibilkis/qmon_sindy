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


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
parser.add_argument("--noise_level", type=float, default=0.1)

args = parser.parse_args()
global itraj
global noise_level
itraj = args.itraj
noise_level = args.noise_level

alphas = [1e-16, 1e-3, 1e-2, 1.]
def int_seed(alpha):
    os.system("{} numerics/NN/modes/sin/in1_15.py --alpha {} --printing 1 --noise_level {}".format(python,alphas[alpha-1], noise_level))

# with mp.Pool(cores) as p:
#     p.map(int_seed, range(0,cores))
int_seed(itraj)
