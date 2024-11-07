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
parser.add_argument("--lr", type=float, default=0.01)

args = parser.parse_args()
global itraj
global lr
itraj = args.itraj
lr = args.lr

alphas = [1e-16, 1e-3, 1e-2, 1.]
def int_seed(alpha):
    os.system("{} numerics/NN/modes/sin/in1_15.py --alpha {} --printing 1 --lr {}".format(python,alphas[alpha-1], lr))

# with mp.Pool(cores) as p:
#     p.map(int_seed, range(0,cores))
int_seed(itraj)
