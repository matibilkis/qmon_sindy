
%load_ext autoreload
%autoreload 2
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from numerics.utilities.misc import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numerics.NN.losses import *
from numerics.NN.misc import *

mode="sin"
id_NN = "in1_3"

params, exp_path = give_params(mode=mode)
gamma, omega, n, eta, kappa, params_force, [periods, ppp] = params
period = (2*np.pi/omega)
total_time = period*periods
dt = period/ppp
times = np.arange(0,total_time+dt,dt)

def kernel(x,id_NN):
    if id_NN=="in1_3":
        f1 = K1.dot(x)
        f2 = K2.dot(x**2)
        f3 = K3.dot(x*np.flip(x,[-1]))
        return f1 + f2 + f3

def rk_step(x,id_NN):
    k1 = kernel(x,id_NN)*dt
    k2 = dt*kernel(x+k1/2.,id_NN)
    k3 = dt*kernel(x+k2/2.,id_NN)
    k4 = dt*kernel(x+k3,id_NN)
    return (k1+2*k2+2*k3+k4)/6.

tmp_net = 0
initial_params_net = w0_net(mode, id_NN, tmp_net)
K1, K2, K3 = initial_params_net[1:]

signal = [np.array(initial_params_net[0])]
for ind,t in enumerate(times[:-1]):
    signal.append(signal[ind] + rk_step(signal[ind],id_NN))
signal = np.stack(signal)

ax=plt.subplot()
ax.plot(times,signal[:,0])
