import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from numerics.utilities.misc import *
import torch
from tqdm import tqdm
from scipy.linalg import solve_continuous_are
from numerics.NN.models.sin.in01234 import *
from numerics.NN.losses import *
from numerics.NN.misc import *
import copy
import argparse
import time

%load_ext autoreload
%autoreload 2

mode="sin"
id_NN = "in01234"

itraj = 1
alpha = 1e-15
lr = 1e-4
noise_level = 1e-10
torch.manual_seed(itraj)
np.random.seed(itraj)

x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)


x.shape

params, exp_path = give_params(mode=mode)
gamma, omega, n, eta, kappa, params_force, [periods, ppp] = params
period = (2*np.pi/omega)
total_time = period*periods
dt = period/ppp
times = np.arange(0,total_time+dt,dt)

[omega_ext] = np.array(params_force[1])
dev = torch.device("cpu")
K1 = cast(np.array([[0,omega_ext],[-omega_ext,0]]) )#+ np.random.rand(2,2)*noise_level)
zero = np.zeros((2,2))
K2 = K3= K4= cast(zero)#give_random_simp() + np.random.rand(2,2)*noise_level)
# K3 = cast(give_random_simp() + np.random.rand(2,2)*noise_level)
# K4 = cast(give_random_simp() + np.random.rand(2,2)*noise_level)

initial_condition = np.array(params_force[0]) + np.random.rand(2)*noise_level
initial_condition=list(initial_condition.astype("float32"))

inputs_cell = [dt,  [gamma, omega, n, eta, kappa, params_force], [initial_condition, K1, K2, K3, K4 ]]
rrn = RecurrentNetwork(inputs_cell)
optimizer = torch.optim.Adam(list(rrn.parameters()), lr=lr)
dys = torch.tensor(data=dy, dtype=torch.float32).to(torch.device("cpu"))

xs_hat, dys_hat, fs_hats = rrn(dys)

ls,lw=15,3
fig2 = plt.figure(figsize=(35,5))
ax=plt.subplot(161)
ax.plot(xs_hat.detach().numpy()[:,0], color="red",marker='.')
ax.plot(x[:,0])
ax=plt.subplot(162)
ax.plot(xs_hat.detach().numpy()[:,1], color="red",marker='.')
ax.plot(x[:,1])
ax=plt.subplot(163)
ax.plot(dys[:,0])
ax.plot(dys_hat.detach().numpy()[:,0], color="red",marker='.')

ax=plt.subplot(164)
ax.plot(fs_hats.detach().numpy()[:,0], color="red",marker='.')
ax.plot(f[:,0])

plt.plot(fs_hats.detach().numpy().squeeze()[:,0])

loss, loss_terms = log_lik(dys, dys_hat, model=rrn, alpha=alpha, dt=dt)
signal_distance = err_f(f[:,0],fs_hats[:,0])

signal_distance
initial_condition


xs_hat








    history = {}
    history["losses"] = [ [loss.item(),loss_terms, err_f(f[:,0],fs_hats[:,0])]  ]
    history["params"] = [[k.detach().data for k in list(rrn.parameters())]]
    history["gradients"] = []
    history["optimizer"] = [optimizer.state_dict()]

    if printing==True:
        print("**** iteration {} ****".format(-1))
        print("losses: ",loss.item(), loss_terms)
        print("MSE signal, true ",signal_distance)
        print("params: ",history["params"][-1])
        print("\n")

    for ind in range(int(1e3)):
        xs_hat, dys_hat, fs_hats = rrn(dys)
        loss, loss_terms = log_lik(dys, dys_hat, dt=dt, alpha=alpha, model=rrn)
        loss.backward()
        signal_distance = err_f(f[:,0],fs_hats[:,0])
        optimizer.step()


        history["losses"].append([loss.item(),loss_terms,signal_distance] )
        history["params"].append([k.detach().data for k in copy.deepcopy(list(rrn.parameters()))])
        history["gradients"].append(copy.deepcopy([k.grad.numpy() for k in list(rrn.parameters())]))
        history["optimizer"].append(optimizer.state_dict())

        optimizer.zero_grad()
        dire = save_history(history, itraj=itraj, exp_path=exp_path,what="{}/{}_{}_{}".format(mode+id_NN,alpha, lr, noise_level))
        if printing==True:#ind%10==0:
            print("saving in {}".format(dire))
            print("ind: ", ind)
            print("**** iteration {} ****".format(ind))
            print("losses: ",loss.item(), loss_terms)
            print("MSE signal, true ",signal_distance)
            print("params: ",history["params"][-1])
            print("\n")
        if (np.abs(loss.item()) < 1+1e-7) or (time.time() - start > 47.9*3600):
            break
