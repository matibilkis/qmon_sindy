import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from numerics.utilities.misc import *
import torch
from tqdm import tqdm
from scipy.linalg import solve_continuous_are
from numerics.NN.models.osc_exp_dec import *
from numerics.NN.losses import *
from numerics.NN.misc import *
import copy
import argparse
import time

if __name__ == "__main__":

    mode="osc-exp-dec"
    
    start = time.time()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    
    args = parser.parse_args()
    itraj = args.itraj ###this determines the seed
    printing=False
    
    torch.manual_seed(0)

    
    x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
    dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
    f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)
    
        ####
    params, exp_path = give_params(mode="osc-exp-dec")
    gamma, omega, n, eta, kappa, b, [periods, ppp] = params
    period = (2*np.pi/omega)
    total_time = period*periods
    dt = period/ppp
    times = np.arange(0,total_time+dt,dt)
    ###
    
    dev = torch.device("cpu")

    gf = .4
    wf=4.
    f0=200.
    inputs_cell = [dt,  [gamma, omega, n, eta, kappa, b], [[f0,0.], [[-gf,wf],[-wf,-gf]]  ]]
    
    rrn = RecurrentNetwork(inputs_cell)

    optimizer = torch.optim.Adam(list(rrn.parameters()), lr=1e-2)

    dys = torch.tensor(data=dy, dtype=torch.float32).to(torch.device("cpu"))

    xs_hat, dys_hat, fs_hats = rrn(dys)
    loss = log_lik(dys, dys_hat)
    history = {}
    history["losses"] = [ [loss.item(),err_f(f[:,0],fs_hats[:,0])]  ]
    history["params"] = [[k.detach().data for k in list(rrn.parameters())]]
    history["gradients"] = []
    
    if printing==True:

        print(loss.item())
        print(err_f(f[:,0],fs_hats[:,0]))
        print(history["params"][-1])
        print("\n")

    for ind in range(3000):
        xs_hat, dys_hat, fs_hats = rrn(dys)
        loss = log_lik(dys, dys_hat, dt=dt)
        loss.backward()
        signal_distance = err_f(f[:,0],fs_hats[:,0])
        optimizer.step()


        history["losses"].append([loss.item(),signal_distance] )
        history["params"].append([k.detach().data for k in copy.deepcopy(list(rrn.parameters()))])
        history["gradients"].append(copy.deepcopy([k.grad.numpy() for k in list(rrn.parameters())]))

        if printing==True:

            print("**** iteration {} ****".format(ind))
            print(loss.item())
            print(signal_distance)
            print(history["params"][-1])
            print("\n")
        optimizer.zero_grad()
        save_history(history, itraj=itraj, exp_path=exp_path,what="osc-dec-exp_Agw")

        if (np.abs(loss.item()) < 1+1e-7) or (time.time() - start > 1.95*3600):
            break

