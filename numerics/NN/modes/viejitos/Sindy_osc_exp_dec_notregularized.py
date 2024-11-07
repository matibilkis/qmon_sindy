import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from numerics.utilities.misc import *
import torch
from tqdm import tqdm
from scipy.linalg import solve_continuous_are
from numerics.NN.models.sindy_osc_exp_dec import *
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
    parser.add_argument("--printing", type=int, default=0)
    args = parser.parse_args()

    printing=args.printing
    printing=[False,True][printing]
    itraj = args.itraj ###this determines the seed

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

    K0 = [-.01,.01]

    gf = .1
    wf=.7
    K1 = np.array([[-gf,wf],[-wf,-gf]])

    K2_0 = K2_1 = 0.1*K1

    inputs_cell = [dt,  [gamma, omega, n, eta, kappa, b], [[200.,0], K0, K1, K2_0,K2_1  ]]
    rrn = RecurrentNetwork(inputs_cell)

    optimizer = torch.optim.Adam(list(rrn.parameters()), lr=1e-2)

    dys = torch.tensor(data=dy, dtype=torch.float32).to(torch.device("cpu"))

    xs_hat, dys_hat, fs_hats = rrn(dys)
    loss = log_lik(dys, dys_hat)
    history = {}
    history["losses"] = [ [loss.item(),err_f(f[:,0],fs_hats[:,0])]  ]
    history["params"] = [[k.detach().data for k in list(rrn.parameters())]]
    history["gradients"] = []
    history["optimizer"] = optimizer.state_dict()
    

    if printing==True:

        print(loss.item())
        print(err_f(f[:,0],fs_hats[:,0]))
        print(history["params"][-1])
        print("\n")

    for ind in range(int(1e5)):
        xs_hat, dys_hat, fs_hats = rrn(dys)
        loss = log_lik(dys, dys_hat, dt=dt)
        loss.backward()
        signal_distance = err_f(f[:,0],fs_hats[:,0])
        optimizer.step()


        history["losses"].append([loss.item(),signal_distance] )
        history["params"].append([k.detach().data for k in copy.deepcopy(list(rrn.parameters()))])
        history["gradients"].append(copy.deepcopy([k.grad.numpy() for k in list(rrn.parameters())]))
        history["optimizer"] = optimizer.state_dict()
        if printing==True:

            print("**** iteration {} ****".format(ind))
            print(loss.item())
            print(signal_distance)
            print(history["params"][-1])
            print("\n")
        optimizer.zero_grad()
        save_history(history, itraj=itraj, exp_path=exp_path,what="osc-dec-sindy")

        if (np.abs(loss.item()) < 1+1e-7) or (time.time() - start > 7.95*3600):
            break
