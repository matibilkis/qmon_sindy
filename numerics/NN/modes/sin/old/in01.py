import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from numerics.utilities.misc import *
import torch
from tqdm import tqdm
from scipy.linalg import solve_continuous_are
from numerics.NN.models.sin.in01 import *
from numerics.NN.losses import *
from numerics.NN.misc import *
import copy
import argparse
import time


if __name__ == "__main__":

    mode="sin"
    id_NN = "in01"

    start = time.time()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    parser.add_argument("--printing", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1e-16)
    parser.add_argument("--noise_level", type=float, default=1e-16)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    testing=False
    if testing is True:
        itraj, alpha, printing = 1, 0., 1
        # %load_ext autoreload
        # %autoreload 2

    printing=args.printing
    itraj = args.itraj ###this determines the seed

    printing=[False,True][printing]

    alpha, lr, noise_level = args.alpha, args.lr, args.noise_level

    torch.manual_seed(itraj)
    np.random.seed(itraj)

    x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
    dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
    f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)

        ####
    params, exp_path = give_params(mode=mode)
    gamma, omega, n, eta, kappa, params_force, [periods, ppp] = params
    period = (2*np.pi/omega)
    total_time = period*periods
    dt = period/ppp
    times = np.arange(0,total_time+dt,dt)
    ###


    [omega_ext] = np.array(params_force[1])
    dev = torch.device("cpu")
    K0 = np.zeros(2)
    K1 = np.array([[0,-omega_ext],[omega_ext,0]])
    K1 = K1 + np.random.randn(*list(K1.shape))*np.std(K1)*noise_level
    K2 = np.random.uniform()*noise_level


    initial_condition = np.array(params_force[0])
    initial_condition+=np.random.randn(*list(initial_condition.shape))*np.std(initial_condition)*noise_level
    initial_condition=list(initial_condition.astype("float32"))

    K0 = K0.astype("float32")
    K1 = K1.astype("float32")

    inputs_cell = [dt,  [gamma, omega, n, eta, kappa, params_force], [initial_condition, K0, K1 ]]
    rrn = RecurrentNetwork(inputs_cell)
    optimizer = torch.optim.Adam(list(rrn.parameters()), lr=lr)
    dys = torch.tensor(data=dy, dtype=torch.float32).to(torch.device("cpu"))

    xs_hat, dys_hat, fs_hats = rrn(dys)
    loss, loss_terms = log_lik(dys, dys_hat, model=rrn, alpha=alpha, dt=dt)
    history = {}
    history["losses"] = [ [loss.item(),loss_terms, err_f(f[:,0],fs_hats[:,0])]  ]
    history["params"] = [[k.detach().data for k in list(rrn.parameters())]]
    history["gradients"] = []
    history["optimizer"] = [optimizer.state_dict()]

    if printing==True:
        print("ind: ", 0)
        print(loss.item())
        print(err_f(f[:,0],fs_hats[:,0]))
        print(history["params"][-1])
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
            print(loss.item(), loss_terms)
            print(signal_distance)
            print(history["params"][-1])
            print("\n")
        if (np.abs(loss.item()) < 1+1e-7) or (time.time() - start > 47.9*3600):
            break
