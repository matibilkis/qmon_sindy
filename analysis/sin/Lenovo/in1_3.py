%load_ext autoreload
%autoreload 2
import numpy as np
import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
import matplotlib.pyplot as plt
from numerics.NN.models.sin.in1_3 import *
from numerics.NN.misc import *
import torch
import numpy as np
from scipy.linalg import solve_continuous_are
from tqdm import tqdm

import matplotlib.pyplot as plt
from numerics.NN.misc import *

itraj=1
mode="sin"
id_NN = "in1_3"
alpha, lr, tmp_net = 1e-16, 1e-3, 0

rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path = get_plot_data_NN(itraj=1, mode="sin",id_NN="in1_3")

fig1, fig2 = load_rnn_and_plot(rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path, itraj=1, alpha=alpha, lr=lr, tmp_net=tmp_net)
os.makedirs("readme_plots/",exist_ok=True)
fig1.savefig("readme_plots/fig1.png")
fig2.savefig("readme_plots/fig2.png")








itraj=1
mode="sin"
id_NN = "in01"
alpha,noise_level, lr = 1e-16, 1e-2, 1e-4#, 1e-5

rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path = get_plot_data_NN(itraj=1, mode="sin",id_NN="in01",noise_level=noise_level)
fig1, fig2 = load_rnn_and_plot(rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path, itraj=1, alpha=alpha, lr=lr, noise_level=noise_level)

fig1


fig2






itraj=1
mode="sin"
id_NN = "in01"
alpha,noise_level, lr = 1e-12, 1e-2, 1e-4#, 1e-5

rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path = get_plot_data_NN(itraj=1, mode="sin",id_NN="in01",noise_level=noise_level)
fig1, fig2 = load_rnn_and_plot(rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path, itraj=1, alpha=alpha, lr=lr, noise_level=noise_level)


fig1


fig2





###

itraj=1
mode="sin"
id_NN = "in01"
alpha,noise_level, lr = 1e-12, 1e-1, 1e-4#, 1e-5

rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path = get_plot_data_NN(itraj=1, mode="sin",id_NN="in01",noise_level=noise_level)
fig1, fig2 = load_rnn_and_plot(rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path, itraj=1, alpha=alpha, lr=lr, noise_level=noise_level)

fig1

fig2














###

itraj=1
mode="sin"
id_NN = "in01"
alpha,noise_level, lr = 1e-12, 1e-1, 1e-2

rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path = get_plot_data_NN(itraj=1, mode="sin",id_NN="in01",noise_level=noise_level)
fig1, fig2 = load_rnn_and_plot(rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path, itraj=1, alpha=alpha, lr=lr, noise_level=noise_level)


fig1

fig2





###

itraj=1
mode="sin"
id_NN = "in01"
alpha,noise_level, lr = 1e-12, 1e-1, 1e-3

rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path = get_plot_data_NN(itraj=1, mode="sin",id_NN="in01",noise_level=noise_level)
fig1, fig2 = load_rnn_and_plot(rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path, itraj=1, alpha=alpha, lr=lr, noise_level=noise_level)


fig1


fig2














#
