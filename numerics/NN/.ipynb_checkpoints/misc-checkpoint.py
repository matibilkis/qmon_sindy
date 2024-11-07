import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
import pickle
import torch

def give_path_model(what="NN",exp_path="",itraj=1, periods=10., ppp=500):
    return get_def_path(what=what)+exp_path + "{}itraj/".format(itraj)

def save_history(history,what="NN",exp_path="", itraj=1, periods=10., ppp=500):

    path_model= give_path_model(what=what,exp_path = exp_path, itraj=itraj)
    os.makedirs(path_model, exist_ok=True)

    with open(path_model+"history.pickle", "wb") as output_file:
       pickle.dump(history, output_file)
    return path_model

def load_history(what="NN",exp_path="",itraj=1, periods=10., ppp=500):
    path_model= give_path_model(what=what,exp_path = exp_path, itraj=itraj)
    with open(path_model+"history.pickle", "rb") as output_file:
       h=pickle.load(output_file)
    return h

def set_params_to_best(rrn, history):
    ll = history["losses"]
    loss_fun = np.array([ll[k][0] for k in range(len(ll))])
    index_favorite = np.argmin(loss_fun)
    news = history["params"][index_favorite]
    with torch.no_grad():

        for j,k in zip(news, list(rrn.parameters())):
            k.data = torch.tensor(j)
    return index_favorite


def give_random_simp():
    aa = np.random.random()
    return np.array([[0,aa],[-aa,0]])

def cast(a):
    return a.astype("float32")



def get_plot_data_NN(itraj, mode="sin",id_NN="in01", noise_level=0.1):
    if mode=="sin":
        if id_NN == "in01":
            from numerics.NN.models.sin.in01 import RecurrentNetwork

            torch.manual_seed(itraj)
            np.random.seed(itraj)

            x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
            dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
            f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)

            params, exp_path = give_params(mode=mode)
            gamma, omega, n, eta, kappa, params_force, [periods, ppp] = params
            period = (2*np.pi/omega)
            total_time = period*periods
            dt = period/ppp
            times = np.arange(0,total_time+dt,dt)

            [omega_ext] = np.array(params_force[1])
            dev = torch.device("cpu")
            K0 = np.zeros(2)
            K1 = np.array([[0,omega_ext],[-omega_ext,0]])
            K1 = K1 + np.random.randn(*list(K1.shape))*np.std(K1)*noise_level

            initial_condition = np.array(params_force[0])
            initial_condition+=np.random.randn(*list(initial_condition.shape))*np.std(initial_condition)*noise_level
            initial_condition=list(initial_condition.astype("float32"))

            K0 = K0.astype("float32")
            K1 = K1.astype("float32")

            inputs_cell = [dt,  [gamma, omega, n, eta, kappa, params_force], [initial_condition, K0, K1 ]]

            rrn = RecurrentNetwork(inputs_cell)
            dys = torch.tensor(data=dy, dtype=torch.float32).to(torch.device("cpu"))
            ixs_hat, idys_hat, ifs_hats = rrn(dys)

            return rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path
        elif id_NN == "in01234":
            from numerics.NN.models.sin.in01234 import RecurrentNetwork

            torch.manual_seed(itraj)
            np.random.seed(itraj)

            x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
            dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
            f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)

            params, exp_path = give_params(mode=mode)
            gamma, omega, n, eta, kappa, params_force, [periods, ppp] = params
            period = (2*np.pi/omega)
            total_time = period*periods
            dt = period/ppp
            times = np.arange(0,total_time+dt,dt)

            def kernelize():
                j = give_random_simp()
                j+=np.mean(j)*np.random.rand(2,2)*0.1
                j*=noise_level
                return cast(j)
            [omega_ext] = np.array(params_force[1])
            dev = torch.device("cpu")
            zero = np.zeros((2,2))
            K1 = cast(np.array([[0,omega_ext],[-omega_ext,0]]) + np.random.rand(2,2)*noise_level)
            K2 = kernelize()#cast(give_random_simp() + np.random.rand(2,2))*noise_level
            K3 = kernelize()#cast(give_random_simp() + np.random.rand(2,2))*noise_level
            K4 = kernelize()#cast(give_random_simp() + np.random.rand(2,2))*noise_level

            initial_condition = np.array(params_force[0]) + np.random.rand(2)*noise_level
            initial_condition=list(initial_condition.astype("float32"))

            inputs_cell = [dt,  [gamma, omega, n, eta, kappa, params_force], [initial_condition, K1, K2, K3, K4 ]]
            rrn = RecurrentNetwork(inputs_cell)
            optimizer = torch.optim.Adam(list(rrn.parameters()), lr=0.01)
            dys = torch.tensor(data=dy, dtype=torch.float32).to(torch.device("cpu"))

            ixs_hat, idys_hat, ifs_hats = rrn(dys)

            return rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path
        elif  id_NN=="in1234":
            from numerics.NN.models.sin.in01234 import RecurrentNetwork

            torch.manual_seed(itraj)
            np.random.seed(itraj)

            x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
            dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
            f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)

            params, exp_path = give_params(mode=mode)
            gamma, omega, n, eta, kappa, params_force, [periods, ppp] = params
            period = (2*np.pi/omega)
            total_time = period*periods
            dt = period/ppp
            times = np.arange(0,total_time+dt,dt)

            def kernelize():
                j = give_random_simp()
                j+=np.mean(j)*np.random.rand(2,2)*0.1
                j*=noise_level
                return cast(j)
            [omega_ext] = np.array(params_force[1])
            dev = torch.device("cpu")
            K1 = cast(np.array([[0,omega_ext],[-omega_ext,0]]) + np.random.rand(2,2)*noise_level)
            K2 = kernelize()#cast(give_random_simp() + np.random.rand(2,2))*noise_level
            K3 = kernelize()#cast(give_random_simp() + np.random.rand(2,2))*noise_level
            K4 = kernelize()#cast(give_random_simp() + np.random.rand(2,2))*noise_level

            initial_condition = np.array(params_force[0]) + 10*np.random.rand(2)*noise_level
            initial_condition=list(initial_condition.astype("float32"))

            inputs_cell = [dt,  [gamma, omega, n, eta, kappa, params_force], [initial_condition, K1, K2, K3, K4 ]]
            rrn = RecurrentNetwork(inputs_cell)
            optimizer = torch.optim.Adam(list(rrn.parameters()), lr=0.01)
            dys = torch.tensor(data=dy, dtype=torch.float32).to(torch.device("cpu"))

            ixs_hat, idys_hat, ifs_hats = rrn(dys)

            return rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path
    else:
        raise NameError("fijate acá")




def load_rnn_and_plot(rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path, itraj=1, alpha=0.0,lr=0.1, noise_level=0.1, mode="sin",id_NN="in01"):

    fig1 = plt.figure(figsize=(25,3))
    ax=plt.subplot(171)
    ax.plot(ixs_hat.detach().numpy()[:,0], color="red",marker='.')
    ax.plot(x[:,0])
    ax=plt.subplot(172)
    ax.plot(ixs_hat.detach().numpy()[:,1], color="red",marker='.')
    ax.plot(x[:,1])
    ax=plt.subplot(173)
    ax.plot(dys[:,0])
    ax.plot(idys_hat.detach().numpy()[:,0], color="red",marker='.')

    ax=plt.subplot(174)
    ax.plot(ifs_hats.detach().numpy()[:,0], color="red",marker='.')
    ax.plot(f[:,0])
    plt.close()

    history = load_history(what="{}/{}_{}_{}".format(mode+id_NN,alpha, lr, noise_level), exp_path=exp_path,itraj=itraj)
    best_ind = set_params_to_best(rrn,history)
    xs_hat, dys_hat, fs_hats = rrn(dys)
    loo = history["losses"]
    ll = np.array([[loo[k][0], loo[k][1], loo[k][2]] for k in range(len(loo))])

    ls,lw=15,3
    fig2 = plt.figure(figsize=(35,5))
    plt.suptitle("alpha = {}, seed = {}, lr = {}, perturbation of eq {}".format(alpha, itraj, history["optimizer"][0]["param_groups"][0]["lr"], noise_level), size=30)
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


    ax=plt.subplot(165)
    p1=ax.plot(np.stack(ll[:,1])[:,0], linewidth=lw,color="blue",label=r'$\ell_0 = \frac{1}{T}\sum_t |dy_t - \hat{dy}_t|^2$')
    ax.tick_params(axis='y', labelcolor="blue")
    ax = ax.twinx()
    p2=ax.plot(np.stack(ll[:,1])[:,1], linewidth=lw, color="red", label=r'$\ell_1 = \alpha \sum_k |\xi_k|$')
    ax.tick_params(axis='y', labelcolor="red")
    ax.set_xlabel("iteration")
    lns = p1+p2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, prop={"size":ls})

    ax=plt.subplot(166)
    p1=ax.plot(ll[:,0], linewidth=lw,color="purple", label=r'$\ell_0 + \ell_1$')
    ax.tick_params(axis='y', labelcolor="purple")
    ax = ax.twinx()
    p2=ax.plot(ll[:,-1], linewidth=lw,color="green", label=r'$\frac{\sum_k |f_k - \hat{f}_k|^2}{N}$')
    ax.tick_params(axis='y', labelcolor="green")
    lns = p1+p2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, prop={"size":ls})
    ax.set_xlabel("iteration")
    plt.close()
    return fig1, fig2
