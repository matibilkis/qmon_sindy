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

def w0_net(mode,id_NN,tmp_net):
    if mode == "sin":
        if id_NN=="in1_3":
            coffs = {}
            if tmp_net==0:
                gin,oin = -.1, 1.*0.99
                ep2, ep3, ep4, ep5 = [.5, .4, -.3, .2]
                initial_condition = np.array([3.,0])
                coffK2= 0.*np.array([-.1, .02, -.02, -.4])
            elif tmp_net == 1:
                gin,oin = 0, 1.
                ep2, ep3, ep4, ep5 = [.5, .4, -.3, .2]
                initial_condition = np.array([.5,-.2])
                coffK2=  np.array([-.1, .02, -.02, -.4])*0.0

            coffs["K1"] = np.array([gin, oin, -oin, gin])
            coffs["K2"] = coffK2
            coffs["K3"] = ep2*coffs["K2"]
        elif id_NN=="in1_6":
            coffs = {}
            if tmp_net==0:
                gin,oin = -.1, 2*0.1
                ep2, ep3, ep4, ep5 = [.5, .4, -.3, .2]
                initial_condition = np.array([-2.,1.])
                coffK2 = np.array([-.1, .02, -.02, -.4])
            elif tmp_net == 1:
                gin,oin = 0, 1.
                ep2, ep3, ep4, ep5 = [.5, .4, -.3, .2]
                initial_condition = np.array([.5,-.2])
                coffK2 = np.array([-.1, .02, -.02, -.4])*0.01

            coffs["K1"] = np.array([gin, oin, -oin, gin])
            coffs["K2"] = coffK2
            coffs["K3"] = ep2*coffs["K2"]

            coffs["K4"] = ep3*np.array(coffs["K1"])
            coffs["K5"] = ep3*np.array(coffs["K1"])
            coffs["K6"] = ep3*np.array(coffs["K1"])

        elif id_NN == "in1_15":
            coffs = {}
            if tmp_net==0:
                gin,oin = -.1, 2*0.1
                ep2, ep3, ep4, ep5 = [.5, .4, -.3, .2]
                initial_condition = np.array([-2.,1.])
                coffs["K2"] = np.array([-.1, .02, -.02, -.4])
            elif tmp_net == 1:
                gin,oin = 0, 1.
                ep2, ep3, ep4, ep5 = [.5, .4, -.3, .2]
                initial_condition = np.array([.5,-.2])
                coffs["K2"] = np.array([-.1, .02, -.02, -.4])*0.01

            coffs["K1"] = np.array([gin, oin, -oin, gin])
            coffs["K3"] = ep2*coffs["K2"]

            coffs["K4"] = ep3*np.array(coffs["K1"])
            coffs["K5"] = ep3*np.array(coffs["K1"])
            coffs["K6"] = ep3*np.array(coffs["K1"])

            coffs["K7"] = ep4*coffs["K2"]
            coffs["K8"] = ep4*coffs["K2"]
            coffs["K9"] = ep4*coffs["K2"]
            coffs["K10"] = ep4*coffs["K2"]

            coffs["K11"] = ep5*np.array(coffs["K1"])
            coffs["K12"] = ep5*np.array(coffs["K1"])
            coffs["K13"] = ep5*np.array(coffs["K1"])
            coffs["K14"] = ep5*np.array(coffs["K1"])
            coffs["K15"] = ep5*np.array(coffs["K1"])

    initial_condition=list(initial_condition.astype("float32"))

    initial_params_net = [initial_condition]
    for k in coffs.values():
        initial_params_net+= [k.reshape((2,2))]

    return initial_params_net



def get_plot_data_NN(itraj, mode="sin",id_NN="in01", tmp_net=0):
    params, exp_path = give_params(mode=mode)

    initial_params_net = w0_net(mode,id_NN,tmp_net)
    if mode=="sin":
        if id_NN == "in1_3":
            from numerics.NN.models.sin.in1_3 import RecurrentNetwork
        elif id_NN == "in1_6":
            from numerics.NN.models.sin.in1_6 import RecurrentNetwork
        elif  id_NN=="in1_15":
            from numerics.NN.models.sin.in1_15 import RecurrentNetwork
    else:
        raise NameError("fijate acá")

    torch.manual_seed(itraj)
    np.random.seed(itraj)

    x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
    dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
    f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)

    params, exp_path = give_params(mode=mode)
    params_sensor, params_force, [periods, ppp], [period, total_time, dt, times] = params

    initial_params_net = w0_net(mode, id_NN, tmp_net)
    inputs_cell = [dt,  [params_sensor, params_force], initial_params_net]

    rrn = RecurrentNetwork(inputs_cell)
    dys = torch.tensor(data=dy, dtype=torch.float32).to(torch.device("cpu"))
    ixs_hat, idys_hat, ifs_hats = rrn(dys)

    return rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path





def load_rnn_and_plot(rrn, ixs_hat, idys_hat, ifs_hats, x, dys, f, exp_path, itraj=1, alpha=0.0,lr=0.1, mode="sin",id_NN="in1_3", tmp_net=0):

    fig1 = plt.figure(figsize=(40,10))
    ax=plt.subplot(141)
    ax.plot(ixs_hat.detach().numpy()[:,0], color="red",marker='.')
    ax.plot(x[:,0])
    ax=plt.subplot(142)
    ax.plot(ixs_hat.detach().numpy()[:,1], color="red",marker='.')
    ax.plot(x[:,1])
    ax=plt.subplot(143)
    ax.plot(dys[:,0])
    ax.plot(idys_hat.detach().numpy()[:,0], color="red",marker='.')

    ax=plt.subplot(144)
    ax.plot(ifs_hats.detach().numpy()[:,0], color="red",marker='.')
    ax.plot(f[:,0])
    plt.close()

    history = load_history(what="{}/{}_{}_{}".format(mode+id_NN,alpha, lr, tmp_net), exp_path=exp_path,itraj=itraj)
    best_ind = set_params_to_best(rrn,history)
    xs_hat, dys_hat, fs_hats = rrn(dys)
    loo = history["losses"]
    ll = np.array([[loo[k][0], loo[k][1], loo[k][2]] for k in range(len(loo))])

    ls,lw=15,3
    fig2 = plt.figure(figsize=(60,10))
    plt.suptitle("alpha = {}, seed = {}, lr = {}, tmp_net {}".format(alpha, itraj, history["optimizer"][0]["param_groups"][0]["lr"], tmp_net), size=30)
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
