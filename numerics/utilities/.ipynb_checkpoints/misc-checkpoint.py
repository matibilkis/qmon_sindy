import numpy as np
import ast
import os
import getpass
import socket

def get_def_path(what="trajectories"):
    user = getpass.getuser()
    uu = socket.gethostname()
    if uu in ["pop-os"]:
        defpath = "/home/{}/qmon_sindy/{}/".format(user,what)
    else:
        defpath = "/data/uab-giq/scratch2/matias/qmon_sindy/{}/".format(what)
    os.makedirs(defpath,exist_ok=True)
    return defpath


def give_params(periods=10., ppp=500, mode="exp-dec"):
    #kappa, gamma, omega, n, eta, b = 1., 1e-4, 1e-4, 1e-3, 1e-5, 0.
#    gamma, omega, n, eta, kappa,b  = 15*2*np.pi, 2*np.pi*1e3, 14., 1., 360*2*np.pi, 0. ##Giulio's

    ## I modify a bit the signal-noise ratio
    #gamma, omega, n, eta, kappa, params_force  = 15*2*np.pi, 2*np.pi*1e2, 14., 1., 360*2*np.pi, [2e2, 5]   ##Giulio's
    if mode == "exp-dec":
        gamma, omega, n, eta, kappa, params_force  = 0.3, 10. , 10., 1.0 , 20., [200., 1., 0.] ##antes kappa = 0.8
    elif mode =="osc-exp-dec":
        gamma, omega, n, eta, kappa, params_force  = 0.3, 10. , 10., 1.0 , 20., [[200.,0.], [-.5, 5.]] ##antes kappa = 0.8
    else:
        raise NameError("define force!")
    params_force.append(mode)
    data_t = [float(periods), ppp]
    p= [gamma, omega, n, eta, kappa, params_force, data_t]
    return p, str(p)+"/"

def load_data(itraj = 1, what="hidden_state.npy",mode="exp-dec"):
    """
    what can be either "dys.npy", "external_signal.npy", or hidden_state.npy
    """
    params, exp_path = give_params(mode=mode)

    ####
    gamma, omega, n, eta, kappa, params_force, [periods, ppp] = params

    path = get_def_path()+ exp_path + "{}itraj/periods_{}_ppp_{}/".format(itraj, float(periods), ppp)
    return np.load(path+what)
