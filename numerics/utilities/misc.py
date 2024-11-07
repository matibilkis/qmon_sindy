import numpy as np
import ast
import os
import getpass
import matplotlib.pyplot as plt
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


def give_params(periods=25, ppp=50, mode="exp-dec"):
    if mode =="sin":
        gamma, omega, n, eta, kappa, params_force  = 0.3, 1. , 10., 1. , .8, [[3., 0.], [.99]]
    elif mode == "exp-dec":
        gamma, omega, n, eta, kappa, params_force  = 0.3, 10. , 10., 1.0 , 20., [200., 1., 0.] ##antes kappa = 0.8
    elif mode =="osc-exp-dec":
        gamma, omega, n, eta, kappa, params_force  = 0.3, 10. , 10., 1 , 20., [[200.,0.], [-.5, 5.]] ##antes kappa = 0.8
    elif mode =="FHN":
        a,b = .7, .8
        tau = 12.5
        I = .5
        delay, zoom = 50., 10.
        gamma, omega, n, eta, kappa, params_force  = 0.3, 10. , 10., 1.0 , 20., [[.8, 1.], [a,b,I,tau, delay, zoom]] ##antes kappa = 0.8
    else:
        raise NameError("define force!")
    params_force.append(mode)
    params_sensor = [gamma, omega, n, eta, kappa]

    period = (2*np.pi/omega)
    total_time = period*periods
    dt = period/ppp
    times = np.arange(0,total_time+dt,dt)
    data_t = [float(periods), ppp] #I use this to save

    p= [params_sensor, params_force, data_t, [period, total_time, dt, times]]
    return p, str(p)+"/"

def load_data(itraj = 1, what="hidden_state.npy",mode="exp-dec"):
    """
    what can be either "dys.npy", "external_signal.npy", or hidden_state.npy
    """
    params, exp_path = give_params(mode=mode)

    ####
    params_sensor, params_force, [periods, ppp], [period, total_time, dt, times] = params

    path = get_def_path()+ exp_path + "{}itraj/periods_{}_ppp_{}/".format(itraj, float(periods), ppp)
    return np.load(path+what)



def plot_integration(x,dy,f,freqs_signal, spectra_signal, params):
    fig1 = plt.figure(figsize=(12,4))
    ax=plt.subplot(231)
    ax.plot(x[:,0])
    ax=plt.subplot(232)
    ax.plot(x[:,1])
    ax=plt.subplot(233)
    ax.plot(dy[:,0])
    ax=plt.subplot(234)
    ax.plot(f[:,0])
    ax=plt.subplot(235)
    ax.plot(freqs_signal, spectra_signal)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax=plt.subplot(236)
    ax.axis("off")
    ax.text(0.2,.5,params[0])
    ax.text(0.2,.2,params[1])
    os.makedirs("analysis/physical_parameters/",exist_ok=True)
    plt.savefig("analysis/physical_parameters/{}_{}.png".format(params[0], params[1]))
    return fig1

def power_spectra(dy,params):
    omega = params[0][1]
    dt = params[3][2]
    spectra_signal = np.abs(np.fft.fft(dy))**2
    freqs_signal = np.fft.fftfreq(n = len(spectra_signal), d= dt)*(2*np.pi)

    cutoff = 10*omega
    cond  = np.logical_and(freqs_signal < cutoff, freqs_signal>=0)
    spectra_signal = spectra_signal[cond]
    freqs_signal = freqs_signal[cond]
    return freqs_signal, spectra_signal
