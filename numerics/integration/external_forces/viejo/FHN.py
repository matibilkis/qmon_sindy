import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
import numpy as np
from tqdm import tqdm
from numba import jit
from scipy.linalg import solve_continuous_are
import argparse

@jit(nopython=True)
def Euler_step_state(x, noise_vector, f):
    return x + A.dot(x)*dt + XiCov.dot(noise_vector) + zoom_f*np.array([0,f[0]])*dt

@jit(nopython=True)
def Euler_step_signal(f):
    #prams_foce = [omega, gamma]  linear oscillator
    v,w = f
    dv = v - v**3/3 - w + I   #i use only R, but it's R*I
    dw = (v + a -b*w)/tau
    df = np.array([dv, dw])
    return f + delay*df*dt#signal_coeff_hidden.dot(f)*dt

def IntLoop(times):
    N = len(times)
    hidden_state = np.zeros((N,2))
    external_signal = np.zeros((N,2))
    external_signal[0] = np.array(fhidden)
    dys = [[0.,0.]]
    for ind, t in enumerate(times[:-1]):
        hidden_state[ind+1] = Euler_step_state(hidden_state[ind], dW[ind], external_signal[ind])
        external_signal[ind+1] = Euler_step_signal(external_signal[ind])
        dys.append(C.dot(hidden_state[ind])*dt + proj_C.dot(dW[ind]))
    return hidden_state, external_signal, dys

def integrate(params, periods=10,ppp=500,  itraj=1, exp_path="",**kwargs):
    global dt, proj_C, A, XiCov, C, dW, params_force, signal_coeff_hidden,fhidden, a,b,I,tau, delay, zoom_f
    gamma, omega, n, eta, kappa, params_force = params
    a,b,I,tau,delay,zoom_f = params_force[1]
    fhidden = params_force[0] #i look at the first component of fhidden, but dx = A-() *dt + fdt, with (0, f) and x=(x,p), so it's a force

    period = (2*np.pi/omega)
    total_time = period*periods
    dt = period/ppp
    times = np.arange(0.,total_time+dt,dt)

    #### generate long trajectory of noises
    np.random.seed(itraj)
    dW = np.sqrt(dt)*np.random.randn(len(times),2)

    A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
    proj_C = np.array([[1.,0.],[0.,0.]])
    C = np.sqrt(4*eta*kappa)*proj_C
    D = np.diag([gamma*(n+0.5) + kappa]*2)
    G = np.zeros((2,2))

    #pHHM=params_force[1]
    #signal_coeff_hidden = np.array([[pHHM[0], pHHM[1]],[-pHHM[1], pHHM[0]]])

    Cov = solve_continuous_are((A-G.dot(C)).T, C.T, D- (G.T).dot(G), np.eye(2)) #### A.T because the way it's implemented!
    XiCov = Cov.dot(C.T) + G.T

    hidden_state, external_signal, dys = IntLoop(times)

    path = get_def_path() + exp_path + "{}itraj/periods_{}_ppp_{}/".format(itraj, periods, ppp)
    os.makedirs(path, exist_ok=True)

    if len(times)>1e8:
        indis = np.linspace(0,len(times)-1, int(1e4)).astype(int)
    else:
        indis = np.arange(0,len(times))

    timind = [times[ind] for ind in indis]

    hidden_state =  np.array([hidden_state[ii] for ii in indis])
    external_signal =  np.array([external_signal[ii] for ii in indis])
    dys =  np.array([dys[ii] for ii in indis])

    np.save(path+"hidden_state",hidden_state)
    np.save(path+"external_signal",external_signal)
    np.save(path+"dys",dys)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    params, exp_path = give_params(mode="FHN")

    ####
    gamma, omega, n, eta, kappa, params_force, [periods, ppp] = params
    print(params_force)
    integrate(params=params[:-1],
              periods= periods,
              ppp=ppp,
              itraj=itraj,
              exp_path = exp_path)



#
