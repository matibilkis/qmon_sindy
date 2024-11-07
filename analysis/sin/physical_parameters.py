%load_ext autoreload
%autoreload 2
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from numerics.utilities.misc import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numerics.NN.losses import *
from numerics.NN.misc import *
import numpy as np
from numerics.NN.misc import *

itraj, mode = 1, "sin"
params, exp_path = give_params(mode="sin")
params_sensor, params_force, [periods, ppp], [period, total_time, dt, times] = params

x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)
freqs_signal, spectra_signal = power_spectra(dy[:,0], params)

fig = plot_integration(x,dy,f,freqs_signal, spectra_signal,params)


itraj, mode = 1, "sin"
params, exp_path = give_params(mode="sin")
params_sensor, params_force, [periods, ppp], [period, total_time, dt, times] = params

x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)
freqs_signal, spectra_signal = power_spectra(dy[:,0], params)

fig = plot_integration(x,dy,f,freqs_signal, spectra_signal,params)




itraj, mode = 1, "sin"
os.system("python3 numerics/integration/external_forces/sin.py")
params, exp_path = give_params(mode="sin")
params_sensor, params_force, [periods, ppp], [period, total_time, dt, times] = params

x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)
freqs_signal, spectra_signal = power_spectra(dy[:,0], params)

fig = plot_integration(x,dy,f,freqs_signal, spectra_signal,params)





itraj, mode = 1, "sin"
os.system("python3 numerics/integration/external_forces/sin.py")
params, exp_path = give_params(mode="sin")
params_sensor, params_force, [periods, ppp], [period, total_time, dt, times] = params

x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)
freqs_signal, spectra_signal = power_spectra(dy[:,0], params)

fig = plot_integration(x,dy,f,freqs_signal, spectra_signal,params)










itraj, mode = 1, "sin"
os.system("python3 numerics/integration/external_forces/sin.py")
params, exp_path = give_params(mode="sin")
params_sensor, params_force, [periods, ppp], [period, total_time, dt, times] = params

x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)
freqs_signal, spectra_signal = power_spectra(dy[:,0], params)

fig = plot_integration(x,dy,f,freqs_signal, spectra_signal,params)






itraj, mode = 1, "sin"
os.system("python3 numerics/integration/external_forces/sin.py")
params, exp_path = give_params(mode="sin")
params_sensor, params_force, [periods, ppp], [period, total_time, dt, times] = params

x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)
freqs_signal, spectra_signal = power_spectra(dy[:,0], params)

fig = plot_integration(x,dy,f,freqs_signal, spectra_signal,params)






itraj, mode = 1, "sin"
os.system("python3 numerics/integration/external_forces/sin.py")
params, exp_path = give_params(mode="sin")
params_sensor, params_force, [periods, ppp], [period, total_time, dt, times] = params

x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)
freqs_signal, spectra_signal = power_spectra(dy[:,0], params)

fig = plot_integration(x,dy,f,freqs_signal, spectra_signal,params)




#
