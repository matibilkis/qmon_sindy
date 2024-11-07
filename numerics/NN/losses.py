import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())

#def log_lik(dys, dys_hat, dt=1e-3):
#    return torch.sum((dys-dys_hat)**2)/(dt*len(dys))


def log_lik(dys, dys_hat, model=None,dt=1e-3, alpha=0.0):
    l0 = torch.sum((dys-dys_hat)**2)/(dt*len(dys))
    if alpha>0.0:
        params = list(model.parameters())[1:]
        l1 = torch.sum(torch.tensor([torch.sum(torch.abs(k)) for k in params]))
        return l0 + alpha*l1, torch.stack([l0, alpha*l1]).detach().numpy()
    else:
        return l0, None

def err_f(f,fhat, one_dim=True):
    if one_dim==True:
        #return np.sum(np.abs(f - fhat[:-1].detach().numpy() ))/np.sum(np.abs(f))
        return np.mean(np.abs(f - fhat[:-1].detach().numpy() )**2)

    else:
        return np.mean(np.abs(f - fhat[:-1,:].detach().numpy() )**2)#/np.sum(np.abs(f))
