"""
Loss functions for quantum SINDy training.

This module implements maximum likelihood estimation for quantum measurement records,
with optional L1 regularization for sparsity promotion in the discovered dynamics.
"""

import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd)


def log_lik(dys, dys_hat, model=None, dt=1e-3, alpha=0.0):
    """
    Maximum likelihood loss function for quantum measurement records.
    
    The loss is the negative log-likelihood of observing dys given predictions dys_hat,
    which for Gaussian noise reduces to mean squared error. Optionally includes L1
    regularization to promote sparsity in the discovered force dynamics.
    
    Args:
        dys: Observed measurement increments [N, 2]
        dys_hat: Predicted measurement increments [N, 2]
        model: PyTorch model (required if alpha > 0)
        dt: Time step size (for proper normalization)
        alpha: L1 regularization strength (0 = no regularization)
        
    Returns:
        loss: Total loss (scalar tensor)
        loss_terms: Optional array [data_loss, reg_loss] if alpha > 0, else None
    """
    # Data fidelity term: mean squared error
    l0 = torch.sum((dys - dys_hat)**2) / (dt * len(dys))
    
    if alpha > 0.0:
        # L1 regularization on trainable parameters (excluding initial condition)
        params = list(model.parameters())[1:]  # Skip initial_state parameter
        l1 = torch.sum(torch.tensor([torch.sum(torch.abs(k)) for k in params]))
        return l0 + alpha * l1, torch.stack([l0, alpha * l1]).detach().numpy()
    else:
        return l0, None


def err_f(f, fhat, one_dim=True):
    """
    Compute mean squared error between true and discovered external force.
    
    This is a diagnostic metric (not used in training) to assess how well
    the model has learned the true force dynamics.
    
    Args:
        f: True external force [N, ...]
        fhat: Discovered external force [N+1, ...] (one extra timestep)
        one_dim: If True, assumes 1D force; if False, handles 2D force vectors
        
    Returns:
        mse: Mean squared error between f and fhat[:-1]
    """
    if one_dim:
        return np.mean(np.abs(f - fhat[:-1].detach().numpy())**2)
    else:
        return np.mean(np.abs(f - fhat[:-1, :].detach().numpy())**2)
