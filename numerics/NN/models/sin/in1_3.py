import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
from scipy.linalg import solve_continuous_are


class GRNN(torch.nn.Module):
    """
    Quantum Recurrent Neural Network (GRNN) cell implementing continuous quantum measurement updates.
    
    This custom PyTorch module combines:
    1. Quantum Kalman filtering for state tracking
    2. Trainable integration kernel for force dynamics discovery (SINDy approach)
    
    The state update follows the stochastic master equation:
        dx = (A - ΞC)x·dt + Ξ·dy + (0, f_t)^T dt
        dΣ = (AΣ + ΣAᵀ + D - ΞΞᵀ)·dt
    
    where Ξ = ΣCᵀ is the Kalman gain coupling measurements to state updates.
    
    Args:
        inputs_cell: List containing [dt, simulation_params, trainable_params]
            - dt: Time step size
            - simulation_params: [params_sensor, params_force] where params_sensor = [gamma, omega, n, eta, kappa]
            - trainable_params: [initial_condition, K1, K2, K3] for force dynamics ansatz
    """
    def __init__(self, inputs_cell):
        super(GRNN, self).__init__()

        self.dt, self.simulation_params, trainable_params = inputs_cell
        K1, K2, K3 = trainable_params[1:]
        [gamma, omega, n, eta, kappa], params_force = self.simulation_params
        [omegaf] = params_force[1]

        # Trainable parameters for force dynamics ansatz (SINDy dictionary)
        self.K1 = torch.nn.Parameter(data=torch.tensor(K1, dtype=torch.float32, requires_grad=True))
        self.K2 = torch.nn.Parameter(data=torch.tensor(K2, dtype=torch.float32, requires_grad=True))
        self.K3 = torch.nn.Parameter(data=torch.tensor(K3, dtype=torch.float32, requires_grad=True))

        # Physical matrices (fixed, non-trainable)
        self.A = torch.tensor(data=[[-gamma/2, omega], [-omega, -gamma/2]], dtype=torch.float32).detach()
        self.proj_C = torch.tensor(data=[[1., 0.], [0., 0.]], dtype=torch.float32).detach()
        self.C = np.sqrt(4*eta*kappa) * self.proj_C.detach()
        self.D = (gamma*(n+0.5) + kappa) * torch.eye(2).detach()
        # Projection matrix: first component of f enters as force in second component of x
        self.proj_F = torch.tensor(data=[[[0, 0], [1, 0]]], dtype=torch.float32).detach()

    def rk_step(self, x):
        """
        Fourth-order Runge-Kutta step for integrating force dynamics.
        
        Args:
            x: Current force state vector
            
        Returns:
            RK4 step update for force dynamics
        """
        k1 = self.kernel(x) * self.dt
        k2 = self.dt * self.kernel(x + k1/2.)
        k3 = self.dt * self.kernel(x + k2/2.)
        k4 = self.dt * self.kernel(x + k3)
        return (k1 + 2*k2 + 2*k3 + k4) / 6.

    def kernel(self, x):
        """
        Trainable integration kernel for force dynamics (SINDy ansatz).
        
        Implements a sparse dictionary of functions:
            df/dt = K1·f + K2·f² + K3·(f_a f_b, f_b f_a)
        
        Args:
            x: Force state vector [f_a, f_b]
            
        Returns:
            Time derivative of force according to learned dynamics
        """
        f1 = torch.squeeze(self.K1).matmul(x)  # Linear terms
        f2 = torch.squeeze(self.K2).matmul(x**2)  # Quadratic terms
        f3 = torch.squeeze(self.K3).matmul(x * torch.flip(x, [-1]))  # Cross terms
        return f1 + f2 + f3

    def forward(self, dy, state, f):
        """
        Forward pass: quantum Kalman filter update + force dynamics integration.
        
        Args:
            dy: Measurement increment (observed data)
            state: Hidden state vector [<q>, <p>, Var[q], Var[p], Cov(q,p), t]
            f: Current external force estimate [f_a, f_b]
            
        Returns:
            nstate: Updated hidden state
            dy_hat: Predicted measurement increment
            fnew: Updated external force estimate
        """
        x = state[:2]
        [vx,vp,cxp] = state[2:5]
        t = state[-1]
        cov = torch.tensor(data = [[vx,cxp],[cxp,vp]], dtype=torch.float32)

        xicov = cov.matmul(self.C.T)
        dx = (self.A - xicov.matmul(self.C)).matmul(x)*self.dt + xicov.matmul(dy)

        fnew = f + self.rk_step(f)

        dx += torch.squeeze(self.proj_F).matmul(fnew)*self.dt
        dcov = self.dt*(cov.matmul(self.A.T) + (self.A).matmul(cov) + self.D - (xicov.matmul(xicov.T)))
        ncov = cov+dcov

        nstate = torch.concatenate([(x + dx), torch.tensor([ncov[0,0],ncov[1,1],ncov[1,0]]), torch.tensor([t+self.dt])])
        dy_hat = self.C.matmul(x)*self.dt
        return nstate, dy_hat, fnew

class RecurrentNetwork(torch.nn.Module):
    """
    Full recurrent network wrapping the GRNN cell for processing entire measurement sequences.
    
    This network processes a full measurement record and returns:
    - Estimated hidden state trajectories
    - Predicted measurement increments
    - Discovered external force dynamics
    
    Args:
        inputs_cell: List containing [dt, simulation_params, trainable_params]
    """
    def __init__(self, inputs_cell):
        super(RecurrentNetwork, self).__init__()
        self.RCell = GRNN(inputs_cell=inputs_cell)
        self.dt, self.simulation_params, trainable_params = inputs_cell
        # Trainable initial condition for external force
        self.initial_state = torch.nn.Parameter(torch.tensor(trainable_params[0]))

    def forward(self, dys):
        """
        Process full measurement sequence through the quantum SINDy network.
        
        Args:
            dys: Tensor of measurement increments [N, 2] where N is sequence length
            
        Returns:
            xs_hat: Stacked estimated hidden states [N+1, 6]
            dys_hat: Stacked predicted measurement increments [N, 2]
            fs_hats: Stacked discovered external forces [N+1, 2]
        """
        dys_hat = []

        # Compute stationary covariance matrix via continuous algebraic Riccati equation
        # This gives the steady-state uncertainty for the quantum filter
        [gamma, omega, n, eta, kappa], b = self.simulation_params
        A = np.array([[-gamma/2, omega], [-omega, -gamma/2]])
        proj_C = np.array([[1., 0.], [0., 0.]])
        C = np.sqrt(4*eta*kappa) * proj_C
        D = np.diag([gamma*(n+0.5) + kappa] * 2)
        G = np.zeros((2, 2))
        # Note: A.T because of scipy.linalg.solve_continuous_are implementation
        Cov = solve_continuous_are((A - G.dot(C)).T, C.T, D - (G.T).dot(G), np.eye(2))
        
        t0 = 0.
        # Initialize state: zero mean, stationary covariance
        xs_hat = [torch.tensor([0., 0., Cov[0, 0], Cov[1, 1], Cov[1, 0], t0], dtype=torch.float32)]
        fs_hat = [self.initial_state]

        x_hat = xs_hat[0]
        f_hat = fs_hat[0]
        
        # Process measurement sequence
        for dy_t in dys:
            x_hat, dy_hat, f_hat = self.RCell(dy_t, x_hat, f_hat)
            dys_hat += [dy_hat]
            xs_hat += [x_hat]
            fs_hat += [f_hat]
            
        return torch.stack(xs_hat), torch.stack(dys_hat), torch.stack(fs_hat)
