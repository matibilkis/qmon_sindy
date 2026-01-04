"""
Tests for neural network models in numerics/NN/models/sin/in1_3.py
"""

import pytest
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())

from numerics.NN.models.sin.in1_3 import GRNN, RecurrentNetwork


class TestGRNN:
    """Test GRNN (quantum recurrent neural network) cell."""
    
    def test_grnn_initialization(self):
        """Test GRNN initialization."""
        dt = 0.01
        params_sensor = [0.3, 1.0, 10.0, 1.0, 0.8]  # gamma, omega, n, eta, kappa
        params_force = [[3.0, 0.0], [0.99], "sin"]
        simulation_params = [params_sensor, params_force]
        
        # Initialize trainable parameters
        initial_condition = np.array([3.0, 0.0])
        K1 = np.array([-0.1, 0.99, -0.99, -0.1])
        K2 = np.array([0.0, 0.0, 0.0, 0.0])
        K3 = np.array([0.0, 0.0, 0.0, 0.0])
        trainable_params = [initial_condition, K1, K2, K3]
        
        inputs_cell = [dt, simulation_params, trainable_params]
        grnn = GRNN(inputs_cell)
        
        # Check that parameters are initialized
        assert grnn.K1 is not None
        assert grnn.K2 is not None
        assert grnn.K3 is not None
        assert grnn.dt == dt
        
        # Check physical matrices
        assert grnn.A.shape == (2, 2)
        assert grnn.C.shape == (2, 2)
        assert grnn.D.shape == (2, 2)
    
    def test_grnn_kernel(self):
        """Test trainable integration kernel."""
        dt = 0.01
        params_sensor = [0.3, 1.0, 10.0, 1.0, 0.8]
        params_force = [[3.0, 0.0], [0.99], "sin"]
        simulation_params = [params_sensor, params_force]
        
        initial_condition = np.array([3.0, 0.0])
        K1 = np.array([-0.1, 0.99, -0.99, -0.1])
        K2 = np.array([0.0, 0.0, 0.0, 0.0])
        K3 = np.array([0.0, 0.0, 0.0, 0.0])
        trainable_params = [initial_condition, K1, K2, K3]
        
        inputs_cell = [dt, simulation_params, trainable_params]
        grnn = GRNN(inputs_cell)
        
        # Test kernel with input force
        f = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        output = grnn.kernel(f)
        
        # Output should have shape (2, 1)
        assert output.shape == (2, 1)
    
    def test_grnn_rk_step(self):
        """Test RK4 integration step."""
        dt = 0.01
        params_sensor = [0.3, 1.0, 10.0, 1.0, 0.8]
        params_force = [[3.0, 0.0], [0.99], "sin"]
        simulation_params = [params_sensor, params_force]
        
        initial_condition = np.array([3.0, 0.0])
        K1 = np.array([-0.1, 0.99, -0.99, -0.1])
        K2 = np.array([0.0, 0.0, 0.0, 0.0])
        K3 = np.array([0.0, 0.0, 0.0, 0.0])
        trainable_params = [initial_condition, K1, K2, K3]
        
        inputs_cell = [dt, simulation_params, trainable_params]
        grnn = GRNN(inputs_cell)
        
        f = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        step = grnn.rk_step(f)
        
        # Step should have same shape as input
        assert step.shape == f.shape
    
    def test_grnn_forward(self):
        """Test forward pass of GRNN."""
        dt = 0.01
        params_sensor = [0.3, 1.0, 10.0, 1.0, 0.8]
        params_force = [[3.0, 0.0], [0.99], "sin"]
        simulation_params = [params_sensor, params_force]
        
        initial_condition = np.array([3.0, 0.0])
        K1 = np.array([-0.1, 0.99, -0.99, -0.1])
        K2 = np.array([0.0, 0.0, 0.0, 0.0])
        K3 = np.array([0.0, 0.0, 0.0, 0.0])
        trainable_params = [initial_condition, K1, K2, K3]
        
        inputs_cell = [dt, simulation_params, trainable_params]
        grnn = GRNN(inputs_cell)
        
        # Create input state: [<q>, <p>, Var[q], Var[p], Cov(q,p), t]
        state = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        dy = torch.tensor([[0.01], [0.0]], dtype=torch.float32)
        f = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        
        nstate, dy_hat, fnew = grnn.forward(dy, state, f)
        
        # Check output shapes
        assert nstate.shape == (6,)  # Updated state
        assert dy_hat.shape == (2, 1)  # Predicted measurement
        assert fnew.shape == (2, 1)  # Updated force
        
        # Check that time advanced
        assert nstate[5].item() == state[5].item() + dt


class TestRecurrentNetwork:
    """Test RecurrentNetwork wrapper."""
    
    def test_recurrent_network_initialization(self):
        """Test RecurrentNetwork initialization."""
        dt = 0.01
        params_sensor = [0.3, 1.0, 10.0, 1.0, 0.8]
        params_force = [[3.0, 0.0], [0.99], "sin"]
        simulation_params = [params_sensor, params_force]
        
        initial_condition = np.array([3.0, 0.0])
        K1 = np.array([-0.1, 0.99, -0.99, -0.1])
        K2 = np.array([0.0, 0.0, 0.0, 0.0])
        K3 = np.array([0.0, 0.0, 0.0, 0.0])
        trainable_params = [initial_condition, K1, K2, K3]
        
        inputs_cell = [dt, simulation_params, trainable_params]
        rnn = RecurrentNetwork(inputs_cell)
        
        # Check that RCell is initialized
        assert rnn.RCell is not None
        assert isinstance(rnn.RCell, GRNN)
        
        # Check that initial_state is a parameter
        assert rnn.initial_state is not None
        assert isinstance(rnn.initial_state, torch.nn.Parameter)
    
    def test_recurrent_network_forward(self):
        """Test forward pass through full sequence."""
        dt = 0.01
        params_sensor = [0.3, 1.0, 10.0, 1.0, 0.8]
        params_force = [[3.0, 0.0], [0.99], "sin"]
        simulation_params = [params_sensor, params_force]
        
        initial_condition = np.array([3.0, 0.0])
        K1 = np.array([-0.1, 0.99, -0.99, -0.1])
        K2 = np.array([0.0, 0.0, 0.0, 0.0])
        K3 = np.array([0.0, 0.0, 0.0, 0.0])
        trainable_params = [initial_condition, K1, K2, K3]
        
        inputs_cell = [dt, simulation_params, trainable_params]
        rnn = RecurrentNetwork(inputs_cell)
        
        # Create measurement sequence
        N = 10
        dys = torch.randn(N, 2) * 0.1  # Small measurements
        
        xs_hat, dys_hat, fs_hats = rnn(dys)
        
        # Check output shapes
        assert xs_hat.shape == (N + 1, 6)  # N+1 states (initial + N updates)
        assert dys_hat.shape == (N, 2)  # N predicted measurements
        assert fs_hats.shape == (N + 1, 2)  # N+1 force estimates
        
        # Check that states evolve
        assert not torch.allclose(xs_hat[0], xs_hat[-1])
    
    def test_recurrent_network_gradient_flow(self):
        """Test that gradients can flow through the network."""
        dt = 0.01
        params_sensor = [0.3, 1.0, 10.0, 1.0, 0.8]
        params_force = [[3.0, 0.0], [0.99], "sin"]
        simulation_params = [params_sensor, params_force]
        
        initial_condition = np.array([3.0, 0.0])
        K1 = np.array([-0.1, 0.99, -0.99, -0.1])
        K2 = np.array([0.0, 0.0, 0.0, 0.0])
        K3 = np.array([0.0, 0.0, 0.0, 0.0])
        trainable_params = [initial_condition, K1, K2, K3]
        
        inputs_cell = [dt, simulation_params, trainable_params]
        rnn = RecurrentNetwork(inputs_cell)
        
        # Create measurement sequence
        N = 5
        dys = torch.randn(N, 2) * 0.1
        dys.requires_grad = False
        
        xs_hat, dys_hat, fs_hats = rnn(dys)
        
        # Compute a simple loss
        loss = torch.sum(dys_hat**2)
        loss.backward()
        
        # Check that gradients exist for trainable parameters
        assert rnn.initial_state.grad is not None
        assert rnn.RCell.K1.grad is not None
        assert rnn.RCell.K2.grad is not None
        assert rnn.RCell.K3.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

