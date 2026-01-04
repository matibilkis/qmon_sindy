"""
Tests for loss functions in numerics/NN/losses.py
"""

import pytest
import torch
import numpy as np
from numerics.NN.losses import log_lik, err_f


class TestLogLik:
    """Test maximum likelihood loss function."""
    
    def test_log_lik_basic(self):
        """Test basic log-likelihood calculation without regularization."""
        N = 100
        dys = torch.randn(N, 2)
        dys_hat = dys + 0.1 * torch.randn(N, 2)  # Small perturbation
        
        loss, loss_terms = log_lik(dys, dys_hat, dt=0.01)
        
        # Loss should be a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() > 0  # Should be positive
        
        # Without regularization, loss_terms should be None
        assert loss_terms is None
    
    def test_log_lik_perfect_prediction(self):
        """Test that perfect prediction gives zero loss."""
        N = 50
        dys = torch.randn(N, 2)
        dys_hat = dys.clone()
        
        loss, _ = log_lik(dys, dys_hat, dt=0.01)
        
        # Should be very close to zero (within numerical precision)
        assert loss.item() < 1e-6
    
    def test_log_lik_with_regularization(self):
        """Test log-likelihood with L1 regularization."""
        N = 100
        dys = torch.randn(N, 2)
        dys_hat = torch.randn(N, 2)
        
        # Create a simple model with parameters
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param1 = torch.nn.Parameter(torch.randn(2, 2))
                self.param2 = torch.nn.Parameter(torch.randn(2, 2))
        
        model = DummyModel()
        alpha = 0.1
        
        loss, loss_terms = log_lik(dys, dys_hat, model=model, dt=0.01, alpha=alpha)
        
        # Loss should be positive
        assert loss.item() > 0
        
        # Loss terms should be an array with [data_loss, reg_loss]
        assert loss_terms is not None
        assert len(loss_terms) == 2
        assert loss_terms[0] > 0  # Data loss
        assert loss_terms[1] > 0  # Regularization loss
        
        # Total loss should be sum of components
        total_from_terms = loss_terms[0] + loss_terms[1]
        assert abs(loss.item() - total_from_terms) < 1e-5
    
    def test_log_lik_regularization_strength(self):
        """Test that regularization strength affects loss."""
        N = 50
        dys = torch.randn(N, 2)
        dys_hat = torch.randn(N, 2)
        
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.ones(2, 2))
        
        model = DummyModel()
        
        loss_small, _ = log_lik(dys, dys_hat, model=model, dt=0.01, alpha=0.01)
        loss_large, _ = log_lik(dys, dys_hat, model=model, dt=0.01, alpha=1.0)
        
        # Larger regularization should give larger loss
        assert loss_large.item() > loss_small.item()
    
    def test_log_lik_dt_dependence(self):
        """Test that loss scales with dt."""
        N = 50
        dys = torch.randn(N, 2)
        dys_hat = dys + 0.1 * torch.randn(N, 2)
        
        loss_dt1, _ = log_lik(dys, dys_hat, dt=0.01)
        loss_dt2, _ = log_lik(dys, dys_hat, dt=0.1)
        
        # Loss should be inversely proportional to dt
        ratio = loss_dt1.item() / loss_dt2.item()
        expected_ratio = 0.1 / 0.01
        assert abs(ratio - expected_ratio) < 0.1  # Allow some numerical error


class TestErrF:
    """Test external force error calculation."""
    
    def test_err_f_one_dim(self):
        """Test error calculation for 1D force."""
        N = 100
        f = np.random.randn(N)
        fhat = torch.tensor(np.random.randn(N + 1))  # One extra timestep
        
        error = err_f(f, fhat, one_dim=True)
        
        # Should be a scalar
        assert isinstance(error, (float, np.floating))
        assert error >= 0
    
    def test_err_f_perfect_prediction(self):
        """Test that perfect prediction gives zero error."""
        N = 50
        f = np.random.randn(N)
        fhat = torch.tensor(np.concatenate([f, [0.0]]))  # Perfect match + one extra
        
        error = err_f(f, fhat, one_dim=True)
        
        # Should be very close to zero
        assert error < 1e-10
    
    def test_err_f_two_dim(self):
        """Test error calculation for 2D force."""
        N = 100
        f = np.random.randn(N, 2)
        fhat = torch.tensor(np.random.randn(N + 1, 2))
        
        error = err_f(f, fhat, one_dim=False)
        
        assert isinstance(error, (float, np.floating))
        assert error >= 0
    
    def test_err_f_shape_handling(self):
        """Test that function handles shape differences correctly."""
        N = 50
        f = np.random.randn(N)
        fhat = torch.tensor(np.random.randn(N + 1))
        
        # Should not raise error
        error = err_f(f, fhat, one_dim=True)
        assert error >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

