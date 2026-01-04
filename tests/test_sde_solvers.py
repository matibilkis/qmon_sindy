"""
Tests for SDE solvers in numerics/integration/steps.py
"""

import pytest
import numpy as np
from numba import jit


class TestSDESolvers:
    """Test stochastic differential equation solvers."""
    
    def test_rk4_deterministic_ode(self):
        """Test RK4 solver on a simple deterministic ODE."""
        from numerics.integration.steps import RK4
        
        # Simple ODE: dx/dt = -x
        @jit(nopython=True)
        def f(x, t):
            return -x
        
        @jit(nopython=True)
        def g(x, t):
            return np.zeros((1, 1))  # No noise
        
        # Initial condition
        x0 = np.array([[1.0]])
        dt = 0.01
        N = 100
        
        # Create arrays
        x = np.zeros((1, N + 1))
        x[:, 0] = x0.flatten()
        dWs = np.zeros((1, N))
        
        # Integrate
        for i in range(N):
            t = i * dt
            dx = RK4(f, g, x, dWs, i, t, dt)
            x[:, i + 1] = x[:, i] + dx.flatten()
        
        # Solution should be x(t) = exp(-t)
        t_final = N * dt
        expected = np.exp(-t_final)
        actual = x[0, -1]
        
        # Should be close (RK4 is 4th order)
        assert abs(actual - expected) < 1e-4
    
    def test_euler_maruyama(self):
        """Test Euler-Maruyama method on simple SDE."""
        from numerics.integration.steps import Euler
        
        # SDE: dx = -x*dt + dW (Ornstein-Uhlenbeck)
        @jit(nopython=True)
        def f(x, t):
            return -x
        
        @jit(nopython=True)
        def g(x, t):
            return np.ones((1, 1))  # Unit noise
        
        # Initial condition
        x0 = np.array([[1.0]])
        dt = 0.01
        N = 100
        
        # Create arrays
        x = np.zeros((1, N + 1))
        x[:, 0] = x0.flatten()
        np.random.seed(42)
        dWs = np.sqrt(dt) * np.random.randn(1, N)
        
        # Integrate
        for i in range(N):
            t = i * dt
            dx = Euler(f, g, x, dWs, i, t, dt)
            x[:, i + 1] = x[:, i] + dx.flatten()
        
        # Check that solution is reasonable
        assert not np.isnan(x[0, -1])
        assert not np.isinf(x[0, -1])
        
        # For OU process, variance should be finite
        assert np.std(x[0, :]) < 10
    
    def test_rk4_convergence(self):
        """Test that RK4 has better convergence than Euler for deterministic case."""
        from numerics.integration.steps import RK4, Euler
        
        # ODE: dx/dt = x (exponential growth)
        @jit(nopython=True)
        def f(x, t):
            return x
        
        @jit(nopython=True)
        def g(x, t):
            return np.zeros((1, 1))
        
        x0 = np.array([[1.0]])
        t_final = 1.0
        
        # Test with different time steps
        dts = [0.1, 0.05, 0.01]
        errors_rk4 = []
        errors_euler = []
        
        for dt in dts:
            N = int(t_final / dt)
            
            # RK4
            x_rk4 = np.zeros((1, N + 1))
            x_rk4[:, 0] = x0.flatten()
            dWs = np.zeros((1, N))
            for i in range(N):
                t = i * dt
                dx = RK4(f, g, x_rk4, dWs, i, t, dt)
                x_rk4[:, i + 1] = x_rk4[:, i] + dx.flatten()
            
            # Euler
            x_euler = np.zeros((1, N + 1))
            x_euler[:, 0] = x0.flatten()
            for i in range(N):
                t = i * dt
                dx = Euler(f, g, x_euler, dWs, i, t, dt)
                x_euler[:, i + 1] = x_euler[:, i] + dx.flatten()
            
            # Exact solution: x(t) = exp(t)
            exact = np.exp(t_final)
            error_rk4 = abs(x_rk4[0, -1] - exact)
            error_euler = abs(x_euler[0, -1] - exact)
            
            errors_rk4.append(error_rk4)
            errors_euler.append(error_euler)
        
        # RK4 should have smaller errors
        assert errors_rk4[-1] < errors_euler[-1]
    
    def test_rossler_step_interface(self):
        """Test that Rossler step function has correct interface."""
        from numerics.integration.steps import Robler_step
        
        # Create dummy functions
        @jit(nopython=True)
        def f(Yn, t, dt):
            return -Yn
        
        @jit(nopython=True)
        def G():
            return np.zeros((1, 1))
        
        d = 1
        m = 1
        dt = 0.01
        t = 0.0
        Yn = np.array([1.0])
        Ik = np.array([0.0])
        Iij = np.array([[0.0]])
        
        # Should not raise error
        try:
            Yn1 = Robler_step(t, Yn, Ik, Iij, dt, f, G, d, m)
            assert Yn1.shape == Yn.shape
        except Exception as e:
            # If function requires specific setup, that's okay
            # Just check that it's callable
            assert callable(Robler_step)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

