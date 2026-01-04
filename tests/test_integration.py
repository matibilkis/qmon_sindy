"""
Tests for quantum trajectory integration functions.
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
import sys
sys.path.insert(0, os.getcwd())

from numerics.integration.external_forces.sin import integrate
from numerics.utilities.misc import give_params, load_data


class TestIntegration:
    """Test quantum trajectory integration."""
    
    def test_integrate_basic(self):
        """Test basic integration with minimal parameters."""
        # Use a temporary directory for test data
        original_get_def_path = None
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Mock get_def_path to use temp directory
            import numerics.utilities.misc as misc_module
            original_get_def_path = misc_module.get_def_path
            
            def mock_get_def_path(what="trajectories"):
                os.makedirs(temp_dir, exist_ok=True)
                return temp_dir + "/"
            
            misc_module.get_def_path = mock_get_def_path
            
            # Generate parameters
            params, exp_path = give_params(periods=1, ppp=10, mode="sin")
            
            # Run integration
            integrate(
                params=params,
                periods=1,
                ppp=10,
                itraj=1,
                exp_path=exp_path
            )
            
            # Check that files were created
            path = temp_dir + exp_path + "1itraj/periods_1.0_ppp_10/"
            assert os.path.exists(path + "hidden_state.npy")
            assert os.path.exists(path + "external_signal.npy")
            assert os.path.exists(path + "dys.npy")
            
            # Check that data can be loaded
            hidden_state = np.load(path + "hidden_state.npy")
            external_signal = np.load(path + "external_signal.npy")
            dys = np.load(path + "dys.npy")
            
            # Check shapes
            assert hidden_state.shape[1] == 2  # [q, p]
            assert external_signal.shape[1] == 2  # [f_a, f_b]
            assert dys.shape[1] == 2  # [dy_1, dy_2]
            
            # Check that we have reasonable number of time steps
            assert len(hidden_state) > 0
            assert len(external_signal) == len(hidden_state)
            assert len(dys) == len(hidden_state)
            
        finally:
            # Restore original function
            if original_get_def_path:
                misc_module.get_def_path = original_get_def_path
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def test_integrate_deterministic_seed(self):
        """Test that same seed produces same results."""
        temp_dir1 = tempfile.mkdtemp()
        temp_dir2 = tempfile.mkdtemp()
        
        try:
            import numerics.utilities.misc as misc_module
            original_get_def_path = misc_module.get_def_path
            
            def mock_get_def_path_1(what="trajectories"):
                return temp_dir1 + "/"
            
            def mock_get_def_path_2(what="trajectories"):
                return temp_dir2 + "/"
            
            params, exp_path = give_params(periods=1, ppp=10, mode="sin")
            
            # First integration
            misc_module.get_def_path = mock_get_def_path_1
            integrate(params=params, periods=1, ppp=10, itraj=42, exp_path=exp_path)
            
            # Second integration with same seed
            misc_module.get_def_path = mock_get_def_path_2
            integrate(params=params, periods=1, ppp=10, itraj=42, exp_path=exp_path)
            
            # Load both results
            path1 = temp_dir1 + exp_path + "42itraj/periods_1.0_ppp_10/"
            path2 = temp_dir2 + exp_path + "42itraj/periods_1.0_ppp_10/"
            
            hidden1 = np.load(path1 + "hidden_state.npy")
            hidden2 = np.load(path2 + "hidden_state.npy")
            
            # Should be identical (within numerical precision)
            np.testing.assert_allclose(hidden1, hidden2, rtol=1e-10)
            
        finally:
            misc_module.get_def_path = original_get_def_path
            for temp_dir in [temp_dir1, temp_dir2]:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
    
    def test_integrate_different_seeds(self):
        """Test that different seeds produce different results."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            import numerics.utilities.misc as misc_module
            original_get_def_path = misc_module.get_def_path
            
            def mock_get_def_path(what="trajectories"):
                return temp_dir + "/"
            
            misc_module.get_def_path = mock_get_def_path
            
            params, exp_path = give_params(periods=1, ppp=10, mode="sin")
            
            # Integrate with different seeds
            integrate(params=params, periods=1, ppp=10, itraj=1, exp_path=exp_path)
            integrate(params=params, periods=1, ppp=10, itraj=2, exp_path=exp_path)
            
            # Load both results
            path1 = temp_dir + exp_path + "1itraj/periods_1.0_ppp_10/"
            path2 = temp_dir + exp_path + "2itraj/periods_1.0_ppp_10/"
            
            hidden1 = np.load(path1 + "hidden_state.npy")
            hidden2 = np.load(path2 + "hidden_state.npy")
            
            # Should be different (stochastic process)
            assert not np.allclose(hidden1, hidden2, rtol=1e-6)
            
        finally:
            misc_module.get_def_path = original_get_def_path
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def test_integrate_force_dynamics(self):
        """Test that external force follows expected dynamics."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            import numerics.utilities.misc as misc_module
            original_get_def_path = misc_module.get_def_path
            
            def mock_get_def_path(what="trajectories"):
                return temp_dir + "/"
            
            misc_module.get_def_path = mock_get_def_path
            
            params, exp_path = give_params(periods=2, ppp=50, mode="sin")
            params_sensor, params_force, _, time_info = params
            omega_f = params_force[1][0]  # Frequency of external force
            
            integrate(params=params, periods=2, ppp=50, itraj=1, exp_path=exp_path)
            
            path = temp_dir + exp_path + "1itraj/periods_2.0_ppp_50/"
            external_signal = np.load(path + "external_signal.npy")
            
            # For sinusoidal force, should see oscillatory behavior
            # Check that signal magnitude is reasonable
            signal_magnitude = np.sqrt(external_signal[:, 0]**2 + external_signal[:, 1]**2)
            assert np.all(signal_magnitude < 10)  # Reasonable bound
            
            # Check that signal is not constant
            assert np.std(external_signal[:, 0]) > 0.1
            
        finally:
            misc_module.get_def_path = original_get_def_path
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def test_integrate_measurement_record(self):
        """Test that measurement record has correct properties."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            import numerics.utilities.misc as misc_module
            original_get_def_path = misc_module.get_def_path
            
            def mock_get_def_path(what="trajectories"):
                return temp_dir + "/"
            
            misc_module.get_def_path = mock_get_def_path
            
            params, exp_path = give_params(periods=1, ppp=20, mode="sin")
            
            integrate(params=params, periods=1, ppp=20, itraj=1, exp_path=exp_path)
            
            path = temp_dir + exp_path + "1itraj/periods_1.0_ppp_20/"
            dys = np.load(path + "dys.npy")
            
            # Measurement record should have noise
            assert np.std(dys[:, 0]) > 0
            
            # Should have correct shape
            assert dys.shape[1] == 2
            
        finally:
            misc_module.get_def_path = original_get_def_path
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

