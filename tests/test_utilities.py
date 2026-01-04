"""
Tests for utility functions in numerics/utilities/misc.py
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
from numerics.utilities.misc import give_params, power_spectra


class TestGiveParams:
    """Test parameter generation functions."""
    
    def test_give_params_sin_mode(self):
        """Test parameter generation for sinusoidal force mode."""
        params, exp_path = give_params(periods=10, ppp=100, mode="sin")
        
        params_sensor, params_force, data_t, time_info = params
        
        # Check sensor parameters
        assert len(params_sensor) == 5
        assert params_sensor[0] == 0.3  # gamma
        assert params_sensor[1] == 1.0  # omega
        assert params_sensor[2] == 10.0  # n
        assert params_sensor[3] == 1.0  # eta
        assert params_sensor[4] == 0.8  # kappa
        
        # Check force parameters
        assert len(params_force) == 3
        assert params_force[0] == [3.0, 0.0]  # initial condition
        assert params_force[1] == [0.99]  # omega_f
        assert params_force[2] == "sin"  # mode
        
        # Check time information
        period, total_time, dt, times = time_info
        assert period == 2 * np.pi / 1.0
        assert total_time == period * 10
        assert dt == period / 100
        assert len(times) > 0
    
    def test_give_params_exp_dec_mode(self):
        """Test parameter generation for exponential decay mode."""
        params, exp_path = give_params(periods=5, ppp=50, mode="exp-dec")
        
        params_sensor, params_force, data_t, time_info = params
        assert params_sensor[1] == 10.0  # omega
        assert params_force[2] == "exp-dec"
    
    def test_give_params_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(NameError):
            give_params(mode="invalid_mode")
    
    def test_give_params_time_calculation(self):
        """Test that time calculations are correct."""
        params, _ = give_params(periods=2, ppp=10, mode="sin")
        _, _, _, time_info = params
        period, total_time, dt, times = time_info
        
        omega = 1.0
        expected_period = 2 * np.pi / omega
        assert abs(period - expected_period) < 1e-10
        assert abs(total_time - expected_period * 2) < 1e-10
        assert abs(dt - expected_period / 10) < 1e-10
        assert len(times) == 21  # 0 to 2*period with step dt


class TestPowerSpectra:
    """Test power spectrum calculation."""
    
    def test_power_spectra_basic(self):
        """Test basic power spectrum calculation."""
        # Create a simple sinusoidal signal
        params, _ = give_params(periods=1, ppp=100, mode="sin")
        omega = params[0][1]
        dt = params[3][2]
        
        times = np.arange(0, 2*np.pi/omega, dt)
        signal = np.sin(omega * times)
        
        freqs, spectra = power_spectra(signal, params)
        
        # Check output shapes
        assert len(freqs) == len(spectra)
        assert len(freqs) > 0
        
        # Check frequency range (should be filtered by cutoff)
        assert np.all(freqs >= 0)
        assert np.all(freqs < 10 * omega)
        
        # Check that spectra are non-negative
        assert np.all(spectra >= 0)
    
    def test_power_spectra_zero_signal(self):
        """Test power spectrum with zero signal."""
        params, _ = give_params(periods=1, ppp=100, mode="sin")
        signal = np.zeros(100)
        
        freqs, spectra = power_spectra(signal, params)
        
        # Should still return valid arrays
        assert len(freqs) == len(spectra)
        assert np.all(spectra == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

