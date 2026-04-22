"""Tests for differential privacy mechanisms."""

import pytest
import numpy as np

from dp_implementation.models.mechanisms import DPMechanisms
from dp_implementation.utils.privacy_accountant import compute_sensitivity, compute_noise_scale


class TestDPMechanisms:
    """Test class for DP mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dp_mechanisms = DPMechanisms(random_seed=42)
        self.test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    def test_laplace_mechanism(self):
        """Test Laplace mechanism."""
        epsilon = 1.0
        delta = 0.0
        
        result, metadata = self.dp_mechanisms.laplace_mechanism(
            self.test_data, epsilon, "mean", delta
        )
        
        # Check that result is a float
        assert isinstance(result, float)
        
        # Check metadata
        assert "true_result" in metadata
        assert "noise" in metadata
        assert "noise_scale" in metadata
        assert "sensitivity" in metadata
        assert "epsilon" in metadata
        assert "delta" in metadata
        assert "mechanism" in metadata
        
        # Check mechanism type
        assert metadata["mechanism"] == "laplace"
        assert metadata["epsilon"] == epsilon
        assert metadata["delta"] == delta
    
    def test_gaussian_mechanism(self):
        """Test Gaussian mechanism."""
        epsilon = 1.0
        delta = 1e-5
        
        result, metadata = self.dp_mechanisms.gaussian_mechanism(
            self.test_data, epsilon, delta, "mean"
        )
        
        # Check that result is a float
        assert isinstance(result, float)
        
        # Check metadata
        assert "true_result" in metadata
        assert "noise" in metadata
        assert "noise_scale" in metadata
        assert "sensitivity" in metadata
        assert "epsilon" in metadata
        assert "delta" in metadata
        assert "mechanism" in metadata
        
        # Check mechanism type
        assert metadata["mechanism"] == "gaussian"
        assert metadata["epsilon"] == epsilon
        assert metadata["delta"] == delta
    
    def test_exponential_mechanism(self):
        """Test exponential mechanism."""
        candidates = ["A", "B", "C"]
        scores = np.array([0.8, 0.6, 0.9])
        epsilon = 1.0
        sensitivity = 1.0
        
        result, metadata = self.dp_mechanisms.exponential_mechanism(
            candidates, scores, epsilon, sensitivity
        )
        
        # Check that result is one of the candidates
        assert result in candidates
        
        # Check metadata
        assert "candidates" in metadata
        assert "scores" in metadata
        assert "probabilities" in metadata
        assert "selected_idx" in metadata
        assert "epsilon" in metadata
        assert "sensitivity" in metadata
        assert "mechanism" in metadata
        
        # Check mechanism type
        assert metadata["mechanism"] == "exponential"
        assert metadata["epsilon"] == epsilon
        assert metadata["sensitivity"] == sensitivity
    
    def test_histogram_mechanism(self):
        """Test histogram mechanism."""
        epsilon = 1.0
        bins = 5
        
        result, metadata = self.dp_mechanisms.histogram_mechanism(
            self.test_data, bins, epsilon
        )
        
        # Check that result is an array
        assert isinstance(result, np.ndarray)
        assert len(result) == bins
        
        # Check metadata
        assert "true_histogram" in metadata
        assert "bin_edges" in metadata
        assert "noise" in metadata
        assert "noise_scale" in metadata
        assert "sensitivity" in metadata
        assert "epsilon" in metadata
        assert "mechanism" in metadata
        
        # Check mechanism type
        assert metadata["mechanism"] == "histogram"
        assert metadata["epsilon"] == epsilon
    
    def test_quantile_mechanism(self):
        """Test quantile mechanism."""
        epsilon = 1.0
        delta = 1e-5
        quantile = 0.5
        
        result, metadata = self.dp_mechanisms.quantile_mechanism(
            self.test_data, quantile, epsilon, delta
        )
        
        # Check that result is a float
        assert isinstance(result, float)
        
        # Check metadata
        assert "true_quantile" in metadata
        assert "quantile" in metadata
        assert "noise" in metadata
        assert "noise_scale" in metadata
        assert "sensitivity" in metadata
        assert "epsilon" in metadata
        assert "delta" in metadata
        assert "mechanism" in metadata
        
        # Check mechanism type
        assert metadata["mechanism"] == "quantile"
        assert metadata["epsilon"] == epsilon
        assert metadata["delta"] == delta
        assert metadata["quantile"] == quantile
    
    def test_compare_mechanisms(self):
        """Test mechanism comparison."""
        epsilon = 1.0
        delta = 1e-5
        function = "mean"
        
        results = self.dp_mechanisms.compare_mechanisms(
            self.test_data, epsilon, delta, function
        )
        
        # Check that results is a dictionary
        assert isinstance(results, dict)
        
        # Check that both mechanisms are present
        assert "laplace" in results
        assert "gaussian" in results
        
        # Check that each result is a tuple
        for mechanism, result in results.items():
            assert isinstance(result, tuple)
            assert len(result) == 2  # (result, metadata)
    
    def test_invalid_function(self):
        """Test with invalid function."""
        epsilon = 1.0
        delta = 0.0
        
        with pytest.raises(ValueError):
            self.dp_mechanisms.laplace_mechanism(
                self.test_data, epsilon, "invalid_function", delta
            )
    
    def test_invalid_mechanism(self):
        """Test with invalid mechanism."""
        epsilon = 1.0
        delta = 1e-5
        
        with pytest.raises(ValueError):
            self.dp_mechanisms.gaussian_mechanism(
                self.test_data, epsilon, delta, "invalid_function"
            )


class TestPrivacyAccountant:
    """Test class for privacy accountant utilities."""
    
    def test_compute_sensitivity(self):
        """Test sensitivity computation."""
        data = np.array([1, 2, 3, 4, 5])
        
        # Test mean sensitivity
        mean_sensitivity = compute_sensitivity(data, "mean")
        expected_mean_sensitivity = (5 - 1) / 5  # (max - min) / n
        assert abs(mean_sensitivity - expected_mean_sensitivity) < 1e-10
        
        # Test sum sensitivity
        sum_sensitivity = compute_sensitivity(data, "sum")
        expected_sum_sensitivity = 5 - 1  # max - min
        assert abs(sum_sensitivity - expected_sum_sensitivity) < 1e-10
        
        # Test count sensitivity
        count_sensitivity = compute_sensitivity(data, "count")
        assert count_sensitivity == 1.0
        
        # Test max sensitivity
        max_sensitivity = compute_sensitivity(data, "max")
        expected_max_sensitivity = 5 - 1  # max - min
        assert abs(max_sensitivity - expected_max_sensitivity) < 1e-10
        
        # Test min sensitivity
        min_sensitivity = compute_sensitivity(data, "min")
        expected_min_sensitivity = 5 - 1  # max - min
        assert abs(min_sensitivity - expected_min_sensitivity) < 1e-10
    
    def test_compute_noise_scale(self):
        """Test noise scale computation."""
        sensitivity = 1.0
        epsilon = 1.0
        delta = 1e-5
        
        # Test Laplace mechanism (delta = 0)
        laplace_scale = compute_noise_scale(sensitivity, epsilon, 0.0)
        expected_laplace_scale = sensitivity / epsilon
        assert abs(laplace_scale - expected_laplace_scale) < 1e-10
        
        # Test Gaussian mechanism (delta > 0)
        gaussian_scale = compute_noise_scale(sensitivity, epsilon, delta)
        expected_gaussian_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        assert abs(gaussian_scale - expected_gaussian_scale) < 1e-10
    
    def test_invalid_function(self):
        """Test with invalid function."""
        data = np.array([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError):
            compute_sensitivity(data, "invalid_function")
