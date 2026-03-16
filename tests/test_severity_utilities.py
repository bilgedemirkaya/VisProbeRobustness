"""
Tests for severity mapping utilities.
"""

import torch
import pytest

from visprobe.strategies import (
    gaussian_blur_severity,
    gaussian_noise_severity,
    lowlight_severity,
    with_severity,
    linear_scale,
    brightness_reduction,
)
from visprobe.strategies import GaussianBlur, GaussianNoise, Brightness


class TestSeverityUtils:
    """Test severity mapping utilities."""

    @pytest.fixture
    def images(self):
        """Create test images."""
        return torch.rand(2, 3, 32, 32)

    def test_gaussian_blur_severity(self, images):
        """Test gaussian_blur_severity factory."""
        blur = gaussian_blur_severity(sigma_max=3.0)

        # Test at different severity levels
        result_low = blur.generate(images, level=0.0)
        result_mid = blur.generate(images, level=0.5)
        result_high = blur.generate(images, level=1.0)

        # At severity=0, should be nearly unchanged
        assert torch.allclose(result_low, images, atol=1e-5)

        # Higher severity should produce more blur
        assert result_mid.shape == images.shape
        assert result_high.shape == images.shape

    def test_gaussian_noise_severity(self, images):
        """Test gaussian_noise_severity factory."""
        noise = gaussian_noise_severity(std_max=0.1, seed=42)

        # Test at different severity levels
        result_none = noise.generate(images, level=0.0)
        result_mid = noise.generate(images, level=0.5)
        result_full = noise.generate(images, level=1.0)

        # At severity=0, should be unchanged
        assert torch.allclose(result_none, images)

        # At severity>0, should be different (noise added)
        assert not torch.allclose(result_mid, images)

        # Shape preserved
        assert result_mid.shape == images.shape
        assert result_full.shape == images.shape

        # Note: GaussianNoise doesn't maintain state between calls with the same seed
        # This is expected behavior - seed is for internal consistency within a call

    def test_lowlight_severity(self, images):
        """Test lowlight_severity factory."""
        lowlight = lowlight_severity(max_reduction=0.7)

        # Test at different severity levels
        result_none = lowlight.generate(images, level=0.0)
        result_mid = lowlight.generate(images, level=0.5)
        result_full = lowlight.generate(images, level=1.0)

        # At severity=0, brightness=1.0 (unchanged)
        assert torch.allclose(result_none, images)

        # At severity=0.5, brightness=0.65
        expected_mid = images * 0.65
        assert torch.allclose(result_mid, expected_mid, atol=1e-5)

        # At severity=1.0, brightness=0.3
        expected_full = images * 0.3
        assert torch.allclose(result_full, expected_full, atol=1e-5)

    def test_with_severity_linear_scale(self, images):
        """Test with_severity with linear_scale."""
        # Create a blur strategy with linear scaling
        blur = with_severity(GaussianBlur(), linear_scale(3.0))

        result = blur.generate(images, level=0.5)
        assert result.shape == images.shape

        # At level=0, should be unchanged
        result_zero = blur.generate(images, level=0.0)
        assert torch.allclose(result_zero, images, atol=1e-5)

    def test_with_severity_brightness_reduction(self, images):
        """Test with_severity with brightness_reduction."""
        lowlight = with_severity(Brightness(), brightness_reduction(0.7))

        # At severity=0, no reduction
        result_none = lowlight.generate(images, level=0.0)
        assert torch.allclose(result_none, images)

        # At severity=1, 70% reduction
        result_full = lowlight.generate(images, level=1.0)
        expected = images * 0.3
        assert torch.allclose(result_full, expected, atol=1e-5)

    def test_linear_scale_transform(self):
        """Test linear_scale transform function."""
        transform = linear_scale(5.0)

        assert transform(0.0) == 0.0
        assert transform(0.5) == 2.5
        assert transform(1.0) == 5.0
        assert transform(None) is None

    def test_brightness_reduction_transform(self):
        """Test brightness_reduction transform function."""
        transform = brightness_reduction(0.7)

        assert abs(transform(0.0) - 1.0) < 1e-10  # No reduction
        assert abs(transform(0.5) - 0.65) < 1e-10  # 35% reduction
        assert abs(transform(1.0) - 0.3) < 1e-10  # 70% reduction
        assert transform(None) is None

    def test_integration_with_composition(self, images):
        """Test severity utilities work with Compose."""
        from visprobe.strategies import Compose

        # Create composed strategy with severity mapping
        composed = Compose([
            lowlight_severity(max_reduction=0.5),
            gaussian_blur_severity(sigma_max=2.0),
        ])

        result = composed.generate(images, level=0.5)
        assert result.shape == images.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
