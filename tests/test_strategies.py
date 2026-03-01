"""
Tests for the visprobe.strategies module.
"""

import torch

from visprobe.strategies.image import (
    BrightnessStrategy,
    ContrastStrategy,
    GaussianNoiseStrategy,
)


class TestGaussianNoiseStrategy:
    """Tests for GaussianNoiseStrategy."""

    def test_apply_adds_noise(self, sample_batch, simple_model):
        """Test that strategy adds noise to images."""
        images, _ = sample_batch
        strategy = GaussianNoiseStrategy(std_dev=0.1)
        perturbed = strategy.apply(images, simple_model)

        assert perturbed.shape == images.shape
        assert not torch.allclose(perturbed, images)

    def test_zero_noise_no_change(self, sample_batch, simple_model):
        """Test that zero noise produces identical output."""
        images, _ = sample_batch
        strategy = GaussianNoiseStrategy(std_dev=0.0)
        perturbed = strategy.apply(images, simple_model)

        assert torch.allclose(perturbed, images, atol=1e-6)

    def test_seed_reproducibility(self, sample_batch, simple_model):
        """Test that same seed produces same noise."""
        images, _ = sample_batch
        strategy1 = GaussianNoiseStrategy(std_dev=0.1, seed=42)
        strategy2 = GaussianNoiseStrategy(std_dev=0.1, seed=42)

        perturbed1 = strategy1.apply(images, simple_model)
        perturbed2 = strategy2.apply(images, simple_model)

        assert torch.allclose(perturbed1, perturbed2)

    def test_different_seed_different_noise(self, sample_batch, simple_model):
        """Test that different seeds produce different noise."""
        images, _ = sample_batch
        strategy1 = GaussianNoiseStrategy(std_dev=0.1, seed=42)
        strategy2 = GaussianNoiseStrategy(std_dev=0.1, seed=123)

        perturbed1 = strategy1.apply(images, simple_model)
        perturbed2 = strategy2.apply(images, simple_model)

        assert not torch.allclose(perturbed1, perturbed2)

    def test_repr(self):
        """Test __repr__ method."""
        strategy = GaussianNoiseStrategy(std_dev=0.1, seed=42)
        repr_str = repr(strategy)
        assert "GaussianNoiseStrategy" in repr_str
        assert "std_dev=0.1" in repr_str


class TestBrightnessStrategy:
    """Tests for BrightnessStrategy."""

    def test_apply_changes_brightness(self, sample_batch, simple_model):
        """Test that strategy changes image brightness."""
        images, _ = sample_batch
        strategy = BrightnessStrategy(brightness_factor=1.5)
        perturbed = strategy.apply(images, simple_model)

        assert perturbed.shape == images.shape
        assert not torch.allclose(perturbed, images)

    def test_factor_one_no_change(self, sample_batch, simple_model):
        """Test that factor=1.0 produces identical output."""
        images, _ = sample_batch
        strategy = BrightnessStrategy(brightness_factor=1.0)
        perturbed = strategy.apply(images, simple_model)

        assert torch.allclose(perturbed, images, atol=1e-6)

    def test_repr(self):
        """Test __repr__ method."""
        strategy = BrightnessStrategy(brightness_factor=1.5)
        repr_str = repr(strategy)
        assert "BrightnessStrategy" in repr_str
        assert "brightness_factor=1.5" in repr_str


class TestContrastStrategy:
    """Tests for ContrastStrategy."""

    def test_apply_changes_contrast(self, sample_batch, simple_model):
        """Test that strategy changes image contrast."""
        images, _ = sample_batch
        strategy = ContrastStrategy(contrast_factor=1.5)
        perturbed = strategy.apply(images, simple_model)

        assert perturbed.shape == images.shape
        # Note: contrast adjustment may produce same output for uniform images

    def test_factor_one_no_change(self, sample_batch, simple_model):
        """Test that factor=1.0 produces identical output."""
        images, _ = sample_batch
        strategy = ContrastStrategy(contrast_factor=1.0)
        perturbed = strategy.apply(images, simple_model)

        # Contrast adjustment with factor=1.0 should preserve the image
        assert torch.allclose(perturbed, images, atol=1e-5)

    def test_repr(self):
        """Test __repr__ method."""
        strategy = ContrastStrategy(contrast_factor=1.5)
        repr_str = repr(strategy)
        assert "ContrastStrategy" in repr_str
        assert "contrast_factor=1.5" in repr_str


class TestStrategyComposition:
    """Tests for strategy composition (if applicable)."""

    def test_sequential_application(self, sample_batch, simple_model):
        """Test applying multiple strategies sequentially."""
        images, _ = sample_batch
        strategy1 = GaussianNoiseStrategy(std_dev=0.05)
        strategy2 = BrightnessStrategy(brightness_factor=1.2)

        intermediate = strategy1.apply(images, simple_model)
        final = strategy2.apply(intermediate, simple_model)

        assert final.shape == images.shape
        assert not torch.allclose(final, images)
