"""
Tests for workflow utilities.
"""

import torch
import pytest
import numpy as np

from visprobe.workflows import (
    SeveritySweep,
    CompositionalTest,
    run_severity_sweep,
    run_compositional_sweep,
    compute_auc,
)
from visprobe.strategies import (
    gaussian_blur_severity,
    gaussian_noise_severity,
    lowlight_severity,
)


class TestMetrics:
    """Test metric computation functions."""

    def test_compute_auc(self):
        """Test AUC computation."""
        severities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        accuracies = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

        auc = compute_auc(severities, accuracies)

        # AUC should be between 0 and 1
        assert 0 <= auc <= 1

        # For linearly decreasing accuracy, AUC should be around 0.75
        assert 0.7 < auc < 0.8

    def test_compute_auc_edge_cases(self):
        """Test AUC edge cases."""
        # All perfect
        auc_perfect = compute_auc([0.0, 0.5, 1.0], [1.0, 1.0, 1.0])
        assert auc_perfect == 1.0

        # All failed
        auc_failed = compute_auc([0.0, 0.5, 1.0], [0.0, 0.0, 0.0])
        assert auc_failed == 0.0

        # Should raise on mismatched lengths
        with pytest.raises(ValueError):
            compute_auc([0.0, 0.5], [1.0, 0.9, 0.8])


class TestSeveritySweep:
    """Test SeveritySweep workflow."""

    @pytest.fixture
    def dummy_model(self):
        """Create a dummy classification model."""
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 32 * 32, 10)
        )

    @pytest.fixture
    def dummy_data(self):
        """Create dummy images and labels."""
        images = torch.rand(20, 3, 32, 32)
        labels = torch.randint(0, 10, (20,))
        return images, labels

    def test_severity_sweep_init(self):
        """Test SeveritySweep initialization."""
        strategy = gaussian_blur_severity(sigma_max=3.0)

        sweep = SeveritySweep(
            strategy=strategy,
            severities=[0.0, 0.5, 1.0],
            batch_size=10,
        )

        assert sweep.strategy is strategy
        assert sweep.severities == [0.0, 0.5, 1.0]
        assert sweep.batch_size == 10

    def test_severity_sweep_run(self, dummy_model, dummy_data):
        """Test single model severity sweep."""
        images, labels = dummy_data
        strategy = gaussian_blur_severity(sigma_max=2.0)

        sweep = SeveritySweep(
            strategy=strategy,
            severities=[0.0, 0.5, 1.0],
            batch_size=10,
            show_progress=False,
        )

        results = sweep.run(
            model=dummy_model,
            images=images,
            labels=labels,
            model_name="TestModel",
        )

        # Should have one result per severity
        assert len(results) == 3

        # Each result should have required fields
        for result in results:
            assert hasattr(result, "accuracy")
            assert hasattr(result, "model_name")
            assert hasattr(result, "scenario")
            assert hasattr(result, "mean_confidence")
            assert hasattr(result, "mean_loss")
            assert result.model_name == "TestModel"

        # Check metadata contains severity
        assert "severity" in results[0].metadata
        assert results[0].metadata["severity"] == 0.0
        assert results[1].metadata["severity"] == 0.5
        assert results[2].metadata["severity"] == 1.0

    def test_severity_sweep_multi_model(self, dummy_model, dummy_data):
        """Test multi-model severity sweep."""
        images, labels = dummy_data

        model2 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 32 * 32, 10)
        )

        models = {"Model1": dummy_model, "Model2": model2}

        sweep = SeveritySweep(
            strategy=gaussian_noise_severity(std_max=0.1, seed=42),
            severities=[0.0, 1.0],
            show_progress=False,
        )

        results = sweep.run_multi_model(models, images, labels)

        # Should have results for both models
        assert "Model1" in results
        assert "Model2" in results

        # Each model should have 2 results (2 severities)
        assert len(results["Model1"]) == 2
        assert len(results["Model2"]) == 2

    def test_compute_auc_from_sweep(self, dummy_model, dummy_data):
        """Test AUC computation from sweep results."""
        images, labels = dummy_data

        sweep = SeveritySweep(
            strategy=lowlight_severity(max_reduction=0.5),
            severities=[0.0, 0.5, 1.0],
            show_progress=False,
        )

        results = sweep.run(dummy_model, images, labels)

        auc = sweep.compute_auc(results)

        # AUC should be between 0 and 1
        assert 0 <= auc <= 1


class TestCompositionalTest:
    """Test CompositionalTest workflow."""

    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model."""
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 32 * 32, 10)
        )

    @pytest.fixture
    def dummy_data(self):
        """Create dummy data."""
        images = torch.rand(20, 3, 32, 32)
        labels = torch.randint(0, 10, (20,))
        return images, labels

    @pytest.fixture
    def dummy_attack(self):
        """Create a dummy attack function."""
        def attack_fn(model, images, labels, eps):
            if eps < 1e-8:
                return images
            # Add small noise as mock attack
            return torch.clamp(images + eps * torch.randn_like(images), 0, 1)

        return attack_fn

    def test_compositional_init(self, dummy_attack):
        """Test CompositionalTest initialization."""
        strategy = lowlight_severity(max_reduction=0.7)
        eps_fn = lambda s: 0.01 * s

        test = CompositionalTest(
            env_strategy=strategy,
            attack_fn=dummy_attack,
            eps_fn=eps_fn,
            severities=[0.0, 0.5, 1.0],
        )

        assert test.attack_fn is dummy_attack
        assert test.eps_fn is eps_fn

    def test_compositional_run(self, dummy_model, dummy_data, dummy_attack):
        """Test compositional test execution."""
        images, labels = dummy_data

        test = CompositionalTest(
            env_strategy=gaussian_blur_severity(sigma_max=2.0),
            attack_fn=dummy_attack,
            eps_fn=lambda s: 0.01 * s,
            severities=[0.0, 0.5, 1.0],
            show_progress=False,
        )

        results = test.run(
            model=dummy_model,
            images=images,
            labels=labels,
            model_name="TestModel",
        )

        # Should have 3 results
        assert len(results) == 3

        # Check metadata contains both severity and eps
        assert "severity" in results[0].metadata
        assert "eps" in results[0].metadata

        # Check eps was computed correctly
        assert results[0].metadata["eps"] == 0.0  # 0.01 * 0.0
        assert abs(results[1].metadata["eps"] - 0.005) < 1e-6  # 0.01 * 0.5
        assert abs(results[2].metadata["eps"] - 0.01) < 1e-6  # 0.01 * 1.0


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    @pytest.fixture
    def dummy_model(self):
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 32 * 32, 10)
        )

    @pytest.fixture
    def dummy_data(self):
        images = torch.rand(10, 3, 32, 32)
        labels = torch.randint(0, 10, (10,))
        return images, labels

    def test_run_severity_sweep(self, dummy_model, dummy_data):
        """Test run_severity_sweep convenience function."""
        images, labels = dummy_data

        results = run_severity_sweep(
            model=dummy_model,
            images=images,
            labels=labels,
            strategy=gaussian_blur_severity(sigma_max=2.0),
            severities=[0.0, 1.0],
            show_progress=False,
        )

        assert len(results) == 2

    def test_run_compositional_sweep(self, dummy_model, dummy_data):
        """Test run_compositional_sweep convenience function."""
        images, labels = dummy_data

        def dummy_attack(model, images, labels, eps):
            return images

        results = run_compositional_sweep(
            model=dummy_model,
            images=images,
            labels=labels,
            env_strategy=lowlight_severity(max_reduction=0.5),
            attack_fn=dummy_attack,
            eps_fn=lambda s: 0.01 * s,
            severities=[0.0, 1.0],
            show_progress=False,
        )

        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
