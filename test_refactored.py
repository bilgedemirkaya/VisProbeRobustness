"""
Simple test script for the refactored VisProbe implementation.
Tests core functionality with minimal dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import shutil


def create_dummy_model():
    """Create a simple dummy model for testing."""
    class DummyModel(nn.Module):
        def __init__(self, n_classes=10):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, n_classes)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    return DummyModel()


def test_checkpoint_manager():
    """Test CheckpointManager functionality."""
    print("Testing CheckpointManager...")

    from visprobe.checkpoint import CheckpointManager

    # Create checkpoint manager
    checkpoint_dir = "./test_checkpoints"
    mgr = CheckpointManager(checkpoint_dir)

    # Save metadata
    metadata = {"test": "metadata", "version": "1.0"}
    mgr.save_metadata(metadata)

    # Save a checkpoint
    test_result = {"accuracy": 0.95, "data": [1, 2, 3]}
    mgr.save_checkpoint(test_result, "model1", "blur", 0.5)

    # Check if completed
    assert mgr.is_completed("model1", "blur", 0.5)
    assert not mgr.is_completed("model1", "blur", 0.7)

    # Load checkpoint
    loaded = mgr.load_checkpoint("model1", "blur", 0.5)
    assert loaded["accuracy"] == 0.95

    # Get resume point
    resume = mgr.get_resume_point()
    assert resume is not None

    # Cleanup
    shutil.rmtree(checkpoint_dir)
    print("✓ CheckpointManager test passed")


def test_memory_manager():
    """Test ModelMemoryManager functionality."""
    print("Testing ModelMemoryManager...")

    from visprobe.memory import ModelMemoryManager

    # Create dummy models
    models = {
        "model1": create_dummy_model(),
        "model2": create_dummy_model(),
    }

    # Create memory manager
    mgr = ModelMemoryManager(models, device="cpu")

    # Load model
    model1 = mgr.load_model("model1")
    assert model1 is not None

    # Switch models
    model2 = mgr.load_model("model2")
    assert mgr.current_model_name == "model2"

    # Estimate memory
    memory_info = mgr.estimate_model_memory("model1")
    assert "total_mb" in memory_info

    # Release all
    mgr.release_all()
    assert mgr.current_model_name is None

    print("✓ ModelMemoryManager test passed")


def test_perturbations():
    """Test perturbation functions."""
    print("Testing perturbations...")

    from visprobe.perturbations import (
        GaussianBlur,
        GaussianNoise,
        LowLight,
        get_minimal_perturbations
    )

    # Create test image
    images = torch.rand(2, 3, 32, 32)

    # Test individual perturbations
    blur = GaussianBlur(sigma_max=3.0)
    blurred = blur(images, severity=0.5)
    assert blurred.shape == images.shape
    assert torch.allclose(blur(images, severity=0.0), images)

    noise = GaussianNoise(std_max=0.1)
    noisy = noise(images, severity=0.5)
    assert noisy.shape == images.shape

    lowlight = LowLight(gamma_max=5.0)
    dark = lowlight(images, severity=0.5)
    assert dark.shape == images.shape
    assert (dark <= images).all()  # Should be darker

    # Test preset
    perturbations = get_minimal_perturbations()
    assert "blur" in perturbations
    assert "noise" in perturbations
    assert "lowlight" in perturbations

    print("✓ Perturbations test passed")


def test_attacks():
    """Test attack factory."""
    print("Testing attacks...")

    from visprobe.attacks import AttackFactory

    # Create test data
    model = create_dummy_model()
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([0, 1])

    # Test PGD attack
    pgd = AttackFactory.create("pgd", eps=0.01)
    perturbed = pgd(model, images, labels)
    assert perturbed.shape == images.shape
    assert (perturbed >= 0).all() and (perturbed <= 1).all()

    # Test no attack
    no_attack = AttackFactory.create("none", eps=0.01)
    unchanged = no_attack(model, images, labels)
    assert torch.allclose(unchanged, images)

    print("✓ Attacks test passed")


def test_results():
    """Test CompositionalResults functionality."""
    print("Testing CompositionalResults...")

    from visprobe.results import CompositionalResults, EvaluationResult

    # Create results container
    results = CompositionalResults()

    # Add some dummy results
    for model in ["model1", "model2"]:
        for scenario in ["blur", "noise"]:
            for severity in [0.0, 0.5, 1.0]:
                result = EvaluationResult(
                    accuracy=1.0 - severity * 0.3,
                    mean_confidence=0.9 - severity * 0.2,
                    mean_loss=severity * 0.5,
                    correct_mask=np.array([True, False, True]),
                    predictions=np.array([0, 1, 0]),
                    confidences=np.array([0.9, 0.8, 0.7]),
                    model_name=model,
                    scenario=scenario,
                    severity=severity,
                    eps=severity * 0.01,
                    n_samples=3,
                    metadata={}
                )
                results.add_result(model, scenario, severity, result)

    # Test getters
    assert "model1" in results.get_models()
    assert "blur" in results.get_scenarios()
    assert 0.5 in results.get_severities("model1", "blur")

    # Test analysis
    gaps = results.protection_gap(baseline="model1")
    assert "model2" in gaps

    crossovers = results.crossover_detection(baseline="model1", threshold=0.5)
    assert "model2" in crossovers

    auc = results.compute_auc("model1", "blur")
    assert 0 <= auc <= 1

    # Test save/load
    save_path = "./test_results"
    results.save(save_path)
    loaded = CompositionalResults.load(save_path)
    assert len(loaded.get_models()) == len(results.get_models())

    # Cleanup
    shutil.rmtree(save_path)

    print("✓ CompositionalResults test passed")


def test_experiment():
    """Test CompositionalExperiment with minimal setup."""
    print("Testing CompositionalExperiment...")

    from visprobe import CompositionalExperiment, get_minimal_perturbations

    # Create minimal test setup
    models = {"test_model": create_dummy_model()}
    images = torch.rand(10, 3, 32, 32)
    labels = torch.randint(0, 10, (10,))

    # Create minimal experiment
    experiment = CompositionalExperiment(
        models=models,
        images=images,
        labels=labels,
        env_strategies={"blur": get_minimal_perturbations()["blur"]},
        attack="none",  # No attack for speed
        severities=[0.0, 1.0],
        checkpoint_dir="./test_exp_checkpoints",
        batch_size=5,
        device="cpu",
        verbose=False
    )

    # Run experiment
    results = experiment.run()

    # Check results
    assert "test_model" in results.get_models()
    assert "blur" in results.get_scenarios()
    assert len(results.get_severities("test_model", "blur")) == 2

    # Test resumption by creating new experiment with same checkpoint dir
    experiment2 = CompositionalExperiment(
        models=models,
        images=images,
        labels=labels,
        env_strategies={"blur": get_minimal_perturbations()["blur"]},
        attack="none",
        severities=[0.0, 0.5, 1.0],  # Add a new severity
        checkpoint_dir="./test_exp_checkpoints",
        batch_size=5,
        device="cpu",
        verbose=False
    )

    # Should only evaluate the new severity
    results2 = experiment2.run()
    assert len(results2.get_severities("test_model", "blur")) == 3

    # Cleanup
    experiment.cleanup()
    shutil.rmtree("./test_exp_checkpoints")

    print("✓ CompositionalExperiment test passed")


def test_quick_test():
    """Test quick_test utility."""
    print("Testing quick_test...")

    from visprobe import quick_test

    model = create_dummy_model()
    images = torch.rand(5, 3, 32, 32)
    labels = torch.randint(0, 10, (5,))

    metrics = quick_test(
        model=model,
        images=images,
        labels=labels,
        attack="none",
        severity=0.5
    )

    assert isinstance(metrics, dict)
    assert any("accuracy" in key for key in metrics)

    print("✓ quick_test passed")


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Refactored VisProbe Implementation")
    print("="*60)

    tests = [
        test_checkpoint_manager,
        test_memory_manager,
        test_perturbations,
        test_attacks,
        test_results,
        test_experiment,
        test_quick_test,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed.append(test.__name__)

    print("\n" + "="*60)
    if not failed:
        print("ALL TESTS PASSED! ✓")
    else:
        print(f"FAILED TESTS: {', '.join(failed)}")
    print("="*60)


if __name__ == "__main__":
    main()