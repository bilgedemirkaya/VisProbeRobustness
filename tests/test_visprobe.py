"""Smoke and integration tests for VisProbe core surface."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


def _dummy_model(n_classes: int = 10) -> nn.Module:
    class DummyModel(nn.Module):
        def __init__(self):
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


def test_checkpoint_roundtrip(tmp_path: Path):
    from visprobe import checkpoint

    root = tmp_path / "ckpt"
    metadata_path = root / "metadata.json"

    checkpoint.save_metadata({"version": "test"}, metadata_path)
    checkpoint.save_cell({"accuracy": 0.95}, root, "model1", "blur", 0.5)

    assert checkpoint.is_completed(root, "model1", "blur", 0.5)
    assert not checkpoint.is_completed(root, "model1", "blur", 0.7)
    assert checkpoint.load_cell(root, "model1", "blur", 0.5)["accuracy"] == 0.95
    assert checkpoint.load_metadata(metadata_path) == {"version": "test"}


def test_memory_manager_swap():
    from visprobe.memory import ModelMemoryManager

    models = {"a": _dummy_model(), "b": _dummy_model()}
    mgr = ModelMemoryManager(models, device="cpu")

    mgr.load_model("a")
    mgr.load_model("b")
    assert mgr.current_model_name == "b"

    assert "total_mb" in mgr.estimate_model_memory("a")

    mgr.release_all()
    assert mgr.current_model_name is None


@pytest.mark.parametrize(
    "name",
    ["blur", "noise", "brightness", "lowlight"],
)
def test_standard_perturbations_shape_and_zero_severity(name):
    from visprobe.perturbations import get_standard_perturbations

    perturbations = get_standard_perturbations()
    assert name in perturbations

    images = torch.rand(2, 3, 32, 32)
    perturbed = perturbations[name](images, severity=0.5)
    assert perturbed.shape == images.shape

    # severity=0 should be a no-op (within floating-point tolerance)
    untouched = perturbations[name](images, severity=0.0)
    assert torch.allclose(untouched, images)


def test_lowlight_darkens():
    from visprobe.perturbations import LowLight

    images = torch.rand(2, 3, 32, 32)
    darker = LowLight(gamma_max=5.0)(images, severity=1.0)
    assert (darker <= images + 1e-6).all()


def test_pgd_and_none_attacks():
    from visprobe import attacks

    model = _dummy_model()
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([0, 1])

    pgd = attacks.build("pgd", eps=0.01)
    perturbed = pgd(model, images, labels)
    assert perturbed.shape == images.shape
    assert (perturbed >= 0).all() and (perturbed <= 1).all()

    none = attacks.build("none", eps=0.01)
    assert torch.allclose(none(model, images, labels), images)


def test_results_save_load_and_auc(tmp_path: Path):
    from visprobe.results import CompositionalResults, EvaluationResult

    results = CompositionalResults()
    for model in ["m1", "m2"]:
        for scenario in ["blur", "noise"]:
            for severity in [0.0, 0.5, 1.0]:
                results.add_result(
                    model,
                    scenario,
                    severity,
                    EvaluationResult(
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
                        metadata={},
                    ),
                )

    assert "m1" in results.get_models()
    assert "blur" in results.get_scenarios()
    assert 0.5 in results.get_severities("m1", "blur")
    assert 0 <= results.compute_auc("m1", "blur") <= 1

    save_path = tmp_path / "results"
    results.save(str(save_path))
    loaded = CompositionalResults.load(str(save_path))
    assert set(loaded.get_models()) == set(results.get_models())


def test_cost_estimate_math_and_threshold():
    from visprobe import cost

    # Tiny sweep: 2 cells, n=10, no attack. Should be far below threshold.
    cheap = cost.estimate("none", n_cells=2, n_samples=10)
    assert cheap["total_hours"] < 0.01
    assert not cost.is_expensive(cheap)
    assert "estimated" in cost.format_estimate(cheap).lower()
    assert "roughly" in cost.format_estimate(cheap).lower()

    # Heavy sweep: full AutoAttack, 5 cells, n=1000.
    # 5 * 1800 = 9000s = 2.5h, well above the 1h gate.
    heavy = cost.estimate("autoattack-standard", n_cells=5, n_samples=1000)
    assert heavy["total_hours"] > cost.THRESHOLD_HOURS
    assert cost.is_expensive(heavy)

    # Linear scaling with n_samples is real.
    a = cost.estimate("pgd", n_cells=1, n_samples=1000)
    b = cost.estimate("pgd", n_cells=1, n_samples=4000)
    assert b["secs_per_cell"] == pytest.approx(a["secs_per_cell"] * 4)


def test_experiment_blocks_when_expensive_without_confirm(tmp_path: Path):
    from visprobe import CompositionalExperiment, get_standard_perturbations

    # 1 model x 4 scenarios x 3 severities x n=1000 x autoattack-standard
    # = 12 cells * 1800s = 21600s = 6h. Well above threshold.
    images = torch.rand(1000, 3, 8, 8)   # tensor never read; just used for len()
    labels = torch.randint(0, 10, (1000,))

    exp = CompositionalExperiment(
        models={"m": _dummy_model()},
        images=images,
        labels=labels,
        env_strategies=get_standard_perturbations(),
        attack="autoattack-standard",
        severities=[0.0, 0.5, 1.0],
        checkpoint_dir=str(tmp_path / "exp"),
        batch_size=50,
        device="cpu",
        verbose=False,
    )

    with pytest.raises(RuntimeError, match="confirm=True"):
        exp.run()


def test_experiment_runs_and_resumes(tmp_path: Path):
    from visprobe import CompositionalExperiment, get_standard_perturbations

    models = {"m": _dummy_model()}
    images = torch.rand(10, 3, 32, 32)
    labels = torch.randint(0, 10, (10,))
    blur_only = {"blur": get_standard_perturbations()["blur"]}
    ckpt = str(tmp_path / "exp")

    exp = CompositionalExperiment(
        models=models,
        images=images,
        labels=labels,
        env_strategies=blur_only,
        attack="none",
        severities=[0.0, 1.0],
        checkpoint_dir=ckpt,
        batch_size=5,
        device="cpu",
        verbose=False,
    )
    results = exp.run()
    assert "blur" in results.get_scenarios()
    assert len(results.get_severities("m", "blur")) == 2

    # Resume with one extra severity; only that new point should run.
    exp2 = CompositionalExperiment(
        models=models,
        images=images,
        labels=labels,
        env_strategies=blur_only,
        attack="none",
        severities=[0.0, 0.5, 1.0],
        checkpoint_dir=ckpt,
        batch_size=5,
        device="cpu",
        verbose=False,
    )
    results2 = exp2.run()
    assert len(results2.get_severities("m", "blur")) == 3

    exp.cleanup()
