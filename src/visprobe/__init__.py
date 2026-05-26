"""
VisProbe: compositional robustness testing for vision models.

Test classification models under environmental degradation x adversarial attack x severity,
with automatic checkpointing and GPU memory management.

Quick start:

    >>> from visprobe import CompositionalExperiment, get_standard_perturbations
    >>>
    >>> experiment = CompositionalExperiment(
    ...     models={"resnet50": model},
    ...     images=images,
    ...     labels=labels,
    ...     env_strategies=get_standard_perturbations(),
    ...     attack="autoattack-apgd-ce",
    ...     checkpoint_dir="./checkpoints",
    ... )
    >>> results = experiment.run()
    >>> results.save("./results")
    >>>
    >>> # Later, load and inspect without a GPU:
    >>> from visprobe import CompositionalResults
    >>> results = CompositionalResults.load("./results")
"""

__version__ = "2.0.0"

from .experiment import CompositionalExperiment
from .results import CompositionalResults, EvaluationResult
from .perturbations import (
    get_standard_perturbations,
    GaussianBlur,
    GaussianNoise,
    Brightness,
    LowLight,
)
from .attacks import AttackFactory

__all__ = [
    "CompositionalExperiment",
    "CompositionalResults",
    "EvaluationResult",
    "get_standard_perturbations",
    "GaussianBlur",
    "GaussianNoise",
    "Brightness",
    "LowLight",
    "AttackFactory",
]
