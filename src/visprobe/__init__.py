"""
VisProbe: Simple, production-ready robustness testing for vision models.

One clear goal: Test vision model robustness under compositional conditions
(environmental degradation + adversarial attack) with automatic checkpointing,
memory management, and parallel execution.

Quick Start:
    >>> from visprobe import CompositionalExperiment, get_standard_perturbations
    >>>
    >>> # Create experiment
    >>> experiment = CompositionalExperiment(
    ...     models={"resnet50": model1, "vit": model2},
    ...     images=images,
    ...     labels=labels,
    ...     env_strategies=get_standard_perturbations(),
    ...     attack="autoattack-apgd-ce",  # Fast mode
    ...     checkpoint_dir="./checkpoints"
    ... )
    >>>
    >>> # Run (auto-resumes if interrupted)
    >>> results = experiment.run()
    >>>
    >>> # Analyze
    >>> results.print_summary()
    >>> results.plot_compositional()
    >>> gaps = results.protection_gap(baseline="resnet50")
    >>>
    >>> # Save for later
    >>> results.save("./results")
    >>>
    >>> # Load without models/GPU
    >>> from visprobe import CompositionalResults
    >>> results = CompositionalResults.load("./results")
"""

__version__ = "2.0.0"

# Main API
from .experiment import CompositionalExperiment, quick_test
from .results import CompositionalResults, EvaluationResult
from .perturbations import (
    get_standard_perturbations,
    get_minimal_perturbations,
    GaussianBlur,
    GaussianNoise,
    Brightness,
    Contrast,
    LowLight,
    MotionBlur,
    SaltPepperNoise,
    Compose
)
from .attacks import AttackFactory

# Keep compatibility with existing analysis functions
try:
    from .analysis import (
        evaluate_detailed,
        DetailedResults,
        SampleResult
    )
except ImportError:
    # Analysis module might not exist yet in refactored version
    pass

__all__ = [
    # Core API
    "CompositionalExperiment",
    "CompositionalResults",
    "EvaluationResult",

    # Perturbations
    "get_standard_perturbations",
    "get_minimal_perturbations",
    "GaussianBlur",
    "GaussianNoise",
    "Brightness",
    "Contrast",
    "LowLight",
    "MotionBlur",
    "SaltPepperNoise",
    "Compose",

    # Attacks
    "AttackFactory",

    # Utilities
    "quick_test",
]