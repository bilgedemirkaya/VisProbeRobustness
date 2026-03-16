"""
Workflows module for common robustness testing patterns.

Provides high-level utilities for:
- Severity sweeps (evaluating across perturbation levels)
- Compositional testing (environmental + adversarial)
- Multi-model comparisons
- Automated experiment orchestration
"""

from .severity_sweep import (
    SeveritySweep,
    CompositionalTest,
    run_severity_sweep,
    run_compositional_sweep,
)
from .metrics import compute_auc, compute_robustness_curve

__all__ = [
    "SeveritySweep",
    "CompositionalTest",
    "run_severity_sweep",
    "run_compositional_sweep",
    "compute_auc",
    "compute_robustness_curve",
]
