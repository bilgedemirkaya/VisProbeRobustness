"""
Severity sweep workflows for robustness testing.

Provides high-level APIs for common experimental patterns:
- Single strategy severity sweeps
- Compositional testing (environmental + adversarial)
- Multi-model comparisons
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from tqdm import tqdm

from ..analysis import evaluate_detailed, EvaluationResult
from ..strategies import Strategy
from .metrics import compute_auc


@dataclass
class SeveritySweepConfig:
    """Configuration for severity sweep experiments."""

    severities: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    batch_size: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    show_progress: bool = True


class SeveritySweep:
    """
    High-level API for running severity sweeps.

    Evaluates model(s) across different perturbation severity levels.

    Example:
        >>> from visprobe.workflows import SeveritySweep
        >>> from visprobe.strategies import gaussian_blur_severity
        >>>
        >>> sweep = SeveritySweep(
        ...     strategy=gaussian_blur_severity(sigma_max=3.0),
        ...     severities=[0.0, 0.5, 1.0]
        ... )
        >>>
        >>> results = sweep.run(model, images, labels)
        >>> print(f"AUC: {sweep.compute_auc(results):.3f}")
    """

    def __init__(
        self,
        strategy: Union[Strategy, Callable],
        severities: Optional[Sequence[float]] = None,
        batch_size: int = 50,
        device: Optional[str] = None,
        show_progress: bool = True,
    ):
        """
        Initialize severity sweep.

        Args:
            strategy: Perturbation strategy (Strategy object or callable)
            severities: Severity levels to test (default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            batch_size: Batch size for evaluation
            device: Device for computation (default: auto-detect)
            show_progress: Whether to show progress bar
        """
        self.strategy = strategy
        self.severities = severities or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.show_progress = show_progress

    def run(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        model_name: str = "model",
        scenario: str = "severity_sweep",
    ) -> List[EvaluationResult]:
        """
        Run severity sweep for a single model.

        Args:
            model: Model to evaluate
            images: Input images
            labels: Ground truth labels
            model_name: Name for logging
            scenario: Scenario name for logging

        Returns:
            List of EvaluationResult objects, one per severity level
        """
        results = []

        iterator = (
            tqdm(self.severities, desc=f"  {model_name[:22]}", leave=False)
            if self.show_progress
            else self.severities
        )

        for severity in iterator:
            # Apply perturbation
            if isinstance(self.strategy, Strategy):
                perturbed = self.strategy.generate(images, level=severity)
            else:
                # Callable interface: (images, severity) -> perturbed
                perturbed = self.strategy(images, severity)

            # Evaluate
            result = evaluate_detailed(
                model=model,
                images=perturbed,
                labels=labels,
                batch_size=self.batch_size,
                device=self.device,
                model_name=model_name,
                scenario=f"{scenario}_s{severity:.2f}",
                severity=severity,  # Stored in metadata
            )

            results.append(result)

        return results

    def run_multi_model(
        self,
        models: Dict[str, torch.nn.Module],
        images: torch.Tensor,
        labels: torch.Tensor,
        scenario: str = "severity_sweep",
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Run severity sweep for multiple models.

        Args:
            models: Dictionary of {model_name: model}
            images: Input images
            labels: Ground truth labels
            scenario: Scenario name for logging

        Returns:
            Dictionary of {model_name: [EvaluationResult]}

        Example:
            >>> models = {"resnet50": model1, "vit": model2}
            >>> results = sweep.run_multi_model(models, images, labels)
            >>> for name, res in results.items():
            ...     auc = sweep.compute_auc(res)
            ...     print(f"{name}: AUC={auc:.3f}")
        """
        all_results = {}

        for model_name, model in models.items():
            results = self.run(
                model=model,
                images=images,
                labels=labels,
                model_name=model_name,
                scenario=scenario,
            )

            all_results[model_name] = results

            # Print summary
            if self.show_progress:
                auc = self.compute_auc(results)
                print(
                    f"  {model_name:28s} {results[0].accuracy:.1%} -> "
                    f"{results[-1].accuracy:.1%}  AUC={auc:.3f}"
                )

        return all_results

    def compute_auc(self, results: List[EvaluationResult]) -> float:
        """
        Compute AUC from sweep results.

        Args:
            results: Results from sweep

        Returns:
            AUC value
        """
        accuracies = [r.accuracy for r in results]
        return compute_auc(self.severities, accuracies)


class CompositionalTest(SeveritySweep):
    """
    Compositional robustness testing: environmental perturbation + adversarial attack.

    Extends SeveritySweep to support two-stage perturbation:
    1. Apply environmental perturbation (blur, noise, lighting, etc.)
    2. Apply adversarial attack on perturbed images

    Example:
        >>> from visprobe.workflows import CompositionalTest
        >>> from visprobe.strategies import gaussian_blur_severity
        >>>
        >>> # Define attack function (e.g., AutoAttack wrapper)
        >>> def run_attack(model, images, labels, eps):
        ...     if eps < 1e-8:
        ...         return images
        ...     # Run AutoAttack
        ...     return attacked_images
        >>>
        >>> test = CompositionalTest(
        ...     env_strategy=gaussian_blur_severity(sigma_max=3.0),
        ...     attack_fn=run_attack,
        ...     eps_fn=lambda s: 0.01 * s,  # Scale attack strength with severity
        ... )
        >>>
        >>> results = test.run(model, images, labels)
    """

    def __init__(
        self,
        env_strategy: Union[Strategy, Callable],
        attack_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, float], torch.Tensor],
        eps_fn: Callable[[float], float],
        severities: Optional[Sequence[float]] = None,
        batch_size: int = 50,
        device: Optional[str] = None,
        show_progress: bool = True,
    ):
        """
        Initialize compositional test.

        Args:
            env_strategy: Environmental perturbation strategy
            attack_fn: Attack function: (model, images, labels, eps) -> attacked_images
            eps_fn: Function mapping severity to attack strength: severity -> eps
            severities: Severity levels to test
            batch_size: Batch size for evaluation
            device: Device for computation
            show_progress: Whether to show progress bar
        """
        super().__init__(
            strategy=env_strategy,
            severities=severities,
            batch_size=batch_size,
            device=device,
            show_progress=show_progress,
        )
        self.attack_fn = attack_fn
        self.eps_fn = eps_fn

    def run(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        model_name: str = "model",
        scenario: str = "compositional",
    ) -> List[EvaluationResult]:
        """
        Run compositional test: env perturbation + attack.

        Args:
            model: Model to evaluate
            images: Input images
            labels: Ground truth labels
            model_name: Name for logging
            scenario: Scenario name for logging

        Returns:
            List of EvaluationResult objects
        """
        results = []

        iterator = (
            tqdm(self.severities, desc=f"  {model_name[:22]}", leave=False)
            if self.show_progress
            else self.severities
        )

        for severity in iterator:
            # Stage 1: Environmental perturbation
            if isinstance(self.strategy, Strategy):
                env_perturbed = self.strategy.generate(images, level=severity)
            else:
                env_perturbed = self.strategy(images, severity)

            # Stage 2: Adversarial attack
            eps = self.eps_fn(severity)
            final_images = self.attack_fn(model, env_perturbed, labels, eps)

            # Evaluate
            result = evaluate_detailed(
                model=model,
                images=final_images,
                labels=labels,
                batch_size=self.batch_size,
                device=self.device,
                model_name=model_name,
                scenario=f"{scenario}_s{severity:.2f}_eps{eps:.4f}",
                severity=severity,  # Stored in metadata
                eps=eps,  # Stored in metadata
            )

            results.append(result)

        return results


def run_severity_sweep(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    strategy: Union[Strategy, Callable],
    severities: Optional[Sequence[float]] = None,
    **kwargs,
) -> List[EvaluationResult]:
    """
    Convenience function for running a severity sweep.

    Args:
        model: Model to evaluate
        images: Input images
        labels: Ground truth labels
        strategy: Perturbation strategy
        severities: Severity levels (default: [0.0, 0.2, ..., 1.0])
        **kwargs: Additional arguments passed to SeveritySweep

    Returns:
        List of EvaluationResult objects

    Example:
        >>> from visprobe.workflows import run_severity_sweep
        >>> from visprobe.strategies import gaussian_blur_severity
        >>>
        >>> results = run_severity_sweep(
        ...     model=my_model,
        ...     images=test_images,
        ...     labels=test_labels,
        ...     strategy=gaussian_blur_severity(sigma_max=3.0),
        ... )
    """
    sweep = SeveritySweep(strategy=strategy, severities=severities, **kwargs)
    return sweep.run(model, images, labels)


def run_compositional_sweep(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    env_strategy: Union[Strategy, Callable],
    attack_fn: Callable,
    eps_fn: Callable[[float], float],
    severities: Optional[Sequence[float]] = None,
    **kwargs,
) -> List[EvaluationResult]:
    """
    Convenience function for running a compositional test.

    Args:
        model: Model to evaluate
        images: Input images
        labels: Ground truth labels
        env_strategy: Environmental perturbation strategy
        attack_fn: Attack function
        eps_fn: Severity -> attack strength mapping
        severities: Severity levels
        **kwargs: Additional arguments

    Returns:
        List of EvaluationResult objects

    Example:
        >>> from visprobe.workflows import run_compositional_sweep
        >>> from visprobe.strategies import lowlight_severity
        >>>
        >>> def autoattack(model, images, labels, eps):
        ...     # Run AutoAttack
        ...     return attacked_images
        >>>
        >>> results = run_compositional_sweep(
        ...     model=my_model,
        ...     images=test_images,
        ...     labels=test_labels,
        ...     env_strategy=lowlight_severity(max_reduction=0.7),
        ...     attack_fn=autoattack,
        ...     eps_fn=lambda s: 0.01 * s,
        ... )
    """
    test = CompositionalTest(
        env_strategy=env_strategy,
        attack_fn=attack_fn,
        eps_fn=eps_fn,
        severities=severities,
        **kwargs,
    )
    return test.run(model, images, labels)
