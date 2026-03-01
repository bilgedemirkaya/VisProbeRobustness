"""
Core search engine for finding failure thresholds.

This module contains the unified SearchEngine class used by the search() function. The actual search algorithms are implemented as
pluggable strategies in search_strategies.py.

Supports three search modes:
- adaptive: Step-halving search (fast, good for unknown ranges)
- binary: Binary search (efficient for known ranges)
- bayesian: Bayesian optimization with GP (query-efficient, provides confidence intervals)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..strategies.base import Strategy
from .normalization import NormalizationHandler
from .search_strategies import (
    EvaluationResult,
    SearchStrategy,
)


class SearchEngine:
    """
    Finds failure threshold for a given perturbation strategy.

    VisProbe tests **accuracy preservation**: models should maintain correct predictions
    when inputs are perturbed. The SearchEngine automatically:
    1. Filters to correctly-classified samples (model_prediction == true_label)
    2. Tests if predictions remain correct under increasing perturbation levels
    3. Finds the threshold where predictions start to fail

    Fallback: If no correct samples exist (e.g., ImageNet model on CIFAR-10),
    SearchEngine uses model predictions as pseudo-labels and tests prediction
    consistency instead.

    Used by:
    - search(): Runs SearchEngine for a single user-specified strategy

    The search algorithm is delegated to a SearchStrategy instance, making
    it easy to add new search methods or customize existing ones.

    Args:
        model: PyTorch model to test
        strategy_factory: Factory function that creates a perturbation strategy at a given level
        property_fn: Function that evaluates if property holds (returns True if passes)
        samples: List of (image, label) tuples to test
        search_method: SearchStrategy instance. Default: AdaptiveSearchStrategy.
        level_lo: Lower bound for perturbation level
        level_hi: Upper bound for perturbation level
        initial_level: Starting level for adaptive search
        step: Initial step size for adaptive search
        min_step: Minimum step size before stopping
        max_queries: Maximum number of search iterations
        pass_threshold: Fraction of samples that must pass (default 0.9 = 90%)
        device: Device to run on
        batch_size: Number of images to process simultaneously (default 32)
        normalization: Normalization config - preset name, dict, or NormalizationHandler.
                      When set, handles denorm→perturb→renorm workflow automatically.

    Example:
        >>> # Using default AdaptiveSearchStrategy
        >>> engine = SearchEngine(model, strategy_factory, property_fn, samples)
        >>>
        >>> # Using custom search strategy
        >>> from visprobe.core.search_strategies import BayesianSearchStrategy
        >>> engine = SearchEngine(
        ...     model, strategy_factory, property_fn, samples,
        ...     search_method=BayesianSearchStrategy(n_initial=10, acquisition="ei")
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        strategy_factory: Callable[[float], Strategy],
        property_fn: Callable,
        samples: List[Tuple[torch.Tensor, int]],
        search_method: Union[str, SearchStrategy, None] = None,
        level_lo: float = 0.0,
        level_hi: float = 1.0,
        initial_level: Optional[float] = None,
        step: Optional[float] = None,
        min_step: float = 0.001,
        max_queries: int = 100,
        pass_threshold: float = 0.9,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        normalization: Union[str, Dict[str, Any], NormalizationHandler, None] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.strategy_factory = strategy_factory
        self.property_fn = property_fn
        self.samples = samples
        self.level_lo = level_lo
        self.level_hi = level_hi
        self.initial_level = initial_level if initial_level is not None else level_lo
        self.step = step if step is not None else (level_hi - level_lo) / 10.0
        self.min_step = min_step
        self.max_queries = max_queries
        self.pass_threshold = pass_threshold
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.verbose = verbose

        # Normalization handling
        self.normalization = NormalizationHandler.from_config(normalization)

        # Validate inputs
        if level_lo >= level_hi:
            raise ValueError(f"level_lo ({level_lo}) must be < level_hi ({level_hi})")
        if len(samples) == 0:
            raise ValueError("samples list cannot be empty")

        # Create search strategy from string or use provided instance
        self.search_strategy = self._create_search_strategy(
            search_method,
            initial_level=self.initial_level,
            step=self.step,
            min_step=min_step,
            pass_threshold=pass_threshold,
        )

        # Baseline accuracy tracking
        self.baseline_accuracy: Optional[float] = None
        self.correct_samples: Optional[List[Tuple[torch.Tensor, int]]] = None
        self.incorrect_indices: List[int] = []

    def _create_search_strategy(
        self,
        search_method: Union[str, "SearchStrategy", None],
        initial_level: float,
        step: float,
        min_step: float,
        pass_threshold: float,
    ) -> "SearchStrategy":
        """Create search strategy from string name or return provided instance."""
        from .search_strategies import (
            AdaptiveSearchStrategy,
            BinarySearchStrategy,
            BayesianSearchStrategy,
        )

        # If already a SearchStrategy instance, use it directly
        if isinstance(search_method, SearchStrategy):
            return search_method

        # Map string to strategy class
        method = search_method or "adaptive"

        if method == "adaptive":
            return AdaptiveSearchStrategy(
                initial_level=initial_level,
                step=step,
                min_step=min_step,
                pass_threshold=pass_threshold,
            )
        elif method == "binary":
            return BinarySearchStrategy(
                tolerance=min_step,
                pass_threshold=pass_threshold,
            )
        elif method == "bayesian":
            return BayesianSearchStrategy(
                pass_threshold=pass_threshold,
            )
        else:
            raise ValueError(
                f"Unknown search method: '{method}'. "
                f"Options: 'adaptive', 'binary', 'bayesian'"
            )

    def _check_baseline_accuracy(self) -> Tuple[float, List[int], List[int], List[int]]:
        """
        Check model accuracy on original (unperturbed) samples.

        Returns:
            (accuracy, correct_indices, incorrect_indices, original_predictions)
            - accuracy: fraction of samples where model prediction matches label
            - correct_indices: indices of correctly classified samples
            - incorrect_indices: indices of incorrectly classified samples
            - original_predictions: model's prediction for each sample (for reference)
        """
        self.model.eval()
        correct_indices = []
        incorrect_indices = []
        original_predictions = []

        with torch.no_grad():
            for batch_start in range(0, len(self.samples), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(self.samples))
                batch_samples = self.samples[batch_start:batch_end]

                batch_imgs = torch.stack([img for img, _ in batch_samples])
                batch_labels = [label for _, label in batch_samples]

                outputs = self.model(batch_imgs)
                preds = torch.argmax(outputs, dim=-1)

                for i, (pred, label) in enumerate(zip(preds, batch_labels)):
                    global_idx = batch_start + i
                    original_predictions.append(int(pred.item()))
                    if pred.item() == label:
                        correct_indices.append(global_idx)
                    else:
                        incorrect_indices.append(global_idx)

        accuracy = len(correct_indices) / len(self.samples) if self.samples else 0.0
        return accuracy, correct_indices, incorrect_indices, original_predictions

    def run(self, progress_bar: Optional[Any] = None) -> Dict[str, Any]:
        """
        Run search to find failure threshold.

        Args:
            progress_bar: Optional tqdm progress bar for updates

        Returns:
            dict with:
            - failure_threshold: float (level where model starts failing)
            - last_pass_level: float (highest level where model passed)
            - robustness_score: float (0-1, normalized score)
            - queries: int (number of search iterations used)
            - failures: list (failure cases with details)
            - search_path: list (history of search iterations)
            - runtime: float (seconds)
            - baseline_accuracy: float (model accuracy on original samples)
            - total_samples: int (original sample count)
            - valid_samples: int (correctly classified samples used for testing)

            Additional for bayesian mode:
            - confidence_interval: tuple (lower, upper bounds)
            - threshold_std: float (posterior standard deviation)
            - gp_levels: list (fine grid of levels)
            - gp_mean: list (GP mean predictions)
            - gp_std: list (GP std predictions)
        """
        start_time = time.time()

        # Check baseline accuracy first
        total_samples = len(self.samples)
        (
            self.baseline_accuracy,
            correct_indices,
            self.incorrect_indices,
            original_preds,
        ) = self._check_baseline_accuracy()

        # Store original samples
        original_samples = self.samples

        # Filter to only correctly classified samples, OR use model predictions as pseudo-labels
        if correct_indices:
            # Use only samples where model's prediction matches the label
            self.correct_samples = [self.samples[i] for i in correct_indices]
            self.samples = self.correct_samples
            excluded_count = total_samples - len(self.samples)
            if excluded_count > 0 and self.verbose:
                print(f"   Baseline accuracy: {self.baseline_accuracy:.1%}")
                print(f"   Testing on {len(self.samples)}/{total_samples} correctly-classified samples")
                print(f"   (Excluded {excluded_count} initially-incorrect samples)")
        elif self.baseline_accuracy == 0.0 and len(original_preds) > 0:
            # No label matches - use model's predictions as "ground truth" for robustness testing
            # This is useful when testing with mismatched label spaces (e.g., ImageNet model on CIFAR-10)
            self.correct_samples = [
                (img, original_preds[i]) for i, (img, _) in enumerate(self.samples)
            ]
            self.samples = self.correct_samples
            # Mark that we're using model predictions as labels
            self.baseline_accuracy = -1.0  # Special marker for "using model predictions"
            if self.verbose:
                print(f"   Mode: Prediction consistency (using model predictions as pseudo-labels)")
                print(f"   Testing all {len(self.samples)} samples")
        else:
            # No correctly classified samples - can't do meaningful robustness test
            self.correct_samples = []
            self.samples = []
            if self.verbose:
                print(f"   WARNING: No correctly-classified samples found!")
                print(f"   Baseline accuracy: {self.baseline_accuracy:.1%}")

        # Run search only if we have valid samples
        if len(self.samples) == 0 and self.baseline_accuracy != -1.0:
            result = {
                "failure_threshold": self.level_lo,
                "last_pass_level": self.level_lo,
                "robustness_score": 0.0,
                "queries": 0,
                "failures": [],
                "search_path": [],
                "unique_failed_samples": 0,
            }
        else:
            # Delegate to search strategy
            search_result = self.search_strategy.search(
                evaluate_fn=self._evaluate_at_level,
                level_lo=self.level_lo,
                level_hi=self.level_hi,
                max_queries=self.max_queries,
                progress_bar=progress_bar,
            )
            result = search_result.to_dict()

        # Restore original samples
        self.samples = original_samples

        # Add baseline metrics to result
        result["runtime"] = time.time() - start_time
        result["total_samples"] = total_samples

        if self.baseline_accuracy == -1.0:
            # Used model predictions as pseudo-labels
            result["baseline_accuracy"] = None  # N/A - using model predictions
            result["valid_samples"] = total_samples
            result["skipped_samples"] = 0
            result["using_model_predictions"] = True
        else:
            result["baseline_accuracy"] = self.baseline_accuracy
            result["valid_samples"] = len(correct_indices)
            result["skipped_samples"] = len(self.incorrect_indices)
            result["using_model_predictions"] = False

        return result

    def _evaluate_at_level(self, level: float) -> EvaluationResult:
        """
        Test if property holds at given perturbation level using batched processing.

        When normalization is configured, handles the workflow:
            normalized_input → denormalize → perturb → clamp → renormalize → model

        Args:
            level: Perturbation level to test

        Returns:
            EvaluationResult with passed, pass_rate, failures, and confidence metrics
        """
        # Instantiate strategy at this level
        strategy = self.strategy_factory(level)

        passed_count = 0
        failed_samples: List[Dict[str, Any]] = []
        n_samples = len(self.samples)

        # Track confidence across all batches
        all_orig_confs: List[float] = []
        all_pert_confs: List[float] = []

        # Process in batches
        for batch_start in range(0, n_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_samples)
            batch_samples = self.samples[batch_start:batch_end]

            # Stack images into batch tensor
            batch_imgs = torch.stack([img for img, _ in batch_samples])
            batch_labels = [label for _, label in batch_samples]

            # Get original predictions for batch
            with torch.no_grad():
                orig_out = self.model(batch_imgs)
                orig_preds = torch.argmax(orig_out, dim=-1)  # [batch_size]

            # Apply perturbation to batch
            # If normalization is set: denormalize → perturb → clamp → renormalize
            # Otherwise: apply perturbation directly
            if self.normalization is not None:
                perturbed_batch = self.normalization.denormalize_perturb_normalize(
                    images=batch_imgs,
                    perturbation_fn=strategy,
                    model=self.model,
                    level=None,  # Strategy already configured for this level
                    clamp=True,
                )
            else:
                # No normalization - apply perturbation directly
                perturbed_batch = strategy.generate(batch_imgs, self.model)

            # Get perturbed predictions for batch
            with torch.no_grad():
                pert_out = self.model(perturbed_batch)
                pert_preds = torch.argmax(pert_out, dim=-1)  # [batch_size]

            # Evaluate property for each sample in batch
            if self.property_fn is not None:
                property_result = self.property_fn(orig_out, pert_out)

                # Handle both single bool and per-sample tensor results
                if isinstance(property_result, bool):
                    batch_passed = [property_result] * len(batch_samples)
                elif isinstance(property_result, torch.Tensor):
                    batch_passed = property_result.tolist()
                else:
                    batch_passed = list(property_result)
            else:
                # Default: label constant (compare predictions)
                batch_passed = (orig_preds == pert_preds).tolist()

            # Compute confidence values
            # For original: use max probability (predicted class)
            # For perturbed: use probability of ORIGINAL correct class (to track degradation)
            orig_softmax = torch.nn.functional.softmax(orig_out, dim=-1)
            pert_softmax = torch.nn.functional.softmax(pert_out, dim=-1)

            orig_conf = orig_softmax.max(dim=-1).values  # Confidence in predicted class
            # Track confidence in ORIGINAL predicted class after perturbation
            pert_conf_correct_class = orig_softmax.gather(1, orig_preds.unsqueeze(1)).squeeze(1)
            pert_conf_after = pert_softmax.gather(1, orig_preds.unsqueeze(1)).squeeze(1)

            conf_drops = (pert_conf_correct_class - pert_conf_after).tolist()
            orig_confs = orig_conf.tolist()
            pert_confs = pert_conf_after.tolist()

            # Record results
            for i, sample_passed in enumerate(batch_passed):
                global_idx = batch_start + i
                if sample_passed:
                    passed_count += 1
                else:
                    failed_samples.append(
                        {
                            "index": global_idx,
                            "level": level,
                            "original_pred": int(orig_preds[i].item()),
                            "perturbed_pred": int(pert_preds[i].item()),
                            "original_label": batch_labels[i],
                            "original_confidence": orig_confs[i],
                            "perturbed_confidence": pert_confs[i],
                            "confidence_drop": conf_drops[i],
                        }
                    )

            # Accumulate confidence stats for this batch
            all_orig_confs.extend(orig_confs)
            all_pert_confs.extend(pert_confs)

        pass_rate = passed_count / n_samples if n_samples > 0 else 0.0
        passed = pass_rate >= self.pass_threshold

        # Compute average confidence metrics
        avg_orig_conf = sum(all_orig_confs) / len(all_orig_confs) if all_orig_confs else 0.0
        avg_pert_conf = sum(all_pert_confs) / len(all_pert_confs) if all_pert_confs else 0.0
        avg_conf_drop = avg_orig_conf - avg_pert_conf

        return EvaluationResult(
            passed=passed,
            pass_rate=pass_rate,
            failures=failed_samples,
            avg_orig_conf=avg_orig_conf,
            avg_pert_conf=avg_pert_conf,
            avg_conf_drop=avg_conf_drop,
        )
