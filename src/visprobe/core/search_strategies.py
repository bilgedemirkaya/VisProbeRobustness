"""
Search strategies for finding failure thresholds.

This module provides pluggable search algorithms for the SearchEngine.
Each strategy implements a different approach to finding the minimal
perturbation level that causes model failures.

Available strategies:
- AdaptiveSearchStrategy: Step-halving with bidirectional exploration (default)
- BinarySearchStrategy: Classic binary search over the range
- BayesianSearchStrategy: Gaussian Process-based Bayesian optimization

Usage:
    # Using string mode (backward compatible)
    search(model, data, strategy, mode="adaptive")

    # Using strategy instance (more control)
    from visprobe.core.search_strategies import BayesianSearchStrategy
    search(model, data, strategy,
           search_method=BayesianSearchStrategy(n_initial=10, acquisition="ei"))
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

# Optional imports for Bayesian mode
try:
    from scipy.stats import norm as scipy_norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvaluationResult:
    """
    Result from evaluating model at a perturbation level.

    Returned by SearchEngine._evaluate_at_level() and consumed by strategies.
    """

    passed: bool
    """Whether pass_rate >= pass_threshold"""

    pass_rate: float
    """Fraction of samples that passed (0.0 to 1.0)"""

    failures: List[Dict[str, Any]]
    """List of failure cases with details (index, predictions, confidences)"""

    avg_orig_conf: float
    """Average confidence on original (unperturbed) samples"""

    avg_pert_conf: float
    """Average confidence on perturbed samples"""

    avg_conf_drop: float
    """Average confidence drop (orig - perturbed)"""


@dataclass
class SearchResult:
    """
    Result from a search strategy.

    Contains the failure threshold, robustness score, and search history.
    """

    failure_threshold: float
    """Lowest perturbation level where model fails"""

    last_pass_level: float
    """Highest perturbation level where model still passes"""

    robustness_score: float
    """Normalized score [0, 1] based on passing range"""

    queries: int
    """Number of evaluation queries used"""

    failures: List[Dict[str, Any]]
    """Failures at the threshold level"""

    search_path: List[Dict[str, Any]]
    """History of all evaluation steps"""

    unique_failed_samples: int = 0
    """Total unique samples that failed across all levels"""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Strategy-specific extra fields (e.g., Bayesian CI)"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for backward compatibility."""
        result = {
            "failure_threshold": self.failure_threshold,
            "last_pass_level": self.last_pass_level,
            "robustness_score": self.robustness_score,
            "queries": self.queries,
            "failures": self.failures,
            "search_path": self.search_path,
            "unique_failed_samples": self.unique_failed_samples,
        }
        result.update(self.extra)
        return result


# =============================================================================
# Base Strategy Class
# =============================================================================


class SearchStrategy(ABC):
    """
    Abstract base class for search strategies.

    Subclasses implement the search() method with their specific algorithm.
    Common utilities are provided in the base class.
    """

    @abstractmethod
    def search(
        self,
        evaluate_fn: Callable[[float], EvaluationResult],
        level_lo: float,
        level_hi: float,
        max_queries: int,
        progress_bar: Optional[Any] = None,
    ) -> SearchResult:
        """
        Run the search algorithm to find failure threshold.

        Args:
            evaluate_fn: Function that evaluates model at a given level
            level_lo: Lower bound of search range
            level_hi: Upper bound of search range
            max_queries: Maximum number of evaluations allowed
            progress_bar: Optional tqdm progress bar

        Returns:
            SearchResult with threshold, score, and history
        """
        pass

    def _compute_robustness_score(
        self, last_pass_level: float, level_lo: float, level_hi: float
    ) -> float:
        """Compute normalized robustness score [0, 1]."""
        if level_hi > level_lo:
            score = (last_pass_level - level_lo) / (level_hi - level_lo)
        else:
            score = 1.0
        return min(max(score, 0.0), 1.0)

    def _record_step(
        self,
        search_path: List[Dict],
        iteration: int,
        level: float,
        result: EvaluationResult,
        progress_bar: Optional[Any] = None,
        **extra_fields,
    ) -> None:
        """Record an evaluation step to search_path."""
        search_path.append(
            {
                "iteration": iteration,
                "level": level,
                "passed": result.passed,
                "pass_rate": result.pass_rate,
                "avg_confidence": result.avg_pert_conf,
                "confidence_drop": result.avg_conf_drop,
                **extra_fields,
            }
        )
        if progress_bar:
            progress_bar.update(1)
            progress_bar.set_postfix(
                level=f"{level:.3f}", pass_rate=f"{result.pass_rate:.2%}"
            )


# =============================================================================
# Adaptive Search Strategy
# =============================================================================


class AdaptiveSearchStrategy(SearchStrategy):
    """
    Step-halving search with bidirectional exploration.

    Algorithm:
    1. Start at initial_level (typically the identity/no-perturbation point)
    2. Search in both directions from initial_level
    3. When failure found, halve step size and refine
    4. Stop when step < min_step or queries >= max_queries

    Best for: Unknown failure range, multiplicative perturbations (brightness, contrast)
    """

    def __init__(
        self,
        initial_level: Optional[float] = None,
        step: Optional[float] = None,
        min_step: float = 0.001,
        pass_threshold: float = 0.9,
    ):
        """
        Args:
            initial_level: Starting point (defaults to level_lo)
            step: Initial step size (defaults to range/10)
            min_step: Minimum step size before stopping
            pass_threshold: Fraction of samples that must pass (default 90%)
        """
        self.initial_level = initial_level
        self.step = step
        self.min_step = min_step
        self.pass_threshold = pass_threshold

    def search(
        self,
        evaluate_fn: Callable[[float], EvaluationResult],
        level_lo: float,
        level_hi: float,
        max_queries: int,
        progress_bar: Optional[Any] = None,
    ) -> SearchResult:
        """Run adaptive step-halving search."""
        # Initialize parameters
        initial = (
            self.initial_level if self.initial_level is not None else level_lo
        )
        step_size = self.step if self.step is not None else (level_hi - level_lo) / 10.0

        queries = 0
        search_path: List[Dict[str, Any]] = []
        failed_sample_info: Dict[int, Dict[str, Any]] = {}
        threshold_failures: List[Dict[str, Any]] = []

        # Determine search directions needed
        # Search upward if initial < level_hi, downward if initial > level_lo
        search_upward = initial < level_hi
        search_downward = initial > level_lo
        bidirectional = search_upward and search_downward

        # Track thresholds in both directions
        upper_fail_level = level_hi
        upper_last_pass = initial
        lower_fail_level = level_lo
        lower_last_pass = initial

        def evaluate_and_record(level: float) -> Tuple[bool, float, List[Dict]]:
            nonlocal queries
            queries += 1
            result = evaluate_fn(level)
            self._record_step(search_path, queries, level, result, progress_bar)
            return result.passed, result.pass_rate, result.failures

        # Test at initial level
        passed, _, level_failures = evaluate_and_record(initial)
        if not passed:
            # Initial level fails - return immediately
            for f in level_failures:
                idx = f["index"]
                if idx not in failed_sample_info:
                    failed_sample_info[idx] = f
            threshold_failures = level_failures
            return SearchResult(
                failure_threshold=initial,
                last_pass_level=level_lo,
                robustness_score=0.0,
                queries=queries,
                failures=sorted(threshold_failures, key=lambda x: x["index"]),
                search_path=search_path,
                unique_failed_samples=len(failed_sample_info),
            )

        # Search upward (increasing level)
        current_up = initial + step_size
        while (
            queries < max_queries
            and step_size >= self.min_step
            and current_up <= level_hi
        ):
            passed, _, level_failures = evaluate_and_record(min(current_up, level_hi))

            if passed:
                upper_last_pass = min(current_up, level_hi)
                current_up += step_size
            else:
                if current_up < upper_fail_level:
                    upper_fail_level = current_up
                    threshold_failures = level_failures
                for f in level_failures:
                    idx = f["index"]
                    if idx not in failed_sample_info:
                        failed_sample_info[idx] = f
                # Refine: step back and halve step size
                current_up -= step_size / 2.0
                step_size /= 2.0

            if current_up > level_hi:
                break

        # Search downward if initial > level_lo (includes bidirectional and downward-only cases)
        if search_downward and queries < max_queries:
            step_size = self.step if self.step is not None else (level_hi - level_lo) / 10.0
            current_down = initial - step_size

            while (
                queries < max_queries
                and step_size >= self.min_step
                and current_down >= level_lo
            ):
                passed, _, level_failures = evaluate_and_record(
                    max(current_down, level_lo)
                )

                if passed:
                    lower_last_pass = max(current_down, level_lo)
                    current_down -= step_size
                else:
                    if current_down > lower_fail_level:
                        lower_fail_level = current_down
                        # Update threshold if closer to identity
                        if upper_fail_level == level_hi or abs(
                            current_down - initial
                        ) < abs(upper_fail_level - initial):
                            threshold_failures = level_failures
                    for f in level_failures:
                        idx = f["index"]
                        if idx not in failed_sample_info:
                            failed_sample_info[idx] = f
                    current_down += step_size / 2.0
                    step_size /= 2.0

                if current_down < level_lo:
                    break

        # Calculate robustness score based on search direction(s)
        total_range = level_hi - level_lo

        if bidirectional:
            # Both directions: find closest failure to identity
            pass_range = upper_last_pass - lower_last_pass
            score = pass_range / total_range if total_range > 0 else 1.0

            fail_dist_up = (
                upper_fail_level - initial
                if upper_fail_level < level_hi
                else float("inf")
            )
            fail_dist_down = (
                initial - lower_fail_level
                if lower_fail_level > level_lo
                else float("inf")
            )

            if fail_dist_up <= fail_dist_down:
                first_fail_level = upper_fail_level
                last_pass_level = upper_last_pass
            else:
                first_fail_level = lower_fail_level
                last_pass_level = lower_last_pass
        elif search_downward and not search_upward:
            # Downward-only (e.g., JPEG where initial == level_hi)
            # For JPEG: quality goes from 100 (identity) down toward 20 (max perturbation)
            # Score = distance from identity to first failure, normalized
            # Failure closer to level_lo (max perturbation) means more robust
            first_fail_level = lower_fail_level
            last_pass_level = lower_last_pass
            # Score: fraction of range that passes (from identity toward max perturbation)
            pass_range = initial - lower_fail_level
            score = pass_range / total_range if total_range > 0 else 1.0
        else:
            # Upward-only (default case)
            score = self._compute_robustness_score(upper_last_pass, level_lo, level_hi)
            first_fail_level = upper_fail_level
            last_pass_level = upper_last_pass

        return SearchResult(
            failure_threshold=first_fail_level,
            last_pass_level=last_pass_level,
            robustness_score=min(max(score, 0.0), 1.0),
            queries=queries,
            failures=sorted(threshold_failures, key=lambda x: x["index"]),
            search_path=search_path,
            unique_failed_samples=len(failed_sample_info),
        )


# =============================================================================
# Binary Search Strategy
# =============================================================================


class BinarySearchStrategy(SearchStrategy):
    """
    Classic binary search over perturbation range.

    Algorithm:
    1. Test midpoint = (lo + hi) / 2
    2. If fails: search in [lo, midpoint]
    3. If passes: search in [midpoint, hi]
    4. Stop when hi - lo < tolerance or queries >= max_queries

    Best for: Known search range, additive perturbations (noise, blur)
    """

    def __init__(
        self,
        tolerance: float = 0.001,
        pass_threshold: float = 0.9,
    ):
        """
        Args:
            tolerance: Minimum range width before stopping
            pass_threshold: Fraction of samples that must pass (default 90%)
        """
        self.tolerance = tolerance
        self.pass_threshold = pass_threshold

    def search(
        self,
        evaluate_fn: Callable[[float], EvaluationResult],
        level_lo: float,
        level_hi: float,
        max_queries: int,
        progress_bar: Optional[Any] = None,
    ) -> SearchResult:
        """Run binary search."""
        lo = level_lo
        hi = level_hi
        queries = 0
        search_path: List[Dict[str, Any]] = []

        last_pass_level = lo
        first_fail_level = hi
        threshold_failures: List[Dict[str, Any]] = []

        while queries < max_queries and (hi - lo) >= self.tolerance:
            queries += 1
            midpoint = (lo + hi) / 2.0

            result = evaluate_fn(midpoint)
            self._record_step(
                search_path,
                queries,
                midpoint,
                result,
                progress_bar,
                lo=lo,
                hi=hi,
            )

            if result.passed:
                last_pass_level = midpoint
                lo = midpoint
            else:
                if midpoint < first_fail_level:
                    first_fail_level = midpoint
                    threshold_failures = result.failures
                hi = midpoint

        score = self._compute_robustness_score(last_pass_level, level_lo, level_hi)

        return SearchResult(
            failure_threshold=first_fail_level,
            last_pass_level=last_pass_level,
            robustness_score=score,
            queries=queries,
            failures=sorted(threshold_failures, key=lambda x: x["index"]),
            search_path=search_path,
        )


# =============================================================================
# Bayesian Search Strategy
# =============================================================================


class BayesianSearchStrategy(SearchStrategy):
    """
    Gaussian Process-based Bayesian optimization.

    Algorithm:
    1. Sample n_initial points uniformly
    2. Fit GP to model: level → pass_rate
    3. Use acquisition function to pick next level
    4. Repeat until budget exhausted
    5. Extract threshold + confidence from GP posterior

    Best for: Expensive evaluations, need confidence intervals
    Requires: scikit-learn and scipy
    """

    def __init__(
        self,
        n_initial: int = 5,
        acquisition: str = "ucb",
        beta: float = 2.0,
        confidence_level: float = 0.95,
        pass_threshold: float = 0.9,
    ):
        """
        Args:
            n_initial: Number of initial random samples
            acquisition: Acquisition function - "ucb" or "ei"
            beta: UCB exploration parameter
            confidence_level: Confidence level for CI (default 0.95)
            pass_threshold: Fraction of samples that must pass (default 90%)
        """
        if not BAYESIAN_AVAILABLE:
            raise ImportError(
                "BayesianSearchStrategy requires scikit-learn and scipy. "
                "Install with: pip install scikit-learn scipy"
            )
        if acquisition not in {"ucb", "ei"}:
            raise ValueError(f"Unknown acquisition: {acquisition}. Must be 'ucb' or 'ei'")

        self.n_initial = n_initial
        self.acquisition = acquisition
        self.beta = beta
        self.confidence_level = confidence_level
        self.pass_threshold = pass_threshold

    def search(
        self,
        evaluate_fn: Callable[[float], EvaluationResult],
        level_lo: float,
        level_hi: float,
        max_queries: int,
        progress_bar: Optional[Any] = None,
    ) -> SearchResult:
        """Run Bayesian optimization search."""
        levels_tested: List[float] = []
        pass_rates: List[float] = []
        all_failures: List[Dict[str, Any]] = []
        search_path: List[Dict[str, Any]] = []
        queries = 0

        # Phase 1: Initial sampling
        initial_levels = np.linspace(level_lo, level_hi, self.n_initial).tolist()

        for level in initial_levels:
            if queries >= max_queries:
                break

            queries += 1
            result = evaluate_fn(level)
            levels_tested.append(level)
            pass_rates.append(result.pass_rate)
            if not result.passed:
                all_failures.extend(result.failures)

            self._record_step(
                search_path, queries, level, result, progress_bar, phase="initial"
            )

        # Phase 2: Bayesian optimization
        while queries < max_queries:
            gp = self._fit_gp(levels_tested, pass_rates, level_lo, level_hi)

            # Generate candidates
            candidates = np.linspace(level_lo, level_hi, 100).reshape(-1, 1)
            mu, sigma = gp.predict(candidates, return_std=True)

            # Compute acquisition function
            if self.acquisition == "ucb":
                scores = self._acquisition_ucb(mu, sigma)
            else:
                scores = self._acquisition_ei(mu, sigma)

            # Select next level
            next_idx = np.argmax(scores)
            next_level = float(candidates[next_idx, 0])

            # Avoid re-evaluating close points
            min_distance = (level_hi - level_lo) / 1000
            for tested in levels_tested:
                if abs(next_level - tested) < min_distance:
                    next_level = float(
                        np.clip(
                            next_level + np.random.uniform(-min_distance, min_distance),
                            level_lo,
                            level_hi,
                        )
                    )
                    break

            queries += 1
            result = evaluate_fn(next_level)
            levels_tested.append(next_level)
            pass_rates.append(result.pass_rate)
            if not result.passed:
                all_failures.extend(result.failures)

            self._record_step(
                search_path,
                queries,
                next_level,
                result,
                progress_bar,
                phase="optimization",
                acquisition_score=float(scores[next_idx]),
            )

        # Phase 3: Extract threshold from final GP
        gp = self._fit_gp(levels_tested, pass_rates, level_lo, level_hi)
        fine_grid = np.linspace(level_lo, level_hi, 1000).reshape(-1, 1)
        mu_fine, sigma_fine = gp.predict(fine_grid, return_std=True)

        # Find threshold
        threshold_idx = np.where(mu_fine < self.pass_threshold)[0]
        if len(threshold_idx) > 0:
            failure_threshold = float(fine_grid[threshold_idx[0], 0])
            threshold_std = float(sigma_fine[threshold_idx[0]])
        else:
            failure_threshold = float(level_hi)
            threshold_std = float(sigma_fine[-1])

        # Find last pass level
        pass_idx = np.where(mu_fine >= self.pass_threshold)[0]
        if len(pass_idx) > 0:
            last_pass_level = float(fine_grid[pass_idx[-1], 0])
        else:
            last_pass_level = float(level_lo)

        # Compute confidence interval
        z = scipy_norm.ppf((1 + self.confidence_level) / 2)
        ci_lower = max(failure_threshold - z * threshold_std, level_lo)
        ci_upper = min(failure_threshold + z * threshold_std, level_hi)

        score = self._compute_robustness_score(last_pass_level, level_lo, level_hi)

        return SearchResult(
            failure_threshold=failure_threshold,
            last_pass_level=last_pass_level,
            robustness_score=score,
            queries=queries,
            failures=all_failures[:10],  # Limit for readability
            search_path=search_path,
            extra={
                "confidence_interval": (ci_lower, ci_upper),
                "threshold_std": threshold_std,
                "confidence_level": self.confidence_level,
                "gp_levels": fine_grid.flatten().tolist(),
                "gp_mean": mu_fine.tolist(),
                "gp_std": sigma_fine.tolist(),
                "observed_levels": levels_tested,
                "observed_pass_rates": pass_rates,
            },
        )

    def _fit_gp(
        self,
        levels: List[float],
        pass_rates: List[float],
        level_lo: float,
        level_hi: float,
    ) -> "GaussianProcessRegressor":
        """Fit Gaussian Process to observations."""
        X = np.array(levels).reshape(-1, 1)
        y = np.array(pass_rates)

        range_size = level_hi - level_lo
        length_scale = range_size / 5.0

        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(0.01, 10.0))
            * RBF(length_scale=length_scale, length_scale_bounds=(range_size / 100, range_size))
            + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 0.5))
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=5,
            normalize_y=True,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, y)

        return gp

    def _acquisition_ucb(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Upper Confidence Bound acquisition for threshold finding."""
        distance_from_threshold = np.abs(mu - self.pass_threshold)
        threshold_proximity = 1.0 / (distance_from_threshold + 0.01)
        return threshold_proximity * (1.0 + self.beta * sigma)

    def _acquisition_ei(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Expected Improvement acquisition for threshold finding."""
        target = self.pass_threshold
        scores = np.zeros_like(mu)

        for i, (m, s) in enumerate(zip(mu, sigma)):
            if s < 1e-10:
                scores[i] = 0.0
                continue

            current_dist = abs(m - target)
            z_upper = (target + current_dist - m) / s
            z_lower = (target - current_dist - m) / s
            prob_improvement = scipy_norm.cdf(z_upper) - scipy_norm.cdf(z_lower)
            scores[i] = prob_improvement * s

        return scores


# =============================================================================
# Factory Function
# =============================================================================


def create_search_strategy(
    mode: str,
    initial_level: Optional[float] = None,
    step: Optional[float] = None,
    min_step: float = 0.001,
    pass_threshold: float = 0.9,
    n_initial: int = 5,
    acquisition: str = "ucb",
    beta: float = 2.0,
    confidence_level: float = 0.95,
) -> SearchStrategy:
    """
    Create a search strategy from mode string.

    Args:
        mode: "adaptive", "binary", or "bayesian"
        **kwargs: Strategy-specific parameters

    Returns:
        Configured SearchStrategy instance
    """
    if mode == "adaptive":
        return AdaptiveSearchStrategy(
            initial_level=initial_level,
            step=step,
            min_step=min_step,
            pass_threshold=pass_threshold,
        )
    elif mode == "binary":
        return BinarySearchStrategy(
            tolerance=min_step,
            pass_threshold=pass_threshold,
        )
    elif mode == "bayesian":
        return BayesianSearchStrategy(
            n_initial=n_initial,
            acquisition=acquisition,
            beta=beta,
            confidence_level=confidence_level,
            pass_threshold=pass_threshold,
        )
    else:
        raise ValueError(
            f"Unknown search mode: {mode}. Must be 'adaptive', 'binary', or 'bayesian'"
        )
