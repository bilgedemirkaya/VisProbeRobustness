"""
Statistical analysis tools including bootstrap confidence intervals.

This module provides rigorous statistical methods for comparing models
and establishing confidence in results.
"""

from typing import Tuple, Optional, Union
import numpy as np
from scipy import stats


def bootstrap_accuracy(
    correct_mask: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for accuracy.

    Args:
        correct_mask: Boolean array of correct predictions
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 95%)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (mean_accuracy, lower_bound, upper_bound)

    Example:
        >>> acc, lower, upper = bootstrap_accuracy(results.correct_mask)
        >>> print(f"Accuracy: {acc:.1%} (95% CI: {lower:.1%} to {upper:.1%})")
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(correct_mask)
    bootstrap_accuracies = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = correct_mask[indices]
        bootstrap_accuracies.append(bootstrap_sample.mean())

    bootstrap_accuracies = np.array(bootstrap_accuracies)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    mean_accuracy = correct_mask.mean()
    lower_bound = np.percentile(bootstrap_accuracies, lower_percentile)
    upper_bound = np.percentile(bootstrap_accuracies, upper_percentile)

    return mean_accuracy, lower_bound, upper_bound


def bootstrap_delta(
    correct_mask_a: np.ndarray,
    correct_mask_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    paired: bool = True,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for difference between two models.

    Args:
        correct_mask_a: Boolean array of correct predictions for model A
        correct_mask_b: Boolean array of correct predictions for model B
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 95%)
        paired: Whether samples are paired (same test set)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (mean_delta, lower_bound, upper_bound)

    Example:
        >>> delta, lower, upper = bootstrap_delta(model_a.correct, model_b.correct)
        >>> if lower > 0:
        ...     print("Model A is significantly better")
    """
    if random_state is not None:
        np.random.seed(random_state)

    if paired and len(correct_mask_a) != len(correct_mask_b):
        raise ValueError("Paired comparison requires equal length arrays")

    n_samples = len(correct_mask_a)
    bootstrap_deltas = []

    for _ in range(n_bootstrap):
        if paired:
            # Paired bootstrap - resample pairs
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample_a = correct_mask_a[indices]
            sample_b = correct_mask_b[indices]
        else:
            # Unpaired bootstrap - resample independently
            indices_a = np.random.choice(len(correct_mask_a), size=len(correct_mask_a), replace=True)
            indices_b = np.random.choice(len(correct_mask_b), size=len(correct_mask_b), replace=True)
            sample_a = correct_mask_a[indices_a]
            sample_b = correct_mask_b[indices_b]

        delta = sample_a.mean() - sample_b.mean()
        bootstrap_deltas.append(delta)

    bootstrap_deltas = np.array(bootstrap_deltas)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    mean_delta = correct_mask_a.mean() - correct_mask_b.mean()
    lower_bound = np.percentile(bootstrap_deltas, lower_percentile)
    upper_bound = np.percentile(bootstrap_deltas, upper_percentile)

    return mean_delta, lower_bound, upper_bound


def bootstrap_confidence_interval(
    values: np.ndarray,
    statistic_fn: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    General bootstrap confidence interval for any statistic.

    Args:
        values: Array of values to bootstrap
        statistic_fn: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 95%)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (statistic, lower_bound, upper_bound)

    Example:
        >>> median, lower, upper = bootstrap_confidence_interval(
        ...     confidences, statistic_fn=np.median
        ... )
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(values)
    bootstrap_statistics = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = values[indices]
        bootstrap_statistics.append(statistic_fn(bootstrap_sample))

    bootstrap_statistics = np.array(bootstrap_statistics)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    statistic = statistic_fn(values)
    lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
    upper_bound = np.percentile(bootstrap_statistics, upper_percentile)

    return statistic, lower_bound, upper_bound


def mcnemar_test(
    correct_mask_a: np.ndarray,
    correct_mask_b: np.ndarray
) -> Tuple[float, float]:
    """
    McNemar's test for paired model comparison.

    Tests if two models have significantly different error rates on the same data.

    Args:
        correct_mask_a: Boolean array of correct predictions for model A
        correct_mask_b: Boolean array of correct predictions for model B

    Returns:
        Tuple of (statistic, p_value)

    Example:
        >>> stat, p_value = mcnemar_test(model_a.correct, model_b.correct)
        >>> if p_value < 0.05:
        ...     print("Models are significantly different")
    """
    if len(correct_mask_a) != len(correct_mask_b):
        raise ValueError("McNemar test requires equal length arrays")

    # Build contingency table
    n00 = np.sum((~correct_mask_a) & (~correct_mask_b))  # Both wrong
    n01 = np.sum((~correct_mask_a) & correct_mask_b)    # A wrong, B right
    n10 = np.sum(correct_mask_a & (~correct_mask_b))    # A right, B wrong
    n11 = np.sum(correct_mask_a & correct_mask_b)       # Both right

    # McNemar statistic
    if n01 + n10 == 0:
        # No disagreements
        return 0.0, 1.0

    # Use continuity correction for small samples
    if n01 + n10 < 25:
        statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    else:
        statistic = (n01 - n10) ** 2 / (n01 + n10)

    # Chi-square test with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return statistic, p_value


def permutation_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    statistic_fn: callable = lambda a, b: a.mean() - b.mean(),
    n_permutations: int = 10000,
    random_state: Optional[int] = None
) -> float:
    """
    Permutation test for comparing two groups.

    Args:
        values_a: Values from group A
        values_b: Values from group B
        statistic_fn: Function to compute test statistic
        n_permutations: Number of permutations
        random_state: Random seed

    Returns:
        p-value for the test

    Example:
        >>> p_value = permutation_test(model_a_scores, model_b_scores)
        >>> if p_value < 0.05:
        ...     print("Significant difference detected")
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Observed statistic
    observed_stat = statistic_fn(values_a, values_b)

    # Combine all values
    combined = np.concatenate([values_a, values_b])
    n_a = len(values_a)

    # Permutation test
    permuted_stats = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        permuted_stats.append(statistic_fn(perm_a, perm_b))

    permuted_stats = np.array(permuted_stats)

    # Two-tailed p-value
    p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))

    return p_value