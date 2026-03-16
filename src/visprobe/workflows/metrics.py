"""
Metrics for robustness evaluation.

Provides utilities for computing robustness metrics from evaluation results.
"""

from typing import List, Union, Sequence
import numpy as np


def compute_auc(
    severities: Union[List[float], np.ndarray],
    accuracies: Union[List[float], np.ndarray],
) -> float:
    """
    Compute Area Under Curve (AUC) for robustness curve.

    Uses trapezoidal integration to compute the area under the
    accuracy vs. severity curve. Higher AUC indicates better robustness.

    Args:
        severities: Severity levels (x-axis)
        accuracies: Accuracy values (y-axis)

    Returns:
        Normalized AUC value (divided by severity range)

    Example:
        >>> severities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        >>> accuracies = [0.95, 0.92, 0.88, 0.82, 0.75, 0.68]
        >>> auc = compute_auc(severities, accuracies)
        >>> print(f"AUC: {auc:.3f}")
        AUC: 0.833
    """
    severities = np.asarray(severities)
    accuracies = np.asarray(accuracies)

    if len(severities) != len(accuracies):
        raise ValueError(
            f"Severities and accuracies must have same length: "
            f"{len(severities)} vs {len(accuracies)}"
        )

    if len(severities) < 2:
        raise ValueError("Need at least 2 points to compute AUC")

    severity_range = severities[-1] - severities[0]
    if severity_range <= 0:
        raise ValueError("Severity range must be positive")

    # Trapezoidal integration
    auc_raw = float(np.trapz(accuracies, severities))

    # Normalize by severity range
    auc_normalized = auc_raw / severity_range

    return auc_normalized


def compute_robustness_curve(
    results: Sequence,
    metric: str = "accuracy",
) -> np.ndarray:
    """
    Extract robustness curve from evaluation results.

    Args:
        results: List of EvaluationResult objects
        metric: Metric to extract ('accuracy', 'mean_confidence', 'mean_loss')

    Returns:
        Array of metric values

    Example:
        >>> results = [eval1, eval2, eval3]  # From severity sweep
        >>> curve = compute_robustness_curve(results, metric='accuracy')
        >>> print(curve)
        [0.95 0.88 0.82]
    """
    valid_metrics = ["accuracy", "mean_confidence", "mean_loss"]
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}, got {metric}")

    return np.array([getattr(r, metric) for r in results])


def compute_relative_robustness(
    baseline_results: Sequence,
    test_results: Sequence,
) -> float:
    """
    Compute relative robustness improvement.

    Compares AUC of test model vs baseline model.

    Args:
        baseline_results: Results from baseline model
        test_results: Results from test model

    Returns:
        Relative improvement ratio (>1 means test is more robust)

    Example:
        >>> baseline_auc = compute_auc(severities, baseline_acc)
        >>> test_auc = compute_auc(severities, test_acc)
        >>> improvement = test_auc / baseline_auc
        >>> print(f"Test model is {improvement:.2f}x more robust")
    """
    baseline_acc = compute_robustness_curve(baseline_results, "accuracy")
    test_acc = compute_robustness_curve(test_results, "accuracy")

    # Assume same severity levels
    severities = np.arange(len(baseline_acc))

    baseline_auc = compute_auc(severities, baseline_acc)
    test_auc = compute_auc(severities, test_acc)

    if baseline_auc <= 0:
        raise ValueError("Baseline AUC must be positive")

    return test_auc / baseline_auc
