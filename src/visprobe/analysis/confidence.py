"""
Confidence analysis and calibration tools.

This module provides tools for analyzing model confidence scores
and assessing calibration quality.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from .detailed_evaluation import SampleResult


@dataclass
class ConfidenceProfile:
    """Profile of model confidence behavior."""

    mean_confidence: float
    mean_confidence_correct: float
    mean_confidence_incorrect: float
    pct_high_confidence_errors: float
    pct_low_confidence_correct: float
    calibration_error: float
    overconfidence_score: float
    underconfidence_score: float
    confidence_histogram: Dict[str, List[float]]


def confidence_profile(
    samples: List[SampleResult],
    high_confidence_threshold: float = 0.8,
    low_confidence_threshold: float = 0.5,
    n_bins: int = 10
) -> ConfidenceProfile:
    """
    Analyze confidence distribution and behavior.

    Args:
        samples: List of sample results with confidence scores
        high_confidence_threshold: Threshold for high confidence
        low_confidence_threshold: Threshold for low confidence
        n_bins: Number of bins for histogram

    Returns:
        ConfidenceProfile with comprehensive confidence analysis

    Example:
        >>> profile = confidence_profile(results.samples)
        >>> if profile.pct_high_confidence_errors > 30:
        ...     print("Model is overconfident on errors!")
    """
    confidences = np.array([s.confidence for s in samples])
    correct_mask = np.array([s.correct for s in samples])

    # Separate confidences by correctness
    conf_correct = confidences[correct_mask]
    conf_incorrect = confidences[~correct_mask]

    # Basic statistics
    mean_confidence = float(confidences.mean())
    mean_conf_correct = float(conf_correct.mean()) if len(conf_correct) > 0 else 0.0
    mean_conf_incorrect = float(conf_incorrect.mean()) if len(conf_incorrect) > 0 else 0.0

    # High confidence errors
    if len(conf_incorrect) > 0:
        high_conf_errors = np.sum(conf_incorrect > high_confidence_threshold)
        pct_high_conf_errors = float(high_conf_errors / len(conf_incorrect) * 100)
    else:
        pct_high_conf_errors = 0.0

    # Low confidence correct
    if len(conf_correct) > 0:
        low_conf_correct = np.sum(conf_correct < low_confidence_threshold)
        pct_low_conf_correct = float(low_conf_correct / len(conf_correct) * 100)
    else:
        pct_low_conf_correct = 0.0

    # Calibration error
    cal_error = calibration_error(samples, n_bins=n_bins)

    # Over/underconfidence scores
    overconfidence = float(max(0, mean_conf_incorrect - 0.5))
    underconfidence = float(max(0, 0.5 - mean_conf_correct))

    # Confidence histogram
    bins = np.linspace(0, 1, n_bins + 1)
    hist_correct, _ = np.histogram(conf_correct, bins=bins)
    hist_incorrect, _ = np.histogram(conf_incorrect, bins=bins)

    confidence_histogram = {
        'bins': bins.tolist(),
        'correct': hist_correct.tolist(),
        'incorrect': hist_incorrect.tolist()
    }

    return ConfidenceProfile(
        mean_confidence=mean_confidence,
        mean_confidence_correct=mean_conf_correct,
        mean_confidence_incorrect=mean_conf_incorrect,
        pct_high_confidence_errors=pct_high_conf_errors,
        pct_low_confidence_correct=pct_low_conf_correct,
        calibration_error=cal_error,
        overconfidence_score=overconfidence,
        underconfidence_score=underconfidence,
        confidence_histogram=confidence_histogram
    )


def calibration_error(
    samples: List[SampleResult],
    n_bins: int = 10,
    method: str = "ECE"
) -> float:
    """
    Compute calibration error (ECE or MCE).

    Args:
        samples: List of sample results
        n_bins: Number of bins for calibration
        method: 'ECE' (Expected) or 'MCE' (Maximum)

    Returns:
        Calibration error value

    Example:
        >>> ece = calibration_error(results.samples, method="ECE")
        >>> print(f"Expected Calibration Error: {ece:.3f}")
    """
    confidences = np.array([s.confidence for s in samples])
    correct_mask = np.array([s.correct for s in samples])

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_errors = []
    bin_counts = []

    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

        if np.sum(in_bin) > 0:
            # Compute accuracy in bin
            bin_accuracy = correct_mask[in_bin].mean()

            # Compute average confidence in bin
            bin_confidence = confidences[in_bin].mean()

            # Calibration error for this bin
            bin_error = abs(bin_accuracy - bin_confidence)

            bin_errors.append(bin_error)
            bin_counts.append(np.sum(in_bin))

    if len(bin_errors) == 0:
        return 0.0

    bin_errors = np.array(bin_errors)
    bin_counts = np.array(bin_counts)

    if method == "ECE":
        # Expected Calibration Error (weighted average)
        weights = bin_counts / bin_counts.sum()
        return float(np.sum(weights * bin_errors))
    elif method == "MCE":
        # Maximum Calibration Error
        return float(np.max(bin_errors))
    else:
        raise ValueError(f"Unknown method: {method}")


def reliability_diagram(
    samples: List[SampleResult],
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for reliability diagram.

    Args:
        samples: List of sample results
        n_bins: Number of bins

    Returns:
        Tuple of (bin_centers, accuracies, confidences)

    Example:
        >>> centers, accs, confs = reliability_diagram(results.samples)
        >>> plt.plot(centers, accs, label='Actual accuracy')
        >>> plt.plot(centers, confs, label='Mean confidence')
    """
    confidences = np.array([s.confidence for s in samples])
    correct_mask = np.array([s.correct for s in samples])

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    accuracies = []
    mean_confidences = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

        if np.sum(in_bin) > 0:
            accuracies.append(correct_mask[in_bin].mean())
            mean_confidences.append(confidences[in_bin].mean())
        else:
            accuracies.append(np.nan)
            mean_confidences.append(np.nan)

    return bin_centers, np.array(accuracies), np.array(mean_confidences)


def confidence_ranking_quality(
    samples: List[SampleResult]
) -> float:
    """
    Measure how well confidence scores rank predictions.

    Higher is better (max 1.0).

    Args:
        samples: List of sample results

    Returns:
        Ranking quality score

    Example:
        >>> quality = confidence_ranking_quality(results.samples)
        >>> print(f"Confidence ranking quality: {quality:.2f}")
    """
    # Sort by confidence (descending)
    sorted_samples = sorted(samples, key=lambda s: s.confidence, reverse=True)

    # Calculate cumulative accuracy
    cumulative_correct = 0
    cumulative_accuracies = []

    for i, sample in enumerate(sorted_samples, 1):
        if sample.correct:
            cumulative_correct += 1
        cumulative_accuracies.append(cumulative_correct / i)

    # Area under cumulative accuracy curve
    auc = np.mean(cumulative_accuracies)

    # Normalize by best possible (all correct first)
    n_correct = sum(s.correct for s in samples)
    n_total = len(samples)

    if n_correct == 0 or n_correct == n_total:
        return 1.0

    best_auc = (n_correct + n_total) / (2 * n_total)

    return float(auc / best_auc)


def temperature_scaling_factor(
    samples: List[SampleResult],
    n_bins: int = 10
) -> float:
    """
    Estimate optimal temperature scaling factor for calibration.

    Args:
        samples: List of sample results
        n_bins: Number of bins for calibration

    Returns:
        Optimal temperature factor

    Example:
        >>> temp = temperature_scaling_factor(results.samples)
        >>> print(f"Optimal temperature: {temp:.2f}")
        >>> # Apply: calibrated_conf = conf ** (1/temp)
    """
    from scipy.optimize import minimize_scalar

    confidences = np.array([s.confidence for s in samples])
    correct_mask = np.array([s.correct for s in samples])

    def ece_with_temperature(temperature):
        # Apply temperature scaling
        if temperature <= 0:
            return float('inf')

        # Convert to logits and back
        eps = 1e-10
        logits = np.log(confidences + eps) - np.log(1 - confidences + eps)
        scaled_logits = logits / temperature
        scaled_confs = 1 / (1 + np.exp(-scaled_logits))

        # Create temp samples for ECE calculation
        temp_samples = []
        for i, sample in enumerate(samples):
            temp_sample = SampleResult(
                index=sample.index,
                label=sample.label,
                prediction=sample.prediction,
                correct=sample.correct,
                confidence=float(scaled_confs[i])
            )
            temp_samples.append(temp_sample)

        return calibration_error(temp_samples, n_bins=n_bins)

    # Find optimal temperature
    result = minimize_scalar(ece_with_temperature, bounds=(0.1, 10.0), method='bounded')

    return float(result.x)