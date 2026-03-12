"""
Crossover detection for model performance curves.

This module identifies points where different models exchange
relative performance under varying conditions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy import interpolate


@dataclass
class CrossoverPoint:
    """Represents a performance crossover between two models."""

    severity: float
    performance_value: float
    model_a_name: str
    model_b_name: str
    confidence: Optional[float] = None
    interpolated: bool = False


def find_crossover(
    severities: np.ndarray,
    performance_a: np.ndarray,
    performance_b: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    interpolate_points: bool = True
) -> Optional[CrossoverPoint]:
    """
    Find the first crossover point between two performance curves.

    Args:
        severities: Array of severity levels
        performance_a: Performance values for model A
        performance_b: Performance values for model B
        model_a_name: Name of model A
        model_b_name: Name of model B
        interpolate_points: Whether to interpolate for precise crossover

    Returns:
        CrossoverPoint if found, None otherwise

    Example:
        >>> crossover = find_crossover(noise_levels, acc_vanilla, acc_robust)
        >>> if crossover:
        ...     print(f"Models cross at severity {crossover.severity:.2f}")
    """
    if len(severities) != len(performance_a) or len(severities) != len(performance_b):
        raise ValueError("All arrays must have the same length")

    # Calculate differences
    differences = performance_a - performance_b

    # Find sign changes
    sign_changes = np.where(np.diff(np.sign(differences)))[0]

    if len(sign_changes) == 0:
        return None

    # Get first crossover
    idx = sign_changes[0]

    if interpolate_points and idx < len(severities) - 1:
        # Linear interpolation for more precise crossover point
        x1, x2 = severities[idx], severities[idx + 1]
        y1_a, y2_a = performance_a[idx], performance_a[idx + 1]
        y1_b, y2_b = performance_b[idx], performance_b[idx + 1]

        # Find intersection point
        # Solve: a1 * x + b1 = a2 * x + b2
        slope_a = (y2_a - y1_a) / (x2 - x1)
        slope_b = (y2_b - y1_b) / (x2 - x1)
        intercept_a = y1_a - slope_a * x1
        intercept_b = y1_b - slope_b * x1

        if slope_a != slope_b:
            crossover_severity = (intercept_b - intercept_a) / (slope_a - slope_b)
            crossover_performance = slope_a * crossover_severity + intercept_a

            return CrossoverPoint(
                severity=float(crossover_severity),
                performance_value=float(crossover_performance),
                model_a_name=model_a_name,
                model_b_name=model_b_name,
                interpolated=True
            )

    # Return discrete crossover point
    return CrossoverPoint(
        severity=float(severities[idx]),
        performance_value=float((performance_a[idx] + performance_b[idx]) / 2),
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        interpolated=False
    )


def find_all_crossovers(
    severities: np.ndarray,
    performance_a: np.ndarray,
    performance_b: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    min_separation: float = 0.1
) -> List[CrossoverPoint]:
    """
    Find all crossover points between two performance curves.

    Args:
        severities: Array of severity levels
        performance_a: Performance values for model A
        performance_b: Performance values for model B
        model_a_name: Name of model A
        model_b_name: Name of model B
        min_separation: Minimum severity separation between crossovers

    Returns:
        List of all CrossoverPoints found

    Example:
        >>> crossovers = find_all_crossovers(severities, perf_a, perf_b)
        >>> print(f"Found {len(crossovers)} crossover points")
    """
    if len(severities) != len(performance_a) or len(severities) != len(performance_b):
        raise ValueError("All arrays must have the same length")

    # Calculate differences
    differences = performance_a - performance_b

    # Find all sign changes
    sign_changes = np.where(np.diff(np.sign(differences)))[0]

    crossovers = []
    last_severity = -np.inf

    for idx in sign_changes:
        if idx < len(severities) - 1:
            # Linear interpolation
            x1, x2 = severities[idx], severities[idx + 1]

            # Check minimum separation
            if x1 - last_severity < min_separation:
                continue

            y1_a, y2_a = performance_a[idx], performance_a[idx + 1]
            y1_b, y2_b = performance_b[idx], performance_b[idx + 1]

            # Find intersection point
            slope_a = (y2_a - y1_a) / (x2 - x1)
            slope_b = (y2_b - y1_b) / (x2 - x1)
            intercept_a = y1_a - slope_a * x1
            intercept_b = y1_b - slope_b * x1

            if slope_a != slope_b:
                crossover_severity = (intercept_b - intercept_a) / (slope_a - slope_b)
                crossover_performance = slope_a * crossover_severity + intercept_a

                crossovers.append(CrossoverPoint(
                    severity=float(crossover_severity),
                    performance_value=float(crossover_performance),
                    model_a_name=model_a_name,
                    model_b_name=model_b_name,
                    interpolated=True
                ))

                last_severity = crossover_severity

    return crossovers


def find_dominance_regions(
    severities: np.ndarray,
    performance_a: np.ndarray,
    performance_b: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B"
) -> List[Tuple[float, float, str]]:
    """
    Find regions where each model dominates.

    Args:
        severities: Array of severity levels
        performance_a: Performance values for model A
        performance_b: Performance values for model B
        model_a_name: Name of model A
        model_b_name: Name of model B

    Returns:
        List of (start_severity, end_severity, dominant_model) tuples

    Example:
        >>> regions = find_dominance_regions(severities, perf_a, perf_b)
        >>> for start, end, model in regions:
        ...     print(f"{model} dominates from {start:.2f} to {end:.2f}")
    """
    crossovers = find_all_crossovers(
        severities, performance_a, performance_b,
        model_a_name, model_b_name
    )

    regions = []
    crossover_points = [0] + [c.severity for c in crossovers] + [severities[-1]]

    for i in range(len(crossover_points) - 1):
        start = crossover_points[i]
        end = crossover_points[i + 1]
        mid_point = (start + end) / 2

        # Find closest severity value
        mid_idx = np.argmin(np.abs(severities - mid_point))

        # Determine dominant model
        if performance_a[mid_idx] > performance_b[mid_idx]:
            dominant = model_a_name
        else:
            dominant = model_b_name

        regions.append((float(start), float(end), dominant))

    return regions


def crossover_stability(
    severities: np.ndarray,
    performance_a: np.ndarray,
    performance_b: np.ndarray,
    window_size: int = 3
) -> float:
    """
    Measure stability of performance difference around crossovers.

    Lower values indicate more stable/reliable crossovers.

    Args:
        severities: Array of severity levels
        performance_a: Performance values for model A
        performance_b: Performance values for model B
        window_size: Size of window around crossover

    Returns:
        Stability score (lower is more stable)

    Example:
        >>> stability = crossover_stability(severities, perf_a, perf_b)
        >>> if stability < 0.1:
        ...     print("Crossover is stable and reliable")
    """
    differences = performance_a - performance_b
    sign_changes = np.where(np.diff(np.sign(differences)))[0]

    if len(sign_changes) == 0:
        return 0.0

    stability_scores = []

    for idx in sign_changes:
        # Get window around crossover
        start = max(0, idx - window_size)
        end = min(len(differences), idx + window_size + 1)

        # Calculate variance in difference
        window_diffs = differences[start:end]
        stability_scores.append(np.std(window_diffs))

    return float(np.mean(stability_scores))