"""
visprobe.analysis
~~~~~~~~~~~~~~~~~

Advanced analysis tools for understanding model behavior under perturbations.

This module provides statistical analysis, confidence profiling, and detailed
evaluation capabilities for robustness testing results.

Example:
    >>> from visprobe.analysis import evaluate_detailed, bootstrap_accuracy
    >>> results = evaluate_detailed(model, images, labels)
    >>> acc, lower, upper = bootstrap_accuracy(results.correct_mask)
    >>> print(f"Accuracy: {acc:.1%} (95% CI: {lower:.1%} to {upper:.1%})")
"""

from .detailed_evaluation import (
    SampleResult,
    DetailedResults,
    evaluate_detailed,
    get_failures,
    get_successes,
)

# Alias for backward compatibility
EvaluationResult = DetailedResults

from .statistical import (
    bootstrap_accuracy,
    bootstrap_delta,
    bootstrap_confidence_interval,
    bootstrap_protection_gap,
)

from .crossover import (
    CrossoverPoint,
    find_crossover,
    find_all_crossovers,
)

from .disagreement import (
    DisagreementAnalysis,
    disagreement_analysis,
    compute_complementarity_score,
)

from .confidence import (
    ConfidenceProfile,
    confidence_profile,
    calibration_error,
    reliability_diagram,
)

# Alias for backward compatibility / common usage
expected_calibration_error = calibration_error

from .vulnerability import (
    ClassVulnerability,
    class_vulnerability,
    worst_case_analysis,
    systematic_failures,
)

from .visualization import (
    plot_accuracy_curves,
    plot_confidence_distribution,
    plot_class_vulnerabilities,
    plot_bootstrap_comparison,
)

__all__ = [
    # Detailed evaluation
    'SampleResult',
    'DetailedResults',
    'EvaluationResult',  # Alias for DetailedResults
    'evaluate_detailed',
    'get_failures',
    'get_successes',

    # Statistical analysis
    'bootstrap_accuracy',
    'bootstrap_delta',
    'bootstrap_confidence_interval',
    'bootstrap_protection_gap',

    # Crossover detection
    'CrossoverPoint',
    'find_crossover',
    'find_all_crossovers',

    # Disagreement analysis
    'DisagreementAnalysis',
    'disagreement_analysis',
    'compute_complementarity_score',

    # Confidence analysis
    'ConfidenceProfile',
    'confidence_profile',
    'calibration_error',
    'expected_calibration_error',  # Alias for calibration_error
    'reliability_diagram',

    # Vulnerability analysis
    'ClassVulnerability',
    'class_vulnerability',
    'worst_case_analysis',
    'systematic_failures',

    # Visualization
    'plot_accuracy_curves',
    'plot_confidence_distribution',
    'plot_class_vulnerabilities',
    'plot_bootstrap_comparison',
]