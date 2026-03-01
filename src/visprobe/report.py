"""
Simple test report for VisProbe robustness testing.

- Pass/fail status
- Robustness score
- Number of failures
- Runtime and query count
- Clear summary
- AI-style insights generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

__all__ = ["Report", "generate_insights"]


@dataclass
class Report:
    """
    Simple robustness test report.

    Contains essential information engineers need to assess model robustness.
    """

    # Test identification
    test_name: str
    test_type: str  # "preset" or "search"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Model info
    model_name: str = "Unknown"
    dataset: str = "Unknown"

    # Test configuration
    preset: Optional[str] = None
    property_name: str = "Unknown"
    strategy: str = "Unknown"

    # Results (most important)
    score: float = 0.0  # Robustness score (0-100)
    total_samples: int = 0
    passed_samples: int = 0
    failed_samples: int = 0

    # Performance
    runtime: float = 0.0  # seconds
    model_queries: int = 0

    # Additional data
    metrics: Dict[str, Any] = field(default_factory=dict)
    search: Dict[str, Any] = field(default_factory=dict)
    failures: List[Dict[str, Any]] = field(default_factory=list)

    # Sample images for dashboard visualization
    sample_images: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived fields."""
        if self.total_samples > 0:
            self.failed_samples = self.total_samples - self.passed_samples
            self.score = (self.passed_samples / self.total_samples) * 100

    @property
    def passed(self) -> bool:
        """Whether the test passed (100% robustness)."""
        return self.passed_samples == self.total_samples

    @property
    def pass_rate(self) -> float:
        """Pass rate as percentage (same as score)."""
        return self.score

    def summary(self) -> str:
        """
        Get a concise text summary of the report.

        Returns:
            Human-readable summary string
        """
        lines = [
            f"\n{'='*60}",
            f"  VisProbe Robustness Test Report",
            f"{'='*60}",
            f"",
            f"Test: {self.test_name}",
            f"Model: {self.model_name}",
            f"Dataset: {self.dataset}",
            f"",
            f"Configuration:",
            f"  Preset: {self.preset or 'N/A'}",
            f"  Property: {self.property_name}",
            f"  Strategy: {self.strategy}",
            f"",
            f"Results:",
            f"  Robustness Score: {self.score:.1f}%",
            f"  Samples Tested: {self.total_samples}",
            f"  Passed: {self.passed_samples}",
            f"  Failed: {self.failed_samples}",
            f"",
            f"Performance:",
            f"  Runtime: {self.runtime:.2f}s",
            f"  Model Queries: {self.model_queries:,}",
            f"  Queries/sec: {self.model_queries/self.runtime if self.runtime > 0 else 0:.1f}",
            f"",
        ]

        # Add threat model info if available
        if self.search and "threat_model" in self.search:
            lines.extend([
                f"Threat Model: {self.search['threat_model']}",
                f"",
            ])

        # Add key metrics if available
        if self.metrics:
            lines.append("Metrics:")
            for key, value in sorted(self.metrics.items()):
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.3f}")
                else:
                    lines.append(f"  {key}: {value}")
            lines.append("")

        # Status
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        lines.extend([
            f"Status: {status}",
            f"{'='*60}\n",
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report to dictionary.

        Returns:
            Dictionary representation of the report
        """
        result = {
            "test_name": self.test_name,
            "test_type": self.test_type,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "dataset": self.dataset,
            "preset": self.preset,
            "property_name": self.property_name,
            "strategy": self.strategy,
            "score": self.score,
            "total_samples": self.total_samples,
            "passed_samples": self.passed_samples,
            "failed_samples": self.failed_samples,
            "runtime": self.runtime,
            "model_queries": self.model_queries,
            "metrics": self.metrics,
            "search": self.search,
            "failures": self.failures,
        }

        # Add sample images if available (for dashboard visualization)
        if self.sample_images:
            result["original_image"] = self.sample_images.get("original")
            result["perturbed_image"] = self.sample_images.get("perturbed")
            result["residual_image"] = self.sample_images.get("residual")

        return result

    def save_json(self, path: str) -> None:
        """
        Save report as JSON.

        Args:
            path: Path to save JSON file
        """
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_csv(self, path: str) -> None:
        """
        Save report summary as CSV (one row).

        Args:
            path: Path to save CSV file
        """
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "test_name", "model_name", "dataset", "preset",
                "score", "passed_samples", "failed_samples", "total_samples",
                "runtime", "model_queries", "timestamp"
            ])
            writer.writeheader()
            writer.writerow({
                "test_name": self.test_name,
                "model_name": self.model_name,
                "dataset": self.dataset,
                "preset": self.preset,
                "score": f"{self.score:.2f}",
                "passed_samples": self.passed_samples,
                "failed_samples": self.failed_samples,
                "total_samples": self.total_samples,
                "runtime": f"{self.runtime:.2f}",
                "model_queries": self.model_queries,
                "timestamp": self.timestamp,
            })

    def __str__(self) -> str:
        """String representation shows summary."""
        return self.summary()

    def __repr__(self) -> str:
        """Repr shows key stats."""
        return (
            f"Report(test={self.test_name}, score={self.score:.1f}%, "
            f"passed={self.passed_samples}/{self.total_samples})"
        )

    def show(self) -> None:
        """Print the summary to console."""
        print(self.summary())

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the report (dict-like access).

        Args:
            key: Key to look up
            default: Default value if key not found

        Returns:
            Value for key or default
        """
        return self.to_dict().get(key, default)

    def add_metric(self, key: str, value: Any) -> None:
        """
        Add a metric to the report.

        Args:
            key: Metric name
            value: Metric value
        """
        self.metrics[key] = value

    def save(self, path: Optional[str] = None, test_file: Optional[str] = None) -> str:
        """
        Save report to JSON file for dashboard visualization.

        Args:
            path: Optional explicit path. If not provided, auto-generates path.
            test_file: Optional test file name (e.g., "test_cifar10.py").
                       If not provided, uses calling script name.

        Returns:
            Path where file was saved

        Example:
            >>> report = search(model, data, strategy)
            >>> report.save()  # Auto-saves to visprobe results directory
        """
        import inspect
        import os

        # Import results dir helper
        try:
            from visprobe.cli.utils import get_results_dir
            results_dir = get_results_dir()
        except ImportError:
            results_dir = os.path.join(os.path.expandvars("$TMPDIR") or "/tmp", "visprobe_results")

        os.makedirs(results_dir, exist_ok=True)

        if path is None:
            # Auto-detect test file name from calling script
            if test_file is None:
                # Walk up the call stack to find the test script
                for frame_info in inspect.stack():
                    filename = os.path.basename(frame_info.filename)
                    # Skip visprobe internal files
                    if filename.startswith("test_") or not filename.startswith(("api.py", "report.py", "__")):
                        if not frame_info.filename.startswith("<"):
                            test_file = filename
                            break

            if test_file:
                module_name = os.path.splitext(test_file)[0]
                # Use strategy name for unique per-strategy files
                path = os.path.join(results_dir, f"{module_name}.{self.strategy}.json")
            else:
                path = os.path.join(results_dir, f"{self.test_name}.json")

        self.save_json(path)
        return path

    def export_failures(self, n: int = 10, output_dir: Optional[str] = None) -> str:
        """
        Export top N failures to a JSON file.

        Args:
            n: Number of failures to export (default 10)
            output_dir: Directory to save to (default visprobe_results/)

        Returns:
            Path where file was saved
        """
        import json
        import os

        if output_dir is None:
            output_dir = "visprobe_results"
        os.makedirs(output_dir, exist_ok=True)

        # Get top N failures
        top_failures = self.failures[:n] if self.failures else []

        export_data = {
            "test_name": self.test_name,
            "total_failures": len(self.failures),
            "exported_count": len(top_failures),
            "failures": top_failures,
        }

        path = os.path.join(output_dir, f"{self.test_name}_failures.json")
        with open(path, "w") as f:
            json.dump(export_data, f, indent=2)

        return path


def generate_insights(reports: List[Report]) -> List[str]:
    """
    Generate AI-style insights by comparing multiple test reports.

    Analyzes patterns across strategies to identify:
    - Which perturbations are most/least effective
    - Which samples are universally vulnerable vs. strategy-specific
    - Confidence patterns and decision boundary behavior
    - Actionable recommendations

    Args:
        reports: List of Report objects from different strategies

    Returns:
        List of insight strings (human-readable observations)

    Example:
        >>> reports = [noise_report, blur_report, brightness_report]
        >>> for insight in generate_insights(reports):
        ...     print(f"• {insight}")
    """
    if not reports:
        return ["No reports provided for analysis."]

    if len(reports) == 1:
        return _single_report_insights(reports[0])

    return _multi_report_insights(reports)


def _single_report_insights(report: Report) -> List[str]:
    """Generate insights for a single report."""
    insights = []

    # Basic stats
    score = report.metrics.get("overall_robustness_score", 0)
    threshold = report.metrics.get("failure_threshold", 0)
    unique_failures = report.metrics.get("unique_failed_samples", 0)

    # Score interpretation
    if score >= 0.9:
        insights.append(f"Excellent robustness ({score:.0%}) - model handles {report.strategy} perturbations well")
    elif score >= 0.7:
        insights.append(f"Good robustness ({score:.0%}) - moderate sensitivity to {report.strategy}")
    elif score >= 0.5:
        insights.append(f"Fair robustness ({score:.0%}) - notable vulnerability to {report.strategy}")
    else:
        insights.append(f"Poor robustness ({score:.0%}) - highly vulnerable to {report.strategy}")

    # Threshold analysis
    if threshold < 0.01:
        insights.append(f"Very low failure threshold ({threshold:.4f}) - even minimal perturbations cause failures")
    elif threshold < 0.05:
        insights.append(f"Low failure threshold ({threshold:.4f}) - model is sensitive to small changes")

    # Confidence analysis from search path
    search_path = report.search.get("search_path", [])
    if search_path:
        confidences = [s.get("avg_confidence", 0) for s in search_path if s.get("avg_confidence")]
        if confidences:
            baseline_conf = confidences[0] if confidences else 0
            if baseline_conf < 0.5:
                insights.append(f"Low baseline confidence ({baseline_conf:.0%}) suggests model is uncertain even on clean inputs")

    return insights


def _multi_report_insights(reports: List[Report]) -> List[str]:
    """Generate comparative insights across multiple reports."""
    insights = []

    # Extract data from each report
    strategy_data = {}
    all_failed_indices: Dict[str, Set[int]] = {}

    for r in reports:
        name = r.strategy
        strategy_data[name] = {
            "score": r.metrics.get("overall_robustness_score", 0),
            "threshold": r.metrics.get("failure_threshold", 0),
            "unique_failures": r.metrics.get("unique_failed_samples", 0),
            "failures": r.failures,
        }
        # Track which sample indices failed
        all_failed_indices[name] = {f["index"] for f in r.failures}

    # 1. Rank strategies by vulnerability (lower score = higher vulnerability)
    ranked = sorted(strategy_data.items(), key=lambda x: x[1]["score"])
    most_vulnerable = ranked[0][0]
    least_vulnerable = ranked[-1][0]

    insights.append(
        f"Highest vulnerability: {most_vulnerable} "
        f"(robustness: {strategy_data[most_vulnerable]['score']:.0%})"
    )
    insights.append(
        f"Lowest vulnerability: {least_vulnerable} "
        f"(robustness: {strategy_data[least_vulnerable]['score']:.0%})"
    )

    # 2. Find universally vulnerable samples (fail under ALL strategies)
    if len(all_failed_indices) > 1:
        common_failures = set.intersection(*all_failed_indices.values())
        if common_failures:
            insights.append(
                f"{len(common_failures)} samples are universally vulnerable "
                f"(fail under all {len(reports)} perturbation types)"
            )

    # 3. Find strategy-specific vulnerabilities
    for name, indices in all_failed_indices.items():
        other_indices = set.union(*[v for k, v in all_failed_indices.items() if k != name]) if len(all_failed_indices) > 1 else set()
        unique_to_strategy = indices - other_indices
        if unique_to_strategy and len(unique_to_strategy) >= 2:
            insights.append(
                f"{len(unique_to_strategy)} samples are uniquely vulnerable to {name} "
                f"(robust to other perturbations)"
            )

    # 4. Compare failure counts
    failure_counts = {name: data["unique_failures"] for name, data in strategy_data.items()}
    max_failures = max(failure_counts.values())
    min_failures = min(failure_counts.values())

    if max_failures > min_failures:
        max_strategy = [k for k, v in failure_counts.items() if v == max_failures][0]
        min_strategy = [k for k, v in failure_counts.items() if v == min_failures][0]
        insights.append(
            f"{max_strategy} causes {max_failures} failures vs {min_strategy}'s {min_failures} - "
            f"{max_failures - min_failures} additional samples are {max_strategy}-sensitive"
        )

    # 5. Threshold comparison
    # Lower threshold = higher sensitivity (model fails at lower perturbation levels)
    thresholds = {name: data["threshold"] for name, data in strategy_data.items()}
    if thresholds and len(thresholds) > 1:
        lowest_thresh = min(thresholds.items(), key=lambda x: x[1])
        highest_thresh = max(thresholds.items(), key=lambda x: x[1])
        # Only report if there's a significant difference (2x or more)
        if highest_thresh[1] > lowest_thresh[1] * 2.0:
            sensitivity_ratio = highest_thresh[1] / lowest_thresh[1]
            insights.append(
                f"Model tolerates {sensitivity_ratio:.1f}x higher {highest_thresh[0]} "
                f"than {lowest_thresh[0]} before failing"
            )

    # 6. Confidence-based insights
    for r in reports:
        search_path = r.search.get("search_path", [])
        if search_path:
            baseline = search_path[0].get("avg_confidence", 0)
            if baseline < 0.3:
                insights.append(
                    f"Warning: Very low baseline confidence ({baseline:.0%}) - "
                    f"model may be operating on out-of-distribution data"
                )
                break  # Only show once

    return insights
