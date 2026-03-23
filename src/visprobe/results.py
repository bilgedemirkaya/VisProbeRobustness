"""
Results container with integrated analysis methods.
Supports both live data and loading from disk.
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime


@dataclass
class EvaluationResult:
    """Result from a single evaluation."""
    accuracy: float
    mean_confidence: float
    mean_loss: float
    correct_mask: np.ndarray
    predictions: np.ndarray
    confidences: np.ndarray
    model_name: str
    scenario: str
    severity: float
    eps: float
    n_samples: int
    metadata: Dict[str, Any] = None


class CompositionalResults:
    """
    Container for compositional test results with integrated analysis.

    Data structure: {model: {scenario: {severity: EvaluationResult}}}
    """

    def __init__(self, data: Dict = None):
        """
        Initialize results container.

        Args:
            data: Nested dictionary of results
        """
        self.data = data or {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'visprobe_version': '2.0.0'
        }

    def add_result(
        self,
        model_name: str,
        scenario: str,
        severity: float,
        result: EvaluationResult
    ):
        """Add a single evaluation result."""
        if model_name not in self.data:
            self.data[model_name] = {}
        if scenario not in self.data[model_name]:
            self.data[model_name][scenario] = {}
        self.data[model_name][scenario][severity] = result

    def get_result(
        self,
        model_name: str,
        scenario: str,
        severity: float
    ) -> Optional[EvaluationResult]:
        """Get a single evaluation result."""
        try:
            return self.data[model_name][scenario][severity]
        except KeyError:
            return None

    def get_models(self) -> List[str]:
        """Get list of model names."""
        return list(self.data.keys())

    def get_scenarios(self, model_name: Optional[str] = None) -> List[str]:
        """Get list of scenarios."""
        if model_name:
            return list(self.data.get(model_name, {}).keys())
        # Get all unique scenarios across all models
        scenarios = set()
        for model_data in self.data.values():
            scenarios.update(model_data.keys())
        return sorted(list(scenarios))

    def get_severities(self, model_name: str, scenario: str) -> List[float]:
        """Get list of severities for a model/scenario."""
        try:
            return sorted(list(self.data[model_name][scenario].keys()))
        except KeyError:
            return []

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def protection_gap(self, baseline: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate protection gap relative to baseline model.

        Protection gap = accuracy difference at each severity level.

        Args:
            baseline: Name of baseline model

        Returns:
            Dictionary: {model: {scenario: {severity: gap}}}
        """
        if baseline not in self.data:
            raise ValueError(f"Baseline model '{baseline}' not found")

        gaps = {}
        baseline_data = self.data[baseline]

        for model_name, model_data in self.data.items():
            if model_name == baseline:
                continue

            gaps[model_name] = {}

            for scenario in model_data:
                if scenario not in baseline_data:
                    continue

                gaps[model_name][scenario] = {}

                for severity, result in model_data[scenario].items():
                    if severity in baseline_data[scenario]:
                        baseline_acc = baseline_data[scenario][severity].accuracy
                        model_acc = result.accuracy
                        gap = model_acc - baseline_acc
                        gaps[model_name][scenario][severity] = gap

        return gaps

    def crossover_detection(
        self,
        baseline: str,
        threshold: float = 0.5
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Find accuracy crossover points between models.

        Crossover point = severity where model accuracy crosses threshold.

        Args:
            baseline: Reference model name
            threshold: Accuracy threshold for crossover

        Returns:
            Dictionary: {model: {scenario: crossover_severity}}
        """
        crossovers = {}

        for model_name, model_data in self.data.items():
            crossovers[model_name] = {}

            for scenario, severity_data in model_data.items():
                # Get sorted severities and accuracies
                severities = sorted(severity_data.keys())
                accuracies = [severity_data[s].accuracy for s in severities]

                # Find crossover point
                crossover = None
                for i, (sev, acc) in enumerate(zip(severities, accuracies)):
                    if acc < threshold:
                        if i == 0:
                            crossover = sev
                        else:
                            # Interpolate between points
                            prev_sev = severities[i-1]
                            prev_acc = accuracies[i-1]
                            # Linear interpolation
                            if prev_acc != acc:
                                crossover = prev_sev + (threshold - prev_acc) * (sev - prev_sev) / (acc - prev_acc)
                            else:
                                crossover = sev
                        break

                crossovers[model_name][scenario] = crossover

        return crossovers

    def confidence_profile(
        self,
        model: str,
        scenario: str,
        severity: float
    ) -> Dict[str, Any]:
        """
        Get confidence statistics for a specific evaluation.

        Args:
            model: Model name
            scenario: Scenario name
            severity: Severity level

        Returns:
            Dictionary with confidence statistics
        """
        result = self.get_result(model, scenario, severity)
        if result is None:
            raise ValueError(f"No result found for {model}/{scenario}/{severity}")

        confidences = result.confidences
        correct = result.correct_mask

        return {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'mean_confidence_correct': float(np.mean(confidences[correct])) if correct.any() else 0.0,
            'mean_confidence_incorrect': float(np.mean(confidences[~correct])) if (~correct).any() else 0.0,
            'confidence_gap': float(np.mean(confidences[correct]) - np.mean(confidences[~correct]))
                             if correct.any() and (~correct).any() else 0.0,
            'overconfidence_rate': float(np.mean((confidences > 0.9) & ~correct)) if (~correct).any() else 0.0,
        }

    def disagreement_analysis(
        self,
        scenario: str = None,
        severity: float = None
    ) -> Dict[str, float]:
        """
        Analyze disagreement between models.

        Args:
            scenario: Specific scenario (or all if None)
            severity: Specific severity (or all if None)

        Returns:
            Dictionary with disagreement metrics
        """
        model_names = self.get_models()
        if len(model_names) < 2:
            return {'disagreement_rate': 0.0}

        # Collect predictions from all models
        all_predictions = []

        for model_name in model_names:
            model_predictions = []

            if scenario:
                scenarios = [scenario]
            else:
                scenarios = self.get_scenarios(model_name)

            for scen in scenarios:
                if severity is not None:
                    severities = [severity]
                else:
                    severities = self.get_severities(model_name, scen)

                for sev in severities:
                    result = self.get_result(model_name, scen, sev)
                    if result:
                        model_predictions.append(result.predictions)

            if model_predictions:
                all_predictions.append(np.concatenate(model_predictions))

        if len(all_predictions) < 2:
            return {'disagreement_rate': 0.0}

        # Calculate disagreement
        min_length = min(len(p) for p in all_predictions)
        all_predictions = [p[:min_length] for p in all_predictions]
        predictions_matrix = np.stack(all_predictions)

        # Disagreement = samples where not all models agree
        mode_predictions = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), 0, predictions_matrix
        )
        disagreement_mask = np.any(predictions_matrix != mode_predictions, axis=0)

        return {
            'disagreement_rate': float(np.mean(disagreement_mask)),
            'n_models': len(model_names),
            'n_samples': min_length
        }

    def compute_auc(
        self,
        model: str,
        scenario: str
    ) -> float:
        """
        Compute area under the robustness curve.

        Args:
            model: Model name
            scenario: Scenario name

        Returns:
            AUC value (higher is more robust)
        """
        severities = self.get_severities(model, scenario)
        if len(severities) < 2:
            return 0.0

        accuracies = [self.data[model][scenario][s].accuracy for s in severities]

        # Trapezoidal integration
        auc = 0.0
        for i in range(len(severities) - 1):
            width = severities[i+1] - severities[i]
            height = (accuracies[i] + accuracies[i+1]) / 2
            auc += width * height

        # Normalize by severity range
        severity_range = severities[-1] - severities[0]
        if severity_range > 0:
            auc /= severity_range

        return auc

    # =========================================================================
    # Visualization
    # =========================================================================

    def plot_compositional(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot compositional robustness results.

        Creates a grid of plots showing accuracy vs severity for each scenario.

        Args:
            save_path: Path to save the figure
            figsize: Figure size
        """
        models = self.get_models()
        scenarios = self.get_scenarios()

        if not models or not scenarios:
            print("No data to plot")
            return

        # Create subplot grid
        n_scenarios = len(scenarios)
        n_cols = min(3, n_scenarios)
        n_rows = (n_scenarios + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        # Plot each scenario
        for idx, scenario in enumerate(scenarios):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col]

            for model_name in models:
                if scenario in self.data[model_name]:
                    severities = sorted(self.data[model_name][scenario].keys())
                    accuracies = [self.data[model_name][scenario][s].accuracy
                                for s in severities]
                    ax.plot(severities, accuracies, marker='o', label=model_name)

            ax.set_xlabel('Severity')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{scenario}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        # Hide empty subplots
        for idx in range(n_scenarios, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].axis('off')

        plt.suptitle('Compositional Robustness Results', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    # =========================================================================
    # Serialization
    # =========================================================================

    def save(self, path: str):
        """
        Save results to disk.

        Args:
            path: Directory path to save results
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Save data
        with open(save_dir / 'results.pkl', 'wb') as f:
            pickle.dump(self.data, f)

        # Save summary as JSON for easy inspection
        summary = self._create_summary()
        with open(save_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {save_dir}")

    @classmethod
    def load(cls, path: str) -> 'CompositionalResults':
        """
        Load results from disk.

        Args:
            path: Directory path containing saved results

        Returns:
            CompositionalResults instance
        """
        load_dir = Path(path)

        # Load metadata
        with open(load_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Load data
        with open(load_dir / 'results.pkl', 'rb') as f:
            data = pickle.load(f)

        # Create instance
        results = cls(data)
        results.metadata = metadata

        print(f"Loaded results from {load_dir}")
        return results

    def _create_summary(self) -> Dict[str, Any]:
        """Create a summary of results for easy inspection."""
        summary = {
            'models': self.get_models(),
            'scenarios': self.get_scenarios(),
            'n_evaluations': sum(
                len(self.data[m][s])
                for m in self.data
                for s in self.data[m]
            ),
            'accuracies': {}
        }

        # Add accuracy ranges
        for model in self.get_models():
            summary['accuracies'][model] = {}
            for scenario in self.get_scenarios(model):
                severities = self.get_severities(model, scenario)
                if severities:
                    accs = [self.data[model][scenario][s].accuracy for s in severities]
                    summary['accuracies'][model][scenario] = {
                        'min': min(accs),
                        'max': max(accs),
                        'auc': self.compute_auc(model, scenario)
                    }

        return summary

    def print_summary(self):
        """Print a summary of results."""
        print("\n" + "="*60)
        print("COMPOSITIONAL RESULTS SUMMARY")
        print("="*60)

        models = self.get_models()
        scenarios = self.get_scenarios()

        print(f"\nModels: {', '.join(models)}")
        print(f"Scenarios: {', '.join(scenarios)}")

        for model in models:
            print(f"\n{model}:")
            for scenario in self.get_scenarios(model):
                severities = self.get_severities(model, scenario)
                if severities:
                    accs = [self.data[model][scenario][s].accuracy for s in severities]
                    auc = self.compute_auc(model, scenario)
                    print(f"  {scenario}:")
                    print(f"    Severity range: [{min(severities):.2f}, {max(severities):.2f}]")
                    print(f"    Accuracy range: [{min(accs):.3f}, {max(accs):.3f}]")
                    print(f"    AUC: {auc:.3f}")

        print("="*60)