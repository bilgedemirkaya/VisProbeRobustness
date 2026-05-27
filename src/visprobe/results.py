"""
Results container with integrated analysis methods.
Supports both live data and loading from disk.
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
import torch
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
    # Analysis
    # =========================================================================

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
    # RobustBench comparison (v3 / Feature A)
    # =========================================================================

    def compare_to_leaderboard(
        self,
        model_name: str,
        dataset: str,
        threat: str,
    ):
        """Rank a protocol-compliant result against the RobustBench leaderboard.

        Finds the cell at ``(model_name, scenario='none', severity=0.0)``,
        validates it satisfies RobustBench's protocol (M3 gate), and looks up
        its rank in the shipped leaderboard snapshot.

        Args:
            model_name: the key under which the result was added (typically
                the name passed to ``CompositionalExperiment(models={...})``
                or ``"_robustbench_target"`` if produced by ``robustbench_eval``).
            dataset: "cifar10" or "imagenet" (case-insensitive).
            threat: "Linf" (case-insensitive).

        Returns:
            ``LeaderboardComparison`` — print it with ``str(...)`` to get a
            paper-pasteable summary.

        Raises:
            ValueError: no cell at ``(model_name, 'none', 0.0)``. The error
                message tells the user to call ``robustbench_eval`` first.
            ProtocolError: the cell exists but does not match RobustBench's
                strict protocol. Always indicates user produced the cell via
                ``CompositionalExperiment`` directly, not via ``robustbench_eval``.
        """
        # Defer import to avoid a top-of-file import cycle: results.py is
        # imported by experiment.py, which leaderboard.py uses via TYPE_CHECKING.
        from .leaderboard import (
            RobustBenchClient,
            LeaderboardComparison,
            validate_protocol,
        )

        cell = self.get_result(model_name, "none", 0.0)
        if cell is None:
            raise ValueError(
                f"No cell found at (model={model_name!r}, scenario='none', severity=0.0). "
                "Produce one with:\n"
                f"    from visprobe import robustbench_eval\n"
                f"    result = robustbench_eval(model, "
                f"dataset={dataset!r}, threat={threat!r}, confirm=True)"
            )

        # Gate. Raises ProtocolError if the cell isn't a strict-protocol result.
        # Canonicalization inside validate_protocol means callers can pass
        # any case for dataset/threat — the gate normalizes both before lookup.
        validate_protocol(cell, dataset, threat)

        client = RobustBenchClient(dataset, threat)
        above, below = client.neighbors(cell.accuracy, k=3)

        return LeaderboardComparison(
            model_name=model_name,
            robust_acc=cell.accuracy,
            rank=client.rank(cell.accuracy),
            total=len(client),
            neighbors_above=above,
            neighbors_below=below,
            snapshot_date=client.snapshot_date,
            dataset=client.dataset,
            threat=client.threat,
            attack=(cell.metadata or {}).get("attack", ""),
            eps=cell.eps,
        )


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