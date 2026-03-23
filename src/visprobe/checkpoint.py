"""
Automatic checkpointing for long-running experiments.
Enables resumption after crashes or interruptions.
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import hashlib


class CheckpointManager:
    """
    Manages automatic checkpointing and resumption of experiments.

    Saves progress after each severity level to enable resumption.
    """

    def __init__(self, checkpoint_dir: str, experiment_id: Optional[str] = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            experiment_id: Unique ID for this experiment (auto-generated if None)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Generate or use provided experiment ID
        if experiment_id is None:
            # Create ID from timestamp and random hash
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hash_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
            self.experiment_id = f"exp_{timestamp}_{hash_suffix}"
        else:
            self.experiment_id = experiment_id

        # Paths for different checkpoint types
        self.metadata_path = self.checkpoint_dir / f"{self.experiment_id}_metadata.json"
        self.progress_path = self.checkpoint_dir / f"{self.experiment_id}_progress.json"
        self.results_dir = self.checkpoint_dir / f"{self.experiment_id}_results"
        self.results_dir.mkdir(exist_ok=True)

    def save_metadata(self, metadata: Dict[str, Any]):
        """Save experiment metadata (models, scenarios, severities, etc.)"""
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load experiment metadata if exists."""
        if not self.metadata_path.exists():
            return None
        with open(self.metadata_path, 'r') as f:
            return json.load(f)

    def save_checkpoint(
        self,
        result: Any,
        model_name: str,
        scenario: str,
        severity: float
    ):
        """
        Save a single evaluation result.

        Args:
            result: The evaluation result to save
            model_name: Name of the model
            scenario: Perturbation scenario name
            severity: Severity level
        """
        # Create unique filename for this checkpoint
        filename = f"{model_name}_{scenario}_{severity:.3f}.pkl"
        filepath = self.results_dir / filename

        # Save the result
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)

        # Update progress tracker
        self._update_progress(model_name, scenario, severity)

    def load_checkpoint(
        self,
        model_name: str,
        scenario: str,
        severity: float
    ) -> Optional[Any]:
        """
        Load a previously saved checkpoint.

        Args:
            model_name: Name of the model
            scenario: Perturbation scenario name
            severity: Severity level

        Returns:
            The saved result if exists, None otherwise
        """
        filename = f"{model_name}_{scenario}_{severity:.3f}.pkl"
        filepath = self.results_dir / filename

        if not filepath.exists():
            return None

        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_resume_point(self) -> Optional[Dict[str, Any]]:
        """
        Find where to resume the experiment.

        Returns:
            Dictionary with last completed model, scenario, severity
        """
        if not self.progress_path.exists():
            return None

        with open(self.progress_path, 'r') as f:
            progress = json.load(f)

        return progress.get('last_completed')

    def is_completed(
        self,
        model_name: str,
        scenario: str,
        severity: float
    ) -> bool:
        """Check if a specific evaluation has been completed."""
        filename = f"{model_name}_{scenario}_{severity:.3f}.pkl"
        filepath = self.results_dir / filename
        return filepath.exists()

    def load_all_results(self) -> Dict[str, Dict[str, Dict[float, Any]]]:
        """
        Load all saved results for this experiment.

        Returns:
            Nested dictionary: {model: {scenario: {severity: result}}}
        """
        results = {}

        for filepath in self.results_dir.glob("*.pkl"):
            # Parse filename
            parts = filepath.stem.split('_')
            if len(parts) < 3:
                continue

            model_name = '_'.join(parts[:-2])  # Handle model names with underscores
            scenario = parts[-2]
            severity = float(parts[-1])

            # Load result
            with open(filepath, 'rb') as f:
                result = pickle.load(f)

            # Store in nested structure
            if model_name not in results:
                results[model_name] = {}
            if scenario not in results[model_name]:
                results[model_name][scenario] = {}
            results[model_name][scenario][severity] = result

        return results

    def _update_progress(self, model_name: str, scenario: str, severity: float):
        """Update the progress tracker."""
        # Load existing progress or create new
        if self.progress_path.exists():
            with open(self.progress_path, 'r') as f:
                progress = json.load(f)
        else:
            progress = {
                'started_at': datetime.now().isoformat(),
                'completed_evaluations': [],
                'last_completed': None
            }

        # Add this evaluation to completed list
        evaluation = {
            'model': model_name,
            'scenario': scenario,
            'severity': severity,
            'completed_at': datetime.now().isoformat()
        }
        progress['completed_evaluations'].append(evaluation)
        progress['last_completed'] = evaluation
        progress['last_updated'] = datetime.now().isoformat()

        # Save updated progress
        with open(self.progress_path, 'w') as f:
            json.dump(progress, f, indent=2)

    def cleanup(self, keep_results: bool = True):
        """
        Clean up checkpoint files.

        Args:
            keep_results: If True, keep result files but remove progress tracking
        """
        if self.progress_path.exists():
            self.progress_path.unlink()

        if self.metadata_path.exists() and not keep_results:
            self.metadata_path.unlink()

        if not keep_results and self.results_dir.exists():
            for filepath in self.results_dir.glob("*.pkl"):
                filepath.unlink()
            self.results_dir.rmdir()

    def estimate_disk_usage(self) -> Dict[str, float]:
        """Estimate disk usage by checkpoints in MB."""
        total_size = 0
        file_count = 0

        for filepath in self.results_dir.glob("*.pkl"):
            total_size += filepath.stat().st_size
            file_count += 1

        return {
            'total_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'average_mb_per_file': (total_size / (1024 * 1024) / file_count) if file_count > 0 else 0
        }