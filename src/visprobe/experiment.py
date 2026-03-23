"""
Main compositional experiment runner.
Handles checkpointing, memory management, and orchestration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Union
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np

from .checkpoint import CheckpointManager
from .memory import ModelMemoryManager, BatchMemoryOptimizer
from .attacks import AttackFactory
from .perturbations import get_standard_perturbations
from .results import CompositionalResults, EvaluationResult

logger = logging.getLogger(__name__)


class CompositionalExperiment:
    """
    Main experiment runner for compositional robustness testing.

    Automatically handles:
    - Checkpointing and resumption
    - GPU memory management with model swapping
    - AutoAttack integration
    - Progress tracking
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        images: torch.Tensor,
        labels: torch.Tensor,
        env_strategies: Optional[Dict[str, Callable]] = None,
        attack: str = "autoattack-apgd-ce",
        severities: Optional[List[float]] = None,
        eps_fn: Optional[Callable] = None,
        checkpoint_dir: str = "./checkpoints",
        batch_size: int = 50,
        device: str = "cuda",
        verbose: bool = True,
        experiment_id: Optional[str] = None,
    ):
        """
        Initialize compositional experiment.

        Args:
            models: Dictionary of model name to model instance
            images: Input images tensor (N, C, H, W)
            labels: Ground truth labels (N,)
            env_strategies: Environmental perturbations (default: standard set)
            attack: Attack type ("autoattack-apgd-ce", "autoattack-standard", "pgd", "none")
            severities: List of severity levels (default: [0, 0.2, 0.4, 0.6, 0.8, 1.0])
            eps_fn: Function mapping severity to epsilon (default: lambda s: (8/255) * s)
            checkpoint_dir: Directory for checkpoints
            batch_size: Batch size for evaluation
            device: Device to use (cuda/cpu)
            verbose: Whether to show progress
            experiment_id: Unique experiment ID (auto-generated if None)
        """
        self.models = models
        self.images = images
        self.labels = labels
        self.env_strategies = env_strategies or get_standard_perturbations()
        self.attack_type = attack
        self.severities = severities or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.eps_fn = eps_fn or (lambda s: (8/255) * s)
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.verbose = verbose

        # Initialize managers
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir, experiment_id)
        self.memory_mgr = ModelMemoryManager(models, self.device)
        self.batch_optimizer = BatchMemoryOptimizer(batch_size)

        # Save experiment metadata
        metadata = {
            'models': list(models.keys()),
            'scenarios': list(self.env_strategies.keys()),
            'severities': self.severities,
            'attack': attack,
            'n_samples': len(images),
            'device': self.device,
        }
        self.checkpoint_mgr.save_metadata(metadata)

        # Setup logging
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

    def run(self, gpus: Optional[List[int]] = None) -> CompositionalResults:
        """
        Run the experiment.

        Args:
            gpus: List of GPU IDs for parallel execution (None for single GPU/CPU)

        Returns:
            CompositionalResults with all evaluation data
        """
        if gpus and len(gpus) > 1:
            return self._run_multi_gpu(gpus)
        else:
            return self._run_single_gpu()

    def _run_single_gpu(self) -> CompositionalResults:
        """Run experiment on single GPU with model swapping."""
        results = CompositionalResults()

        # Check for existing results
        resume_point = self.checkpoint_mgr.get_resume_point()
        if resume_point:
            logger.info(f"Resuming from checkpoint: {resume_point}")
            # Load existing results
            existing_data = self.checkpoint_mgr.load_all_results()
            results.data = existing_data

        # Total number of evaluations
        total_evals = len(self.models) * len(self.env_strategies) * len(self.severities)
        completed_evals = 0

        # Progress bar
        if self.verbose:
            pbar = tqdm(total=total_evals, desc="Compositional Test")
            # Update with already completed evaluations
            if resume_point:
                for model in results.data:
                    for scenario in results.data[model]:
                        completed_evals += len(results.data[model][scenario])
                pbar.update(completed_evals)
        else:
            pbar = None

        # Iterate through all combinations
        for model_name in self.models:
            # Check if model already completed
            if self._is_model_complete(model_name, results):
                if self.verbose:
                    logger.info(f"Skipping completed model: {model_name}")
                continue

            # Load model to GPU
            model = self.memory_mgr.load_model(model_name)

            for scenario_name, env_strategy in self.env_strategies.items():
                for severity in self.severities:
                    # Check if already completed
                    if self.checkpoint_mgr.is_completed(model_name, scenario_name, severity):
                        existing_result = self.checkpoint_mgr.load_checkpoint(
                            model_name, scenario_name, severity
                        )
                        if existing_result:
                            results.add_result(model_name, scenario_name, severity, existing_result)
                            if pbar:
                                pbar.update(1)
                            continue

                    # Run evaluation
                    if self.verbose:
                        pbar.set_description(f"{model_name}/{scenario_name}/sev={severity:.2f}")

                    result = self._evaluate_single(
                        model=model,
                        model_name=model_name,
                        scenario=scenario_name,
                        env_strategy=env_strategy,
                        severity=severity
                    )

                    # Save result and checkpoint
                    results.add_result(model_name, scenario_name, severity, result)
                    self.checkpoint_mgr.save_checkpoint(result, model_name, scenario_name, severity)

                    if pbar:
                        pbar.update(1)

                    # Clear GPU cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if pbar:
            pbar.close()

        # Print summary
        if self.verbose:
            results.print_summary()

        return results

    def _evaluate_single(
        self,
        model: nn.Module,
        model_name: str,
        scenario: str,
        env_strategy: Callable,
        severity: float
    ) -> EvaluationResult:
        """Evaluate a single model/scenario/severity combination."""
        # Stage 1: Environmental perturbation
        env_perturbed = env_strategy(self.images, severity)

        # Stage 2: Adversarial attack
        eps = self.eps_fn(severity)
        if eps < 1e-10 or self.attack_type == "none":
            # Skip attack if epsilon too small
            final_images = env_perturbed
        else:
            # Create attack
            attack_fn = AttackFactory.create(
                self.attack_type,
                eps=eps,
                batch_size=self.batch_size
            )
            # Run attack
            final_images = attack_fn(model, env_perturbed, self.labels)

        # Stage 3: Evaluate
        accuracy, predictions, confidences, correct_mask, loss = self._evaluate_batch(
            model, final_images, self.labels
        )

        # Create result
        result = EvaluationResult(
            accuracy=accuracy,
            mean_confidence=float(np.mean(confidences)),
            mean_loss=loss,
            correct_mask=correct_mask,
            predictions=predictions,
            confidences=confidences,
            model_name=model_name,
            scenario=scenario,
            severity=severity,
            eps=eps,
            n_samples=len(self.labels),
            metadata={'attack': self.attack_type}
        )

        return result

    def _evaluate_batch(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple:
        """Evaluate model on a batch of images."""
        model.eval()

        all_predictions = []
        all_confidences = []
        all_correct = []
        total_loss = 0.0
        n_batches = 0

        # Process in batches
        n_samples = len(images)
        for i in range(0, n_samples, self.batch_size):
            batch_images = images[i:i+self.batch_size].to(self.device)
            batch_labels = labels[i:i+self.batch_size].to(self.device)

            with torch.no_grad():
                outputs = model(batch_images)
                loss = F.cross_entropy(outputs, batch_labels)

                # Get predictions and confidences
                probs = F.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probs, dim=1)

                # Check correctness
                correct = predictions == batch_labels

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_correct.extend(correct.cpu().numpy())
                total_loss += loss.item()
                n_batches += 1

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_confidences = np.array(all_confidences)
        all_correct = np.array(all_correct)
        accuracy = float(np.mean(all_correct))
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        return accuracy, all_predictions, all_confidences, all_correct, avg_loss

    def _is_model_complete(self, model_name: str, results: CompositionalResults) -> bool:
        """Check if a model has all evaluations complete."""
        if model_name not in results.data:
            return False

        expected_count = len(self.env_strategies) * len(self.severities)
        actual_count = sum(
            len(results.data[model_name].get(scenario, {}))
            for scenario in self.env_strategies
        )
        return actual_count >= expected_count

    def _run_multi_gpu(self, gpus: List[int]) -> CompositionalResults:
        """
        Run experiment in parallel across multiple GPUs.

        This is a placeholder for future implementation.
        """
        logger.warning("Multi-GPU support not yet implemented. Falling back to single GPU.")
        return self._run_single_gpu()

    def cleanup(self):
        """Clean up resources."""
        self.memory_mgr.release_all()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def quick_test(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    attack: str = "pgd",
    severity: float = 0.5
) -> Dict[str, float]:
    """
    Quick test function for single evaluation.

    Args:
        model: Model to test
        images: Input images
        labels: Ground truth labels
        attack: Attack type
        severity: Perturbation severity

    Returns:
        Dictionary with accuracy metrics
    """
    from .perturbations import get_minimal_perturbations

    # Create simple experiment
    experiment = CompositionalExperiment(
        models={'model': model},
        images=images,
        labels=labels,
        env_strategies=get_minimal_perturbations(),
        attack=attack,
        severities=[severity],
        verbose=False
    )

    # Run
    results = experiment.run()

    # Extract metrics
    metrics = {}
    for scenario in results.get_scenarios():
        result = results.get_result('model', scenario, severity)
        if result:
            metrics[f'{scenario}_accuracy'] = result.accuracy

    experiment.cleanup()
    return metrics