"""
Main compositional experiment runner.

Orchestrates the sweep over (model, scenario, severity), delegating persistence
to the ``checkpoint`` module and CPU<->GPU swapping to ``ModelMemoryManager``.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from . import attacks, checkpoint
from .memory import ModelMemoryManager
from .perturbations import get_standard_perturbations
from .results import CompositionalResults, EvaluationResult

logger = logging.getLogger(__name__)


class CompositionalExperiment:
    """
    Sweep `model x scenario x severity`, applying environmental perturbation
    composed with an adversarial attack at each cell. Cells are checkpointed
    per `(model, scenario, severity)` triple so reruns resume where they stopped.
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
    ):
        self.models = models
        self.images = images
        self.labels = labels
        self.env_strategies = env_strategies or get_standard_perturbations()
        self.attack_type = attack
        self.severities = severities or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.eps_fn = eps_fn or (lambda s: (8 / 255) * s)
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.verbose = verbose

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.checkpoint_dir / "metadata.json"

        self.memory_mgr = ModelMemoryManager(models, self.device)

        checkpoint.save_metadata(
            {
                "models": list(models.keys()),
                "scenarios": list(self.env_strategies.keys()),
                "severities": self.severities,
                "attack": attack,
                "n_samples": len(images),
                "device": self.device,
            },
            self._metadata_path,
        )

        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

    def run(self) -> CompositionalResults:
        """Run the sweep, resuming any cells already on disk in ``checkpoint_dir``."""
        results = CompositionalResults()
        results.data = checkpoint.load_all(self.checkpoint_dir)

        total = len(self.models) * len(self.env_strategies) * len(self.severities)
        already_done = sum(
            len(scenarios.get(s, {}))
            for scenarios in results.data.values()
            for s in scenarios
        )
        if already_done:
            logger.info("Resuming: %d/%d cells already complete", already_done, total)

        pbar = tqdm(total=total, desc="Compositional Test") if self.verbose else None
        if pbar:
            pbar.update(already_done)

        for model_name in self.models:
            if self._is_model_complete(model_name, results):
                if self.verbose:
                    logger.info("Skipping completed model: %s", model_name)
                continue

            model = self.memory_mgr.load_model(model_name)

            for scenario_name, env_strategy in self.env_strategies.items():
                for severity in self.severities:
                    if checkpoint.is_completed(
                        self.checkpoint_dir, model_name, scenario_name, severity
                    ):
                        cached = checkpoint.load_cell(
                            self.checkpoint_dir, model_name, scenario_name, severity
                        )
                        if cached is not None:
                            results.add_result(model_name, scenario_name, severity, cached)
                            if pbar:
                                pbar.update(1)
                            continue

                    if pbar:
                        pbar.set_description(
                            f"{model_name}/{scenario_name}/sev={severity:.2f}"
                        )

                    result = self._evaluate_single(
                        model, model_name, scenario_name, env_strategy, severity
                    )
                    results.add_result(model_name, scenario_name, severity, result)
                    checkpoint.save_cell(
                        result, self.checkpoint_dir, model_name, scenario_name, severity
                    )

                    if pbar:
                        pbar.update(1)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if pbar:
            pbar.close()
        if self.verbose:
            results.print_summary()
        return results

    def _evaluate_single(
        self,
        model: nn.Module,
        model_name: str,
        scenario: str,
        env_strategy: Callable,
        severity: float,
    ) -> EvaluationResult:
        env_perturbed = env_strategy(self.images, severity)

        eps = self.eps_fn(severity)
        attack_fn = attacks.build(self.attack_type, eps=eps, batch_size=self.batch_size)
        final_images = attack_fn(model, env_perturbed, self.labels)

        accuracy, predictions, confidences, correct_mask, loss = self._evaluate_batch(
            model, final_images, self.labels
        )

        return EvaluationResult(
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
            metadata={"attack": self.attack_type},
        )

    def _evaluate_batch(
        self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor
    ):
        model.eval()
        all_predictions, all_confidences, all_correct = [], [], []
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size].to(self.device)
            batch_labels = labels[i : i + self.batch_size].to(self.device)

            with torch.no_grad():
                outputs = model(batch_images)
                loss = F.cross_entropy(outputs, batch_labels)
                probs = F.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probs, dim=1)
                correct = predictions == batch_labels

                all_predictions.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_correct.extend(correct.cpu().numpy())
                total_loss += loss.item()
                n_batches += 1

        all_predictions = np.array(all_predictions)
        all_confidences = np.array(all_confidences)
        all_correct = np.array(all_correct)
        accuracy = float(np.mean(all_correct))
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        return accuracy, all_predictions, all_confidences, all_correct, avg_loss

    def _is_model_complete(self, model_name: str, results: CompositionalResults) -> bool:
        if model_name not in results.data:
            return False
        expected = len(self.env_strategies) * len(self.severities)
        actual = sum(
            len(results.data[model_name].get(scenario, {}))
            for scenario in self.env_strategies
        )
        return actual >= expected

    def cleanup(self) -> None:
        self.memory_mgr.release_all()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
