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

from . import attacks, checkpoint, cost
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

    def run(self, *, confirm: bool = False) -> CompositionalResults:
        """
        Run the sweep, resuming any cells already on disk in ``checkpoint_dir``.

        Prints a rough cost estimate at start. If the *remaining* work exceeds
        ~1 hour or ~$5 of A100 time, raises unless ``confirm=True`` is passed.
        Short sweeps (debugging, smoke tests) run silently with no gate.
        """
        results = CompositionalResults()
        results.data = checkpoint.load_all(self.checkpoint_dir)

        total = len(self.models) * len(self.env_strategies) * len(self.severities)
        already_done = sum(
            len(scenarios.get(s, {}))
            for scenarios in results.data.values()
            for s in scenarios
        )
        remaining = total - already_done

        est = cost.estimate(self.attack_type, remaining, len(self.images))
        msg = cost.format_estimate(est)
        if already_done:
            msg = f"Resuming: {already_done}/{total} cells already complete. {msg}"

        if cost.is_expensive(est) and not confirm:
            raise RuntimeError(
                f"{msg}\n\n"
                f"Estimated cost exceeds the auto-confirm threshold "
                f"(>{cost.THRESHOLD_HOURS}h or >${cost.THRESHOLD_USD}). "
                f"Pass confirm=True to proceed, or pick a faster attack "
                f"(autoattack-apgd-ce, pgd) / smaller sweep."
            )
        if self.verbose:
            print(msg)
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


# ---------------------------------------------------------------------------
# robustbench_eval (v3 / Feature A entry point)
#
# Lazy-imports the optional `robustbench` package only when the user reaches
# the actual compute path. Errors and case normalization happen BEFORE the
# import attempt so users get the most actionable failure first:
#   1. invalid (dataset, threat) → ValueError (programmer error)
#   2. confirm=False             → RuntimeError with cost estimate
#   3. robustbench not installed → ImportError with install hint
# ---------------------------------------------------------------------------

# Realistic cost ranges per (dataset, threat). Numbers come from observed
# full-AutoAttack runs on A100-80GB. The range covers small (WRN-28-10) to
# large (WRN-70-16 / Swin-L) robust models.
_ROBUSTBENCH_COST_RANGES = {
    ("cifar10",  "Linf"): {"hours": "5-25", "usd": "$15-70"},
    ("imagenet", "Linf"): {"hours": "3-12", "usd": "$10-35"},
}


def _format_robustbench_cost(dataset: str, threat: str, n_samples: int) -> str:
    """Return the cost-estimate message that prefixes every robustbench_eval call.

    Spell out the attack ('full AutoAttack' with sub-attacks listed) so users
    cannot read 'AutoAttack' and assume APGD-CE alone. Show a range — the cost
    depends heavily on model size, and a single number would be misleading.
    """
    cost = _ROBUSTBENCH_COST_RANGES.get((dataset, threat), {"hours": "many", "usd": "expensive"})
    return (
        f"\nRobustBench protocol — {dataset}/{threat}:\n"
        f"  {n_samples} samples, full AutoAttack "
        f"(APGD-CE + APGD-DLR + FAB + Square)\n"
        f"  Estimated time: {cost['hours']} hours on A100 "
        f"(depends heavily on model size)\n"
        f"  Estimated cost: {cost['usd']} on cloud A100 (~$2.78/hr)\n"
    )


def _tag_robustbench_metadata(
    result: EvaluationResult, dataset: str, threat: str
) -> None:
    """Add protocol/dataset/threat tags to a result's metadata in place.

    These tags are exactly what ``validate_protocol`` looks for. Existing keys
    (e.g. ``"attack"`` set by CompositionalExperiment) are preserved.
    """
    result.metadata = {
        **(result.metadata or {}),
        "protocol": "robustbench",
        "dataset": dataset,
        "threat": threat,
    }


def robustbench_eval(
    model: nn.Module,
    dataset: str,
    threat: str,
    *,
    batch_size: int = 50,
    device: str = "cuda",
    confirm: bool = False,
    data_dir: str = "./data",
    checkpoint_dir: Optional[str] = None,
) -> EvaluationResult:
    """Evaluate ``model`` under RobustBench's strict published protocol.

    This is the entrypoint for Feature A. It pins every protocol-relevant
    parameter (n_samples, eps, attack, no env perturbation) so the resulting
    ``EvaluationResult`` can be passed to ``compare_to_leaderboard()`` without
    tripping the protocol gate.

    Expensive — full AutoAttack on the full RobustBench test split takes
    hours on an A100 and costs $10-$70 of cloud GPU time depending on model
    size. ``confirm=True`` is required so the cost is visible BEFORE compute
    begins.

    Args:
        model: the classifier to evaluate, already on the target device or
            movable to it.
        dataset: "cifar10" or "imagenet" (case-insensitive).
        threat: "Linf" (case-insensitive; others land in v3.1).
        batch_size: forwarded to CompositionalExperiment.
        device: "cuda" or "cpu". Defaults to cuda; falls back to cpu inside
            CompositionalExperiment if CUDA is unavailable.
        confirm: must be True to proceed past the cost gate.
        data_dir: where ``robustbench.data.load_*`` caches downloads.
        checkpoint_dir: where intermediate results land. Defaults to
            ``./checkpoints/robustbench_<dataset>_<threat>``.

    Returns:
        An ``EvaluationResult`` tagged with ``metadata["protocol"] = "robustbench"``
        plus dataset / threat. ``validate_protocol(result, dataset, threat)``
        will accept it.

    Raises:
        ValueError: unknown (dataset, threat) pair.
        RuntimeError: ``confirm`` is not True. Message includes the cost estimate.
        ImportError: ``robustbench`` package is not installed. Message includes
            the install command.
    """
    from . import leaderboard

    # 1. Canonicalize + look up the protocol — this raises ValueError on a
    # bad pair, BEFORE we burn user attention on the cost message.
    canonical_dataset, canonical_threat = leaderboard._canonicalize(dataset, threat)
    spec = leaderboard.get_protocol_spec(canonical_dataset, canonical_threat)

    # 2. Cost gate. Print the message AND raise if not confirmed, so the user
    # sees the same text whether confirm is True (printed) or False (raised).
    cost_msg = _format_robustbench_cost(
        canonical_dataset, canonical_threat, spec["n_samples"]
    )
    if not confirm:
        raise RuntimeError(cost_msg + "\nPass confirm=True to proceed.")
    print(cost_msg)

    # 3. Lazy-import robustbench data loaders. Importing inside the function
    # keeps the package importable without robustbench installed.
    try:
        if canonical_dataset == "cifar10":
            from robustbench.data import load_cifar10
            x_test, y_test = load_cifar10(
                n_examples=spec["n_samples"], data_dir=data_dir
            )
        elif canonical_dataset == "imagenet":
            from robustbench.data import load_imagenet
            x_test, y_test = load_imagenet(
                n_examples=spec["n_samples"], data_dir=data_dir
            )
        else:
            # _canonicalize + get_protocol_spec above should have rejected this,
            # so reaching here is an internal bug, not a user error.
            raise ValueError(
                f"Internal error: unsupported dataset {canonical_dataset!r} "
                "passed the protocol spec lookup"
            )
    except ImportError as e:
        raise ImportError(
            "robustbench_eval requires the `robustbench` package. "
            "Install with: pip install visprobe[robustbench]  "
            "(or pip install git+https://github.com/RobustBench/robustbench.git "
            "if the PyPI version is stale)."
        ) from e

    # 4. Run the experiment with pinned protocol params. The identity env
    # strategy below is critical — an empty dict would mean zero loop
    # iterations and silently produce no result.
    if checkpoint_dir is None:
        checkpoint_dir = (
            f"./checkpoints/robustbench_{canonical_dataset}_{canonical_threat.lower()}"
        )

    experiment = CompositionalExperiment(
        models={"_robustbench_target": model},
        images=x_test,
        labels=y_test,
        env_strategies={"none": lambda images, severity: images},  # identity, NOT {}
        attack=spec["attack"],
        severities=[0.0],
        eps_fn=lambda s: spec["eps"],   # fixed eps; severity does not modulate it
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        device=device,
        verbose=True,
    )
    results = experiment.run()
    experiment.cleanup()

    # 5. Pull the single result and tag its metadata so validate_protocol
    # will accept it downstream in compare_to_leaderboard (M7).
    result = results.get_result("_robustbench_target", "none", 0.0)
    if result is None:
        raise RuntimeError(
            "robustbench_eval did not produce a result — this is a bug; "
            "the experiment loop should always yield one cell."
        )
    _tag_robustbench_metadata(result, canonical_dataset, canonical_threat)
    return result
