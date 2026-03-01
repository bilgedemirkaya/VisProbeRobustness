"""
CIFAR-10 Compositional Attack Protection Gap Analysis.

This script produces the "protection gap" table for your article: how much
of a robust model's adversarial advantage survives under compositional attacks.

Models (loaded from RobustBench — no training needed):
  vanilla   Standard clean-trained WRN-28-10         (~95% clean accuracy)
  robust    Gowal2020Uncovering_70_16 WRN-70-16   (~65% AutoAttack robust acc)

Optional: replace `robust` with your trained CrossMax checkpoint.

Four scenarios (VisProbe adaptive threshold search):
  1. adversarial_only   Pure PGD eps sweep         — should show large robust advantage
  2. lowlight_pgd       Fixed low-light + PGD eps  — hypothesis: advantage shrinks
  3. blur_pgd           Fixed blur    + PGD eps    — hypothesis: advantage shrinks
  4. noise_pgd          Fixed noise   + PGD eps    — hypothesis: may persist (AT covers noise)

The "protection gap" = robust_threshold − vanilla_threshold on each scenario.
A shrinking or negative gap under compositional attacks is the article finding.

Usage:
  # Path A: RobustBench models (no training, runs immediately):
  pip install robustbench
  python experiments/cifar10_protection_gap.py

  # Path B: Use your trained CrossMax checkpoint instead of RobustBench robust model:
  python experiments/cifar10_protection_gap.py \\
      --crossmax-checkpoint checkpoints/crossmax_wrn28.pt

  # Quick smoke test (10 samples, 5 queries):
  python experiments/cifar10_protection_gap.py --debug

  # Full run for article (500 samples, 40 queries per experiment):
  python experiments/cifar10_protection_gap.py --n-samples 500 --max-queries 40

Output files:
  results/protection_gap/protection_gap.json   — all thresholds + scores
  results/protection_gap/article_table.txt     — ready to paste into your paper
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent))
from crossmax_cifar import CIFAR10_MEAN, CIFAR10_STD, CrossMaxSelfEnsemble, load_crossmax  # noqa: E402

from visprobe import search
from visprobe.strategies.adversarial import PGDStrategy
from visprobe.strategies.base import Strategy
from visprobe.strategies.blur import GaussianBlur
from visprobe.strategies.lighting import LowLight
from visprobe.strategies.noise import GaussianNoise

logger = logging.getLogger(__name__)

CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Search range: 0 → 2× paper's L∞ budget
PGD_LO: float = 0.0
PGD_HI: float = 16 / 255   # ≈ 0.063

# Fixed natural corruption level for compositional scenarios
NATURAL_LEVEL: float = 0.5


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class _NormalizedWrapper(nn.Module):
    """
    Wraps a model that expects CIFAR-10 standardized input, adding internal
    normalization so it accepts raw [0,1] pixel inputs.

    This gives every model the same interface: forward(x) where x ∈ [0,1].
    VisProbe search() then uses normalization=None for all models.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)  # type: ignore[operator]


def load_robustbench_model(
    model_name: str,
    dataset: str = "cifar10",
    threat_model: str = "Linf",
) -> nn.Module:
    """
    Load a model from RobustBench and wrap it with internal normalization.

    Raises ImportError with install instructions if robustbench is missing.
    """
    try:
        from robustbench.utils import load_model as rb_load
    except ImportError as e:
        raise ImportError(
            "RobustBench is required for pre-trained models.\n"
            "Install with: pip install git+https://github.com/RobustBench/robustbench.git\n"
            f"Original error: {e}"
        ) from e

    logger.info(f"Loading '{model_name}' from RobustBench ({dataset}, {threat_model})...")
    model = rb_load(model_name=model_name, dataset=dataset, threat_model=threat_model)
    model.eval()
    # RobustBench models expect pre-normalized CIFAR-10 input.
    # Wrap to add internal normalization so they accept [0,1] pixel inputs.
    wrapped = _NormalizedWrapper(model)
    wrapped.eval()
    logger.info(f"  Loaded and wrapped: {model_name}")
    return wrapped


def load_vanilla_robustbench() -> nn.Module:
    """
    Load the standard (clean-trained) CIFAR-10 model from RobustBench.

    This is a WideResNet-28-10 trained with standard cross-entropy,
    ~95% clean accuracy, ~0% AutoAttack accuracy. Uses 'corruptions'
    threat model since Standard is listed there.
    """
    return load_robustbench_model(
        model_name="Standard",
        dataset="cifar10",
        threat_model="corruptions",
    )


def load_robust_robustbench(model_name: str = "Gowal2020Uncovering_28_10_extra") -> nn.Module:
    """
    Load a robust CIFAR-10 model from RobustBench.

    Default: Gowal2020Uncovering_28_10_extra — WideResNet-28-10 with extra data,
    ~62% AutoAttack accuracy. Same architecture as the vanilla model, making
    comparison fair and inference fast enough for local CPU runs.

    For the definitive article run (GPU recommended):
      Gowal2020Uncovering_70_16 — WRN-70-16, ~65% AutoAttack (~7× slower on CPU).
    """
    return load_robustbench_model(
        model_name=model_name,
        dataset="cifar10",
        threat_model="Linf",
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_cifar10_test_samples(
    data_dir: str = "./data",
    n_samples: int = 500,
) -> List[Tuple[torch.Tensor, int]]:
    """
    Load CIFAR-10 test samples as CPU tensors in [0,1] pixel space.

    Kept on CPU because ART (PGDStrategy) is numpy-based.
    All models accept [0,1] pixel inputs via internal normalization wrappers.
    """
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=transforms.ToTensor(),
    )
    samples = []
    for i in range(min(n_samples, len(dataset))):
        img, label = dataset[i]
        samples.append((img, label))
    logger.info(f"Loaded {len(samples)} CIFAR-10 test samples")
    return samples


def get_mutually_correct_samples(
    models: Dict[str, nn.Module],
    data_dir: str = "./data",
    n_samples: int = 500,
) -> List[Tuple[torch.Tensor, int]]:
    """
    Filter to samples that ALL models classify correctly.

    For a fair comparison: all models see only samples they can handle
    under clean conditions. If you skip this, models with lower clean
    accuracy would appear artificially robust (they already fail on clean).

    Args:
        models:    {name: model} dict, all with [0,1] pixel input interface
        data_dir:  CIFAR-10 data root
        n_samples: Target number of samples

    Returns:
        List of (image, label) tuples correctly classified by every model
    """
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=transforms.ToTensor(),
    )
    model_list = list(models.values())
    for m in model_list:
        m.eval()

    mutual: List[Tuple[torch.Tensor, int]] = []
    scanned = 0

    with torch.no_grad():
        for scanned, (img, label) in enumerate(dataset):
            if len(mutual) >= n_samples:
                break
            img_batch = img.unsqueeze(0)  # [1,3,32,32] on CPU
            all_correct = all(
                m.cpu()(img_batch).argmax().item() == label for m in model_list
            )
            if all_correct:
                mutual.append((img, label))

    logger.info(
        f"Scanned {scanned} samples, found {len(mutual)} "
        f"mutually correct across {len(models)} models"
    )
    return mutual


# ---------------------------------------------------------------------------
# Strategy factories (same as crossmax_visprobe_eval.py)
# ---------------------------------------------------------------------------

class _FixedNaturalPGD(Strategy):
    """Apply fixed natural perturbation then variable-eps PGD."""

    def __init__(
        self,
        natural_strategy: Strategy,
        natural_level: float,
        pgd_eps: float,
        pgd_iter: int = 20,
        strategy_name: str = "natural+pgd",
    ) -> None:
        super().__init__()
        self._natural = natural_strategy
        self._natural_level = natural_level
        self._pgd = PGDStrategy(
            eps=pgd_eps,
            eps_step=max(pgd_eps / 10, 1e-7),
            max_iter=pgd_iter,
        )
        self.name = strategy_name

    def generate(
        self,
        imgs: torch.Tensor,
        model: nn.Module,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        imgs_corrupted = self._natural.generate(imgs, model, self._natural_level)
        return self._pgd.generate(imgs_corrupted, model, None)


SCENARIOS: List[Tuple[str, str, Callable]] = [
    (
        "adversarial_only",
        "PGD only (replicates paper baseline)",
        lambda n_iter: (
            lambda eps: PGDStrategy(
                eps=eps, eps_step=max(eps / 10, 1e-7), max_iter=n_iter
            )
        ),
    ),
    (
        "lowlight_pgd",
        f"Low-light (fixed level={NATURAL_LEVEL}) + PGD",
        lambda n_iter: (
            lambda eps: _FixedNaturalPGD(
                LowLight(), NATURAL_LEVEL, eps, n_iter,
                f"lowlight+pgd(eps={eps:.5f})",
            )
        ),
    ),
    (
        "blur_pgd",
        f"Gaussian blur (fixed sigma=1.5) + PGD",
        lambda n_iter: (
            lambda eps: _FixedNaturalPGD(
                GaussianBlur(), 1.5, eps, n_iter,
                f"blur+pgd(eps={eps:.5f})",
            )
        ),
    ),
    (
        "noise_pgd",
        f"Gaussian noise (fixed std=0.05) + PGD",
        lambda n_iter: (
            lambda eps: _FixedNaturalPGD(
                GaussianNoise(), 0.05, eps, n_iter,
                f"noise+pgd(eps={eps:.5f})",
            )
        ),
    ),
]


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    model: nn.Module,
    samples: List[Tuple[torch.Tensor, int]],
    factory: Callable[[float], Strategy],
    strategy_name: str,
    max_queries: int = 40,
) -> Dict:
    """
    Run one VisProbe threshold search for (model, attack scenario).

    Model is temporarily moved to CPU for ART compatibility, then restored.
    normalization=None because all models normalize internally.
    """
    original_device = next(model.parameters()).device
    model.cpu()

    try:
        report = search(
            model=model,
            data=samples,
            strategy=factory,
            level_lo=PGD_LO,
            level_hi=PGD_HI,
            max_queries=max_queries,
            device="cpu",
            normalization=None,
            class_names=CIFAR10_CLASSES,
            strategy_name=strategy_name,
            verbose=False,
        )
    finally:
        model.to(original_device)

    threshold = report.metrics.get("failure_threshold", float("nan"))
    score = report.score
    logger.info(f"      eps threshold: {threshold:.5f}   score: {score:.1f}/100")
    return report.to_dict()


def run_protection_gap(
    models: Dict[str, nn.Module],
    samples: List[Tuple[torch.Tensor, int]],
    output_dir: str = "results/protection_gap",
    n_pgd_iter: int = 20,
    max_queries: int = 40,
) -> Dict[str, Dict[str, Dict]]:
    """
    Run the full 4-scenario battery across all models.

    Returns:
        {scenario_id: {model_name: report_dict}}
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Dict[str, Dict]] = {}

    for scenario_id, scenario_desc, factory_maker in SCENARIOS:
        logger.info(f"\n{'='*65}")
        logger.info(f"Scenario: {scenario_desc}")
        logger.info(f"{'='*65}")

        factory = factory_maker(n_pgd_iter)
        all_results[scenario_id] = {}

        for model_name, model in models.items():
            logger.info(f"\n  {model_name}:")
            result = run_experiment(
                model=model,
                samples=samples,
                factory=factory,
                strategy_name=f"{scenario_id}_{model_name}",
                max_queries=max_queries,
            )
            all_results[scenario_id][model_name] = result

    return all_results


# ---------------------------------------------------------------------------
# Article-ready output
# ---------------------------------------------------------------------------

def _threshold(results: Dict, scenario: str, model: str) -> float:
    try:
        # VisProbe stores the threshold in metrics, not in the search sub-dict
        return results[scenario][model]["metrics"]["failure_threshold"]
    except (KeyError, TypeError):
        return float("nan")


def _score(results: Dict, scenario: str, model: str) -> float:
    try:
        return results[scenario][model]["score"]
    except (KeyError, TypeError):
        return float("nan")


def print_article_table(
    results: Dict[str, Dict[str, Dict]],
    model_names: List[str],
    output_dir: str = "results/protection_gap",
) -> None:
    """
    Print and save the article-ready protection gap table.

    The table shows:
      - Failure threshold (eps) per model per scenario
      - Gap between robust and vanilla
      - How the gap changes from adversarial-only to compositional

    The "gap shrinkage" row is the key article finding:
      Under compositional attacks, the robust model's advantage closes by X%.
    """
    lines = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    SEP = "=" * 80
    emit(SEP)
    emit("  PROTECTION GAP ANALYSIS: COMPOSITIONAL ROBUSTNESS")
    emit("  Failure threshold = L∞ epsilon at which model drops below 90% pass rate")
    emit("  Higher threshold = more robust. Gap = Robust − Vanilla.")
    emit(SEP)

    # Header
    col_w = 14
    header = f"  {'Scenario':<28}"
    for name in model_names:
        header += f"  {name[:col_w]:>{col_w}}"
    if len(model_names) == 2:
        header += f"  {'Gap':>{col_w}}"
    emit(header)
    emit(f"  {'-'*28}" + f"  {'-'*col_w}" * (len(model_names) + (1 if len(model_names) == 2 else 0)))

    scenario_labels = {
        "adversarial_only": "Adversarial only",
        "lowlight_pgd":     "Low-light + PGD",
        "blur_pgd":         "Blur + PGD",
        "noise_pgd":        "Noise + PGD",
    }

    baseline_gap = None
    rows = {}

    for scenario_id, label in scenario_labels.items():
        thresholds = [_threshold(results, scenario_id, m) for m in model_names]
        row = f"  {label:<28}"
        for t in thresholds:
            row += f"  {t:>{col_w}.5f}" if not (t != t) else f"  {'N/A':>{col_w}}"

        gap = None
        if len(model_names) == 2:
            try:
                gap = thresholds[1] - thresholds[0]  # robust - vanilla
                row += f"  {gap:>+{col_w}.5f}"
                if scenario_id == "adversarial_only":
                    baseline_gap = gap
            except (TypeError, ValueError):
                row += f"  {'N/A':>{col_w}}"

        emit(row)
        rows[scenario_id] = {"thresholds": thresholds, "gap": gap}

    emit(SEP)

    # Gap shrinkage analysis (the key finding)
    if baseline_gap is not None and len(model_names) == 2:
        emit("\n  GAP SHRINKAGE vs ADVERSARIAL-ONLY BASELINE")
        emit(f"  {'Scenario':<28}  {'Gap':>14}  {'Shrinkage':>14}  {'% of baseline':>14}")
        emit(f"  {'-'*28}  {'-'*14}  {'-'*14}  {'-'*14}")

        for scenario_id, label in scenario_labels.items():
            gap = rows[scenario_id]["gap"]
            if gap is not None and baseline_gap is not None and baseline_gap != 0:
                shrinkage = baseline_gap - gap
                pct = (shrinkage / baseline_gap) * 100
                flag = " **" if pct > 30 else ""
                emit(
                    f"  {label:<28}  {gap:>+14.5f}  "
                    f"{shrinkage:>+14.5f}  {pct:>13.1f}%{flag}"
                )
            else:
                emit(f"  {label:<28}  {'N/A':>14}  {'N/A':>14}  {'N/A':>14}")

        emit(SEP)
        emit("\n  ** = Gap shrinks >30% relative to adversarial-only baseline.")
        emit("       This is the protection gap — the article's key finding.\n")

    # Robustness scores
    emit("\n  VISPROBE ROBUSTNESS SCORES (0-100, higher = more robust)")
    emit(f"  {'Scenario':<28}" + "".join(f"  {n[:col_w]:>{col_w}}" for n in model_names))
    emit(f"  {'-'*28}" + f"  {'-'*col_w}" * len(model_names))

    for scenario_id, label in scenario_labels.items():
        row = f"  {label:<28}"
        for name in model_names:
            s = _score(results, scenario_id, name)
            row += f"  {s:>{col_w}.1f}" if not (s != s) else f"  {'N/A':>{col_w}}"
        emit(row)

    emit(SEP)

    # Article interpretation
    emit("\n  ARTICLE INTERPRETATION")
    emit("  ─────────────────────────────────────────────────────────────────")
    emit("  1. On adversarial_only: robust model shows large advantage (expected).")
    emit("  2. Under compositional attacks: measure how much that advantage shrinks.")
    emit("  3. If gap shrinks >30%: compositional attacks expose a protection gap.")
    emit("  4. This gap is the motivation for property-based testing (VisProbe).")
    emit("  5. Standard AT evaluation (single attack) overstates real-world robustness.")
    emit(SEP)

    # Save
    table_path = Path(output_dir) / "article_table.txt"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"\nArticle table saved: {table_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CIFAR-10 compositional attack protection gap analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output-dir", default="results/protection_gap")
    p.add_argument(
        "--n-samples", type=int, default=500,
        help="Test samples per experiment (500 for article, 10 for debug)",
    )
    p.add_argument(
        "--max-queries", type=int, default=40,
        help="VisProbe search budget per experiment (40 for article, 5 for debug)",
    )
    p.add_argument(
        "--pgd-iter", type=int, default=20,
        help="PGD iterations per evaluation call",
    )
    p.add_argument(
        "--crossmax-checkpoint",
        help="Path to trained CrossMax .pt checkpoint. "
             "If provided, replaces the RobustBench robust model.",
    )
    p.add_argument(
        "--robust-model", default="Gowal2020Uncovering_28_10_extra",
        help="RobustBench model name to use as the robust baseline (default: "
             "Gowal2020Uncovering_28_10_extra, WRN-28-10, ~62%% AutoAttack). "
             "For the strongest baseline use Gowal2020Uncovering_70_16 (GPU recommended). "
             "Ignored if --crossmax-checkpoint is provided.",
    )
    p.add_argument(
        "--mutual-correct", action="store_true",
        help="Filter to samples correctly classified by BOTH models. "
             "Recommended for fair comparison; adds scanning overhead.",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Quick smoke test: n=10 samples, max_queries=5, pgd_iter=3",
    )
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args()

    if args.debug:
        args.n_samples = 10
        args.max_queries = 5
        args.pgd_iter = 3
        logger.info("DEBUG mode: n=10, queries=5, pgd_iter=3")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load vanilla model ──────────────────────────────────────────────────
    try:
        logger.info("\nLoading vanilla (Standard) CIFAR-10 model...")
        vanilla = load_vanilla_robustbench()
    except ImportError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Load robust model ───────────────────────────────────────────────────
    if args.crossmax_checkpoint:
        logger.info(f"\nLoading CrossMax checkpoint: {args.crossmax_checkpoint}")
        robust = load_crossmax(args.crossmax_checkpoint)
        # CrossMax already has internal normalization — no wrapper needed
        robust_name = "crossmax_wrn28"
    else:
        logger.info(f"\nLoading RobustBench model: {args.robust_model}")
        robust = load_robust_robustbench(model_name=args.robust_model)
        robust_name = args.robust_model.split("20")[0].lower()  # e.g. "gowal"

    models: Dict[str, nn.Module] = {
        "vanilla": vanilla,
        robust_name: robust,
    }

    # ── Load test samples ───────────────────────────────────────────────────
    if args.mutual_correct:
        logger.info("\nFinding mutually correctly classified samples...")
        samples = get_mutually_correct_samples(models, args.data_dir, args.n_samples)
    else:
        samples = get_cifar10_test_samples(args.data_dir, args.n_samples)

    if len(samples) == 0:
        logger.error("No valid samples found. Try --n-samples or remove --mutual-correct.")
        sys.exit(1)

    # ── Run protection gap experiments ──────────────────────────────────────
    logger.info(f"\nRunning {len(SCENARIOS)} scenarios × {len(models)} models...")
    results = run_protection_gap(
        models=models,
        samples=samples,
        output_dir=args.output_dir,
        n_pgd_iter=args.pgd_iter,
        max_queries=args.max_queries,
    )

    # ── Save raw results ────────────────────────────────────────────────────
    raw_path = Path(args.output_dir) / "protection_gap.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nRaw results: {raw_path}")

    # ── Print + save article table ──────────────────────────────────────────
    print_article_table(results, list(models.keys()), args.output_dir)


if __name__ == "__main__":
    main()
