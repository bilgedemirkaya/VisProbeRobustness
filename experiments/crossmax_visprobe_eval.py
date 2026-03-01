"""
CrossMax Robustness Evaluation with VisProbe — Compositional Attack Analysis.

Tests the hypothesis: CrossMax's multi-resolution defense generalises to
compositional perturbations, or whether it closes the gap with a vanilla AT model.

The paper (Fort & Lakshminarayanan, 2024) reports ~78% accuracy under standard
AutoAttack (L∞, eps=8/255) in isolation. VisProbe's adaptive threshold search
reveals the *failure boundary* under four scenarios:

  Experiment 1 — adversarial_only:
      PGD eps sweep, replicates the paper's evaluation.
      Expected: CrossMax threshold >> vanilla (paper's claim).

  Experiment 2 — lowlight_pgd:
      Fixed low-light degradation (brightness ≈ 65%) + variable PGD eps.
      Hypothesis: multi-resolution prior is tuned for standard illumination;
      low-light degrades the low-res channels more, weakening the defense.

  Experiment 3 — blur_pgd:
      Fixed Gaussian blur (sigma=1.5) + variable PGD eps.
      Hypothesis: CrossMax already applies blur-like downsampling internally,
      so input blur may be redundant — or it may destructively interfere.

  Experiment 4 — noise_pgd:
      Fixed Gaussian noise (std=0.05) + variable PGD eps.
      Hypothesis: CrossMax's training augmentation includes noise, so this
      compositional attack may be *less* effective — a positive result for
      CrossMax and an interesting methodological finding.

Usage:
  # With trained CrossMax checkpoint:
  python experiments/crossmax_visprobe_eval.py \\
      --checkpoint checkpoints/crossmax_wrn28.pt

  # Quick smoke test (untrained model, n=10 samples):
  python experiments/crossmax_visprobe_eval.py --debug

  # Against a trained AT vanilla ResNet-18:
  python experiments/crossmax_visprobe_eval.py \\
      --checkpoint checkpoints/crossmax_wrn28.pt \\
      --vanilla-checkpoint checkpoints/vanilla_resnet18.pt

Run from repo root:
  python -m experiments.crossmax_visprobe_eval
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
import torch.nn.functional as F
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as transforms

# Make crossmax_cifar importable when running as a script
sys.path.insert(0, str(Path(__file__).parent))
from crossmax_cifar import (  # noqa: E402
    CIFAR10_MEAN,
    CIFAR10_STD,
    AT_EPS,
    CrossMaxSelfEnsemble,
    Normalize,
    load_crossmax,
    pgd_attack,
)

from visprobe import search
from visprobe.strategies.adversarial import PGDStrategy
from visprobe.strategies.base import Strategy
from visprobe.strategies.blur import GaussianBlur
from visprobe.strategies.composition import Compose
from visprobe.strategies.lighting import LowLight
from visprobe.strategies.noise import GaussianNoise

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Adversarial search range: 0 → 2× paper's budget
PGD_LEVEL_LO: float = 0.0
PGD_LEVEL_HI: float = 16 / 255  # ≈ 0.063

# Fixed natural corruption severity for compositional scenarios
NATURAL_STRESS_LEVEL: float = 0.5  # maps to ~65% brightness for LowLight


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def get_test_samples(
    data_dir: str = "./data",
    n_samples: int = 500,
) -> List[Tuple[torch.Tensor, int]]:
    """
    Load CIFAR-10 test samples in [0,1] pixel space for VisProbe.

    VisProbe expects: ``List[Tuple[Tensor[3,H,W], int]]`` with images in [0,1].

    Samples are intentionally kept on CPU:
      - ART (PGDStrategy) converts to numpy internally, requiring CPU tensors.
      - VisProbe's search() with ``device="cpu"`` manages device placement.
      - Both models normalize internally, so no pre-normalization needed here.

    Args:
        data_dir:  CIFAR-10 data root directory
        n_samples: Number of samples to load

    Returns:
        List of (image_tensor [3,32,32], label_int) tuples on CPU
    """
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor(),  # [0,1], no normalization, CPU tensors
    )
    samples: List[Tuple[torch.Tensor, int]] = []
    for i in range(min(n_samples, len(dataset))):
        img, label = dataset[i]
        samples.append((img, label))
    return samples


def load_vanilla_resnet18(
    device: torch.device,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Load a ResNet-18 adapted for CIFAR-10 with internal normalization.

    The model is wrapped to accept [0,1] pixel inputs (matching
    CrossMaxSelfEnsemble's interface), applying CIFAR-10 normalization
    internally so both models are comparable in VisProbe experiments.

    Args:
        device:          Target device
        checkpoint_path: Optional path to a trained state_dict (.pt).
                         If None, returns untrained model (for debug only).

    Returns:
        Wrapped ResNet-18 in eval mode
    """

    class NormalizedResNet18(nn.Module):
        """ResNet-18 for CIFAR-10 with internal pixel-space normalization."""

        def __init__(self) -> None:
            super().__init__()
            self.normalize = Normalize(CIFAR10_MEAN, CIFAR10_STD)
            backbone = tv_models.resnet18(weights=None)
            # CIFAR adaptation: remove initial 7×7 stride, maxpool
            backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            backbone.maxpool = nn.Identity()
            backbone.fc = nn.Linear(512, 10)
            self.backbone = backbone

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone(self.normalize(x))

    model = NormalizedResNet18()

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        # Handle both raw state_dict and checkpoint dict formats
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        logger.info(f"Loaded vanilla ResNet-18 from '{checkpoint_path}'")
    else:
        logger.warning(
            "Vanilla ResNet-18 is untrained. "
            "Train with crossmax_cifar.py adapted for vanilla architecture, "
            "or results will only be meaningful for CrossMax."
        )

    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Compositional Strategy Factories
# ---------------------------------------------------------------------------
# Each factory is a Callable[[float], Strategy] passed to visprobe.search().
#
# Search variable: PGD epsilon (level ∈ [PGD_LEVEL_LO, PGD_LEVEL_HI])
# Fixed variable:  natural perturbation severity (constant across search)
#
# VisProbe calls factory(level) → Strategy, then strategy.generate(imgs, model, level).
# We use a custom Strategy subclass that IGNORES the passed level in generate()
# for the natural component, ensuring only PGD eps varies during the search.
# ---------------------------------------------------------------------------


class _FixedNaturalPGD(Strategy):
    """
    Base class for fixed-natural + variable-PGD compositional strategies.

    The natural perturbation is applied at a fixed severity (baked in at
    construction), then PGD is applied at the factory-configured eps.
    The search level passed by VisProbe to generate() is intentionally
    ignored — all variability is encoded at factory(eps) time.

    Args:
        natural_strategy: Pre-instantiated natural perturbation strategy
        natural_level:    Fixed severity for the natural perturbation
        pgd_eps:          PGD L∞ budget (set by factory at each search level)
        pgd_iter:         PGD iterations
        strategy_name:    Human-readable name for logging
    """

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
        level: Optional[float] = None,  # noqa: F841  — intentionally unused
    ) -> torch.Tensor:
        """
        Apply fixed natural perturbation then PGD at baked-in eps.

        The ``level`` argument passed by VisProbe's search engine is ignored
        here because both perturbation magnitudes are fully configured at
        factory(level) construction time.
        """
        # 1. Natural corruption at fixed severity
        imgs_corrupted = self._natural.generate(imgs, model, self._natural_level)
        # 2. Adversarial attack at factory-configured eps (level=None → use self.eps)
        return self._pgd.generate(imgs_corrupted, model, None)


def make_adversarial_only_factory(n_iter: int = 20) -> Callable[[float], PGDStrategy]:
    """
    Baseline: adversarial perturbation only. Replicates the paper's AutoAttack
    evaluation using VisProbe's adaptive threshold search.

    search level → PGD eps
    """
    def factory(eps: float) -> PGDStrategy:
        return PGDStrategy(
            eps=eps,
            eps_step=max(eps / 10, 1e-7),
            max_iter=n_iter,
        )
    return factory


def make_lowlight_pgd_factory(
    fixed_ll_level: float = NATURAL_STRESS_LEVEL,
    n_iter: int = 20,
) -> Callable[[float], _FixedNaturalPGD]:
    """
    Fixed low-light + variable adversarial eps.

    low-light level=0.5 → brightness ≈ 1 - 0.5×0.7 = 0.65 (65% of original).

    Compositional hypothesis: CrossMax's multi-resolution prior relies on the
    low-res channels retaining discriminative signal. Under low-light, the
    low-res views lose more signal (they encode coarse brightness structure),
    potentially weakening CrossMax's defensive advantage vs. vanilla AT.
    """
    def factory(eps: float) -> _FixedNaturalPGD:
        return _FixedNaturalPGD(
            natural_strategy=LowLight(),
            natural_level=fixed_ll_level,
            pgd_eps=eps,
            pgd_iter=n_iter,
            strategy_name=f"lowlight(ll={fixed_ll_level:.2f})+pgd(eps={eps:.5f})",
        )
    return factory


def make_blur_pgd_factory(
    fixed_blur_sigma: float = 1.5,
    n_iter: int = 20,
) -> Callable[[float], _FixedNaturalPGD]:
    """
    Fixed Gaussian blur + variable adversarial eps.

    CrossMax already applies blur-like downsampling internally (bilinear
    resize to 16/8/4 px then upscale). Pre-blurring the input may:
      - Constructively degrade: blurring removes information from high-res channel
      - Destructively degrade: smoothing makes the adversarial signal weaker

    sigma=1.5 on 32×32 is moderate (≈ half-pixel FWHM visible blurring).
    """
    def factory(eps: float) -> _FixedNaturalPGD:
        return _FixedNaturalPGD(
            natural_strategy=GaussianBlur(),
            natural_level=fixed_blur_sigma,
            pgd_eps=eps,
            pgd_iter=n_iter,
            strategy_name=f"blur(sigma={fixed_blur_sigma:.1f})+pgd(eps={eps:.5f})",
        )
    return factory


def make_noise_pgd_factory(
    fixed_noise_std: float = 0.05,
    n_iter: int = 20,
) -> Callable[[float], _FixedNaturalPGD]:
    """
    Fixed Gaussian noise + variable adversarial eps.

    CrossMax's multi-resolution stochastic augmentation includes Gaussian noise
    during training (noise_std ≈ 0.1). Pre-applying noise at inference time
    may be *less* effective against CrossMax than against vanilla — a finding
    that would support Fort et al.'s training-distribution robustness claim.

    std=0.05 is visually noticeable (PSNR ≈ 26 dB) but sub-pixel on 32×32.
    """
    def factory(eps: float) -> _FixedNaturalPGD:
        return _FixedNaturalPGD(
            natural_strategy=GaussianNoise(),
            natural_level=fixed_noise_std,
            pgd_eps=eps,
            pgd_iter=n_iter,
            strategy_name=f"noise(std={fixed_noise_std:.3f})+pgd(eps={eps:.5f})",
        )
    return factory


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    model: nn.Module,
    samples: List[Tuple[torch.Tensor, int]],
    strategy_factory: Callable[[float], Strategy],
    strategy_name: str,
    level_lo: float = PGD_LEVEL_LO,
    level_hi: float = PGD_LEVEL_HI,
    max_queries: int = 30,
) -> Dict:
    """
    Run one VisProbe adaptive threshold search and return the report dict.

    Key configuration notes:

    1. ``normalization=None``: both CrossMaxSelfEnsemble and the vanilla ResNet-18
       wrapper normalise internally. Passing "cifar10" here would double-normalize.

    2. ``device="cpu"``: ART (used by PGDStrategy) is numpy-based and does not
       support MPS. VisProbe's search must run on CPU regardless of training device.
       The model is temporarily moved to CPU for the search and restored afterwards.

    Args:
        model:            Model with internal normalization (pixel [0,1] input)
        samples:          List of (image, label) in [0,1] pixel space (CPU tensors)
        strategy_factory: Callable[eps → Strategy] for VisProbe threshold search
        strategy_name:    Label for report display
        level_lo:         Search lower bound (min PGD eps)
        level_hi:         Search upper bound (max PGD eps)
        max_queries:      VisProbe adaptive search budget

    Returns:
        report.to_dict() containing failure_threshold, robustness_score, etc.
    """
    logger.info(f"    strategy: {strategy_name}")

    # ART (PGDStrategy) is numpy-based and requires CPU.
    # Move model to CPU for VisProbe search; restore device afterwards.
    original_device = next(model.parameters()).device
    model.cpu()

    try:
        report = search(
            model=model,
            data=samples,
            strategy=strategy_factory,
            level_lo=level_lo,
            level_hi=level_hi,
            max_queries=max_queries,
            device="cpu",
            # normalization=None: both models normalize internally.
            # Do NOT use normalization="cifar10" here.
            normalization=None,
            class_names=CIFAR10_CLASSES,
            strategy_name=strategy_name,
            verbose=False,
        )
    finally:
        # Always restore original device even if search raises
        model.to(original_device)

    threshold = report.search.get("failure_threshold", float("nan"))
    score = report.score
    logger.info(f"    → failure threshold: {threshold:.6f} | score: {score:.1f}/100")

    return report.to_dict()


def run_all_experiments(
    crossmax_model: nn.Module,
    vanilla_model: nn.Module,
    samples: List[Tuple[torch.Tensor, int]],
    output_dir: str = "results/crossmax_eval",
    n_pgd_iter: int = 20,
    max_queries: int = 30,
) -> Dict[str, Dict[str, Dict]]:
    """
    Run the full experiment battery, comparing CrossMax vs. vanilla ResNet-18.

    For each scenario, both models are tested and the failure thresholds are
    compared. The key interpretable metric is:

      gap = crossmax_threshold − vanilla_threshold

      gap > 0: CrossMax is more robust (advantage preserved)
      gap ≈ 0: No advantage from CrossMax ensembling
      gap < 0: CrossMax is *less* robust (defense breakdown under composition)

    Args:
        crossmax_model: Trained CrossMax model (pixel-space input interface)
        vanilla_model:  Vanilla/AT ResNet-18 (same input interface)
        samples:        CIFAR-10 test samples in [0,1] pixel space
        output_dir:     Directory for individual and combined JSON results
        n_pgd_iter:     PGD iterations in adversarial strategies
        max_queries:    VisProbe search budget per experiment

    Returns:
        Nested dict: {scenario_name: {model_name: report_dict}}
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    scenarios = [
        (
            "adversarial_only",
            "Adversarial only (baseline, replicates paper)",
            make_adversarial_only_factory(n_iter=n_pgd_iter),
            PGD_LEVEL_LO,
            PGD_LEVEL_HI,
        ),
        (
            "lowlight_pgd",
            "Low-light (level=0.5) + PGD",
            make_lowlight_pgd_factory(fixed_ll_level=0.5, n_iter=n_pgd_iter),
            PGD_LEVEL_LO,
            PGD_LEVEL_HI,
        ),
        (
            "blur_pgd",
            "Gaussian blur (sigma=1.5) + PGD",
            make_blur_pgd_factory(fixed_blur_sigma=1.5, n_iter=n_pgd_iter),
            PGD_LEVEL_LO,
            PGD_LEVEL_HI,
        ),
        (
            "noise_pgd",
            "Gaussian noise (std=0.05) + PGD",
            make_noise_pgd_factory(fixed_noise_std=0.05, n_iter=n_pgd_iter),
            PGD_LEVEL_LO,
            PGD_LEVEL_HI,
        ),
    ]

    models = {
        "crossmax_wrn28": crossmax_model,
        "vanilla_resnet18": vanilla_model,
    }

    all_results: Dict[str, Dict[str, Dict]] = {}

    for scenario_id, scenario_desc, factory, lo, hi in scenarios:
        logger.info(f"\n{'='*65}")
        logger.info(f"Scenario: {scenario_desc}")
        logger.info(f"{'='*65}")
        all_results[scenario_id] = {}

        for model_name, model in models.items():
            logger.info(f"\n  Model: {model_name}")
            result = run_single_experiment(
                model=model,
                samples=samples,
                strategy_factory=factory,
                strategy_name=f"{scenario_id}_{model_name}",
                level_lo=lo,
                level_hi=hi,
                max_queries=max_queries,
            )
            all_results[scenario_id][model_name] = result

        # Save per-scenario file
        scenario_path = Path(output_dir) / f"{scenario_id}.json"
        with open(scenario_path, "w") as f:
            json.dump(all_results[scenario_id], f, indent=2, default=str)
        logger.info(f"\n  Saved: {scenario_path}")

    # Summary table
    _print_summary_table(all_results)

    # Combined results file
    combined_path = Path(output_dir) / "crossmax_comparison.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nCombined results: {combined_path}")

    return all_results


def _print_summary_table(results: Dict[str, Dict[str, Dict]]) -> None:
    """
    Print a formatted comparison table of failure thresholds.

    The 'gap' column is the most important: it shows whether CrossMax's
    advantage grows, shrinks, or reverses under compositional attacks.
    """
    SEP = "=" * 75

    print(f"\n{SEP}")
    print("  CROSSMAX vs VANILLA: FAILURE THRESHOLD (L∞ eps) BY ATTACK SCENARIO")
    print(f"  Higher threshold = more robust. Gap = CrossMax − Vanilla.")
    print(SEP)
    print(
        f"  {'Scenario':<28} {'CrossMax WRN-28':>16} {'Vanilla ResNet-18':>18} {'Gap':>9}"
    )
    print(f"  {'-'*28} {'-'*16} {'-'*18} {'-'*9}")

    for scenario, model_results in results.items():
        cm = model_results.get("crossmax_wrn28", {})
        van = model_results.get("vanilla_resnet18", {})

        cm_thresh = cm.get("search", {}).get("failure_threshold", float("nan"))
        van_thresh = van.get("search", {}).get("failure_threshold", float("nan"))

        try:
            gap = cm_thresh - van_thresh
            gap_str = f"{gap:+.5f}"
        except (TypeError, ValueError):
            gap_str = "     N/A"

        cm_str = f"{cm_thresh:.5f}" if isinstance(cm_thresh, float) else str(cm_thresh)
        van_str = f"{van_thresh:.5f}" if isinstance(van_thresh, float) else str(van_thresh)

        print(f"  {scenario:<28} {cm_str:>16} {van_str:>18} {gap_str:>9}")

    print(SEP)
    print("\n  Interpretation:")
    print("    gap >> 0 on adversarial_only: reproduces paper's L∞ robustness claim")
    print("    gap shrinks on compositional: defense does not transfer across domains")
    print("    gap < 0 on any scenario:      compositional attack reverses the defense")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VisProbe compositional robustness evaluation: CrossMax vs. vanilla",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        help="Trained CrossMax checkpoint (.pt). "
             "Omit or use --debug for a quick smoke test with untrained model.",
    )
    p.add_argument(
        "--vanilla-checkpoint",
        help="Optional trained vanilla ResNet-18 checkpoint (.pt). "
             "If omitted, uses untrained model (comparison will not be meaningful).",
    )
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output-dir", default="results/crossmax_eval")
    p.add_argument("--n-samples", type=int, default=500,
                   help="Number of CIFAR-10 test samples")
    p.add_argument("--max-queries", type=int, default=30,
                   help="VisProbe adaptive search budget per experiment")
    p.add_argument("--pgd-iter", type=int, default=20,
                   help="PGD iterations in adversarial strategies")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument(
        "--debug",
        action="store_true",
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

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Device: {device}")

    # Debug overrides
    n_samples = 10 if args.debug else args.n_samples
    max_queries = 5 if args.debug else args.max_queries
    pgd_iter = 3 if args.debug else args.pgd_iter

    if args.debug:
        logger.info("DEBUG mode: n_samples=10, max_queries=5, pgd_iter=3")

    # ── Load CrossMax model ─────────────────────────────────────────────────
    if args.checkpoint:
        crossmax_model = load_crossmax(args.checkpoint, device)
    else:
        logger.warning(
            "No --checkpoint provided. Using UNTRAINED CrossMax model. "
            "Threshold comparisons will be random — use --debug for quick API check."
        )
        crossmax_model = CrossMaxSelfEnsemble(num_classes=10).to(device)
        crossmax_model.eval()

    # ── Load vanilla baseline ───────────────────────────────────────────────
    vanilla_model = load_vanilla_resnet18(device, args.vanilla_checkpoint)

    # ── Load test samples (pixel space [0,1]) ───────────────────────────────
    logger.info(f"Loading {n_samples} CIFAR-10 test samples...")
    samples = get_test_samples(args.data_dir, n_samples)
    logger.info(f"Loaded {len(samples)} samples")

    # ── Run experiments ─────────────────────────────────────────────────────
    run_all_experiments(
        crossmax_model=crossmax_model,
        vanilla_model=vanilla_model,
        samples=samples,
        output_dir=args.output_dir,
        n_pgd_iter=pgd_iter,
        max_queries=max_queries,
    )


if __name__ == "__main__":
    main()
