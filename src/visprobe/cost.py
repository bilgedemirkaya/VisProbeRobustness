"""
Rough cost estimation for VisProbe sweeps.

Calibrated against the May 2026 pilot run (CIFAR-10, WRN-class models,
A100-80GB on Modal). Numbers are approximations and should be presented to
users as such -- the public API uses "estimated" / "roughly" language and
never claims precision.

The estimate is used in two places:

  - ``CompositionalExperiment.run()`` -- prints the estimate; raises if it
    exceeds ``THRESHOLD_HOURS`` or ``THRESHOLD_USD`` and ``confirm=True`` was
    not passed.
  - ``robustbench_eval()`` -- always gates on ``confirm=True`` since strict
    RobustBench evals are always expensive; this module provides the message.
"""

from typing import Dict

# Seconds per cell at n=1000 samples on a single A100-80GB.
# Calibrated against the pilot: APGD-CE on WRN-70-16-class models averaged
# ~10 min per cell. Full AutoAttack is ~3x slower (4 sub-attacks vs 1).
_SECS_PER_CELL_N1000_A100: Dict[str, float] = {
    "autoattack-standard": 1800.0,   # ~30 min/cell  (full AutoAttack)
    "autoattack-apgd-ce":   600.0,   # ~10 min/cell  (APGD-CE only)
    "pgd":                   30.0,
    "none":                   2.0,
}

# A100-80GB on-demand spot price as of the pilot (Modal, May 2026).
_A100_USD_PER_HOUR = 2.78

# Thresholds above which CompositionalExperiment.run() requires confirm=True.
# Either trips the gate -- time threshold is the binding constraint at A100
# prices; USD threshold protects users on more expensive hardware.
THRESHOLD_HOURS = 1.0
THRESHOLD_USD = 5.0


def estimate(attack: str, n_cells: int, n_samples: int) -> Dict[str, float]:
    """
    Rough estimate of total wall-clock time and dollar cost for a sweep.

    Args:
        attack: Attack-type string ("autoattack-standard", etc.).
        n_cells: Total cells in the sweep (models x scenarios x severities).
        n_samples: Number of input samples per cell.

    Returns:
        dict with keys: n_cells, secs_per_cell, total_hours, total_usd.
    """
    base = _SECS_PER_CELL_N1000_A100.get(attack, 600.0)
    secs_per_cell = base * (n_samples / 1000.0)
    total_secs = secs_per_cell * n_cells
    total_hours = total_secs / 3600.0
    total_usd = total_hours * _A100_USD_PER_HOUR
    return {
        "n_cells": float(n_cells),
        "secs_per_cell": secs_per_cell,
        "total_hours": total_hours,
        "total_usd": total_usd,
    }


def format_estimate(est: Dict[str, float], *, prefix: str = "Estimated") -> str:
    """Human-readable one-liner. Always frames the numbers as approximate."""
    hours = est["total_hours"]
    if hours < 1.0:
        duration = f"~{hours * 60:.0f} min"
    else:
        duration = f"~{hours:.1f} h"
    return (
        f"{prefix}: {int(est['n_cells'])} cells x ~{est['secs_per_cell']:.0f}s each "
        f"~= {duration} on A100, roughly ${est['total_usd']:.0f} "
        f"(reference: A100-80GB at ${_A100_USD_PER_HOUR:.2f}/hr)"
    )


def is_expensive(est: Dict[str, float]) -> bool:
    """True if the estimate exceeds either threshold."""
    return est["total_hours"] > THRESHOLD_HOURS or est["total_usd"] > THRESHOLD_USD
