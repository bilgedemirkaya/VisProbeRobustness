"""
This module provides general-purpose utility classes and functions for the API.
"""

import json
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy and Torch types. This allows reports
    containing tensors to be serialized correctly.
    """

    def default(self, obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        return super().default(obj)


def to_image_space(imgs: torch.Tensor, mean, std) -> torch.Tensor:
    """
    Convert normalized tensors to image space [0,1] using mean/std.
    mean/std can be sequences of length 3 or tensors broadcastable to imgs.
    """
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, device=imgs.device, dtype=imgs.dtype)
    if not torch.is_tensor(std):
        std = torch.tensor(std, device=imgs.device, dtype=imgs.dtype)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    return imgs * std + mean


def to_model_space(imgs: torch.Tensor, mean, std) -> torch.Tensor:
    """
    Normalize image-space tensors using mean/std expected by the model.
    mean/std can be sequences of length 3 or tensors broadcastable to imgs.
    """
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, device=imgs.device, dtype=imgs.dtype)
    if not torch.is_tensor(std):
        std = torch.tensor(std, device=imgs.device, dtype=imgs.dtype)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    return (imgs - mean) / std


# ---- Dataset helpers for correct rendering ----

# CIFAR-10 normalization commonly used for training
CIFAR10_MEAN: List[float] = [0.4914, 0.4822, 0.4465]
CIFAR10_STD: List[float] = [0.2470, 0.2435, 0.2616]


def load_cifar10_label_names(meta_path: str) -> List[str]:
    """
    Loads CIFAR-10 label names from batches.meta using latin1 encoding.
    Reference: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="latin1")
    names = meta.get("label_names") or meta.get(b"label_names")
    return [str(n) for n in names]


def build_torchvision_collate_stack() -> Callable[[Any], torch.Tensor]:
    """Returns a collate_fn that stacks image tensors from (img, label) pairs."""

    def _fn(batch: List[Tuple[torch.Tensor, Any]]) -> torch.Tensor:
        return torch.stack([img for img, _ in batch])

    return _fn


def cifar10_data_source(
    dataset: Any,
    *,
    normalized: bool = False,
    meta_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> Tuple[Any, Callable[[Any], Any], Optional[List[str]], List[float], List[float]]:
    """
    Convenience helper to feed CIFAR-10 into @data_source with correct mean/std
    so images render accurately in the dashboard.

    - If your dataset transform includes Normalize(CIFAR10_MEAN, CIFAR10_STD),
      pass normalized=True. Otherwise we assume raw ToTensor() (no normalization).
    - class_names will be inferred from dataset.classes or batches.meta if provided.
    """
    # Collate: stack image tensors from (img, label) pairs
    collate_fn = build_torchvision_collate_stack()

    # Resolve class names
    if class_names is None:
        if hasattr(dataset, "classes") and isinstance(dataset.classes, list):
            class_names = [str(c) for c in dataset.classes]
        elif meta_path:
            try:
                class_names = load_cifar10_label_names(meta_path)
            except Exception:
                class_names = None

    # Mean/std for correct de-normalization in ImageData.from_tensors
    if normalized:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    return dataset, collate_fn, class_names, mean, std


# ---- Report assembly builders ----


def build_final_strategy(runner, spec: Any, level: Optional[float]):
    """Resolves the final perturbation object at the chosen level.

    - Supports factory callables: fn(level) -> Strategy
    - Falls back to Strategy.resolve(spec) and sets eps/level attribute
    """
    from ..strategies.base import Strategy as _Strategy

    final = None
    if level is not None and callable(spec):
        try:
            candidate = spec(level)
            if callable(getattr(candidate, "generate", None)):
                final = candidate
        except TypeError:
            final = None
    if final is None:
        final = _Strategy.resolve(spec)
        if hasattr(final, "eps") and level is not None:
            final.eps = float(level)
        elif hasattr(final, "level") and level is not None:
            final.level = float(level)
    return final


def build_visuals(
    runner, batch_tensor: torch.Tensor, clean_out: torch.Tensor, search_results: Dict[str, Any]
):
    """Creates original/perturbed images and residual panel; returns
    (original_img, perturbed_img, residual_panel, residual_metrics, fail_idx).
    """
    fail_idx = runner._select_first_failing_index(
        clean_out,
        search_results.get("perturbed_output"),
        default_index=int(search_results.get("first_failing_index", 0) or 0),
    )
    original_img, perturbed_img = runner._create_image_data_pair(
        batch_tensor,
        clean_out,
        search_results.get("perturbed_tensor"),
        search_results.get("perturbed_output"),
        index=fail_idx,
    )
    residual_panel, residual_metrics = runner._build_residual_panel_from_batches(
        batch_tensor, search_results.get("perturbed_tensor"), fail_idx
    )
    return original_img, perturbed_img, residual_panel, residual_metrics, fail_idx


def build_search_blocks(
    runner,
    mode: str,
    params: Dict[str, Any],
    search_results: Dict[str, Any],
    ensemble: Optional[Dict[str, float]],
):
    """Builds the search, metrics, and aggregate blocks for the report."""
    search_block = {
        "initial_level": params.get("initial_level"),
        "step": params.get("step"),
        "min_step": params.get("min_step"),
        "max_queries": params.get("max_queries"),
        "history": [
            {"level": float(e.get("level", 0.0)), "pass": bool(e.get("passed", False))}
            for e in search_results.get("path", [])
        ],
        "best_failure_level": search_results.get("failure_threshold"),
        "mode": mode,
        "level_lo": params.get("level_lo"),
        "level_hi": params.get("level_hi"),
        "num_levels": params.get("num_levels"),
        "num_samples": params.get("num_samples"),
        "reduce": params.get("reduce"),
        "per_sample_thresholds": search_results.get("per_sample_thresholds"),
    }
    metrics_block = {
        "topk_overlap": search_results.get("top_k_path"),
        "layer_cosine": ensemble,
    }
    aggregates_block = runner._compute_threshold_quantiles(
        search_results.get("per_sample_thresholds")
    )
    return search_block, metrics_block, aggregates_block
