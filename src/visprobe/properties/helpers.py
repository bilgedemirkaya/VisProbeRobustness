"""
This module provides helper functions for properties, primarily for extracting
information from model outputs in a framework-agnostic way.
"""

from typing import Any, Tuple

import torch
import torch.nn.functional as F


def get_topk_predictions(output: torch.Tensor, k: int) -> torch.Tensor:
    """
    Gets the top-k prediction indices.

    Args:
        output: The raw output tensor from the model (logits).
               Shape: [batch_size, num_classes] or [num_classes]
        k: The number of top predictions to return.

    Returns:
        Tensor of top-k indices.
        Shape: [batch_size, k] for batched input, [k] for single sample.

    Raises:
        ValueError: If k < 1 or output is empty
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if isinstance(output, tuple):
        output = output[0]
    if output.numel() == 0:
        raise ValueError("Cannot get top-k predictions from empty tensor")

    _, pred_indices = torch.topk(output, k, dim=-1)
    return pred_indices


def get_top_prediction(output: torch.Tensor) -> Tuple[int, float]:
    """Returns the top-1 class index and confidence.

    If a batch is provided, returns the top-1 for the first element to keep
    backward compatibility with call sites that expect a scalar result.
    """
    if isinstance(output, tuple):
        output = output[0]
    probabilities = F.softmax(output, dim=-1)
    if probabilities.ndim == 2:
        probabilities = probabilities[0]
    conf, idx = torch.max(probabilities, dim=-1)
    return int(idx.item()), float(conf.item())


def extract_logits(obj: Any) -> torch.Tensor:
    """
    Extracts logits tensor from flexible inputs:
    - dict with key "output"
    - tuple (logits, features)
    - raw tensor

    Raises:
        TypeError: If obj is not a supported type
    """
    if isinstance(obj, dict):
        if "output" not in obj:
            raise TypeError(f"Dict must contain 'output' key, got keys: {list(obj.keys())}")
        obj = obj["output"]
    if isinstance(obj, tuple):
        obj = obj[0]
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(obj).__name__}")
    return obj
