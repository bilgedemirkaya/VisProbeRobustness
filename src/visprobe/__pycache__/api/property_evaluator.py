"""
Property evaluation logic for robustness testing.

This module handles the evaluation of robustness properties on clean and perturbed
model outputs, supporting both vectorized and per-sample evaluation modes.
"""

from typing import Callable, List, Tuple

import torch


class PropertyEvaluator:
    """Evaluates robustness properties on model outputs."""

    @staticmethod
    def evaluate_count(
        user_func: Callable,
        clean_logits: torch.Tensor,
        pert_logits: torch.Tensor,
        batch_size: int,
        vectorized: bool = False,
    ) -> int:
        """
        Evaluate property and return count of passed samples.

        Args:
            user_func: User's test function
            clean_logits: Clean model outputs
            pert_logits: Perturbed model outputs
            batch_size: Number of samples in batch
            vectorized: Whether to evaluate in batched mode

        Returns:
            Number of samples that passed the property test
        """
        if vectorized:
            try:
                user_func({"output": clean_logits}, {"output": pert_logits})
                return batch_size
            except AssertionError:
                return 0

        # Per-sample evaluation
        passed = 0
        clean_logits = PropertyEvaluator._ensure_batch_dim(clean_logits)
        pert_logits = PropertyEvaluator._ensure_batch_dim(pert_logits)

        for i in range(batch_size):
            try:
                user_func(
                    {"output": clean_logits[i : i + 1]}, {"output": pert_logits[i : i + 1]}
                )
                passed += 1
            except AssertionError:
                pass
        return passed

    @staticmethod
    def evaluate_mask(
        user_func: Callable, clean_logits: torch.Tensor, pert_logits: torch.Tensor
    ) -> List[bool]:
        """
        Evaluate property and return per-sample pass/fail mask.

        Args:
            user_func: User's test function
            clean_logits: Clean model outputs
            pert_logits: Perturbed model outputs

        Returns:
            List of boolean values indicating pass (True) or fail (False) per sample
        """
        clean_logits = PropertyEvaluator._ensure_batch_dim(clean_logits)
        pert_logits = PropertyEvaluator._ensure_batch_dim(pert_logits)

        mask: List[bool] = []
        for i in range(clean_logits.shape[0]):
            try:
                user_func(
                    {"output": clean_logits[i : i + 1]}, {"output": pert_logits[i : i + 1]}
                )
                mask.append(True)
            except AssertionError:
                mask.append(False)
        return mask

    @staticmethod
    def evaluate_mask_from_results(
        user_func: Callable, clean_results: Tuple, pert_results: Tuple, batch_size: int
    ) -> List[bool]:
        """
        Evaluate property from (output, features) tuples.

        Args:
            user_func: User's test function
            clean_results: Tuple of (clean_output, clean_features)
            pert_results: Tuple of (pert_output, pert_features)
            batch_size: Number of samples in batch

        Returns:
            List of boolean values indicating pass (True) or fail (False) per sample
        """
        clean_out, _ = clean_results
        pert_out, _ = pert_results

        mask: List[bool] = []
        for i in range(batch_size):
            try:
                user_func({"output": clean_out[i : i + 1]}, {"output": pert_out[i : i + 1]})
                mask.append(True)
            except AssertionError:
                mask.append(False)
        return mask

    @staticmethod
    def _ensure_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has batch dimension."""
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def extract_logits(output):
        """Extract logits from output (handles both tensor and tuple formats)."""
        return output[0] if isinstance(output, tuple) else output
