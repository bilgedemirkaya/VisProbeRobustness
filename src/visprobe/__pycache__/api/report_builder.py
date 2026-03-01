"""
Report construction utilities for test results.

This module provides functions for building report components including
images, metrics, residuals, and metadata.
"""

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import torch

from .report import ImageData, PanelImage, PerturbationInfo
from .strategy_utils import StrategyConfig

logger = logging.getLogger(__name__)


class ReportBuilder:
    """Builds components for test reports."""

    @staticmethod
    def create_image_data_pair(
        clean_tensor: torch.Tensor,
        clean_logits: torch.Tensor,
        pert_tensor: Optional[torch.Tensor],
        pert_logits: Optional[torch.Tensor],
        context: Dict[str, Any],
        index: int = 0,
    ) -> Tuple[Optional[ImageData], Optional[ImageData]]:
        """
        Create pair of ImageData objects for original and perturbed images.

        Args:
            clean_tensor: Clean image tensor
            clean_logits: Clean model outputs
            pert_tensor: Perturbed image tensor (optional)
            pert_logits: Perturbed model outputs (optional)
            context: Test context dictionary
            index: Index of sample to use

        Returns:
            Tuple of (original_image, perturbed_image)
        """
        if clean_tensor is None or clean_logits is None:
            return None, None

        original_image = ImageData.from_tensors(
            tensor=clean_tensor[index],
            output=clean_logits[index],
            class_names=context["class_names"],
            mean=context["mean"],
            std=context["std"],
        )

        if pert_tensor is None or pert_logits is None:
            return original_image, None

        perturbed_image = ImageData.from_tensors(
            tensor=pert_tensor[index],
            output=pert_logits[index],
            class_names=context["class_names"],
            mean=context["mean"],
            std=context["std"],
        )

        return original_image, perturbed_image

    @staticmethod
    def create_residual_panel(
        clean_batch: torch.Tensor,
        adv_batch: Optional[torch.Tensor],
        index: int,
        denorm_fn,
    ) -> Tuple[Optional[PanelImage], Optional[Dict[str, float]]]:
        """
        Create residual visualization panel and metrics.

        Args:
            clean_batch: Clean image batch
            adv_batch: Adversarial image batch (optional)
            index: Sample index to visualize
            denorm_fn: Function to denormalize tensors

        Returns:
            Tuple of (residual_panel, residual_metrics)
        """
        if adv_batch is None:
            return None, None

        try:
            clean_px = denorm_fn(clean_batch[index : index + 1])  # [1,3,H,W] in [0,1]
            adv_px = denorm_fn(adv_batch[index : index + 1])

            diff = (adv_px - clean_px).float()
            linf_norm = diff.abs().max().item()
            l2_norm = diff.norm().item()

            # Scale for visualization
            q = 0.995
            a = torch.quantile(diff.abs().flatten(), q).item()
            a = max(a, 1e-8)
            signed = (diff / (2.0 * a) + 0.5).clamp(0.0, 1.0)

            caption = (
                f"Residual r = x_adv - x_clean (signed, q={q:.3f}). "
                f"Display scaled for visibility; L∞={linf_norm:.4f}, L2={l2_norm:.4f}"
            )
            panel = PanelImage.from_tensor(signed, caption=caption)
            metrics = {"linf_norm": linf_norm, "l2_norm": l2_norm, "scaling_factor": a}
            return panel, metrics
        except Exception as e:
            logger.debug(f"Could not build residual panel: {e}")
            return None, None

    @staticmethod
    def select_first_failing_index(
        clean_out: torch.Tensor, pert_out: Optional[torch.Tensor], default_index: int = 0
    ) -> int:
        """
        Find index of first sample where predictions differ.

        Args:
            clean_out: Clean model outputs
            pert_out: Perturbed model outputs (optional)
            default_index: Index to return if no failures found

        Returns:
            Index of first failing sample, or default_index
        """
        if pert_out is None:
            return default_index

        try:
            if pert_out.dim() == 1:
                pert_out = pert_out.unsqueeze(0)
            clean_pred = torch.argmax(clean_out, dim=1)
            pert_pred = torch.argmax(pert_out, dim=1)
            mismatches = (clean_pred != pert_pred).nonzero(as_tuple=False).view(-1)
            if mismatches.numel() > 0:
                return int(mismatches[0].item())
        except Exception as e:
            logger.debug(f"Could not find failure index: {e}")
        return default_index

    @staticmethod
    def create_perturbation_info(strategy_obj) -> Optional[PerturbationInfo]:
        """
        Create PerturbationInfo from strategy object.

        Args:
            strategy_obj: Strategy object to serialize

        Returns:
            PerturbationInfo object or None on failure
        """
        try:
            name, params = StrategyConfig.serialize(strategy_obj)
            return PerturbationInfo(name=name, params=params)
        except Exception:
            try:
                return PerturbationInfo(name=type(strategy_obj).__name__, params={})
            except Exception:
                return None

    @staticmethod
    def build_run_metadata(
        user_func, context: Dict[str, Any], strategy_obj, strength_level: Optional[float]
    ) -> Dict[str, Any]:
        """
        Build metadata dictionary for test run.

        Args:
            user_func: User's test function
            context: Test context dictionary
            strategy_obj: Strategy object used
            strength_level: Perturbation strength level

        Returns:
            Dictionary containing run metadata
        """
        device = context.get("device", str(context["batch_tensor"].device))

        # Get git commit hash
        commit_hash = None
        try:
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd())
                .decode()
                .strip()
            )
        except Exception:
            pass

        # Get torch seed
        seed = None
        try:
            seed = torch.initial_seed()
        except Exception:
            pass

        module_name = os.environ.get("VISPROBE_MODULE_NAME", user_func.__module__)

        return {
            "test_name": f"{module_name}.{user_func.__name__}",
            "commit_hash": commit_hash,
            "seed": seed,
            "device": device,
            "torch_version": torch.__version__,
            "torchvision_version": ReportBuilder._get_torchvision_version(),
            "strategy_name": strategy_obj.__class__.__name__ if strategy_obj else None,
            "strength_units": StrategyConfig.infer_strength_units(strategy_obj) if strategy_obj else None,
            "strength_level": strength_level,
        }

    @staticmethod
    def _get_torchvision_version() -> Optional[str]:
        """Get torchvision version if available."""
        try:
            import torchvision
            return torchvision.__version__
        except Exception:
            return None

    @staticmethod
    def compute_per_sample_metrics(
        clean_out: torch.Tensor,
        pert_out: torch.Tensor,
        class_names: Optional[List[str]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute per-sample metrics for given test.

        Args:
            clean_out: Clean model outputs
            pert_out: Perturbed model outputs
            class_names: Optional class names for labels
            top_k: Optional k for top-k overlap computation

        Returns:
            List of per-sample metric dictionaries
        """
        if clean_out.dim() == 1:
            clean_out = clean_out.unsqueeze(0)
        if pert_out.dim() == 1:
            pert_out = pert_out.unsqueeze(0)

        batch = clean_out.shape[0]
        per_sample: List[Dict[str, Any]] = []

        for i in range(batch):
            clean_probs = torch.softmax(clean_out[i], dim=0)
            pert_probs = torch.softmax(pert_out[i], dim=0)
            clean_label = int(torch.argmax(clean_probs).item())
            pert_label = int(torch.argmax(pert_probs).item())
            conf_drop = float(torch.max(clean_probs).item() - torch.max(pert_probs).item())

            sample: Dict[str, Any] = {
                "index": i,
                "passed": bool(clean_label == pert_label),
                "threshold_estimate": None,
                "queries": None,
                "topk_overlap": None,
                "confidence_drop": conf_drop,
                "clean_label": class_names[clean_label] if class_names else str(clean_label),
                "pert_label": class_names[pert_label] if class_names else str(pert_label),
                "trace": None,
            }

            # Compute top-k overlap if requested
            if top_k and top_k > 0:
                k = min(top_k, clean_probs.shape[0])
                _, c_top = torch.topk(clean_probs, k)
                _, p_top = torch.topk(pert_probs, k)
                sample["topk_overlap"] = int(
                    len(set(c_top.tolist()).intersection(set(p_top.tolist())))
                )

            per_sample.append(sample)

        return per_sample

    @staticmethod
    def compute_threshold_quantiles(
        per_sample_thresholds: Optional[List[Optional[float]]]
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Compute threshold quantiles from per-sample thresholds.

        Args:
            per_sample_thresholds: List of threshold values per sample

        Returns:
            Dictionary with p05, median, p95 quantiles or None
        """
        try:
            import numpy as np

            per_th = [t for t in (per_sample_thresholds or []) if t is not None]
            if not per_th:
                return None
            q05, q50, q95 = np.quantile(per_th, [0.05, 0.5, 0.95]).tolist()
            return {
                "threshold_quantiles": {"p05": float(q05), "median": float(q50), "p95": float(q95)}
            }
        except Exception:
            return None
