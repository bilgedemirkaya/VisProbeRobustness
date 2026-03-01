"""
Analysis helper functions used by the TestRunner.
These are extracted to keep runner.py concise.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .model_wrap import _ModelWithIntermediateOutput
from .query_counter import QueryCounter


def run_ensemble_analysis(runner, clean_features, perturbed_features) -> Optional[Dict[str, float]]:
    """Compares intermediate layer activations if available."""
    if not clean_features or not perturbed_features:
        return None

    analysis: Dict[str, float] = {}
    for layer in runner.context.get("capture_layers", []):
        if layer not in clean_features or layer not in perturbed_features:
            continue
        a = clean_features[layer]
        b = perturbed_features[layer]
        a_flat = a.detach().flatten()
        b_flat = b.detach().flatten()
        denom = torch.norm(a_flat) * torch.norm(b_flat) + 1e-12
        sim = torch.sum(a_flat * b_flat) / denom
        analysis[layer] = float(sim.item())
    return analysis if analysis else None


def run_resolution_impact(
    runner, resolutions, perturbation_obj, clean_results
) -> Optional[Dict[str, Dict[str, float]]]:
    """Analyzes robustness across different image resolutions."""
    if not resolutions or perturbation_obj is None:
        return None
    ctx = runner.context
    m, batch_tensor = ctx["model"], ctx["batch_tensor"]
    impact: Dict[str, Dict[str, float]] = {}

    from torchvision.transforms.functional import resize  # lazy import

    for res in resolutions:
        pixel_clean = runner._denorm(batch_tensor)
        resized_pixel = resize(pixel_clean, res)
        resized_clean_tensor = runner._renorm(resized_pixel)

        m.eval()
        with QueryCounter(m) as qc:
            resized_pert_tensor = perturbation_obj.generate(resized_clean_tensor, model=m)
            clean_model_output = m(resized_clean_tensor)
            if isinstance(m, _ModelWithIntermediateOutput):
                resized_clean_out, _ = clean_model_output
            else:
                resized_clean_out = clean_model_output

            pert_model_output = m(resized_pert_tensor)
            if isinstance(m, _ModelWithIntermediateOutput):
                resized_pert_out, _ = pert_model_output
            else:
                resized_pert_out = pert_model_output

        runner.query_count += qc.extra

        clean_pred = torch.argmax(resized_clean_out, dim=1)
        pert_pred = torch.argmax(resized_pert_out, dim=1)
        passed_samples = (clean_pred == pert_pred).sum().item()
        accuracy = passed_samples / ctx["batch_size"]

        impact[f"{res[0]}x{res[1]}"] = {"accuracy": float(accuracy)}

    return impact or None


def run_noise_sweep(runner, noise_sweep_params, clean_results) -> Optional[List[Dict[str, float]]]:
    """Analyzes robustness against random Gaussian noise."""
    if not noise_sweep_params:
        return None
    ctx = runner.context
    m, batch_tensor = ctx["model"], ctx["batch_tensor"]
    clean_out, _ = clean_results

    levels = noise_sweep_params.get("levels", 10)
    min_level = noise_sweep_params.get("min_level", 0.0)
    max_level = noise_sweep_params.get("max_level", 0.5)

    results: List[Dict[str, float]] = []
    for level in np.linspace(min_level, max_level, levels):
        pixel_clean = runner._denorm(batch_tensor)
        noise = torch.randn_like(pixel_clean) * level
        noisy_pixel = torch.clamp(pixel_clean + noise, 0, 1)
        noisy_tensor = runner._renorm(noisy_pixel)

        m.eval()
        noisy_model_output = m(noisy_tensor)
        if isinstance(m, _ModelWithIntermediateOutput):
            noisy_out, _ = noisy_model_output
        else:
            noisy_out = noisy_model_output
        runner.query_count += 1

        clean_pred = torch.argmax(clean_out, dim=1)
        noisy_pred = torch.argmax(noisy_out, dim=1)
        passed_samples = (clean_pred == noisy_pred).sum().item()
        accuracy = passed_samples / ctx["batch_size"]
        results.append({"level": float(level), "accuracy": float(accuracy)})

    return results if results else None


def run_corruption_sweep(  # noqa: C901
    runner, corruptions, clean_results
) -> Optional[Dict[str, List[Dict[str, float]]]]:
    """Evaluates robustness across CIFAR-10-C style corruptions over severities 1..5."""
    try:
        if not corruptions:
            return None
        ctx = runner.context
        m, batch_tensor = ctx["model"], ctx["batch_tensor"]
        clean_out, _ = clean_results
        clean_pred = torch.argmax(clean_out, dim=1)

        pixel_clean = runner._denorm(batch_tensor)

        def apply_corruption(name: str, x: torch.Tensor, severity: int) -> torch.Tensor:
            sev = max(1, min(5, int(severity)))
            if name == "gaussian_noise":
                std_map = {1: 0.04, 2: 0.06, 3: 0.08, 4: 0.09, 5: 0.10}
                std = std_map[sev]
                out = (x + torch.randn_like(x) * std).clamp(0, 1)
                return out
            if name == "brightness":
                delta_map = {1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25}
                delta = delta_map[sev]
                out = (x + delta).clamp(0, 1)
                return out
            if name == "contrast":
                factor_map = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5}
                factor = factor_map[sev]
                mean = x.mean(dim=(2, 3), keepdim=True)
                out = (mean + (x - mean) * factor).clamp(0, 1)
                return out
            return x

        results: Dict[str, List[Dict[str, float]]] = {}
        for name in corruptions:
            rows: List[Dict[str, float]] = []
            for s in range(1, 6):
                x_corr = apply_corruption(str(name), pixel_clean, s)
                x_corr_tensor = runner._renorm(x_corr)
                m.eval()
                with QueryCounter(m) as qc:
                    out_corr = m(x_corr_tensor)
                    if isinstance(m, _ModelWithIntermediateOutput):
                        out_corr, _ = out_corr
                runner.query_count += qc.extra
                corr_pred = torch.argmax(out_corr, dim=1)
                passed_samples = (clean_pred == corr_pred).sum().item()
                acc = passed_samples / ctx["batch_size"]
                rows.append({"severity": s, "accuracy": float(acc)})
            results[str(name)] = rows
        return results or None
    except Exception:
        return None


def run_top_k_analysis(
    runner, k: Optional[int], clean_output: torch.Tensor, perturbed_output: torch.Tensor
) -> Optional[float]:
    """Calculates overlap count of top-k predictions between clean and perturbed outputs."""
    if k is None or k == 0:
        return None
    if clean_output is None or perturbed_output is None:
        return None

    if clean_output.dim() == 1:
        clean_output = clean_output.unsqueeze(0)
    if perturbed_output.dim() == 1:
        perturbed_output = perturbed_output.unsqueeze(0)

    batch_size = clean_output.shape[0]
    total_overlap_score = 0.0

    for i in range(batch_size):
        clean_probs = torch.softmax(clean_output[i], dim=0)
        pert_probs = torch.softmax(perturbed_output[i], dim=0)
        num_classes = clean_probs.shape[0]
        actual_k = min(k, num_classes)
        _, clean_top_k_indices = torch.topk(clean_probs, actual_k)
        _, pert_top_k_indices = torch.topk(pert_probs, actual_k)
        overlap_count = len(
            set(clean_top_k_indices.tolist()).intersection(set(pert_top_k_indices.tolist()))
        )
        total_overlap_score += overlap_count

    if batch_size == 1:
        return float(overlap_count)
    return total_overlap_score / batch_size
