"""
Attack builders for VisProbe.

Public entry point: ``build(attack_type, eps, **kwargs)`` returns a callable with
signature ``(model, images, labels) -> perturbed_images``. Each call to build
produces a fresh closure; there is no shared state.

Supported attack types:
    - "none"               : identity (no perturbation)
    - "pgd"                : projected gradient descent
    - "autoattack-standard": full AutoAttack (APGD-CE + APGD-DLR + FAB + Square)
    - "autoattack-apgd-ce" : AutoAttack restricted to APGD-CE (~5x faster)
"""

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def build(attack_type: str, eps: float, **kwargs) -> Callable:
    """Construct an attack callable from a string name and an epsilon."""
    if attack_type == "none" or eps < 1e-10:
        return _identity
    if attack_type == "pgd":
        return _make_pgd(eps, **kwargs)
    if attack_type == "autoattack-standard":
        return _make_autoattack(eps, version="standard", **kwargs)
    if attack_type == "autoattack-apgd-ce":
        return _make_autoattack(
            eps, version="custom", attacks_to_run=["apgd-ce"], **kwargs
        )
    raise ValueError(f"Unknown attack: {attack_type!r}")


def _identity(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return images


def _make_autoattack(
    eps: float,
    *,
    version: str = "standard",
    attacks_to_run: Optional[list] = None,
    norm: str = "Linf",
    verbose: bool = False,
    batch_size: int = 50,
) -> Callable:
    try:
        from autoattack import AutoAttack
    except ImportError as e:
        raise ImportError(
            "AutoAttack not installed. "
            "Install with: pip install git+https://github.com/fra31/auto-attack"
        ) from e

    def attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        aa_kwargs = dict(model=model, norm=norm, eps=eps, version=version, verbose=verbose)
        if attacks_to_run is not None:
            aa_kwargs["attacks_to_run"] = attacks_to_run
        aa = AutoAttack(**aa_kwargs)
        # Modern AutoAttack stores eps on each sub-attack, not on the wrapper.
        for sub_name in ("apgd", "apgd_targeted", "fab", "square"):
            sub = getattr(aa, sub_name, None)
            if sub is not None:
                sub.eps = eps
        if hasattr(aa, "apgd"):
            aa.apgd.bs = batch_size
        with torch.no_grad():
            return aa.run_standard_evaluation(images, labels, bs=batch_size)

    return attack


def _make_pgd(
    eps: float,
    *,
    norm: str = "Linf",
    eps_step: Optional[float] = None,
    max_iter: int = 20,
    random_start: bool = True,
    batch_size: int = 50,
) -> Callable:
    step = eps_step if eps_step is not None else eps / 4

    def attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        model.eval()
        x_adv = images.clone().detach()

        if random_start:
            if norm == "Linf":
                x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
            else:  # L2
                noise = torch.randn_like(x_adv)
                noise = noise / noise.view(len(x_adv), -1).norm(2, 1).view(-1, 1, 1, 1)
                x_adv = x_adv + eps * noise
            x_adv = torch.clamp(x_adv, 0, 1)

        for _ in range(max_iter):
            x_adv.requires_grad = True
            outputs = model(x_adv)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            with torch.no_grad():
                if norm == "Linf":
                    x_adv = x_adv + step * x_adv.grad.sign()
                    x_adv = torch.max(torch.min(x_adv, images + eps), images - eps)
                else:  # L2
                    grad_norm = x_adv.grad.view(len(x_adv), -1).norm(2, 1).view(-1, 1, 1, 1)
                    x_adv = x_adv + step * x_adv.grad / (grad_norm + 1e-10)
                    delta = x_adv - images
                    delta_norm = delta.view(len(delta), -1).norm(2, 1).view(-1, 1, 1, 1)
                    factor = torch.min(torch.ones_like(delta_norm), eps / (delta_norm + 1e-10))
                    x_adv = images + delta * factor
                x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv.detach()

    return attack
