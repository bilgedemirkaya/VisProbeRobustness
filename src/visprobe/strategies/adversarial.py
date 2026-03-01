"""
Adversarial attack strategies using the Adversarial Robustness Toolbox (ART).

Provides gradient-based and score-based attacks for robustness testing.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional, Type

import torch
import torch.nn as nn

from .base import Strategy

logger = logging.getLogger(__name__)

# Lazy import flag for ART
_ART_AVAILABLE = False
_ART_IMPORT_ERROR: Optional[str] = None

try:
    from art.attacks.evasion import (
        AutoProjectedGradientDescent,
        BasicIterativeMethod,
        FastGradientMethod,
        ProjectedGradientDescent,
        SquareAttack,
    )
    from art.estimators.classification import PyTorchClassifier

    _ART_AVAILABLE = True
except ImportError as e:
    _ART_IMPORT_ERROR = str(e)
    # Placeholders for type hints
    PyTorchClassifier = None
    FastGradientMethod = None
    ProjectedGradientDescent = None
    BasicIterativeMethod = None
    AutoProjectedGradientDescent = None
    SquareAttack = None


def _check_art_available() -> None:
    """Raise ImportError if ART is not installed."""
    if not _ART_AVAILABLE:
        raise ImportError(
            "Adversarial strategies require the Adversarial Robustness Toolbox (ART).\n"
            "\n"
            "Install options:\n"
            "  pip install visprobe[adversarial]  # Install ART only\n"
            "  pip install visprobe[all]          # Install all optional dependencies\n"
            "\n"
            f"Original error: {_ART_IMPORT_ERROR}"
        )


def _create_art_classifier(
    model: nn.Module, imgs: torch.Tensor, loss_fn: Optional[nn.Module] = None
) -> "PyTorchClassifier":
    """
    Create a PyTorchClassifier for ART attacks.

    Args:
        model: PyTorch model (or wrapper with .model attribute)
        imgs: Sample images to infer input shape and num classes
        loss_fn: Loss function (defaults to CrossEntropyLoss)

    Returns:
        Configured PyTorchClassifier instance
    """
    _check_art_available()

    inner_model = getattr(model, "model", model)

    try:
        model_device = next(inner_model.parameters()).device
    except StopIteration:
        model_device = imgs.device

    sample = imgs[0:1].to(model_device)
    with torch.no_grad():
        output = inner_model(sample)
        if isinstance(output, tuple):
            output = output[0]

    return PyTorchClassifier(
        model=inner_model,
        loss=loss_fn or nn.CrossEntropyLoss(),
        input_shape=tuple(imgs[0].shape),
        nb_classes=output.shape[1],
        clip_values=(0.0, 1.0),
    )


class _ARTStrategyBase(Strategy):
    """
    Base class for ART-based adversarial strategies.

    Subclasses only need to set `attack_class` and optionally override
    `_get_attack_kwargs()` for attack-specific parameters.
    """

    attack_class: ClassVar[Optional[Type]] = None
    default_max_iter: ClassVar[int] = 100

    def __init__(
        self,
        eps: float,
        max_iter: Optional[int] = None,
        targeted: bool = False,
        **kwargs: Any,
    ):
        _check_art_available()
        self._cached_classifier = None
        self._cache_key = None
        self.eps = eps
        self.max_iter = max_iter if max_iter is not None else self.default_max_iter
        self.targeted = targeted
        self.art_kwargs = kwargs

    def _get_classifier(self, model: nn.Module, imgs: torch.Tensor) -> "PyTorchClassifier":
        """Get or create cached ART classifier."""
        key = (id(model), imgs.shape[1:], str(imgs.device))
        if self._cache_key != key:
            self._cached_classifier = _create_art_classifier(model, imgs)
            self._cache_key = key
        return self._cached_classifier

    def _get_attack_kwargs(self, level: Optional[float] = None) -> dict:
        """
        Build kwargs for the ART attack constructor.

        Override in subclasses that need special parameters.
        """
        return {
            "eps": level if level is not None else self.eps,
            **self.art_kwargs,
        }

    def _run_attack(self, attack, imgs: torch.Tensor) -> torch.Tensor:
        """Run ART attack and convert back to original device."""
        device = imgs.device
        adv_np = attack.generate(x=imgs.detach().cpu().numpy())
        return torch.from_numpy(adv_np).to(device)

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate adversarial examples.

        Args:
            imgs: Input images
            model: Model to attack
            level: Optional epsilon value (overrides instance value)

        Returns:
            Adversarial examples
        """
        if self.attack_class is None:
            raise NotImplementedError("Subclass must set attack_class")

        estimator = self._get_classifier(model, imgs)
        attack = self.attack_class(estimator=estimator, **self._get_attack_kwargs(level))
        return self._run_attack(attack, imgs)

    def query_cost(self) -> int:
        """Number of model queries required."""
        return self.max_iter

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps:.4f})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps}, max_iter={self.max_iter})"


class FGSMStrategy(_ARTStrategyBase):
    """
    Fast Gradient Sign Method (FGSM) attack.

    Single-step attack that perturbs inputs in the direction of the loss gradient.

    Args:
        eps: Maximum perturbation (L-inf norm), default 2/255
        targeted: If True, minimize loss for target class
    """

    attack_class = FastGradientMethod
    default_max_iter = 1

    def __init__(self, eps: float = 2 / 255, targeted: bool = False, **kwargs: Any):
        super().__init__(eps=eps, max_iter=1, targeted=targeted, **kwargs)

    def _get_attack_kwargs(self, level: Optional[float] = None) -> dict:
        return {
            "eps": level if level is not None else self.eps,
            "targeted": self.targeted,
            **self.art_kwargs,
        }


class PGDStrategy(_ARTStrategyBase):
    """
    Projected Gradient Descent (PGD) attack.

    Iterative attack that takes multiple smaller steps with projection.

    Args:
        eps: Maximum perturbation (L-inf norm)
        eps_step: Step size per iteration (default: eps/10)
        max_iter: Maximum iterations (default: 100)
    """

    attack_class = ProjectedGradientDescent
    default_max_iter = 100

    def __init__(
        self,
        eps: float,
        eps_step: Optional[float] = None,
        max_iter: int = 100,
        **kwargs: Any,
    ):
        super().__init__(eps=eps, max_iter=max_iter, **kwargs)
        self.eps_step = eps_step if eps_step is not None else eps / 10

    def _get_attack_kwargs(self, level: Optional[float] = None) -> dict:
        return {
            "eps": level if level is not None else self.eps,
            "eps_step": self.eps_step,
            "max_iter": self.max_iter,
            **self.art_kwargs,
        }


class BIMStrategy(_ARTStrategyBase):
    """
    Basic Iterative Method (BIM) attack.

    Also known as I-FGSM. Iteratively applies FGSM with smaller steps.

    Args:
        eps: Maximum perturbation (L-inf norm)
        eps_step: Step size per iteration (default: eps/max_iter)
        max_iter: Maximum iterations (default: 10)
    """

    attack_class = BasicIterativeMethod
    default_max_iter = 10

    def __init__(
        self,
        eps: float,
        eps_step: Optional[float] = None,
        max_iter: int = 10,
        **kwargs: Any,
    ):
        super().__init__(eps=eps, max_iter=max_iter, **kwargs)
        self.eps_step = eps_step if eps_step is not None else eps / max(1, max_iter)

    def _get_attack_kwargs(self, level: Optional[float] = None) -> dict:
        return {
            "eps": level if level is not None else self.eps,
            "eps_step": self.eps_step,
            "max_iter": self.max_iter,
            **self.art_kwargs,
        }


class APGDStrategy(_ARTStrategyBase):
    """
    Auto-PGD (APGD) attack.

    Adaptive step-size PGD with automatic hyperparameter tuning.

    Args:
        eps: Maximum perturbation (L-inf norm)
        max_iter: Maximum iterations (default: 100)
    """

    attack_class = AutoProjectedGradientDescent
    default_max_iter = 100

    def __init__(self, eps: float, max_iter: int = 100, **kwargs: Any):
        super().__init__(eps=eps, max_iter=max_iter, **kwargs)

    def _get_attack_kwargs(self, level: Optional[float] = None) -> dict:
        return {
            "eps": level if level is not None else self.eps,
            "max_iter": self.max_iter,
            **self.art_kwargs,
        }


class SquareAttackStrategy(_ARTStrategyBase):
    """
    Square Attack (score-based black-box attack).

    Query-efficient black-box attack using random square-shaped perturbations.
    Does not require gradients.

    Args:
        eps: Maximum perturbation (L-inf norm)
        max_iter: Maximum queries (default: 5000)
    """

    attack_class = SquareAttack
    default_max_iter = 5000

    def __init__(self, eps: float, max_iter: int = 5000, **kwargs: Any):
        super().__init__(eps=eps, max_iter=max_iter, **kwargs)

    def _get_attack_kwargs(self, level: Optional[float] = None) -> dict:
        return {
            "eps": level if level is not None else self.eps,
            "max_iter": self.max_iter,
            **self.art_kwargs,
        }
