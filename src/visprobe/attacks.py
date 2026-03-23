"""
First-class attack integration for VisProbe.
Provides AutoAttack and PGD as built-in, optimized attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try importing AutoAttack
try:
    from autoattack import AutoAttack
    AUTOATTACK_AVAILABLE = True
except ImportError:
    AUTOATTACK_AVAILABLE = False
    logger.warning("AutoAttack not installed. Install with: pip install autoattack")


class AttackFactory:
    """Factory for creating attack functions."""

    @staticmethod
    def create(
        attack_type: str,
        eps: float,
        norm: str = "Linf",
        **kwargs
    ) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Create an attack function.

        Args:
            attack_type: Type of attack (autoattack-standard, autoattack-apgd-ce, pgd, none)
            eps: Epsilon value for attack
            norm: Norm type (Linf, L2)
            **kwargs: Additional attack-specific parameters

        Returns:
            Attack function with signature (model, images, labels) -> perturbed_images
        """
        if attack_type == "none":
            return NoAttack()

        if attack_type.startswith("autoattack"):
            if not AUTOATTACK_AVAILABLE:
                raise ImportError("AutoAttack not installed. Use 'pip install autoattack'")

            if attack_type == "autoattack-standard":
                return AutoAttackStandard(eps, norm, **kwargs)
            elif attack_type == "autoattack-apgd-ce":
                return AutoAttackAPGD(eps, norm, **kwargs)
            elif attack_type == "autoattack-apgd-dlr":
                return AutoAttackAPGDDLR(eps, norm, **kwargs)
            else:
                raise ValueError(f"Unknown AutoAttack variant: {attack_type}")

        elif attack_type == "pgd":
            return PGDAttack(eps, norm=norm, **kwargs)

        else:
            raise ValueError(f"Unknown attack type: {attack_type}")


class NoAttack:
    """No attack - returns original images."""

    def __call__(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        return images


class AutoAttackStandard:
    """
    Standard AutoAttack with all 4 attacks.

    Includes: APGD-CE, APGD-DLR, FAB, Square Attack
    """

    def __init__(
        self,
        eps: float,
        norm: str = "Linf",
        version: str = "standard",
        verbose: bool = False,
        batch_size: int = 50
    ):
        """
        Initialize AutoAttack.

        Args:
            eps: Epsilon for attack
            norm: Norm type (Linf or L2)
            version: AutoAttack version
            verbose: Whether to print progress
            batch_size: Batch size for attack
        """
        self.eps = eps
        self.norm = norm
        self.version = version
        self.verbose = verbose
        self.batch_size = batch_size
        self._aa_cache = {}  # Cache AutoAttack instances per model

    def __call__(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Run AutoAttack."""
        # Skip if epsilon is too small
        if self.eps < 1e-10:
            return images

        # Get or create AutoAttack instance for this model
        model_id = id(model)
        if model_id not in self._aa_cache:
            logger.info(f"Creating AutoAttack instance for model (eps={self.eps})")
            self._aa_cache[model_id] = AutoAttack(
                model,
                norm=self.norm,
                eps=self.eps,
                version=self.version,
                verbose=self.verbose
            )
            # Set batch size
            if hasattr(self._aa_cache[model_id], 'apgd'):
                self._aa_cache[model_id].apgd.bs = self.batch_size

        aa = self._aa_cache[model_id]

        # Update epsilon if changed
        if aa.eps != self.eps:
            aa.eps = self.eps
            # Update for all sub-attacks
            if hasattr(aa, 'apgd'):
                aa.apgd.eps = self.eps
            if hasattr(aa, 'apgd_targeted'):
                aa.apgd_targeted.eps = self.eps
            if hasattr(aa, 'fab'):
                aa.fab.eps = self.eps
            if hasattr(aa, 'square'):
                aa.square.eps = self.eps

        # Run attack
        with torch.no_grad():
            x_adv = aa.run_standard(images, labels, bs=self.batch_size)

        return x_adv


class AutoAttackAPGD:
    """
    APGD-CE only attack from AutoAttack.

    ~5x faster than standard AutoAttack, recommended for initial sweeps.
    """

    def __init__(
        self,
        eps: float,
        norm: str = "Linf",
        verbose: bool = False,
        n_iter: int = 100,
        batch_size: int = 50
    ):
        """
        Initialize APGD-CE attack.

        Args:
            eps: Epsilon for attack
            norm: Norm type
            verbose: Whether to print progress
            n_iter: Number of iterations
            batch_size: Batch size for attack
        """
        self.eps = eps
        self.norm = norm
        self.verbose = verbose
        self.n_iter = n_iter
        self.batch_size = batch_size
        self._aa_cache = {}

    def __call__(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Run APGD-CE attack."""
        # Skip if epsilon is too small
        if self.eps < 1e-10:
            return images

        # Get or create AutoAttack instance
        model_id = id(model)
        if model_id not in self._aa_cache:
            logger.info(f"Creating APGD-CE instance for model (eps={self.eps})")
            self._aa_cache[model_id] = AutoAttack(
                model,
                norm=self.norm,
                eps=self.eps,
                version='custom',
                attacks_to_run=['apgd-ce'],
                verbose=self.verbose
            )
            # Configure APGD parameters
            if hasattr(self._aa_cache[model_id], 'apgd'):
                self._aa_cache[model_id].apgd.n_iter = self.n_iter
                self._aa_cache[model_id].apgd.bs = self.batch_size

        aa = self._aa_cache[model_id]

        # Update epsilon if changed
        if aa.eps != self.eps:
            aa.eps = self.eps
            if hasattr(aa, 'apgd'):
                aa.apgd.eps = self.eps

        # Run attack
        with torch.no_grad():
            x_adv = aa.run_standard(images, labels, bs=self.batch_size)

        return x_adv


class AutoAttackAPGDDLR:
    """APGD-DLR only attack from AutoAttack."""

    def __init__(
        self,
        eps: float,
        norm: str = "Linf",
        verbose: bool = False,
        n_iter: int = 100,
        batch_size: int = 50
    ):
        self.eps = eps
        self.norm = norm
        self.verbose = verbose
        self.n_iter = n_iter
        self.batch_size = batch_size
        self._aa_cache = {}

    def __call__(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Run APGD-DLR attack."""
        if self.eps < 1e-10:
            return images

        model_id = id(model)
        if model_id not in self._aa_cache:
            logger.info(f"Creating APGD-DLR instance for model (eps={self.eps})")
            self._aa_cache[model_id] = AutoAttack(
                model,
                norm=self.norm,
                eps=self.eps,
                version='custom',
                attacks_to_run=['apgd-dlr'],
                verbose=self.verbose
            )

        aa = self._aa_cache[model_id]
        if aa.eps != self.eps:
            aa.eps = self.eps

        with torch.no_grad():
            x_adv = aa.run_standard(images, labels, bs=self.batch_size)

        return x_adv


class PGDAttack:
    """
    Projected Gradient Descent attack.

    Fast, simple adversarial attack for testing.
    """

    def __init__(
        self,
        eps: float,
        norm: str = "Linf",
        eps_step: Optional[float] = None,
        max_iter: int = 20,
        random_start: bool = True,
        batch_size: int = 50
    ):
        """
        Initialize PGD attack.

        Args:
            eps: Maximum perturbation
            norm: Norm type (Linf or L2)
            eps_step: Step size (default: eps/4)
            max_iter: Number of iterations
            random_start: Whether to start from random perturbation
            batch_size: Batch size for attack
        """
        self.eps = eps
        self.norm = norm
        self.eps_step = eps_step if eps_step is not None else eps / 4
        self.max_iter = max_iter
        self.random_start = random_start
        self.batch_size = batch_size

    def __call__(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Run PGD attack."""
        # Skip if epsilon is too small
        if self.eps < 1e-10:
            return images

        # Ensure model is in eval mode
        model.eval()

        # Clone images to avoid modifying originals
        x_adv = images.clone().detach()

        # Random start
        if self.random_start:
            if self.norm == "Linf":
                x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            else:  # L2
                random_noise = torch.randn_like(x_adv)
                random_noise = random_noise / random_noise.view(len(x_adv), -1).norm(2, 1).view(-1, 1, 1, 1)
                x_adv = x_adv + self.eps * random_noise

            x_adv = torch.clamp(x_adv, 0, 1)

        # PGD iterations
        for _ in range(self.max_iter):
            x_adv.requires_grad = True

            # Forward pass
            outputs = model(x_adv)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass
            loss.backward()

            # Generate perturbation
            with torch.no_grad():
                if self.norm == "Linf":
                    # Linf attack
                    x_adv = x_adv + self.eps_step * x_adv.grad.sign()
                    # Project back to epsilon ball
                    x_adv = torch.max(torch.min(x_adv, images + self.eps), images - self.eps)
                else:  # L2
                    # L2 attack
                    grad_norm = x_adv.grad.view(len(x_adv), -1).norm(2, 1).view(-1, 1, 1, 1)
                    x_adv = x_adv + self.eps_step * x_adv.grad / (grad_norm + 1e-10)
                    # Project back to epsilon ball
                    delta = x_adv - images
                    delta_norm = delta.view(len(delta), -1).norm(2, 1).view(-1, 1, 1, 1)
                    factor = torch.min(torch.ones_like(delta_norm), self.eps / (delta_norm + 1e-10))
                    x_adv = images + delta * factor

                # Clamp to valid range
                x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv.detach()


def create_adaptive_attack(
    fast_attack: str = "autoattack-apgd-ce",
    full_attack: str = "autoattack-standard",
    crossover_threshold: float = 0.5,
    full_validation_severities: Optional[list] = None,
    **kwargs
) -> Callable:
    """
    Create an adaptive attack that uses fast mode for exploration
    and full mode for validation at critical points.

    Args:
        fast_attack: Attack type for initial exploration
        full_attack: Attack type for thorough validation
        crossover_threshold: Accuracy threshold to trigger full validation
        full_validation_severities: Specific severities for full validation
        **kwargs: Additional attack parameters

    Returns:
        Adaptive attack function
    """

    class AdaptiveAttack:
        def __init__(self):
            self.fast_attacker = AttackFactory.create(fast_attack, eps=0.0, **kwargs)
            self.full_attacker = AttackFactory.create(full_attack, eps=0.0, **kwargs)
            self.results_cache = {}
            self.full_severities = full_validation_severities or [0.0, 1.0]

        def __call__(
            self,
            model: nn.Module,
            images: torch.Tensor,
            labels: torch.Tensor,
            eps: float,
            severity: Optional[float] = None
        ) -> torch.Tensor:
            # Determine which attack to use
            use_full = False

            if severity is not None:
                # Use full attack for specific severities
                if severity in self.full_severities:
                    use_full = True

                # Use full attack if near crossover (based on cached results)
                model_id = id(model)
                if model_id in self.results_cache:
                    prev_accs = self.results_cache[model_id]
                    if len(prev_accs) > 0:
                        last_acc = prev_accs[-1]
                        if abs(last_acc - crossover_threshold) < 0.1:
                            use_full = True

            # Select and run attack
            if use_full:
                logger.info(f"Using full attack at severity={severity}, eps={eps}")
                attacker = self.full_attacker
            else:
                attacker = self.fast_attacker

            # Update epsilon
            if hasattr(attacker, 'eps'):
                attacker.eps = eps

            return attacker(model, images, labels)

    return AdaptiveAttack()