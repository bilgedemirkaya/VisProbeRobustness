"""
CrossMax Self-Ensemble for CIFAR-10 Adversarial Training.

Reproduces the core defense from:
  Fort & Lakshminarayanan, "Ensemble Everything Everywhere:
  Multi-Scale Aggregation for Adversarial Robustness" (2024)

Architecture:
  - WideResNet-28-10 backbone with 12-channel first conv (4 multi-res inputs)
  - Linear probes at each of the 3 WRN stages + final head = 4 predictors
  - CrossMax double-normalization aggregation (top-k=3)

Training:
  - PGD-10 adversarial training (Madry et al.)
  - SGD with Nesterov momentum + cosine annealing LR
  - 100 epochs minimum for meaningful robustness (~75%+ PGD-20 on CIFAR-10)

Usage:
  # Train:
  python experiments/crossmax_cifar.py --epochs 100 --output checkpoints/crossmax_wrn28.pt

  # Eval only:
  python experiments/crossmax_cifar.py --eval --checkpoint checkpoints/crossmax_wrn28.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CIFAR-10 normalization (applied inside the model, so PGD stays in [0,1])
# ---------------------------------------------------------------------------

CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)

# ---------------------------------------------------------------------------
# Adversarial training defaults (L∞ threat model, matching Fort et al.)
# ---------------------------------------------------------------------------

AT_EPS: float = 8 / 255      # L∞ budget
AT_STEP: float = 2 / 255     # PGD step size
AT_N_ITER: int = 10          # PGD iterations during training


# ---------------------------------------------------------------------------
# In-model Normalizer
# ---------------------------------------------------------------------------

class Normalize(nn.Module):
    """
    Pixel-to-standardized normalization placed inside the model.

    Keeping normalization inside the model means adversarial attacks (PGD)
    can operate cleanly in [0, 1] pixel space without needing to know about
    the model's expected input distribution.

    Args:
        mean: Per-channel mean (C,)
        std:  Per-channel std  (C,)
    """

    def __init__(self, mean: Tuple[float, ...], std: Tuple[float, ...]) -> None:
        super().__init__()
        # Buffers move automatically with model.to(device)
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Multi-Resolution Input Transform
# ---------------------------------------------------------------------------

class MultiResolutionTransform(nn.Module):
    """
    Stacks multiple downsample→upsample views of the input, channel-wise.

    Fort & Lakshminarayanan (2024) §3: each resolution provides a progressively
    blurred view, smoothing high-frequency adversarial perturbations. The 4×3=12
    channel stack is fed to a WRN with a 12-channel first convolution.

    Args:
        resolutions: Spatial resolutions to generate (descending).
                     First entry must equal the input spatial size.

    Shape:
        Input:  [B, 3, H, W]  (already normalized)
        Output: [B, 3·N, H, W]  where N = len(resolutions)

    Example:
        >>> t = MultiResolutionTransform((32, 16, 8, 4))
        >>> out = t(torch.rand(2, 3, 32, 32))
        >>> out.shape
        torch.Size([2, 12, 32, 32])
    """

    def __init__(self, resolutions: Tuple[int, ...] = (32, 16, 8, 4)) -> None:
        super().__init__()
        self.resolutions = resolutions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        channels: List[torch.Tensor] = []

        for r in self.resolutions:
            if r == h and r == w:
                channels.append(x)
            else:
                # Downsample then upsample back to original resolution
                downsampled = F.interpolate(
                    x, size=(r, r), mode="bilinear", align_corners=False
                )
                upsampled = F.interpolate(
                    downsampled, size=(h, w), mode="bilinear", align_corners=False
                )
                channels.append(upsampled)

        return torch.cat(channels, dim=1)  # [B, 3·N, H, W]


# ---------------------------------------------------------------------------
# WideResNet-28-10
# ---------------------------------------------------------------------------

class _WRNBlock(nn.Module):
    """Pre-activation WideResNet residual block."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Shortcut: project when dimensions change
        self.shortcut: nn.Module
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    """
    WideResNet (Zagoruyko & Komodakis, 2016) adapted for CIFAR (32×32 inputs).

    Modified to accept a variable number of input channels (`in_channels`) to
    support multi-resolution channel stacking (12 channels for 4 resolutions).

    Args:
        depth:        Network depth. Must satisfy (depth - 4) % 6 == 0.
        widen_factor: Channel width multiplier.
        dropout_rate: Dropout inside residual blocks.
        num_classes:  Output classes (10 for CIFAR-10, 100 for CIFAR-100).
        in_channels:  Input channels (3 for standard, 12 for 4-scale multi-res).
    """

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        dropout_rate: float = 0.0,
        num_classes: int = 10,
        in_channels: int = 12,
    ) -> None:
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must satisfy (depth-4) % 6 == 0"
        n = (depth - 4) // 6

        # Channel widths: initial → stage1 → stage2 → stage3
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv0 = nn.Conv2d(
            in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.stage1 = self._make_stage(widths[0], widths[1], n, stride=1, dropout_rate=dropout_rate)
        self.stage2 = self._make_stage(widths[1], widths[2], n, stride=2, dropout_rate=dropout_rate)
        self.stage3 = self._make_stage(widths[2], widths[3], n, stride=2, dropout_rate=dropout_rate)

        self.bn_final = nn.BatchNorm2d(widths[3])
        self.relu_final = nn.ReLU(inplace=True)
        self.fc = nn.Linear(widths[3], num_classes)

        # Expose for probe registration in CrossMaxSelfEnsemble
        self.stage_channels: Tuple[int, int, int] = (widths[1], widths[2], widths[3])

        self._init_weights()

    def _make_stage(
        self,
        in_planes: int,
        out_planes: int,
        num_blocks: int,
        stride: int,
        dropout_rate: float,
    ) -> nn.Sequential:
        layers = [_WRNBlock(in_planes, out_planes, stride, dropout_rate)]
        for _ in range(num_blocks - 1):
            layers.append(_WRNBlock(out_planes, out_planes, 1, dropout_rate))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward_stages(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass, returning each stage's feature map.

        Returns:
            (stage1_feats, stage2_feats, stage3_feats, head_logits)
            where stage_i_feats are pre-pooled spatial tensors.
        """
        out = self.conv0(x)
        f1 = self.stage1(out)                                          # [B, 160, 32, 32]
        f2 = self.stage2(f1)                                           # [B, 320, 16, 16]
        f3 = self.stage3(f2)                                           # [B, 640,  8,  8]
        f_final = self.relu_final(self.bn_final(f3))                   # BN + ReLU before pool
        logits = self.fc(F.adaptive_avg_pool2d(f_final, 1).flatten(1)) # [B, num_classes]
        return f1, f2, f3, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, logits = self.forward_stages(x)
        return logits


# ---------------------------------------------------------------------------
# CrossMax Aggregation
# ---------------------------------------------------------------------------

def crossmax(
    logits_stack: torch.Tensor,
    k: int = 3,
    use_median: bool = False,
) -> torch.Tensor:
    """
    CrossMax robust aggregation (Fort & Lakshminarayanan, 2024, Algorithm 1).

    Double-normalization eliminates per-predictor scale bias and per-class
    systematic advantage before aggregating the top-k predictors.

    Steps:
      1. Row-max subtraction:    z_ic ← z_ic − max_c(z_ic)   [predictor normalization]
      2. Column-max subtraction: z_ic ← z_ic − max_i(z_ic)   [class normalization]
      3. Top-k mean aggregation (or median for independent ensembles)

    Args:
        logits_stack: Raw predictor logits [B, N, C]
        k:            Top-k predictors for self-ensemble aggregation
        use_median:   Use median instead of top-k (for independent model ensembles)

    Returns:
        Aggregated logits [B, C]
    """
    # Step 1: subtract per-predictor max (row-wise normalization)
    z = logits_stack - logits_stack.max(dim=2, keepdim=True).values

    # Step 2: subtract per-class max (column-wise normalization)
    z = z - z.max(dim=1, keepdim=True).values

    # Step 3: aggregate
    if use_median:
        return z.median(dim=1).values
    else:
        k_actual = min(k, z.shape[1])
        topk_vals, _ = z.topk(k_actual, dim=1, largest=True)
        return topk_vals.mean(dim=1)


# ---------------------------------------------------------------------------
# CrossMax Self-Ensemble (full model)
# ---------------------------------------------------------------------------

class CrossMaxSelfEnsemble(nn.Module):
    """
    CrossMax Self-Ensemble: WideResNet-28-10 with multi-resolution input
    and intermediate linear probes aggregated via CrossMax.

    Full forward pipeline:
      1. Normalize      [0,1] pixel → standardized (CIFAR-10 mean/std)
      2. MultiResolutionTransform  [B, 3, 32, 32] → [B, 12, 32, 32]
      3. WideResNet backbone       3 stages of residual blocks
      4. Linear probes at each stage + final head = 4 predictors
      5. CrossMax aggregation      [B, 4, C] → [B, C]

    VisProbe interface note:
      This model accepts [B, 3, 32, 32] pixel-space input in [0, 1].
      Pass ``normalization=None`` to visprobe.search() since normalization
      is handled internally here.

    Args:
        num_classes:   Number of output classes (10 for CIFAR-10)
        wrn_depth:     WideResNet depth (28)
        wrn_width:     WideResNet widen_factor (10)
        dropout_rate:  Dropout in WRN blocks (use 0.3 if underfitting)
        crossmax_k:    Top-k for CrossMax aggregation
        resolutions:   Multi-resolution input scales

    Example:
        >>> model = CrossMaxSelfEnsemble(num_classes=10)
        >>> x = torch.rand(4, 3, 32, 32)   # CIFAR-10 batch in [0,1]
        >>> logits = model(x)              # [4, 10]
    """

    def __init__(
        self,
        num_classes: int = 10,
        wrn_depth: int = 28,
        wrn_width: int = 10,
        dropout_rate: float = 0.0,
        crossmax_k: int = 3,
        resolutions: Tuple[int, ...] = (32, 16, 8, 4),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.crossmax_k = crossmax_k
        self.wrn_depth = wrn_depth
        self.wrn_width = wrn_width

        n_res = len(resolutions)

        # Internal normalization (pixel → standardized)
        self.normalize = Normalize(CIFAR10_MEAN, CIFAR10_STD)

        # Multi-resolution channel stacking
        self.multi_res = MultiResolutionTransform(resolutions=resolutions)

        # WRN backbone with 12-channel first conv
        self.backbone = WideResNet(
            depth=wrn_depth,
            widen_factor=wrn_width,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            in_channels=3 * n_res,
        )

        # Intermediate linear probes (one per WRN stage)
        # WRN-28-10 stage channels: (160, 320, 640)
        ch1, ch2, ch3 = self.backbone.stage_channels
        self.probe1 = nn.Linear(ch1, num_classes)  # After stage 1
        self.probe2 = nn.Linear(ch2, num_classes)  # After stage 2
        self.probe3 = nn.Linear(ch3, num_classes)  # After stage 3 (pre-BN/ReLU)
        # probe4 = backbone.fc  (after stage 3 BN+ReLU)

    @staticmethod
    def _global_avg_pool(x: torch.Tensor) -> torch.Tensor:
        """Global average pool spatial tensor to [B, C]."""
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images in pixel space [B, 3, H, W], values in [0, 1].

        Returns:
            CrossMax-aggregated class logits [B, num_classes].
        """
        # 1. Standardize
        x_norm = self.normalize(x)

        # 2. Multi-resolution channel expansion
        x_multi = self.multi_res(x_norm)

        # 3. Backbone forward, capturing intermediate stages
        f1, f2, f3, logits_head = self.backbone.forward_stages(x_multi)

        # 4. Probe each stage via global avg pool + linear
        logits1 = self.probe1(self._global_avg_pool(f1))  # Stage 1 features
        logits2 = self.probe2(self._global_avg_pool(f2))  # Stage 2 features
        logits3 = self.probe3(self._global_avg_pool(f3))  # Stage 3 features (pre-BN)
        # logits_head: stage 3 features after BN+ReLU (backbone.fc)

        # 5. Stack and CrossMax aggregate: [B, 4, num_classes]
        logits_stack = torch.stack(
            [logits1, logits2, logits3, logits_head], dim=1
        )
        return crossmax(logits_stack, k=self.crossmax_k)


# ---------------------------------------------------------------------------
# PGD Adversarial Attack (pure PyTorch, no ART dependency for training)
# ---------------------------------------------------------------------------

def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = AT_EPS,
    eps_step: float = AT_STEP,
    n_iter: int = AT_N_ITER,
    random_start: bool = True,
) -> torch.Tensor:
    """
    Untargeted L∞ PGD attack in pixel space [0, 1].

    The model is expected to handle normalization internally. This attack
    perturbs in [0, 1] and the model receives perturbed pixel-space inputs.

    Args:
        model:        Model accepting [0,1] inputs (normalizes internally)
        images:       Clean images [B, 3, H, W] in [0, 1]
        labels:       Ground-truth labels [B]
        eps:          L∞ perturbation budget (default 8/255)
        eps_step:     PGD step size (default 2/255)
        n_iter:       PGD iterations (default 10)
        random_start: Initialize from random noise (recommended True for AT)

    Returns:
        Adversarial examples [B, 3, H, W] in [0, 1]
    """
    images = images.detach()
    loss_fn = nn.CrossEntropyLoss()

    if random_start:
        delta = torch.empty_like(images).uniform_(-eps, eps)
        x_adv = torch.clamp(images + delta, 0.0, 1.0).requires_grad_(True)
    else:
        x_adv = images.clone().requires_grad_(True)

    for _ in range(n_iter):
        logits = model(x_adv)
        loss = loss_fn(logits, labels)
        loss.backward()

        with torch.no_grad():
            # Gradient sign update
            x_adv = x_adv + eps_step * x_adv.grad.sign()
            # Project to L∞ ball centered on original images
            x_adv = torch.clamp(x_adv, images - eps, images + eps)
            # Clip to valid pixel range
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv.detach()


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 train and test DataLoaders.

    Returns images in [0, 1] pixel space with NO normalization applied —
    CrossMaxSelfEnsemble normalizes internally, and training PGD operates
    in [0, 1] space.

    Training augmentation: random crop (pad=4) + horizontal flip.

    Args:
        data_dir:    Directory for CIFAR-10 download/cache
        batch_size:  Batch size
        num_workers: DataLoader worker count

    Returns:
        (train_loader, test_loader)
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),   # → [0, 1], C×H×W
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_clean(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Clean accuracy over the full loader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return correct / total


def evaluate_pgd(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = AT_EPS,
    n_iter: int = 20,
    n_samples: int = 1000,
) -> float:
    """
    PGD-{n_iter} adversarial accuracy on up to n_samples test examples.

    Uses more iterations than training (20 vs 10) for a more reliable estimate.
    For a full AutoAttack evaluation, use the crossmax_visprobe_eval.py script.

    Args:
        model:     Model in eval mode
        loader:    Test DataLoader
        device:    Computation device
        eps:       L∞ budget (default 8/255)
        n_iter:    PGD iterations (20 recommended for eval)
        n_samples: Max samples to evaluate

    Returns:
        Adversarial accuracy in [0, 1]
    """
    model.eval()
    correct = total = 0

    for images, labels in loader:
        if total >= n_samples:
            break
        images, labels = images.to(device), labels.to(device)

        adv_images = pgd_attack(
            model, images, labels,
            eps=eps, eps_step=eps / 4, n_iter=n_iter,
        )
        with torch.no_grad():
            preds = model(adv_images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return correct / total


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    output_path: str = "checkpoints/crossmax_wrn28.pt",
    eval_interval: int = 10,
    max_batches: Optional[int] = None,
    at_pgd_iter: int = AT_N_ITER,
) -> None:
    """
    PGD adversarial training for CrossMaxSelfEnsemble.

    Training strategy:
      - SGD with Nesterov momentum
      - Cosine annealing LR schedule (T_max=epochs)
      - PGD-{at_pgd_iter} (L∞, eps=8/255) adversarial examples
      - Gradient clipping (max_norm=1.0) for stability
      - Best checkpoint saved by PGD-20 adversarial accuracy

    Args:
        model:         CrossMaxSelfEnsemble (untrained)
        train_loader:  CIFAR-10 training DataLoader
        test_loader:   CIFAR-10 test DataLoader
        device:        Computation device
        epochs:        Training epochs (100+ for meaningful robustness)
        lr:            Initial learning rate (0.1 standard for SGD+CIFAR)
        weight_decay:  L2 regularization weight
        momentum:      SGD momentum
        output_path:   Best checkpoint save path
        eval_interval: Evaluate every N epochs
        max_batches:   Cap batches per epoch (None = full epoch). Use 2-5 for
                       quick smoke tests without running the full dataset.
        at_pgd_iter:   PGD iterations for adversarial training (default 10).
                       Use 2-3 to speed up smoke tests on local hardware.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_pgd_acc = 0.0
    training_log: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples.
            # Brief eval mode for PGD avoids dropout/BN stochasticity during attack.
            model.eval()
            adv_images = pgd_attack(
                model, images, labels,
                eps=AT_EPS, eps_step=AT_STEP, n_iter=at_pgd_iter,
            )
            model.train()

            optimizer.zero_grad()
            logits = model(adv_images)
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(labels)
            epoch_correct += (logits.argmax(dim=1) == labels).sum().item()
            epoch_total += len(labels)

        scheduler.step()

        avg_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Loss {avg_loss:.4f} | Train {train_acc:.4f} | LR {current_lr:.6f}"
        )

        if epoch % eval_interval == 0 or epoch == epochs:
            clean_acc = evaluate_clean(model, test_loader, device)
            pgd_acc = evaluate_pgd(model, test_loader, device)

            logger.info(
                f"  >> Eval {epoch}: Clean={clean_acc:.4f}  PGD-20={pgd_acc:.4f}"
            )
            training_log.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "clean_acc": clean_acc,
                "pgd20_acc": pgd_acc,
            })

            if pgd_acc > best_pgd_acc:
                best_pgd_acc = pgd_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "clean_acc": clean_acc,
                        "pgd20_acc": pgd_acc,
                        "config": {
                            "num_classes": model.num_classes,
                            "crossmax_k": model.crossmax_k,
                            "wrn_depth": model.wrn_depth,
                            "wrn_width": model.wrn_width,
                        },
                    },
                    output_path,
                )
                logger.info(f"  >> Checkpoint saved (PGD-20={pgd_acc:.4f})")

    log_path = str(output_path).replace(".pt", "_training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    logger.info(f"\nTraining complete. Best PGD-20: {best_pgd_acc:.4f}")
    logger.info(f"Checkpoint: {output_path}")
    logger.info(f"Training log: {log_path}")


# ---------------------------------------------------------------------------
# Checkpoint Loading
# ---------------------------------------------------------------------------

def load_crossmax(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> CrossMaxSelfEnsemble:
    """
    Load a trained CrossMaxSelfEnsemble from a checkpoint file.

    Args:
        checkpoint_path: Path to .pt checkpoint saved by train()
        device:          Target device (auto-selected if None)

    Returns:
        CrossMaxSelfEnsemble in eval mode, moved to device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})

    model = CrossMaxSelfEnsemble(
        num_classes=cfg.get("num_classes", 10),
        crossmax_k=cfg.get("crossmax_k", 3),
        wrn_depth=cfg.get("wrn_depth", 28),
        wrn_width=cfg.get("wrn_width", 10),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(
        f"Loaded checkpoint '{checkpoint_path}' "
        f"(epoch={checkpoint.get('epoch', '?')}, "
        f"clean={checkpoint.get('clean_acc', '?'):.4f}, "
        f"pgd20={checkpoint.get('pgd20_acc', '?'):.4f})"
    )
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CrossMax CIFAR-10 adversarial training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output", default="checkpoints/crossmax_wrn28.pt")
    p.add_argument("--epochs", type=int, default=100,
                   help="Training epochs (100 minimum for ~75%+ PGD-20)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--wrn-depth", type=int, default=28)
    p.add_argument("--wrn-width", type=int, default=10)
    p.add_argument("--crossmax-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-batches", type=int, default=None,
        help="Cap batches per epoch. Use 2-5 for a quick smoke test on local hardware.",
    )
    p.add_argument(
        "--at-pgd-iter", type=int, default=AT_N_ITER,
        help="PGD iterations during adversarial training (default 10). "
             "Use 2-3 for quick local tests.",
    )
    # Eval-only mode
    p.add_argument("--eval", action="store_true",
                   help="Evaluate existing checkpoint instead of training")
    p.add_argument("--checkpoint", help="Checkpoint path for --eval mode")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    # ── Eval-only mode ──────────────────────────────────────────────────────
    if args.eval:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required with --eval")
        _, test_loader = get_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers)
        model = load_crossmax(args.checkpoint, device)

        clean = evaluate_clean(model, test_loader, device)
        logger.info(f"Clean accuracy:   {clean:.4f}  ({clean * 100:.2f}%)")

        pgd = evaluate_pgd(model, test_loader, device, eps=AT_EPS, n_iter=20)
        logger.info(f"PGD-20 accuracy:  {pgd:.4f}  ({pgd * 100:.2f}%)")
        return

    # ── Training mode ────────────────────────────────────────────────────────
    train_loader, test_loader = get_cifar10_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    model = CrossMaxSelfEnsemble(
        num_classes=10,
        wrn_depth=args.wrn_depth,
        wrn_width=args.wrn_width,
        crossmax_k=args.crossmax_k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"CrossMax WRN-{args.wrn_depth}-{args.wrn_width}: {n_params:,} parameters"
    )

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_path=args.output,
        eval_interval=args.eval_interval,
        max_batches=args.max_batches,
        at_pgd_iter=args.at_pgd_iter,
    )


if __name__ == "__main__":
    main()
