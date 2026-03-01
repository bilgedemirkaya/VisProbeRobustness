#!/usr/bin/env python3
"""
Simple robustness testing for ImageNet models.

Tests model robustness against:
- Gaussian blur (natural perturbation)
- FGSM attack (adversarial perturbation)
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from visprobe import search

# Smart device selection
# CUDA: works with everything
# MPS: works for blur, not for ART-based attacks
# CPU: fallback
if torch.cuda.is_available():
    device = "cuda"
    fgsm_device = "cuda"  # ART supports CUDA
elif torch.backends.mps.is_available():
    device = "mps"
    fgsm_device = "cpu"  # ART doesn't support MPS
else:
    device = "cpu"
    fgsm_device = "cpu"

# Load model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Load ImageNet data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageFolder(root="/Users/bilgedemirkaya/imagenet_data/val", transform=transform)

# Get 100 correctly-classified samples
test_data = []
model.to(device)
class_counts = {}

with torch.no_grad():
    for i in range(len(dataset)):
        img, label = dataset[i]
        pred = model(img.unsqueeze(0).to(device)).argmax().item()

        if pred == label:
            test_data.append((img, label))
            class_counts[label] = class_counts.get(label, 0) + 1

        if len(test_data) >= 100:
            break

print(f"Loaded {len(test_data)} correctly-classified samples")
print(f"Class distribution: {len(class_counts)} unique classes")
print(f"Top 5 classes: {sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")

# Test 1: Gaussian Blur (natural perturbation)
blur_report = search(
    model=model,
    data=test_data,
    perturbation="gaussian_blur",
    device=device,
    normalization="imagenet",
)

print(f"\nGaussian Blur:")
print(f"  Robustness: {blur_report.metrics['overall_robustness_score']:.1%}")
print(f"  Failure threshold: σ={blur_report.metrics['failure_threshold']:.2f}")
print(f"  Report: {blur_report.save()}")

# Test 2: FGSM Attack (adversarial perturbation)
from visprobe.strategies.adversarial import FGSMStrategy

if fgsm_device != device:
    model.to(fgsm_device)  # Move to FGSM-compatible device

fgsm_report = search(
    model=model,
    data=test_data,
    strategy=lambda eps: FGSMStrategy(eps=eps),
    level_lo=0.0,
    level_hi=0.03,
    device=fgsm_device,
    normalization="imagenet",
)

print(f"\nFGSM Attack:")
print(f"  Robustness: {fgsm_report.metrics['overall_robustness_score']:.1%}")
print(f"  Failure threshold: ε={fgsm_report.metrics['failure_threshold']:.4f}")
print(f"  Report: {fgsm_report.save()}")
