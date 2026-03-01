#!/usr/bin/env python3
"""
Simple preset testing for ImageNet models.

Tests comprehensive preset (natural + adversarial) on ImageNet data.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from visprobe import search

# Smart device selection (ART needs CPU on MPS systems)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "cpu"  # Use CPU for comprehensive (has adversarial attacks)
else:
    device = "cpu"

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

with torch.no_grad():
    for i in range(len(dataset)):
        img, label = dataset[i]
        pred = model(img.unsqueeze(0).to(device)).argmax().item()

        if pred == label:
            test_data.append((img, label))

        if len(test_data) >= 100:
            break

print(f"Loaded {len(test_data)} correctly-classified samples\n")

# Test comprehensive preset (natural + adversarial)
report = search(
    model=model,
    data=test_data,
    preset="comprehensive",  # Tests both natural and adversarial
    device=device,
    normalization="imagenet",
)

# Print results
print(f"\n{'='*60}")
print(f"Preset: comprehensive")
print(f"{'='*60}")
print(f"Overall Score: {report.score:.1f}%")
print(f"Failed Samples: {report.failed_samples}/{report.total_samples}")

# Save and print report path
report_path = report.save()
print(f"\nReport saved to:\n  {report_path}")
