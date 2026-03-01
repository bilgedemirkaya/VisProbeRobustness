#!/usr/bin/env python3
"""
Custom Model Testing Template

This example shows how to test YOUR own custom model with VisProbe.
Simply replace the model loading and data preparation sections with your own.
"""

import torch
import torch.nn as nn
from visprobe import search

# ============================================================================
# STEP 1: Load YOUR model
# ============================================================================

# Option A: Load from checkpoint
# model = YourModelClass()
# model.load_state_dict(torch.load('path/to/checkpoint.pth'))
# model.eval()

# Option B: Use a simple example model (replace with yours!)
class SimpleModel(nn.Module):
    """Example model - replace with your actual model!"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 56 * 56, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

print("Loading your custom model...")
model = SimpleModel()
model.eval()
print("✓ Model loaded")

# ============================================================================
# STEP 2: Prepare YOUR test data
# ============================================================================

# Option A: Load from your dataset
# from your_dataset import load_test_data
# test_data = load_test_data()

# Option B: Use DataLoader
# from torch.utils.data import DataLoader
# test_loader = DataLoader(your_dataset, batch_size=32)
# test_data = test_loader  # VisProbe accepts DataLoaders!

# Option C: Create list of (image, label) tuples (most flexible)
print("\nPreparing test data...")
# Replace this with your actual data!
test_images = torch.randn(50, 3, 224, 224)  # Your images
test_labels = torch.randint(0, 10, (50,))    # Your labels

test_data = [(img, int(label.item())) for img, label in zip(test_images, test_labels)]
print(f"✓ Loaded {len(test_data)} test samples")

# ============================================================================
# STEP 3: Configure normalization (IMPORTANT!)
# ============================================================================

# Use the SAME normalization your model was trained with!

# Common normalizations:
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Set your normalization here:
YOUR_MEAN = IMAGENET_MEAN  # ← Change this!
YOUR_STD = IMAGENET_STD    # ← Change this!

# ============================================================================
# STEP 4: Choose a preset based on your use case
# ============================================================================

# Available presets:
# - "standard": General robustness (brightness, blur, noise, compression + compositional)
# - "lighting": Outdoor cameras, varying daylight (brightness, contrast, gamma)
# - "blur": Motion/defocus (Gaussian blur, motion blur, compression)
# - "corruption": Lossy transmission (noise, compression, degradation)

YOUR_PRESET = "standard"  # ← Change this based on your application!

# ============================================================================
# STEP 5: Run robustness test
# ============================================================================

print(f"\nRunning robustness test with '{YOUR_PRESET}' preset...")
print("(This may take a few minutes...)\n")

report = search(
    model=model,
    data=test_data,
    preset=YOUR_PRESET,
    budget=1000,        # Increase for more thorough testing
    device="auto",      # Will use GPU if available
    mean=YOUR_MEAN,     # Your normalization
    std=YOUR_STD
)

# ============================================================================
# STEP 6: View and analyze results
# ============================================================================

print("\n" + "="*70)
print("RESULTS")
print("="*70)
report.show()

print("\n" + "="*70)
print("WHAT DO THESE RESULTS MEAN?")
print("="*70)

score = report.score

if score > 0.8:
    print(f"✅ EXCELLENT (score: {score:.1%})")
    print("   Your model is highly robust to these perturbations.")
    print("   Consider testing with other presets or more samples.")
elif score > 0.6:
    print(f"✅ GOOD (score: {score:.1%})")
    print("   Your model is reasonably robust, but has some weaknesses.")
    print("   Review the failures to identify patterns.")
elif score > 0.4:
    print(f"⚠️  MODERATE (score: {score:.1%})")
    print("   Your model has significant robustness issues.")
    print("   Consider:")
    print("   - Adding data augmentation during training")
    print("   - Retraining with exported failure cases")
    print("   - Using a more robust architecture")
else:
    print(f"❌ POOR (score: {score:.1%})")
    print("   Your model is very fragile to perturbations.")
    print("   STRONGLY RECOMMENDED:")
    print("   - Review your training data and augmentation")
    print("   - Retrain with robust training techniques")
    print("   - Consider using the exported failures as hard negatives")

# ============================================================================
# STEP 7: Export failures for improvement
# ============================================================================

if report.failures:
    print(f"\n📊 Found {len(report.failures)} failure cases")
    print("   Exporting for analysis...")

    export_path = report.export_failures(n=20)
    print(f"   ✓ Exported to: {export_path}")

    print("\n💡 HOW TO USE THESE FAILURES:")
    print("   1. Review the failure cases to find patterns")
    print("   2. Add similar examples to your training set")
    print("   3. Increase data augmentation in these areas")
    print("   4. Retrain and test again to verify improvement")

print("\n" + "="*70)
print("✅ Testing complete!")
print("="*70)

print("\n📚 NEXT STEPS:")
print("   1. Try different presets to test other failure modes")
print("   2. Increase budget for more precise threshold finding")
print("   3. Test on more samples for statistical confidence")
print("   4. Use report.summary dict for automated CI/CD checks")
