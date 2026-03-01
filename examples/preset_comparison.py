#!/usr/bin/env python3
"""
Preset Comparison Example

This example compares all 4 presets to understand your model's
robustness across different perturbation types.
"""

import torch
import torchvision.models as models
from visprobe import search, presets

print("="*70)
print("VisProbe Preset Comparison")
print("="*70)

# 1. Load model
print("\n1. Loading model...")
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()
print("   ✓ Model: ResNet-18 (ImageNet pretrained)")

# 2. Create test data
print("\n2. Creating test data...")
num_samples = 50  # Use more for production testing
test_images = torch.randn(num_samples, 3, 224, 224)
test_labels = torch.randint(0, 1000, (num_samples,))
test_data = [(img, int(label.item())) for img, label in zip(test_images, test_labels)]
print(f"   ✓ Created {num_samples} test samples")

# 3. List available presets
print("\n3. Available presets:")
for name, description in presets.list_presets():
    print(f"   • {name:12s}: {description}")

# 4. Test with all presets
print("\n4. Running tests with all presets...")
print("   (This will take a few minutes...)\n")

results = {}
preset_names = ["standard", "lighting", "blur", "corruption"]

for preset_name in preset_names:
    print(f"   Testing with '{preset_name}' preset...")

    report = search(
        model=model,
        data=test_data,
        preset=preset_name,
        budget=500,  # Smaller budget for faster comparison
        device="auto"
    )

    results[preset_name] = {
        'score': report.score,
        'failures': len(report.failures),
        'runtime': report.summary['runtime_sec'],
        'queries': report.summary['model_queries']
    }

    print(f"      → Score: {report.score:.1%}, Failures: {len(report.failures)}, "
          f"Time: {report.summary['runtime_sec']:.1f}s")

# 5. Compare results
print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)

print("\n┌─────────────┬───────────┬──────────┬──────────┬─────────┐")
print("│ Preset      │   Score   │ Failures │ Runtime  │ Queries │")
print("├─────────────┼───────────┼──────────┼──────────┼─────────┤")

for preset_name in preset_names:
    r = results[preset_name]
    print(f"│ {preset_name:11s} │ {r['score']:7.1%}   │ {r['failures']:8d} │ "
          f"{r['runtime']:6.1f}s  │ {r['queries']:7d} │")

print("└─────────────┴───────────┴──────────┴──────────┴─────────┘")

# 6. Identify weaknesses
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

# Find weakest preset
weakest_preset = min(results.items(), key=lambda x: x[1]['score'])
strongest_preset = max(results.items(), key=lambda x: x[1]['score'])

print(f"\n🎯 WEAKEST AREA: {weakest_preset[0].upper()}")
print(f"   Score: {weakest_preset[1]['score']:.1%}")
print(f"   → Your model struggles most with {weakest_preset[0]} perturbations")

if weakest_preset[0] == "lighting":
    print("   💡 Recommendation: Add brightness/contrast augmentation during training")
elif weakest_preset[0] == "blur":
    print("   💡 Recommendation: Add blur/compression augmentation during training")
elif weakest_preset[0] == "corruption":
    print("   💡 Recommendation: Add noise/corruption augmentation during training")
elif weakest_preset[0] == "standard":
    print("   💡 Recommendation: The compositional perturbations are challenging!")
    print("      Consider testing individual perturbations to find specific weaknesses")

print(f"\n✅ STRONGEST AREA: {strongest_preset[0].upper()}")
print(f"   Score: {strongest_preset[1]['score']:.1%}")
print(f"   → Your model is most robust to {strongest_preset[0]} perturbations")

# 7. Overall assessment
overall_score = sum(r['score'] for r in results.values()) / len(results)
print(f"\n📊 OVERALL ROBUSTNESS: {overall_score:.1%}")

if overall_score > 0.8:
    print("   ✅ Excellent! Your model is highly robust across all perturbation types.")
elif overall_score > 0.6:
    print("   ✅ Good! Your model is reasonably robust, with some room for improvement.")
elif overall_score > 0.4:
    print("   ⚠️  Moderate. Your model has significant robustness gaps.")
else:
    print("   ❌ Poor. Your model is fragile and needs robustness improvements.")

# 8. Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("\n1. FOCUS AREAS:")
# Sort by score (ascending) to show weakest first
sorted_presets = sorted(results.items(), key=lambda x: x[1]['score'])
for i, (preset_name, r) in enumerate(sorted_presets[:2], 1):
    print(f"   {i}. Improve {preset_name} robustness (current: {r['score']:.1%})")

print("\n2. NEXT STEPS:")
print("   • Run individual presets with higher budget for precise thresholds")
print("   • Export failures from weakest presets for retraining")
print("   • Add targeted data augmentation for weak areas")
print("   • Re-test after improvements to verify gains")

print("\n3. PRODUCTION DEPLOYMENT:")
if overall_score > 0.7:
    print("   ✅ Model is ready for deployment with monitoring")
else:
    print("   ⚠️  Consider improving robustness before deployment")
    print("   → Use exported failures as test cases in CI/CD")

print("\n" + "="*70)
print("✅ Comparison complete!")
print("="*70)
