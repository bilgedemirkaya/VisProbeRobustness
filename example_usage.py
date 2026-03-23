"""
Example usage of the refactored VisProbe library.

This demonstrates how to use the new simplified API for compositional
robustness testing with automatic checkpointing and memory management.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageNet

# Import the new VisProbe API
from visprobe import (
    CompositionalExperiment,
    CompositionalResults,
    get_standard_perturbations,
    get_minimal_perturbations
)


def main():
    """Main example demonstrating VisProbe usage."""

    # =========================================================================
    # 1. PREPARE MODELS AND DATA
    # =========================================================================

    print("Loading models...")

    # Load multiple models for comparison
    models_dict = {
        "resnet50": models.resnet50(pretrained=True),
        "vit_b_16": models.vit_b_16(pretrained=True),
        # Add more models as needed
    }

    # Ensure models are in eval mode
    for model in models_dict.values():
        model.eval()

    print(f"Loaded {len(models_dict)} models")

    # Load ImageNet validation data (or any dataset)
    print("Loading data...")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # For demonstration, we'll use a subset of ImageNet
    # In practice, replace this with your dataset
    try:
        dataset = ImageNet(
            root="./data",
            split="val",
            transform=transform
        )
        # Use subset for faster testing
        subset_indices = list(range(500))  # First 500 images
        dataset = Subset(dataset, subset_indices)
    except:
        # Fallback to synthetic data for demonstration
        print("Using synthetic data for demonstration")
        images = torch.randn(100, 3, 224, 224)
        labels = torch.randint(0, 1000, (100,))

    # Extract images and labels (for simplicity in this example)
    if 'dataset' in locals():
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        images_list, labels_list = [], []
        for imgs, lbls in dataloader:
            images_list.append(imgs)
            labels_list.append(lbls)
        images = torch.cat(images_list)
        labels = torch.cat(labels_list)

    print(f"Data shape: {images.shape}")

    # =========================================================================
    # 2. BASIC USAGE: Simple compositional test
    # =========================================================================

    print("\n" + "="*60)
    print("EXAMPLE 1: Basic compositional test")
    print("="*60)

    # Create experiment with minimal perturbations
    experiment = CompositionalExperiment(
        models=models_dict,
        images=images,
        labels=labels,
        env_strategies=get_minimal_perturbations(),  # Just blur, noise, lowlight
        attack="pgd",  # Fast PGD attack
        severities=[0.0, 0.5, 1.0],  # Just 3 severity levels for speed
        eps_fn=lambda s: (4/255) * s,  # Smaller epsilon for testing
        checkpoint_dir="./demo_checkpoints",
        batch_size=50,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Run the experiment (auto-saves checkpoints)
    print("Running experiment...")
    results = experiment.run()

    # Print summary
    results.print_summary()

    # Save results
    results.save("./demo_results")
    print("Results saved to ./demo_results")

    # =========================================================================
    # 3. ADVANCED USAGE: Full compositional test with AutoAttack
    # =========================================================================

    print("\n" + "="*60)
    print("EXAMPLE 2: Full test with AutoAttack")
    print("="*60)

    # Create experiment with all standard perturbations
    full_experiment = CompositionalExperiment(
        models=models_dict,
        images=images[:50],  # Smaller subset for AutoAttack (slower)
        labels=labels[:50],
        env_strategies=get_standard_perturbations(),  # All perturbations
        attack="autoattack-apgd-ce",  # Fast AutoAttack variant
        severities=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Full severity range
        eps_fn=lambda s: (8/255) * s,  # Standard epsilon mapping
        checkpoint_dir="./full_checkpoints",
        batch_size=25,
        verbose=True
    )

    # Run with automatic checkpointing
    print("Running full experiment (this will take longer)...")
    full_results = full_experiment.run()

    # =========================================================================
    # 4. ANALYSIS: Working with results
    # =========================================================================

    print("\n" + "="*60)
    print("EXAMPLE 3: Analyzing results")
    print("="*60)

    # Protection gap analysis
    if len(models_dict) > 1:
        baseline_model = list(models_dict.keys())[0]
        gaps = full_results.protection_gap(baseline=baseline_model)
        print(f"\nProtection gaps relative to {baseline_model}:")
        for model, model_gaps in gaps.items():
            for scenario, scenario_gaps in model_gaps.items():
                avg_gap = sum(scenario_gaps.values()) / len(scenario_gaps) if scenario_gaps else 0
                print(f"  {model}/{scenario}: {avg_gap:.3f}")

    # Crossover detection
    crossovers = full_results.crossover_detection(
        baseline=list(models_dict.keys())[0],
        threshold=0.5
    )
    print("\nCrossover points (accuracy < 0.5):")
    for model, model_crossovers in crossovers.items():
        for scenario, severity in model_crossovers.items():
            if severity is not None:
                print(f"  {model}/{scenario}: severity={severity:.2f}")

    # Confidence analysis for specific evaluation
    if full_results.get_models() and full_results.get_scenarios():
        model = full_results.get_models()[0]
        scenario = full_results.get_scenarios()[0]
        severity = 0.4
        try:
            confidence_stats = full_results.confidence_profile(model, scenario, severity)
            print(f"\nConfidence stats for {model}/{scenario}/severity={severity}:")
            for key, value in confidence_stats.items():
                print(f"  {key}: {value:.3f}")
        except:
            pass

    # Model disagreement
    disagreement = full_results.disagreement_analysis()
    print(f"\nModel disagreement rate: {disagreement['disagreement_rate']:.3f}")

    # Visualization
    print("\nGenerating plots...")
    full_results.plot_compositional(save_path="./compositional_plot.png")

    # =========================================================================
    # 5. RESUMPTION: Demonstrate checkpoint recovery
    # =========================================================================

    print("\n" + "="*60)
    print("EXAMPLE 4: Resuming from checkpoint")
    print("="*60)

    # Simulate interruption by creating a new experiment with same ID
    # It will automatically detect and resume from checkpoints
    resumed_experiment = CompositionalExperiment(
        models=models_dict,
        images=images[:50],
        labels=labels[:50],
        env_strategies=get_standard_perturbations(),
        attack="autoattack-apgd-ce",
        severities=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        checkpoint_dir="./full_checkpoints",  # Same checkpoint dir
        experiment_id="resumed_exp"  # Can specify ID for consistency
    )

    print("Resuming experiment (will skip completed evaluations)...")
    resumed_results = resumed_experiment.run()
    print("Resume complete!")

    # =========================================================================
    # 6. LOADING FROM DISK: Work without models or GPU
    # =========================================================================

    print("\n" + "="*60)
    print("EXAMPLE 5: Loading and analyzing saved results")
    print("="*60)

    # Load previously saved results
    loaded_results = CompositionalResults.load("./demo_results")
    print("Loaded results from disk")

    # Can perform all analyses without models or GPU
    loaded_results.print_summary()

    # Compute AUC for robustness curves
    for model in loaded_results.get_models():
        for scenario in loaded_results.get_scenarios(model):
            auc = loaded_results.compute_auc(model, scenario)
            print(f"AUC for {model}/{scenario}: {auc:.3f}")

    # =========================================================================
    # 7. CUSTOM PERTURBATIONS: Define your own
    # =========================================================================

    print("\n" + "="*60)
    print("EXAMPLE 6: Custom perturbations")
    print("="*60)

    from visprobe import GaussianBlur, GaussianNoise, Compose

    # Create custom perturbation combinations
    custom_perturbations = {
        "strong_blur": GaussianBlur(sigma_max=5.0),
        "heavy_noise": GaussianNoise(std_max=0.2),
        "blur_and_noise": Compose([
            GaussianBlur(sigma_max=2.0),
            GaussianNoise(std_max=0.1)
        ])
    }

    custom_experiment = CompositionalExperiment(
        models={"test_model": list(models_dict.values())[0]},
        images=images[:20],
        labels=labels[:20],
        env_strategies=custom_perturbations,
        attack="none",  # No attack, just environmental
        severities=[0.5, 1.0],
        checkpoint_dir="./custom_checkpoints"
    )

    print("Running custom perturbation test...")
    custom_results = custom_experiment.run()
    custom_results.print_summary()

    # =========================================================================
    # 8. QUICK TEST: Single evaluation
    # =========================================================================

    print("\n" + "="*60)
    print("EXAMPLE 7: Quick single test")
    print("="*60)

    from visprobe import quick_test

    # Quick test with single model and severity
    quick_metrics = quick_test(
        model=list(models_dict.values())[0],
        images=images[:10],
        labels=labels[:10],
        attack="pgd",
        severity=0.5
    )

    print("Quick test results:")
    for metric, value in quick_metrics.items():
        print(f"  {metric}: {value:.3f}")

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()