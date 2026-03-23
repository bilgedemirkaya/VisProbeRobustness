"""
Comprehensive example showing all VisProbe features.
"""

import torch
import torchvision.models as models
from visprobe import (
    CompositionalExperiment,
    CompositionalResults,
    get_standard_perturbations,
    GaussianBlur,
    GaussianNoise,
    Compose
)


def main():
    # =========================================================================
    # 1. Multi-model comparison with automatic memory management
    # =========================================================================

    print("Loading models...")
    models_dict = {
        "resnet50": models.resnet50(pretrained=True),
        "vit_b_16": models.vit_b_16(pretrained=True),
    }

    for model in models_dict.values():
        model.eval()

    # Load your data
    images = torch.randn(200, 3, 224, 224)  # Replace with real data
    labels = torch.randint(0, 1000, (200,))

    # =========================================================================
    # 2. Run experiment with automatic checkpointing
    # =========================================================================

    experiment = CompositionalExperiment(
        models=models_dict,
        images=images,
        labels=labels,
        env_strategies=get_standard_perturbations(),  # All 7 perturbations
        attack="autoattack-apgd-ce",  # Fast AutoAttack (5x faster than standard)
        severities=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        eps_fn=lambda s: (8/255) * s,  # Map severity to epsilon
        checkpoint_dir="./exp_checkpoints",  # Auto-saves and resumes
        batch_size=50,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Running experiment (auto-saves progress)...")
    results = experiment.run()

    # =========================================================================
    # 3. Analysis - all methods work on live or loaded data
    # =========================================================================

    # Protection gap analysis
    gaps = results.protection_gap(baseline="resnet50")
    print("\nProtection gaps relative to resnet50:")
    for model, scenarios in gaps.items():
        for scenario, severities in scenarios.items():
            avg_gap = sum(severities.values()) / len(severities) if severities else 0
            print(f"  {model}/{scenario}: {avg_gap:.3f}")

    # Crossover detection (when accuracy drops below threshold)
    crossovers = results.crossover_detection(baseline="resnet50", threshold=0.5)
    print("\nCrossover points (accuracy < 0.5):")
    for model, scenarios in crossovers.items():
        for scenario, severity in scenarios.items():
            if severity is not None:
                print(f"  {model}/{scenario}: severity={severity:.2f}")

    # Model disagreement
    disagreement = results.disagreement_analysis()
    print(f"\nModel disagreement rate: {disagreement['disagreement_rate']:.3f}")

    # =========================================================================
    # 4. Visualization and saving
    # =========================================================================

    results.plot_compositional(save_path="compositional_results.png")
    results.save("./experiment_results")
    print("\nResults saved to ./experiment_results")

    # =========================================================================
    # 5. Load and analyze without models/GPU
    # =========================================================================

    print("\n" + "="*60)
    print("Loading saved results (no GPU/models needed)...")
    loaded = CompositionalResults.load("./experiment_results")
    loaded.print_summary()

    # Compute AUC for robustness curves
    for model in loaded.get_models():
        for scenario in loaded.get_scenarios(model):
            auc = loaded.compute_auc(model, scenario)
            print(f"AUC for {model}/{scenario}: {auc:.3f}")

    # =========================================================================
    # 6. Custom perturbations
    # =========================================================================

    print("\n" + "="*60)
    print("Testing custom perturbations...")

    custom_perturbations = {
        "strong_blur": GaussianBlur(sigma_max=5.0),
        "heavy_noise": GaussianNoise(std_max=0.2),
        "combined": Compose([
            GaussianBlur(sigma_max=2.0),
            GaussianNoise(std_max=0.1)
        ])
    }

    custom_exp = CompositionalExperiment(
        models={"resnet50": models_dict["resnet50"]},
        images=images[:50],  # Smaller subset
        labels=labels[:50],
        env_strategies=custom_perturbations,
        attack="pgd",  # Fast PGD instead of AutoAttack
        severities=[0.5, 1.0],
        checkpoint_dir="./custom_checkpoints"
    )

    custom_results = custom_exp.run()
    custom_results.print_summary()

    print("\nAll examples complete!")


if __name__ == "__main__":
    main()