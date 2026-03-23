"""
Simple example of using VisProbe for compositional robustness testing.
"""

import torch
import torchvision.models as models
from visprobe import CompositionalExperiment, get_minimal_perturbations


def main():
    # Load model
    model = models.resnet50(pretrained=True)
    model.eval()

    # Create sample data (replace with your dataset)
    images = torch.randn(100, 3, 224, 224)
    labels = torch.randint(0, 1000, (100,))

    # Run compositional test
    experiment = CompositionalExperiment(
        models={"resnet50": model},
        images=images,
        labels=labels,
        env_strategies=get_minimal_perturbations(),  # blur, noise, lowlight
        attack="autoattack-apgd-ce",  # Fast AutoAttack mode
        severities=[0.0, 0.5, 1.0],
        checkpoint_dir="./checkpoints"  # Auto-saves progress
    )

    # Run experiment (auto-resumes if interrupted)
    results = experiment.run()

    # Analyze and save
    results.print_summary()
    results.plot_compositional(save_path="results.png")
    results.save("./results")

    print("Done! Results saved to ./results")


if __name__ == "__main__":
    main()