"""
Pilot: do RobustBench CIFAR-10 Linf rankings survive under env perturbation + attack?

Five models spanning the leaderboard rank range, Gaussian noise, three severities.
Uses APGD-CE (not full AutoAttack) -- faster pilot protocol; the severity=0.0 cell
gives a clean-attack baseline but is not a strict RobustBench replication.

Usage:
    pip install robustbench   # research dep, not part of visprobe core
    python scripts/pilot/run_pilot.py
"""

import torch

from robustbench.utils import load_model
from robustbench.data import load_cifar10

from visprobe import CompositionalExperiment, GaussianNoise

# Five models spanning the RobustBench CIFAR-10 Linf ranking.
# Ranks are approximate -- check the live leaderboard at robustbench.github.io
# before publishing the paper; they shift as new entries land.
MODEL_NAMES = [
    "Wang2023Better_WRN-70-16",              # ~#1, diffusion-augmented WRN
    "Cui2023Decoupled_WRN-28-10",            # ~#5, distillation (DKD)
    "Rebuffi2021Fixing_70_16_cutmix_extra",  # ~#15, earlier diffusion era
    "Gowal2020Uncovering_70_16_extra",       # ~#30, pre-diffusion + extra data
    "Carmon2019Unlabeled",                   # ~#50, classic AT + SVHN
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 1000           # RobustBench standard for CIFAR-10
BATCH_SIZE = 50
EPS = 8 / 255              # RobustBench Linf budget for CIFAR-10

print(f"Loading {N_SAMPLES} CIFAR-10 test images...")
x_test, y_test = load_cifar10(n_examples=N_SAMPLES, data_dir="./data")
x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)

print("Loading models from RobustBench...")
models = {}
for name in MODEL_NAMES:
    print(f"  - {name}")
    models[name] = (
        load_model(model_name=name, dataset="cifar10", threat_model="Linf")
        .to(DEVICE)
        .eval()
    )

# Single scenario, three severities.
# severity=0.0 -> no env, attack only -> clean-attack baseline (not strict RobustBench replication)
# severity=0.5 -> mid-degradation
# severity=1.0 -> max env from VisProbe's standard scale
env_strategies = {"noise": GaussianNoise(std_max=0.1)}
severities = [0.0, 0.5, 1.0]

experiment = CompositionalExperiment(
    models=models,
    images=x_test,
    labels=y_test,
    env_strategies=env_strategies,
    attack="autoattack-apgd-ce",    # faster pilot protocol; not directly comparable to RobustBench numbers
    severities=severities,
    eps_fn=lambda s: EPS,           # FIXED eps; only the env varies across severities
    checkpoint_dir="./checkpoints/pilot_apgd",
    batch_size=BATCH_SIZE,
    device=DEVICE,
    verbose=True,
)

results = experiment.run()
results.save("./results/pilot_apgd")
experiment.cleanup()
print("\nDone. Run scripts/pilot/analyze_pilot.py next.")
