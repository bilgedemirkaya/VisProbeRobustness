"""
Modal wrapper for the RobustBench audit pilot.

Runs the same logic as run_pilot.py but on a cloud A100 via Modal, so you
don't need a local GPU. Results land in a persistent Modal volume; pull them
locally and run analyze_pilot.py on your laptop afterward.

One-time setup (you've already done the modal auth):
    pip install modal
    modal setup

Run the pilot (detached so it survives your terminal closing):
    modal run --detach scripts/pilot/modal_runner.py

Watch logs:
    modal app logs visprobe-pilot

Pull results back to your laptop when finished:
    modal volume get visprobe-pilot results/pilot ./results/pilot

Then analyze locally:
    python scripts/pilot/analyze_pilot.py
"""

import modal

app = modal.App("visprobe-pilot")

# Everything the pilot needs. autoattack and robustbench come from GitHub
# (neither is reliably on PyPI). visprobe is pulled from this repo's main branch
# so we get the AutoAttack-compatibility fix that didn't make it into PyPI v2.0.0.
# Bump to a tagged commit (or PyPI v2.0.1) once a fixed release is out.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("git+https://github.com/fra31/auto-attack")
    .pip_install("git+https://github.com/RobustBench/robustbench.git")
    # Pin to a specific commit -- pip URL changes on every push, so Modal's layer
    # cache invalidates automatically. Bump this SHA whenever you push a new fix.
    .pip_install("git+https://github.com/bilgedemirkaya/VisProbeRobustness.git@307c813")
    .pip_install("pandas")
)

# Persistent storage for: CIFAR-10 cache, model weight cache, checkpoints, results.
# Created on first run; reused on subsequent runs.
volume = modal.Volume.from_name("visprobe-pilot", create_if_missing=True)

MODEL_NAMES = [
    "Wang2023Better_WRN-70-16",              # ~#1
    "Cui2023Decoupled_WRN-28-10",            # ~#5
    "Rebuffi2021Fixing_70_16_cutmix_extra",  # ~#15
    "Gowal2020Uncovering_70_16_extra",       # ~#30
    "Carmon2019Unlabeled",                   # ~#50
]


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=6 * 60 * 60,                 # 6h hard cap; pilot should finish in 4-5h
    volumes={"/workspace": volume},
    retries=modal.Retries(max_retries=5, initial_delay=30.0, backoff_coefficient=1.5),
)
def run_pilot():
    import os
    import torch

    from robustbench.utils import load_model
    from robustbench.data import load_cifar10
    from visprobe import CompositionalExperiment, GaussianNoise

    os.chdir("/workspace")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    n_samples = 1000
    batch_size = 50
    eps = 8 / 255
    device = "cuda"

    print(f"\nLoading {n_samples} CIFAR-10 test images (cached on volume)...")
    x_test, y_test = load_cifar10(n_examples=n_samples, data_dir="./data")
    x_test, y_test = x_test.to(device), y_test.to(device)

    print("\nLoading models from RobustBench (cached on volume after first run)...")
    models = {}
    for name in MODEL_NAMES:
        print(f"  - {name}")
        models[name] = (
            load_model(
                model_name=name,
                dataset="cifar10",
                threat_model="Linf",
                model_dir="./models",   # cache weights on the volume
            )
            .to(device)
            .eval()
        )

    experiment = CompositionalExperiment(
        models=models,
        images=x_test,
        labels=y_test,
        env_strategies={"noise": GaussianNoise(std_max=0.1)},
        attack="autoattack-apgd-ce",    # faster pilot protocol; not directly comparable to RobustBench numbers
        severities=[0.0, 0.5, 1.0],
        eps_fn=lambda s: eps,           # fixed eps; only the env varies
        checkpoint_dir="./checkpoints/pilot_apgd",
        batch_size=batch_size,
        device=device,
        verbose=True,
    )

    results = experiment.run()
    results.save("./results/pilot_apgd")
    experiment.cleanup()

    # Commit the volume so the data persists across function invocations
    # and is visible to `modal volume get`.
    volume.commit()

    print("\n" + "=" * 60)
    print("Pilot complete.")
    print("=" * 60)
    print("Pull results to your laptop with:")
    print("  modal volume get visprobe-pilot results/pilot_apgd ./results/pilot_apgd")
    print("Then run:")
    print("  python scripts/pilot/analyze_pilot.py")


@app.local_entrypoint()
def main():
    run_pilot.remote()
