
"""
================================================================================
Compositional Robustness of Adversarially Trained Vision Models
================================================================================

Evaluates whether adversarial training creates environmental vulnerabilities
invisible to standard benchmarks — across 3 architecture families using
VisProbe's analysis infrastructure and AutoAttack.

Models (5):
  Vanilla:  ResNet-50  (deployment baseline)
            Swin-B     (architecture control for same-family comparison)
  Robust:   Liu2023Comprehensive_Swin-B          (rank #8,  56.16%)
            Singh2023Revisiting_ConvNeXt-B-ConvStem (rank #9,  56.14%)
            Singh2023Revisiting_ViT-B-ConvStem     (rank #12, 54.66%)

Attack: AutoAttack (standard) — subsumes PGD via APGD.

Scenarios (12):
  Baseline:       Pure AutoAttack | Pure Low-Light | Pure Blur | Pure Noise
  Compositional:  Low-Light + AA | Blur + AA | Noise + AA | Strong Noise + AA
  Ablation:       Fixed Dark + AA Sweep | Full AA + Brightness |
                  Full AA + Noise | Noise + Fixed AA

Severity: s ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}  (6 points)
Samples:  500 mutually correct ImageNet val images (seed=42)

All evaluation through VisProbe:
  - visprobe.analysis.evaluate_detailed   → per-sample tracking
  - visprobe.analysis.bootstrap_accuracy  → CIs on accuracy
  - visprobe.analysis.bootstrap_delta     → CIs on model differences
  - visprobe.analysis.bootstrap_protection_gap → protection gap CIs
  - visprobe.analysis.find_crossover      → robustness reversal detection
  - visprobe.analysis.confidence_profile  → calibration analysis
  - visprobe.analysis.class_vulnerability → per-class failure analysis
  - visprobe.analysis.expected_calibration_error → ECE

Estimated compute: ~40–50 A100 GPU-hours
================================================================================
"""

# %% [markdown]
# # Step 1 · Installation

# %%
# !pip install -q torch torchvision
# !pip install -q robustbench
# !pip install -q git+https://github.com/fra31/auto-attack
# !pip install -q git+https://github.com/bilgedemirkaya/VisProbeRobustness
# !pip install -q matplotlib seaborn tqdm

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import json, time, warnings
warnings.filterwarnings("ignore")

# ── AutoAttack ──
from autoattack import AutoAttack

# ── RobustBench ──
from robustbench.utils import load_model

# ── VisProbe analysis module ──
from visprobe.analysis import (
    evaluate_detailed,
    EvaluationResult,
    bootstrap_accuracy,
    bootstrap_delta,
    bootstrap_protection_gap,
    find_crossover,
    disagreement_analysis,
    confidence_profile,
    class_vulnerability,
    expected_calibration_error,
)

# ── Reproducibility ──
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU:  {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"  VRAM: {mem:.1f} GB")


# %% [markdown]
# # Step 2 · Configuration

# %%
@dataclass
class Config:
    imagenet_root: str = "/data/imagenet"          # ← SET THIS
    n_samples: int = 500
    batch_size: int = 50
    num_workers: int = 4

    # Severity sweep (6 points for AUC)
    severity_levels: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    )

    # AutoAttack
    aa_version: str = "standard"
    aa_eps_benchmark: float = 4 / 255       # RobustBench standard

    # Compositional budget (below training budget of 4/255 ≈ 0.016)
    comp_eps_max: float = 0.01              # ε = 0.01·s

    # Environmental maxima
    brightness_reduction: float = 0.7
    blur_sigma_max: float = 3.0
    noise_std_max: float = 0.1
    strong_noise_std_max: float = 0.15

    # Ablation
    ablation_fixed_dark: float = 0.3
    ablation_full_eps: float = 0.03

    # VisProbe bootstrap settings
    n_bootstrap: int = 10_000
    ci_level: float = 0.95

    save_dir: Path = Path("results_compositional_v2")
    seed: int = 42

CFG = Config()
CFG.save_dir.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# # Step 3 · Model Loading
#
# All models accept **[0, 1] inputs**.
# - Vanilla models: wrapped with ImageNet normalisation.
# - RobustBench models: normalise internally.

# %%
class NormalizedModel(nn.Module):
    """Wraps a torchvision model to accept [0,1] inputs."""
    def __init__(self, backbone, mean, std):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_all_models(device) -> Dict[str, nn.Module]:
    models_dict = {}

    # ── Vanilla baselines ─────────────────────────────────────────────
    print("Loading vanilla baselines...")

    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    models_dict["Vanilla_ResNet50"] = (
        NormalizedModel(resnet, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    )

    swin = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    models_dict["Vanilla_SwinB"] = (
        NormalizedModel(swin, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    )

    print(f"  ✓ 2 vanilla baselines")

    # ── Adversarially-trained (RobustBench, ImageNet Linf) ────────────
    print("Loading adversarially-trained models from RobustBench...")

    ROBUST = {
        "Liu2023_SwinB":            "Liu2023Comprehensive_Swin-B",
        "Singh2023_ConvNeXtB":      "Singh2023Revisiting_ConvNeXt-B-ConvStem",
        "Singh2023_ViTB":           "Singh2023Revisiting_ViT-B-ConvStem",
    }

    for short_name, rb_name in ROBUST.items():
        try:
            m = load_model(model_name=rb_name, dataset="imagenet",
                           threat_model="Linf")
            models_dict[short_name] = m.to(device).eval()
            print(f"  ✓ {short_name}  ({rb_name})")
        except Exception as e:
            print(f"  ✗ {short_name}: {e}")

    print(f"\nTotal: {len(models_dict)} models loaded")
    return models_dict


MODELS = load_all_models(DEVICE)

# ── Architecture families ────────────────────────────────────────────
FAMILIES = {
    "Swin-B": {
        "vanilla": "Vanilla_SwinB",
        "robust":  ["Liu2023_SwinB"],
    },
    "ConvNeXt-B": {
        "vanilla": "Vanilla_ResNet50",      # deployment baseline
        "robust":  ["Singh2023_ConvNeXtB"],
    },
    "ViT-B": {
        "vanilla": "Vanilla_ResNet50",      # deployment baseline
        "robust":  ["Singh2023_ViTB"],
    },
}

# Swin-B also gets compared against ResNet50 for deployment context
DEPLOYMENT_BASELINE = "Vanilla_ResNet50"


# %% [markdown]
# # Step 4 · Load ImageNet Validation

# %%
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

print("Loading ImageNet validation set...")
dataset = ImageNet(root=CFG.imagenet_root, split="val", transform=transform)
print(f"  ✓ {len(dataset)} images")

# Class names for vulnerability analysis
class_names = {i: dataset.classes[i][0] for i in range(len(dataset.classes))}


# %% [markdown]
# # Step 5 · Find Mutually Correct Samples

# %%
@torch.no_grad()
def find_mutually_correct(
    models_dict: Dict[str, nn.Module],
    dataset,
    n_target: int,
    device=DEVICE,
    max_scan: int = 10_000,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Return (images, labels, indices) correctly classified by every model."""
    imgs, labs, idxs = [], [], []

    for i in tqdm(range(min(len(dataset), max_scan)), desc="Mutual correctness"):
        if len(imgs) >= n_target:
            break
        img, label = dataset[i]
        x = img.unsqueeze(0).to(device)
        if all(m(x).argmax(1).item() == label for m in models_dict.values()):
            imgs.append(img)
            labs.append(label)
            idxs.append(i)

    images = torch.stack(imgs)
    labels = torch.tensor(labs, dtype=torch.long)
    print(f"  ✓ {len(labels)} samples from {i+1} scanned")
    print(f"  ✓ {len(set(labs))} unique classes covered")
    return images, labels, idxs


IMAGES, LABELS, SAMPLE_IDX = find_mutually_correct(
    MODELS, dataset, CFG.n_samples
)


# %% [markdown]
# # Step 6 · Perturbation Functions
#
# Applied in **[0, 1] pixel space** before the adversarial attack.

# %%
def apply_low_light(images, severity, max_red=0.7):
    return (images * (1.0 - max_red * severity)).clamp(0, 1)

def apply_blur(images, severity, sigma_max=3.0):
    if severity < 0.01:
        return images
    sigma = sigma_max * severity
    k = max(int(6 * sigma + 1) | 1, 3)          # odd, ≥ 3
    return transforms.functional.gaussian_blur(images, kernel_size=k, sigma=sigma)

def apply_noise(images, severity, std_max=0.1):
    if severity < 1e-8:
        return images
    return (images + torch.randn_like(images) * std_max * severity).clamp(0, 1)

def identity(images, severity, **kw):
    return images


# %% [markdown]
# # Step 7 · Core Evaluation Pipeline
#
# Uses **VisProbe `evaluate_detailed`** for every single evaluation,
# producing `EvaluationResult` objects with per-sample tracking.

# %%
def run_autoattack(model, images, labels, eps, device=DEVICE):
    """Run AutoAttack (standard) and return adversarial images on CPU."""
    if eps < 1e-8:
        return images
    aa = AutoAttack(model, norm="Linf", eps=eps,
                    version=CFG.aa_version, verbose=False)
    aa.seed = SEED
    x_adv = aa.run_standard_evaluation(
        images.to(device), labels.to(device), bs=CFG.batch_size
    )
    return x_adv.cpu()


def evaluate_scenario(
    model: nn.Module,
    model_name: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    scenario: str,
    severity: float,
    perturbation_fn: Callable,
    eps: float = 0.0,
) -> EvaluationResult:
    """
    Pipeline:  env perturbation → AutoAttack → VisProbe evaluate_detailed.
    """
    t0 = time.time()

    # 1 · Environmental perturbation in pixel space
    x_env = perturbation_fn(images, severity)

    # 2 · Adversarial attack on degraded images
    if eps > 1e-8:
        x_final = run_autoattack(model, x_env, labels, eps)
    else:
        x_final = x_env

    # 3 · Evaluate with VisProbe — per-sample tracking
    result = evaluate_detailed(
        model=model,
        images=x_final,
        labels=labels,
        batch_size=CFG.batch_size,
        device=DEVICE,
        scenario=f"{scenario}_s{severity:.2f}_eps{eps:.4f}",
        severity=severity,
        eps=eps,
    )

    wall = time.time() - t0
    return result


# %% [markdown]
# # Step 8 · AUC Helper

# %%
def compute_auc(severities, accuracies):
    """Trapezoidal AUC normalised to [0, 1]."""
    return float(np.trapz(accuracies, severities) / (severities[-1] - severities[0]))

S = CFG.severity_levels


# %% [markdown]
# # Step 9 · Phase 1 — Pure Adversarial (AutoAttack ε = 4/255)

# %%
print("\n" + "=" * 80)
print("PHASE 1: PURE ADVERSARIAL (AutoAttack ε = 4/255)")
print("=" * 80)

pure_adv: Dict[str, List[EvaluationResult]] = {}

for mname, model in MODELS.items():
    print(f"\n  {mname}:")
    results = []
    for eps in [0.0, CFG.aa_eps_benchmark]:
        r = evaluate_scenario(
            model, mname, IMAGES, LABELS,
            scenario="pure_adv", severity=0.0,
            perturbation_fn=identity, eps=eps,
        )
        tag = "clean" if eps < 1e-8 else f"ε={eps:.4f}"
        print(f"    {tag}: {r.accuracy:.1%}")
        results.append(r)
    pure_adv[mname] = results

print("\n✅ Phase 1 complete")


# %% [markdown]
# # Step 10 · Phase 2 — Pure Environmental (No Attack)

# %%
print("\n" + "=" * 80)
print("PHASE 2: PURE ENVIRONMENTAL")
print("=" * 80)

ENV = {
    "pure_low_light": lambda img, s: apply_low_light(img, s, CFG.brightness_reduction),
    "pure_blur":      lambda img, s: apply_blur(img, s, CFG.blur_sigma_max),
    "pure_noise":     lambda img, s: apply_noise(img, s, CFG.noise_std_max),
}

pure_env: Dict[str, Dict[str, List[EvaluationResult]]] = {}

for sc_name, pert_fn in ENV.items():
    print(f"\n  {sc_name}")
    pure_env[sc_name] = {}
    for mname, model in MODELS.items():
        results = []
        for s in tqdm(S, desc=f"    {mname[:25]}", leave=False):
            r = evaluate_scenario(model, mname, IMAGES, LABELS,
                                  scenario=sc_name, severity=s,
                                  perturbation_fn=pert_fn, eps=0.0)
            results.append(r)
        pure_env[sc_name][mname] = results
        auc = compute_auc(S, [r.accuracy for r in results])
        print(f"    {mname:30s}  {results[0].accuracy:.1%} → "
              f"{results[-1].accuracy:.1%}  AUC={auc:.3f}")

print("\n✅ Phase 2 complete")


# %% [markdown]
# # Step 11 · Phase 3 — Compositional (Env + AutoAttack)
#
# Environmental degradation first, then AutoAttack on degraded image.
# ε = 0.01·s — max ε = 0.01, **below** training budget of 4/255.

# %%
print("\n" + "=" * 80)
print("PHASE 3: COMPOSITIONAL (Env + AutoAttack)")
print("=" * 80)

COMP = {
    "comp_lowlight_aa": {
        "fn":    lambda img, s: apply_low_light(img, s, CFG.brightness_reduction),
        "eps":   lambda s: CFG.comp_eps_max * s,
        "label": "Low-Light + AA",
    },
    "comp_blur_aa": {
        "fn":    lambda img, s: apply_blur(img, s, CFG.blur_sigma_max),
        "eps":   lambda s: CFG.comp_eps_max * s,
        "label": "Blur + AA",
    },
    "comp_noise_aa": {
        "fn":    lambda img, s: apply_noise(img, s, CFG.noise_std_max),
        "eps":   lambda s: CFG.comp_eps_max * s,
        "label": "Noise + AA",
    },
    "comp_strong_noise_aa": {
        "fn":    lambda img, s: apply_noise(img, s, CFG.strong_noise_std_max),
        "eps":   lambda s: CFG.comp_eps_max * s,
        "label": "Strong Noise + AA",
    },
}

comp: Dict[str, Dict[str, List[EvaluationResult]]] = {}

for sc_name, sc in COMP.items():
    print(f"\n  {sc['label']}")
    comp[sc_name] = {}
    for mname, model in MODELS.items():
        results = []
        for s in tqdm(S, desc=f"    {mname[:25]}", leave=False):
            r = evaluate_scenario(model, mname, IMAGES, LABELS,
                                  scenario=sc_name, severity=s,
                                  perturbation_fn=sc["fn"],
                                  eps=sc["eps"](s))
            results.append(r)
        comp[sc_name][mname] = results
        auc = compute_auc(S, [r.accuracy for r in results])
        print(f"    {mname:30s}  {results[0].accuracy:.1%} → "
              f"{results[-1].accuracy:.1%}  AUC={auc:.3f}")

print("\n✅ Phase 3 complete")


# %% [markdown]
# # Step 12 · Phase 4 — Ablations

# %%
print("\n" + "=" * 80)
print("PHASE 4: ABLATIONS")
print("=" * 80)

abl: Dict[str, Dict[str, List[EvaluationResult]]] = {}
ABL_EPS_AXIS = np.linspace(0, CFG.comp_eps_max, 6)

# ── A: Fixed darkness (brightness=0.3) + AA sweep ────────────────────
print("\n  A: Fixed darkness (0.3) + AA sweep")
abl["abl_fixdark_aa"] = {}
for mname, model in MODELS.items():
    results = []
    for eps in tqdm(ABL_EPS_AXIS, desc=f"    {mname[:25]}", leave=False):
        r = evaluate_scenario(
            model, mname, IMAGES, LABELS,
            scenario="abl_fixdark_aa", severity=1.0,
            perturbation_fn=lambda img, s: apply_low_light(img, 1.0, CFG.brightness_reduction),
            eps=eps,
        )
        results.append(r)
    abl["abl_fixdark_aa"][mname] = results
    print(f"    {mname:30s}  {results[0].accuracy:.1%} → {results[-1].accuracy:.1%}")

# ── B: Full AA (ε=0.03) + brightness sweep ───────────────────────────
print("\n  B: Full AA (ε=0.03) + brightness sweep")
abl["abl_fullaa_bright"] = {}
for mname, model in MODELS.items():
    results = []
    for s in tqdm(S, desc=f"    {mname[:25]}", leave=False):
        r = evaluate_scenario(
            model, mname, IMAGES, LABELS,
            scenario="abl_fullaa_bright", severity=s,
            perturbation_fn=lambda img, sv: apply_low_light(img, sv, CFG.brightness_reduction),
            eps=CFG.ablation_full_eps,
        )
        results.append(r)
    abl["abl_fullaa_bright"][mname] = results
    auc = compute_auc(S, [r.accuracy for r in results])
    print(f"    {mname:30s}  {results[0].accuracy:.1%} → "
          f"{results[-1].accuracy:.1%}  AUC={auc:.3f}")

# ── C: Full AA (ε=0.03) + noise sweep ────────────────────────────────
print("\n  C: Full AA (ε=0.03) + noise sweep")
abl["abl_fullaa_noise"] = {}
for mname, model in MODELS.items():
    results = []
    for s in tqdm(S, desc=f"    {mname[:25]}", leave=False):
        r = evaluate_scenario(
            model, mname, IMAGES, LABELS,
            scenario="abl_fullaa_noise", severity=s,
            perturbation_fn=lambda img, sv: apply_noise(img, sv, CFG.noise_std_max),
            eps=CFG.ablation_full_eps,
        )
        results.append(r)
    abl["abl_fullaa_noise"][mname] = results
    auc = compute_auc(S, [r.accuracy for r in results])
    print(f"    {mname:30s}  {results[0].accuracy:.1%} → "
          f"{results[-1].accuracy:.1%}  AUC={auc:.3f}")

# ── D: Noise sweep + fixed AA (ε=0.01) ──────────────────────────────
print("\n  D: Noise sweep + fixed AA (ε=0.01)")
abl["abl_noise_fixaa"] = {}
for mname, model in MODELS.items():
    results = []
    for s in tqdm(S, desc=f"    {mname[:25]}", leave=False):
        r = evaluate_scenario(
            model, mname, IMAGES, LABELS,
            scenario="abl_noise_fixaa", severity=s,
            perturbation_fn=lambda img, sv: apply_noise(img, sv, CFG.noise_std_max),
            eps=CFG.comp_eps_max,
        )
        results.append(r)
    abl["abl_noise_fixaa"][mname] = results
    auc = compute_auc(S, [r.accuracy for r in results])
    print(f"    {mname:30s}  {results[0].accuracy:.1%} → "
          f"{results[-1].accuracy:.1%}  AUC={auc:.3f}")

print("\n✅ Phase 4 complete")


# %% [markdown]
# # Step 13 · VisProbe Analysis — Protection Gaps, Crossovers, Calibration
#
# All statistics computed with **VisProbe's analysis module**.

# %%
print("\n" + "=" * 80)
print("PHASE 5: VISPROBE ANALYSIS")
print("=" * 80)

# ── 13a · AUC Summary Table ──────────────────────────────────────────
AUC = {}

AUC["Pure AA (ε=4/255)"] = {
    m: pure_adv[m][-1].accuracy for m in MODELS
}
for sc_name, sc_data in pure_env.items():
    label = sc_name.replace("pure_", "Pure ").replace("_", " ").title()
    AUC[label] = {m: compute_auc(S, [r.accuracy for r in sc_data[m]]) for m in MODELS}
for sc_name, sc_cfg in COMP.items():
    AUC[sc_cfg["label"]] = {m: compute_auc(S, [r.accuracy for r in comp[sc_name][m]]) for m in MODELS}

ABL_LABELS = {
    "abl_fixdark_aa":    "Fixed Dark + AA Sweep",
    "abl_fullaa_bright": "Full AA + Brightness",
    "abl_fullaa_noise":  "Full AA + Noise",
    "abl_noise_fixaa":   "Noise + Fixed AA",
}
for sc_name, sc_data in abl.items():
    x = ABL_EPS_AXIS if sc_name == "abl_fixdark_aa" else S
    AUC[ABL_LABELS[sc_name]] = {m: compute_auc(x, [r.accuracy for r in sc_data[m]]) for m in MODELS}

print("\nAUC TABLE")
print("-" * 100)
header = f"{'Scenario':35s}" + "".join(f"  {m[:14]:>14s}" for m in MODELS)
print(header)
print("-" * len(header))
for scenario, scores in AUC.items():
    best = max(scores.values())
    row = f"{scenario:35s}"
    for m in MODELS:
        v = scores[m]
        mark = " *" if abs(v - best) < 1e-4 else "  "
        row += f"  {v:>12.3f}{mark}"
    print(row)


# ── 13b · Protection Gap (VisProbe bootstrap_protection_gap) ─────────
print("\n\nPROTECTION GAP — PER ARCHITECTURE")
print("-" * 80)

gap_scenes = ["comp_lowlight_aa", "comp_blur_aa", "comp_noise_aa"]
protection_gaps = {}

robust_models = [m for m in MODELS if not m.startswith("Vanilla")]
van = DEPLOYMENT_BASELINE

for rob in robust_models:
    # Pure adversarial masks
    adv_rob_mask  = pure_adv[rob][-1].correct_mask
    adv_van_mask  = pure_adv[van][-1].correct_mask
    adv_advantage = adv_rob_mask.mean() - adv_van_mask.mean()

    # Average compositional advantage
    comp_advs = []
    for sc in gap_scenes:
        rob_auc = compute_auc(S, [r.accuracy for r in comp[sc][rob]])
        van_auc = compute_auc(S, [r.accuracy for r in comp[sc][van]])
        comp_advs.append(rob_auc - van_auc)
    mean_comp = np.mean(comp_advs)

    # Protection gap as percentage of adversarial advantage lost
    gap_pct = (1 - mean_comp / adv_advantage) * 100 if adv_advantage > 0 else float("nan")

    # Bootstrap CI on gap using worst compositional scenario
    worst_sc = gap_scenes[np.argmin(comp_advs)]
    comp_rob_mask = comp[worst_sc][rob][-1].correct_mask
    comp_van_mask = comp[worst_sc][van][-1].correct_mask

    gap_val, gap_lo, gap_hi = bootstrap_protection_gap(
        adv_rob_mask, adv_van_mask,
        comp_rob_mask, comp_van_mask,
        n_bootstrap=CFG.n_bootstrap,
    )

    protection_gaps[rob] = {
        "adv_advantage": float(adv_advantage),
        "comp_advantages": comp_advs,
        "mean_comp_advantage": float(mean_comp),
        "gap_pct": gap_pct,
        "gap_bootstrap": (gap_val, gap_lo, gap_hi),
    }

    print(f"\n  {rob}")
    print(f"    Adv advantage:    {adv_advantage:+.3f}")
    print(f"    Comp advantages:  {[f'{a:+.3f}' for a in comp_advs]}")
    print(f"    Protection gap:   {gap_pct:.1f}%")
    print(f"    Bootstrap gap:    {gap_val:.3f} [{gap_lo:.3f}, {gap_hi:.3f}]")


# ── 13c · Crossover Detection (VisProbe find_crossover) ──────────────
print("\n\nCROSSOVER DETECTION")
print("-" * 80)

crossovers = {}
for sc_name, sc_cfg in COMP.items():
    for rob in robust_models:
        accs_rob = np.array([r.accuracy for r in comp[sc_name][rob]])
        accs_van = np.array([r.accuracy for r in comp[sc_name][van]])

        cross = find_crossover(S, accs_rob, accs_van)
        key = f"{sc_cfg['label']} | {rob}"

        if cross is not None:
            crossovers[key] = cross.severity
            print(f"  ✗ {key}: reversal at s={cross.severity:.3f}")
        elif accs_van[-1] > accs_rob[-1]:
            print(f"  ~ {key}: vanilla ends ahead ({accs_van[-1]:.1%} vs {accs_rob[-1]:.1%})")
        else:
            print(f"  ✓ {key}: robust dominates")


# ── 13d · Confidence Calibration (VisProbe confidence_profile + ECE) ─
print("\n\nCONFIDENCE CALIBRATION")
print("-" * 80)

calibration = {}
for mname in MODELS:
    clean = pure_adv[mname][0]             # clean evaluation
    prof = confidence_profile(clean.samples)
    ece = expected_calibration_error(clean.samples)

    calibration[mname] = {
        "ece": ece,
        "conf_correct": prof.mean_confidence_correct,
        "conf_incorrect": prof.mean_confidence_incorrect,
        "high_conf_errors": prof.pct_high_confidence_errors,
    }

    print(f"  {mname:30s}  ECE={ece:.3f}  "
          f"conf(✓)={prof.mean_confidence_correct:.3f}  "
          f"conf(✗)={prof.mean_confidence_incorrect:.3f}  "
          f"high-conf-err={prof.pct_high_confidence_errors:.1f}%")


# ── 13e · Class Vulnerability (VisProbe class_vulnerability) ─────────
print("\n\nCLASS VULNERABILITY (top 5 per model)")
print("-" * 80)

vuln = {}
for mname in robust_models:
    clean_res  = pure_adv[mname][0]
    attack_res = pure_adv[mname][-1]       # AA at ε=4/255

    reports = class_vulnerability(
        clean_res.samples, attack_res.samples,
        class_names=class_names, top_k=5,
    )
    vuln[mname] = reports

    print(f"\n  {mname}:")
    for rp in reports:
        print(f"    {rp.class_name:25s}  drop={rp.accuracy_drop:.1%}")


# ── 13f · Disagreement Analysis (VisProbe disagreement_analysis) ─────
print("\n\nDISAGREEMENT ANALYSIS (strong noise, max severity)")
print("-" * 80)

sc = "comp_strong_noise_aa"
for rob in robust_models:
    van_res = comp[sc][van][-1]
    rob_res = comp[sc][rob][-1]
    dis = disagreement_analysis(van_res, rob_res)
    print(f"  {van} vs {rob}:")
    print(f"    Both correct:  {dis.both_correct:.1%}")
    print(f"    Only vanilla:  {dis.only_a_correct:.1%}")
    print(f"    Only robust:   {dis.only_b_correct:.1%}")
    print(f"    Both wrong:    {dis.both_wrong:.1%}")


# ── 13g · Bootstrap CIs on Key Comparisons ───────────────────────────
print("\n\nBOOTSTRAP CIs")
print("-" * 80)

for rob in robust_models:
    # Clean accuracy
    d, lo, hi = bootstrap_delta(
        pure_adv[rob][0].correct_mask.astype(float),
        pure_adv[van][0].correct_mask.astype(float),
        n_bootstrap=CFG.n_bootstrap, ci=CFG.ci_level,
    )
    sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
    print(f"  Clean Δ({rob} − vanilla): {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {sig}")

    # Strong noise + AA at max severity
    sc = "comp_strong_noise_aa"
    d, lo, hi = bootstrap_delta(
        comp[sc][rob][-1].correct_mask.astype(float),
        comp[sc][van][-1].correct_mask.astype(float),
        n_bootstrap=CFG.n_bootstrap, ci=CFG.ci_level,
    )
    sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
    print(f"  StrongNoise max Δ:        {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {sig}\n")


# %% [markdown]
# # Step 14 · Visualisation

# %%
plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "Vanilla_ResNet50":     "#2196F3",
    "Vanilla_SwinB":        "#64B5F6",
    "Liu2023_SwinB":        "#E53935",
    "Singh2023_ConvNeXtB":  "#FF9800",
    "Singh2023_ViTB":       "#AB47BC",
}
MARKERS = {
    "Vanilla_ResNet50":     "o",
    "Vanilla_SwinB":        "D",
    "Liu2023_SwinB":        "s",
    "Singh2023_ConvNeXtB":  "^",
    "Singh2023_ViTB":       "v",
}

def plot_curves(ax, data_dict, title, xlabel, x_vals=S):
    for mname, results in data_dict.items():
        accs = [r.accuracy * 100 for r in results]
        ax.plot(x_vals, accs, marker=MARKERS.get(mname, "o"),
                color=COLORS.get(mname, "gray"),
                label=mname, linewidth=2, markersize=5)
        # Annotate final value
        ax.annotate(f"{accs[-1]:.0f}%", (x_vals[-1], accs[-1]),
                    textcoords="offset points", xytext=(8, 0), fontsize=7)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)

# ── Fig 1: Pure environmental ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_curves(axes[0], pure_env["pure_low_light"],
            "Pure Low-Light", f"Severity s (brightness = 1−{CFG.brightness_reduction}s)")
plot_curves(axes[1], pure_env["pure_blur"],
            "Pure Blur", f"Severity s (σ = {CFG.blur_sigma_max}s)")
plot_curves(axes[2], pure_env["pure_noise"],
            "Pure Noise", f"Severity s (std = {CFG.noise_std_max}s)")
axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
fig.suptitle("Environmental Robustness — No Adversarial Attack", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(CFG.save_dir / "fig1_pure_env.pdf", bbox_inches="tight", dpi=300)
plt.savefig(CFG.save_dir / "fig1_pure_env.png", bbox_inches="tight", dpi=300)
plt.show()

# ── Fig 2: Compositional ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
for i, (sc_name, sc_cfg) in enumerate(COMP.items()):
    plot_curves(axes[i], comp[sc_name], sc_cfg["label"],
                f"Severity s (ε = {CFG.comp_eps_max}s)")
axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
fig.suptitle("Compositional Robustness — Environmental + AutoAttack",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(CFG.save_dir / "fig2_compositional.pdf", bbox_inches="tight", dpi=300)
plt.savefig(CFG.save_dir / "fig2_compositional.png", bbox_inches="tight", dpi=300)
plt.show()

# ── Fig 3: Ablations ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
abl_plots = [
    ("abl_fixdark_aa",    "Fixed Dark + AA Sweep",  "ε",              ABL_EPS_AXIS),
    ("abl_fullaa_bright", "Full AA + Brightness",   "Darkness sev s", S),
    ("abl_fullaa_noise",  "Full AA + Noise",         "Noise sev s",   S),
    ("abl_noise_fixaa",   "Noise + Fixed AA",        "Noise sev s",   S),
]
for i, (sc, title, xlabel, xv) in enumerate(abl_plots):
    plot_curves(axes[i], abl[sc], title, xlabel, x_vals=xv)
axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
fig.suptitle("Ablation Studies", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(CFG.save_dir / "fig3_ablations.pdf", bbox_inches="tight", dpi=300)
plt.savefig(CFG.save_dir / "fig3_ablations.png", bbox_inches="tight", dpi=300)
plt.show()

# ── Fig 4: Protection gap bar chart ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
names = list(protection_gaps.keys())
gaps  = [protection_gaps[n]["gap_pct"] for n in names]
colors = ["#e74c3c" if g > 50 else "#f39c12" if g > 25 else "#4caf50" for g in gaps]
bars = ax.bar(range(len(names)), gaps, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Protection Gap (%)", fontsize=11)
ax.set_title("Protection Gap by Model", fontsize=13, fontweight="bold")
ax.axhline(50, color="gray", ls="--", alpha=0.5, label="50% threshold")
for bar, val in zip(bars, gaps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(CFG.save_dir / "fig4_protection_gap.pdf", bbox_inches="tight", dpi=300)
plt.savefig(CFG.save_dir / "fig4_protection_gap.png", bbox_inches="tight", dpi=300)
plt.show()

# ── Fig 5: Confidence calibration grouped bar ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(MODELS))
w = 0.35
cc = [calibration[m]["conf_correct"]   for m in MODELS]
ci = [calibration[m]["conf_incorrect"] for m in MODELS]
ax.bar(x - w/2, cc, w, label="Correct",   color="#4caf50", alpha=0.8)
ax.bar(x + w/2, ci, w, label="Incorrect", color="#f44336", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(list(MODELS.keys()), rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Mean Confidence", fontsize=11)
ax.set_title("Confidence Calibration (Clean)", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(CFG.save_dir / "fig5_calibration.pdf", bbox_inches="tight", dpi=300)
plt.savefig(CFG.save_dir / "fig5_calibration.png", bbox_inches="tight", dpi=300)
plt.show()


# %% [markdown]
# # Step 15 · LaTeX Table

# %%
def make_latex(auc_table, model_names):
    n = len(model_names)
    short = {m: m.replace("Vanilla_", "V-").replace("Singh2023_", "S23-").replace("Liu2023_", "L23-")
             for m in model_names}

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{AUC scores across all 12 scenarios (higher = more robust). "
        r"\textbf{Bold} = best per row. "
        r"\underline{Underline} = robust model below deployment baseline (negative transfer).}",
        r"\label{tab:auc_multi_arch}",
        f"\\begin{{tabular}}{{l{'c' * n}}}",
        r"\toprule",
    ]
    lines.append("Scenario" + "".join(f" & {short[m]}" for m in model_names) + r" \\")
    lines.append(r"\midrule")

    van_baseline = DEPLOYMENT_BASELINE
    for scenario, scores in auc_table.items():
        best = max(scores.values())
        van_val = scores.get(van_baseline, 0)
        row = scenario
        for m in model_names:
            v = scores[m]
            cell = f"{v:.3f}"
            if abs(v - best) < 1e-4:
                cell = r"\textbf{" + cell + "}"
            if not m.startswith("Vanilla") and v < van_val:
                cell = r"\underline{" + cell + "}"
            row += f" & {cell}"
        row += r" \\"
        lines.append(row)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

latex = make_latex(AUC, list(MODELS.keys()))
print(latex)

with open(CFG.save_dir / "table_auc.tex", "w") as f:
    f.write(latex)
print(f"\n✓ Saved to {CFG.save_dir / 'table_auc.tex'}")


# %% [markdown]
# # Step 16 · Save All Results

# %%
def to_json(obj):
    if isinstance(obj, EvaluationResult):
        return {
            "accuracy": obj.accuracy,
            "mean_confidence": obj.mean_confidence,
            "mean_loss": obj.mean_loss,
            "correct_mask": obj.correct_mask.tolist(),
        }
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json(i) for i in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj

results_all = {
    "config": {
        "n_samples": CFG.n_samples,
        "severity_levels": S.tolist(),
        "aa_version": CFG.aa_version,
        "aa_eps_benchmark": CFG.aa_eps_benchmark,
        "comp_eps_max": CFG.comp_eps_max,
        "models": list(MODELS.keys()),
        "seed": CFG.seed,
    },
    "auc_table":        AUC,
    "protection_gaps":  to_json(protection_gaps),
    "crossovers":       crossovers,
    "calibration":      calibration,
    "vulnerability":    {m: [(r.class_name, r.accuracy_drop) for r in v] for m, v in vuln.items()},
    "pure_adversarial": to_json(pure_adv),
    "pure_environmental": to_json(pure_env),
    "compositional":    to_json(comp),
    "ablations":        to_json(abl),
    "sample_indices":   SAMPLE_IDX,
    "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
}

with open(CFG.save_dir / "results_full.json", "w") as f:
    json.dump(results_all, f, indent=2)

with open(CFG.save_dir / "results_summary.json", "w") as f:
    json.dump({
        "auc_table": AUC, "protection_gaps": to_json(protection_gaps),
        "crossovers": crossovers, "calibration": calibration,
        "n_samples": CFG.n_samples,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, f, indent=2)

print(f"✓ Saved to {CFG.save_dir}/")


# %% [markdown]
# # Step 17 · Final Report

# %%
print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)

print(f"\nDataset:   ImageNet validation, {CFG.n_samples} mutually correct samples")
print(f"Attack:    AutoAttack ({CFG.aa_version})")
print(f"Models:    {len(MODELS)}")
for m in MODELS:
    aa_acc = pure_adv[m][-1].accuracy
    print(f"  {m:30s}  AA@4/255 = {aa_acc:.1%}")

print(f"\n{'─'*60}")
print("PROTECTION GAPS")
for rob, g in protection_gaps.items():
    print(f"  {rob:30s}  {g['gap_pct']:.1f}%")
mean_gap = np.mean([g["gap_pct"] for g in protection_gaps.values()
                     if not np.isnan(g["gap_pct"])])
print(f"  {'MEAN':30s}  {mean_gap:.1f}%")

print(f"\n{'─'*60}")
print("ROBUSTNESS REVERSALS (strong noise + AA, max severity)")
sc = "comp_strong_noise_aa"
van_acc = comp[sc][van][-1].accuracy
for rob in robust_models:
    rob_acc = comp[sc][rob][-1].accuracy
    tag = "REVERSED ✗" if van_acc > rob_acc else "Robust ✓"
    print(f"  vanilla={van_acc:.1%}  {rob}={rob_acc:.1%}  → {tag}")

print(f"\n{'─'*60}")
print("OUTPUT FILES")
for f in sorted(CFG.save_dir.glob("*")):
    print(f"  {f.name}")
print("=" * 80)
