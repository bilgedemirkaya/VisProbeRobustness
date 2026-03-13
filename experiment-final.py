"""
================================================================================
Compositional Robustness of Adversarially Trained Vision Models
Complete experiment — 5 models × 12 scenarios × 6 severity levels
================================================================================

Models (5):
  Vanilla_ResNet50          — deployment baseline
  Vanilla_SwinB             — architecture control
  Liu2023_SwinB             — RobustBench rank #8  (56.16%)
  Singh2023_ConvNeXtB       — RobustBench rank #9  (56.14%)
  Singh2023_ViTB            — RobustBench rank #12 (54.66%)

Scenarios (12):
  Baseline (4):      Pure AA | Pure Low-Light | Pure Blur | Pure Noise
  Compositional (4): Low-Light+AA | Blur+AA | Noise+AA | Strong Noise+AA
  Ablation (4):      FixedDark+AA sweep | FullAA+Brightness |
                     FullAA+Noise | Noise+FixedAA

Attack:   AutoAttack standard (APGD-CE + APGD-T + FAB-T + Square)
Severity: s ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
Samples:  500 mutually correct ImageNet val images (seed=42)
================================================================================
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL 1 — Installation
# ═══════════════════════════════════════════════════════════════════════
# Uncomment for Colab:
# !pip install -q torch torchvision
# !pip install -q robustbench
# !pip install -q git+https://github.com/fra31/auto-attack
# !pip install -q git+https://github.com/bilgedemirkaya/VisProbeRobustness
# !pip install -q matplotlib seaborn tqdm
#
# from google.colab import drive
# drive.mount('/content/drive')

# ═══════════════════════════════════════════════════════════════════════
# CELL 2 — Imports
# ═══════════════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable
import json, time, warnings
warnings.filterwarnings("ignore")

from autoattack import AutoAttack
from robustbench.utils import load_model

# ── VisProbe import with fallback ─────────────────────────────────────
# Try importing from installed package first.
# If signatures differ, the fallback re-implements the essential API.
try:
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
    VISPROBE_SOURCE = "package"
    print("✓ VisProbe analysis module imported from installed package")
except ImportError:
    VISPROBE_SOURCE = "inline"
    print("⚠ VisProbe not installed — using inline implementation")
    # Inline fallback is defined in the next cell

# ── Reproducibility ───────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU:  {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════
# CELL 3 — VisProbe inline fallback (only runs if import failed)
# ═══════════════════════════════════════════════════════════════════════
if VISPROBE_SOURCE == "inline":
    from collections import defaultdict

    @dataclass
    class SampleResult:
        index: int; label: int; prediction: int
        correct: bool; confidence: float

    @dataclass
    class EvaluationResult:
        model_name: str; scenario: str
        samples: List[SampleResult]
        accuracy: float; correct_mask: np.ndarray
        mean_confidence: float = 0.0; mean_loss: float = 0.0

    @torch.no_grad()
    def evaluate_detailed(model, images, labels, batch_size=50,
                          device=None, **kw) -> EvaluationResult:
        dev = device or next(model.parameters()).device
        model.eval()
        samples, correct_all, confs, losses = [], [], [], []
        for i in range(0, len(images), batch_size):
            xb = images[i:i+batch_size].to(dev)
            yb = labels[i:i+batch_size].to(dev)
            out = model(xb)
            prb = F_torch.softmax(out, dim=1)
            cf, pr = prb.max(1)
            loss = F_torch.cross_entropy(out, yb, reduction="none")
            for j in range(len(xb)):
                ok = pr[j].item() == yb[j].item()
                samples.append(SampleResult(i+j, yb[j].item(),
                               pr[j].item(), ok, cf[j].item()))
                correct_all.append(ok); confs.append(cf[j].item())
                losses.append(loss[j].item())
        mask = np.array(correct_all)
        return EvaluationResult(
            model_name=kw.get("model_name", ""),
            scenario=kw.get("scenario", ""),
            samples=samples, accuracy=mask.mean(),
            correct_mask=mask, mean_confidence=np.mean(confs),
            mean_loss=np.mean(losses))

    def bootstrap_accuracy(mask, n_bootstrap=10000, ci=0.95):
        bs = [np.random.choice(mask, len(mask), True).mean()
              for _ in range(n_bootstrap)]
        a = (1-ci)/2
        return mask.mean(), np.percentile(bs, a*100), np.percentile(bs,(1-a)*100)

    def bootstrap_delta(a, b, n_bootstrap=10000, ci=0.95, **kw):
        ds = [np.random.choice(a,len(a),True).mean() -
              np.random.choice(b,len(b),True).mean() for _ in range(n_bootstrap)]
        al = (1-ci)/2
        return a.mean()-b.mean(), np.percentile(ds,al*100), np.percentile(ds,(1-al)*100)

    def bootstrap_protection_gap(ar, av, cr, cv, n_bootstrap=10000):
        gs = []
        for _ in range(n_bootstrap):
            ix = np.random.choice(len(ar), len(ar), True)
            ad = ar[ix].mean()-av[ix].mean()
            cd = cr[ix].mean()-cv[ix].mean()
            gs.append((ad-cd)/ad if ad > 0 else 0.0)
        g = np.array(gs)
        return g.mean(), np.percentile(g,2.5), np.percentile(g,97.5)

    def find_crossover(sevs, pa, pb):
        d = pa - pb; sc = np.where(np.diff(np.sign(d)))[0]
        if len(sc)==0: return None
        i = sc[0]; x1,x2 = sevs[i],sevs[i+1]; y1,y2 = d[i],d[i+1]
        sv = x1 - y1*(x2-x1)/(y2-y1)
        return type("X",(),{"severity":float(sv)})()

    @dataclass
    class ConfidenceProfile:
        mean_confidence_correct: float
        mean_confidence_incorrect: float
        pct_high_confidence_errors: float

    def confidence_profile(samples, threshold=0.8):
        cor = [s.confidence for s in samples if s.correct]
        inc = [s.confidence for s in samples if not s.correct]
        pct = 100*sum(1 for c in inc if c>threshold)/len(inc) if inc else 0
        return ConfidenceProfile(
            np.mean(cor) if cor else 0,
            np.mean(inc) if inc else 0, pct)

    def expected_calibration_error(samples, n_bins=10):
        cf = np.array([s.confidence for s in samples])
        ok = np.array([s.correct for s in samples])
        edges = np.linspace(0,1,n_bins+1); ece = 0.0
        for i in range(n_bins):
            m = (cf>edges[i])&(cf<=edges[i+1])
            if m.sum()>0:
                ece += m.sum()/len(samples)*abs(ok[m].mean()-cf[m].mean())
        return ece

    @dataclass
    class ClassVuln:
        class_name: str; accuracy_drop: float

    def class_vulnerability(clean_s, atk_s, class_names=None, top_k=5):
        cs = defaultdict(lambda: {"c":[],"a":[]})
        for c,a in zip(clean_s, atk_s):
            cs[c.label]["c"].append(c.correct); cs[c.label]["a"].append(a.correct)
        vs = []
        for cid, st in cs.items():
            if len(st["c"])>=3:
                d = np.mean(st["c"])-np.mean(st["a"])
                nm = class_names.get(cid,f"Class {cid}") if class_names else f"Class {cid}"
                vs.append(ClassVuln(nm, d))
        vs.sort(key=lambda v: v.accuracy_drop, reverse=True)
        return vs[:top_k]

    @dataclass
    class DisagreementResult:
        both_correct: float; only_a_correct: float
        only_b_correct: float; both_wrong: float

    def disagreement_analysis(ra, rb):
        a, b = ra.correct_mask, rb.correct_mask
        return DisagreementResult(
            (a&b).mean(), (a&~b).mean(), (~a&b).mean(), (~a&~b).mean())

    print("  Inline VisProbe analysis functions loaded")


# ═══════════════════════════════════════════════════════════════════════
# CELL 4 — Configuration
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    imagenet_root: str = "/content/drive/MyDrive/imagenet/val"  # ← SET THIS
    n_samples: int = 500
    batch_size: int = 50

    severity_levels: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))

    aa_version: str = "standard"
    aa_eps_benchmark: float = 4 / 255
    comp_eps_max: float = 0.01

    brightness_reduction: float = 0.7
    blur_sigma_max: float = 3.0
    noise_std_max: float = 0.1
    strong_noise_std_max: float = 0.15

    ablation_full_eps: float = 0.03

    n_bootstrap: int = 10_000
    ci_level: float = 0.95

    save_dir: Path = Path("results_compositional_v2")
    seed: int = 42

CFG = Config()
CFG.save_dir.mkdir(parents=True, exist_ok=True)
S = CFG.severity_levels


# ═══════════════════════════════════════════════════════════════════════
# CELL 5 — Model loading
# ═══════════════════════════════════════════════════════════════════════
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class NormalizedModel(nn.Module):
    """Wraps torchvision model to accept [0,1] inputs."""
    def __init__(self, backbone, mean, std):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor(std).view(1,3,1,1))
    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)

def load_all_models(device):
    m = {}
    print("Loading vanilla baselines...")
    r50 = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    m["Vanilla_ResNet50"] = NormalizedModel(r50, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    sw = tv_models.swin_b(weights=tv_models.Swin_B_Weights.IMAGENET1K_V1)
    m["Vanilla_SwinB"] = NormalizedModel(sw, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    print("  ✓ 2 vanilla baselines")

    print("Loading adversarially-trained models...")
    ROBUST = {
        "Liu2023_SwinB":        "Liu2023Comprehensive_Swin-B",
        "Singh2023_ConvNeXtB":  "Singh2023Revisiting_ConvNeXt-B-ConvStem",
        "Singh2023_ViTB":       "Singh2023Revisiting_ViT-B-ConvStem",
    }
    for short, rb in ROBUST.items():
        try:
            mdl = load_model(model_name=rb, dataset="imagenet", threat_model="Linf")
            m[short] = mdl.to(device).eval()
            print(f"  ✓ {short}")
        except Exception as e:
            print(f"  ✗ {short}: {e}")

    print(f"\nTotal: {len(m)} models")
    return m

MODELS = load_all_models(DEVICE)
DEPLOYMENT_BASELINE = "Vanilla_ResNet50"
ROBUST_NAMES = [n for n in MODELS if not n.startswith("Vanilla")]


# ═══════════════════════════════════════════════════════════════════════
# CELL 6 — Load ImageNet validation + find mutually correct samples
# ═══════════════════════════════════════════════════════════════════════
transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

print(f"Loading ImageNet from {CFG.imagenet_root}...")
dataset = ImageFolder(root=CFG.imagenet_root, transform=transform)
print(f"  ✓ {len(dataset)} images")
class_names = {i: n for i, n in enumerate(dataset.classes)}

@torch.no_grad()
def find_mutually_correct(models_dict, dataset, n_target, max_scan=10_000):
    imgs, labs, idxs = [], [], []
    for i in tqdm(range(min(len(dataset), max_scan)), desc="Mutual correctness"):
        if len(imgs) >= n_target:
            break
        img, label = dataset[i]
        x = img.unsqueeze(0).to(DEVICE)
        if all(mdl(x).argmax(1).item() == label for mdl in models_dict.values()):
            imgs.append(img); labs.append(label); idxs.append(i)
    images = torch.stack(imgs)
    labels = torch.tensor(labs, dtype=torch.long)
    print(f"  ✓ {len(labels)} samples, {len(set(labs))} classes")
    return images, labels, idxs

IMAGES, LABELS, SAMPLE_IDX = find_mutually_correct(MODELS, dataset, CFG.n_samples)


# ═══════════════════════════════════════════════════════════════════════
# CELL 7 — Perturbation functions (using functools.partial, no lambdas)
# ═══════════════════════════════════════════════════════════════════════
def _low_light(images, severity, max_red):
    return (images * (1.0 - max_red * severity)).clamp(0, 1)

def _blur(images, severity, sigma_max):
    if severity < 0.01:
        return images
    sigma = sigma_max * severity
    k = max(int(6*sigma+1) | 1, 3)
    return transforms.functional.gaussian_blur(images, kernel_size=k, sigma=sigma)

def _noise(images, severity, std_max):
    if severity < 1e-8:
        return images
    return (images + torch.randn_like(images) * std_max * severity).clamp(0, 1)

def _identity(images, severity):
    return images

# Pre-bound perturbation functions — no lambda closure bugs
pert_lowlight      = partial(_low_light, max_red=CFG.brightness_reduction)
pert_blur          = partial(_blur, sigma_max=CFG.blur_sigma_max)
pert_noise         = partial(_noise, std_max=CFG.noise_std_max)
pert_strong_noise  = partial(_noise, std_max=CFG.strong_noise_std_max)
pert_fixed_dark    = partial(_low_light, max_red=CFG.brightness_reduction)


# ═══════════════════════════════════════════════════════════════════════
# CELL 8 — Core evaluation pipeline
# ═══════════════════════════════════════════════════════════════════════
def run_autoattack(model, images, labels, eps):
    if eps < 1e-8:
        return images
    aa = AutoAttack(model, norm="Linf", eps=eps,
                    version=CFG.aa_version, verbose=False)
    aa.seed = SEED
    return aa.run_standard_evaluation(
        images.to(DEVICE), labels.to(DEVICE), bs=CFG.batch_size).cpu()

def evaluate_scenario(model, model_name, images, labels,
                      scenario, severity, perturbation_fn, eps=0.0):
    """Env perturbation → AutoAttack → VisProbe evaluate_detailed."""
    x_env = perturbation_fn(images, severity)
    x_final = run_autoattack(model, x_env, labels, eps) if eps > 1e-8 else x_env
    return evaluate_detailed(
        model=model, images=x_final, labels=labels,
        batch_size=CFG.batch_size, device=DEVICE,
        model_name=model_name,
        scenario=f"{scenario}_s{severity:.2f}_eps{eps:.4f}",
        severity=severity, eps=eps)

def compute_auc(sevs, accs):
    return float(np.trapz(accs, sevs) / (sevs[-1] - sevs[0]))

def _run_phase(name, models, images, labels, severities, pert_fn, eps_fn):
    """Run one model×severity sweep. Returns {model_name: [EvaluationResult]}."""
    out = {}
    for mname, model in models.items():
        results = []
        for s in tqdm(severities, desc=f"  {mname[:22]}", leave=False):
            r = evaluate_scenario(model, mname, images, labels,
                                  name, s, pert_fn, eps_fn(s))
            results.append(r)
        out[mname] = results
        auc = compute_auc(severities, [r.accuracy for r in results])
        print(f"  {mname:28s} {results[0].accuracy:.1%} → "
              f"{results[-1].accuracy:.1%}  AUC={auc:.3f}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# CELL 9 — PHASE 1: Pure Adversarial (AutoAttack ε=4/255)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("PHASE 1: PURE ADVERSARIAL")
print("="*80)

pure_adv = {}
for mname, model in MODELS.items():
    results = []
    for eps in [0.0, CFG.aa_eps_benchmark]:
        r = evaluate_scenario(model, mname, IMAGES, LABELS,
                              "pure_adv", 0.0, _identity, eps)
        tag = "clean" if eps < 1e-8 else f"ε={eps:.4f}"
        print(f"  {mname:28s} {tag}: {r.accuracy:.1%}")
        results.append(r)
    pure_adv[mname] = results
print("✅ Phase 1 complete")


# ═══════════════════════════════════════════════════════════════════════
# CELL 10 — PHASE 2: Pure Environmental (no attack)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("PHASE 2: PURE ENVIRONMENTAL")
print("="*80)

ENV_SCENARIOS = {
    "pure_low_light": pert_lowlight,
    "pure_blur":      pert_blur,
    "pure_noise":     pert_noise,
}

pure_env = {}
no_eps = lambda s: 0.0
for sc_name, fn in ENV_SCENARIOS.items():
    print(f"\n  {sc_name}")
    pure_env[sc_name] = _run_phase(sc_name, MODELS, IMAGES, LABELS, S, fn, no_eps)
print("\n✅ Phase 2 complete")


# ═══════════════════════════════════════════════════════════════════════
# CELL 11 — PHASE 3: Compositional (Env + AutoAttack)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("PHASE 3: COMPOSITIONAL")
print("="*80)

comp_eps = lambda s: CFG.comp_eps_max * s

COMP_SCENARIOS = {
    "comp_lowlight_aa":      {"fn": pert_lowlight,     "eps": comp_eps,
                              "label": "Low-Light + AA"},
    "comp_blur_aa":          {"fn": pert_blur,         "eps": comp_eps,
                              "label": "Blur + AA"},
    "comp_noise_aa":         {"fn": pert_noise,        "eps": comp_eps,
                              "label": "Noise + AA"},
    "comp_strong_noise_aa":  {"fn": pert_strong_noise, "eps": comp_eps,
                              "label": "Strong Noise + AA"},
}

comp = {}
for sc_name, sc in COMP_SCENARIOS.items():
    print(f"\n  {sc['label']}")
    comp[sc_name] = _run_phase(sc_name, MODELS, IMAGES, LABELS,
                               S, sc["fn"], sc["eps"])
print("\n✅ Phase 3 complete")


# ═══════════════════════════════════════════════════════════════════════
# CELL 12 — PHASE 4: Ablations
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("PHASE 4: ABLATIONS")
print("="*80)

ABL_EPS_AXIS = np.linspace(0, CFG.comp_eps_max, 6)
abl = {}

# A — Fixed darkness (brightness=0.3), sweep AutoAttack ε
print("\n  A: Fixed dark (brightness=0.3) + AA sweep")
abl["abl_fixdark_aa"] = {}
for mname, model in MODELS.items():
    results = []
    for eps in tqdm(ABL_EPS_AXIS, desc=f"  {mname[:22]}", leave=False):
        # severity=1.0 gives max brightness reduction (factor = 1-0.7 = 0.3)
        r = evaluate_scenario(model, mname, IMAGES, LABELS,
                              "abl_fixdark_aa", 1.0, pert_fixed_dark, eps)
        results.append(r)
    abl["abl_fixdark_aa"][mname] = results
    print(f"  {mname:28s} {results[0].accuracy:.1%} → {results[-1].accuracy:.1%}")

# B — Full AA (ε=0.03) + brightness sweep
print("\n  B: Full AA (ε=0.03) + brightness sweep")
full_eps = lambda s: CFG.ablation_full_eps
abl["abl_fullaa_bright"] = _run_phase(
    "abl_fullaa_bright", MODELS, IMAGES, LABELS, S, pert_lowlight, full_eps)

# C — Full AA (ε=0.03) + noise sweep
print("\n  C: Full AA (ε=0.03) + noise sweep")
abl["abl_fullaa_noise"] = _run_phase(
    "abl_fullaa_noise", MODELS, IMAGES, LABELS, S, pert_noise, full_eps)

# D — Noise sweep + fixed AA (ε=0.01)
print("\n  D: Noise sweep + fixed AA (ε=0.01)")
fix_eps = lambda s: CFG.comp_eps_max
abl["abl_noise_fixaa"] = _run_phase(
    "abl_noise_fixaa", MODELS, IMAGES, LABELS, S, pert_noise, fix_eps)

print("\n✅ Phase 4 complete")


# ═══════════════════════════════════════════════════════════════════════
# CELL 13 — AUC Summary Table
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("AUC SUMMARY TABLE")
print("="*80)

AUC = {}
AUC["Pure AA (ε=4/255)"] = {m: pure_adv[m][-1].accuracy for m in MODELS}

for sc, data in pure_env.items():
    AUC[sc.replace("pure_","Pure ").replace("_"," ").title()] = {
        m: compute_auc(S, [r.accuracy for r in data[m]]) for m in MODELS}

for sc, cfg in COMP_SCENARIOS.items():
    AUC[cfg["label"]] = {
        m: compute_auc(S, [r.accuracy for r in comp[sc][m]]) for m in MODELS}

ABL_LABELS = {"abl_fixdark_aa": "Fixed Dark + AA Sweep",
              "abl_fullaa_bright": "Full AA + Brightness",
              "abl_fullaa_noise": "Full AA + Noise",
              "abl_noise_fixaa": "Noise + Fixed AA"}
for sc, data in abl.items():
    x = ABL_EPS_AXIS if sc == "abl_fixdark_aa" else S
    AUC[ABL_LABELS[sc]] = {
        m: compute_auc(x, [r.accuracy for r in data[m]]) for m in MODELS}

# Print
hdr = f"{'Scenario':32s}" + "".join(f" {m[:13]:>13s}" for m in MODELS)
print(hdr); print("-"*len(hdr))
for scenario, scores in AUC.items():
    best = max(scores.values())
    row = f"{scenario:32s}"
    for m in MODELS:
        v = scores[m]
        row += f" {v:>11.3f}{'*' if abs(v-best)<1e-4 else ' '} "
    print(row)


# ═══════════════════════════════════════════════════════════════════════
# CELL 14 — Protection Gap (with bootstrap CIs)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("PROTECTION GAP")
print("="*80)

van = DEPLOYMENT_BASELINE
gap_scenes = ["comp_lowlight_aa", "comp_blur_aa", "comp_noise_aa"]
protection_gaps = {}

for rob in ROBUST_NAMES:
    # Pure adversarial masks at ε=4/255
    ar = pure_adv[rob][-1].correct_mask
    av = pure_adv[van][-1].correct_mask
    adv_adv = ar.mean() - av.mean()

    # Compositional AUC advantages
    comp_advs = []
    for sc in gap_scenes:
        ra = compute_auc(S, [r.accuracy for r in comp[sc][rob]])
        va = compute_auc(S, [r.accuracy for r in comp[sc][van]])
        comp_advs.append(ra - va)
    mean_comp = np.mean(comp_advs)
    gap_pct = (1 - mean_comp / adv_adv) * 100 if adv_adv > 0 else float("nan")

    # Bootstrap CI on protection gap (worst compositional scenario)
    worst_sc = gap_scenes[int(np.argmin(comp_advs))]
    cr = comp[worst_sc][rob][-1].correct_mask
    cv = comp[worst_sc][van][-1].correct_mask
    g_val, g_lo, g_hi = bootstrap_protection_gap(
        ar, av, cr, cv, n_bootstrap=CFG.n_bootstrap)

    protection_gaps[rob] = {
        "adv_advantage": float(adv_adv),
        "comp_advantages": comp_advs,
        "mean_comp": float(mean_comp),
        "gap_pct": gap_pct,
        "bootstrap": (g_val, g_lo, g_hi),
    }
    print(f"\n  {rob}")
    print(f"    Adv advantage:  {adv_adv:+.3f}")
    print(f"    Comp advs:      {[f'{a:+.3f}' for a in comp_advs]}")
    print(f"    Protection gap: {gap_pct:.1f}%")
    print(f"    Bootstrap:      {g_val:.3f} [{g_lo:.3f}, {g_hi:.3f}]")


# ═══════════════════════════════════════════════════════════════════════
# CELL 15 — Crossover Detection
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("CROSSOVER DETECTION")
print("="*80)

crossovers = {}
for sc_name, sc_cfg in COMP_SCENARIOS.items():
    for rob in ROBUST_NAMES:
        ar = np.array([r.accuracy for r in comp[sc_name][rob]])
        av = np.array([r.accuracy for r in comp[sc_name][van]])
        cross = find_crossover(S, ar, av)
        key = f"{sc_cfg['label']} | {rob}"
        if cross is not None:
            crossovers[key] = cross.severity
            print(f"  ✗ {key}: reversal at s={cross.severity:.3f}")
        elif av[-1] > ar[-1]:
            print(f"  ~ {key}: vanilla ends ahead ({av[-1]:.1%} vs {ar[-1]:.1%})")
        else:
            print(f"  ✓ {key}: robust dominates")


# ═══════════════════════════════════════════════════════════════════════
# CELL 16 — Confidence Calibration
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("CONFIDENCE CALIBRATION")
print("="*80)

calibration = {}
for mname in MODELS:
    clean = pure_adv[mname][0]
    prof = confidence_profile(clean.samples)
    ece = expected_calibration_error(clean.samples)
    calibration[mname] = {
        "ece": ece,
        "conf_correct": prof.mean_confidence_correct,
        "conf_incorrect": prof.mean_confidence_incorrect,
        "high_conf_errors": prof.pct_high_confidence_errors,
    }
    print(f"  {mname:28s}  ECE={ece:.3f}  "
          f"conf(✓)={prof.mean_confidence_correct:.3f}  "
          f"conf(✗)={prof.mean_confidence_incorrect:.3f}  "
          f"hi-conf-err={prof.pct_high_confidence_errors:.1f}%")


# ═══════════════════════════════════════════════════════════════════════
# CELL 17 — Class Vulnerability
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("CLASS VULNERABILITY (top 5 per robust model)")
print("="*80)

vuln = {}
for rob in ROBUST_NAMES:
    clean = pure_adv[rob][0]
    attacked = pure_adv[rob][-1]
    reps = class_vulnerability(clean.samples, attacked.samples,
                               class_names=class_names, top_k=5)
    vuln[rob] = reps
    print(f"\n  {rob}:")
    for rp in reps:
        print(f"    {rp.class_name:25s}  drop={rp.accuracy_drop:.1%}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 18 — Disagreement Analysis (strong noise, max severity)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("DISAGREEMENT ANALYSIS (strong noise + AA, max severity)")
print("="*80)

sc = "comp_strong_noise_aa"
for rob in ROBUST_NAMES:
    van_r = comp[sc][van][-1]
    rob_r = comp[sc][rob][-1]
    dis = disagreement_analysis(van_r, rob_r)
    print(f"\n  {van} vs {rob}:")
    print(f"    Both correct: {dis.both_correct:.1%}")
    print(f"    Only vanilla: {dis.only_a_correct:.1%}")
    print(f"    Only robust:  {dis.only_b_correct:.1%}")
    print(f"    Both wrong:   {dis.both_wrong:.1%}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 19 — Bootstrap CIs on key comparisons
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("BOOTSTRAP CIs")
print("="*80)

for rob in ROBUST_NAMES:
    d, lo, hi = bootstrap_delta(
        pure_adv[rob][0].correct_mask.astype(float),
        pure_adv[van][0].correct_mask.astype(float),
        n_bootstrap=CFG.n_bootstrap, ci=CFG.ci_level)
    sig = "SIG" if (lo>0 or hi<0) else "n.s."
    print(f"  Clean Δ({rob}−vanilla): {d:+.3f} [{lo:+.3f},{hi:+.3f}] {sig}")

    d2, lo2, hi2 = bootstrap_delta(
        comp["comp_strong_noise_aa"][rob][-1].correct_mask.astype(float),
        comp["comp_strong_noise_aa"][van][-1].correct_mask.astype(float),
        n_bootstrap=CFG.n_bootstrap, ci=CFG.ci_level)
    sig2 = "SIG" if (lo2>0 or hi2<0) else "n.s."
    print(f"  StrongNoise Δ:          {d2:+.3f} [{lo2:+.3f},{hi2:+.3f}] {sig2}\n")


# ═══════════════════════════════════════════════════════════════════════
# CELL 20 — Visualisation
# ═══════════════════════════════════════════════════════════════════════
plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "Vanilla_ResNet50": "#2196F3", "Vanilla_SwinB": "#64B5F6",
    "Liu2023_SwinB": "#E53935", "Singh2023_ConvNeXtB": "#FF9800",
    "Singh2023_ViTB": "#AB47BC",
}
MARKERS = {
    "Vanilla_ResNet50": "o", "Vanilla_SwinB": "D",
    "Liu2023_SwinB": "s", "Singh2023_ConvNeXtB": "^",
    "Singh2023_ViTB": "v",
}

def plot_curves(ax, data, title, xlabel, xv=S):
    for mn, res in data.items():
        ac = [r.accuracy*100 for r in res]
        ax.plot(xv, ac, marker=MARKERS.get(mn,"o"), color=COLORS.get(mn,"gray"),
                label=mn, linewidth=2, markersize=5)
        ax.annotate(f"{ac[-1]:.0f}%", (xv[-1], ac[-1]),
                    textcoords="offset points", xytext=(6,0), fontsize=7)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_ylim(-2, 105); ax.grid(True, alpha=0.3)

# ── Fig 1: Pure environmental (3 panels) ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_curves(axes[0], pure_env["pure_low_light"], "Pure Low-Light",
            f"Severity s (brightness = 1−{CFG.brightness_reduction}s)")
plot_curves(axes[1], pure_env["pure_blur"], "Pure Blur",
            f"Severity s (σ = {CFG.blur_sigma_max}s)")
plot_curves(axes[2], pure_env["pure_noise"], "Pure Noise",
            f"Severity s (std = {CFG.noise_std_max}s)")
axes[2].legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize=7)
fig.suptitle("Environmental Robustness — No Adversarial Attack",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(CFG.save_dir/"fig1_pure_env.pdf", bbox_inches="tight", dpi=300)
plt.savefig(CFG.save_dir/"fig1_pure_env.png", bbox_inches="tight", dpi=300)
plt.show()

# ── Fig 2: Compositional (4 panels) ──────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
for i, (sc, cfg) in enumerate(COMP_SCENARIOS.items()):
    plot_curves(axes[i], comp[sc], cfg["label"],
                f"Severity s (ε={CFG.comp_eps_max}s)")
axes[-1].legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize=7)
fig.suptitle("Compositional Robustness — Environmental + AutoAttack",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(CFG.save_dir/"fig2_compositional.pdf", bbox_inches="tight", dpi=300)
plt.savefig(CFG.save_dir/"fig2_compositional.png", bbox_inches="tight", dpi=300)
plt.show()

# ── Fig 3: Ablations (4 panels) ──────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
abl_specs = [
    ("abl_fixdark_aa",    "Fixed Dark + AA Sweep",  "ε",            ABL_EPS_AXIS),
    ("abl_fullaa_bright", "Full AA + Brightness",   "Darkness sev", S),
    ("abl_fullaa_noise",  "Full AA + Noise",        "Noise sev",    S),
    ("abl_noise_fixaa",   "Noise + Fixed AA",       "Noise sev",    S),
]
for i, (sc, title, xlabel, xv) in enumerate(abl_specs):
    plot_curves(axes[i], abl[sc], title, xlabel, xv=xv)
axes[-1].legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize=7)
fig.suptitle("Ablation Studies", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(CFG.save_dir/"fig3_ablations.pdf", bbox_inches="tight", dpi=300)
plt.savefig(CFG.save_dir/"fig3_ablations.png", bbox_inches="tight", dpi=300)
plt.show()

# ── Fig 4: Protection gap bar chart ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
names = list(protection_gaps.keys())
gaps = [protection_gaps[n]["gap_pct"] for n in names]
cols = ["#e74c3c" if g>50 else "#f39c12" if g>25 else "#4caf50" for g in gaps]
bars = ax.bar(range(len(names)), gaps, color=cols, edgecolor="white")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Protection Gap (%)", fontsize=11)
ax.set_title("Protection Gap by Model", fontsize=13, fontweight="bold")
ax.axhline(50, color="gray", ls="--", alpha=.5, label="50%")
for bar, val in zip(bars, gaps):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig(CFG.save_dir/"fig4_protection_gap.pdf", bbox_inches="tight", dpi=300)
plt.savefig(CFG.save_dir/"fig4_protection_gap.png", bbox_inches="tight", dpi=300)
plt.show()

# ── Fig 5: Confidence calibration ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(MODELS)); w = 0.35
cc = [calibration[m]["conf_correct"] for m in MODELS]
ci = [calibration[m]["conf_incorrect"] for m in MODELS]
ax.bar(x-w/2, cc, w, label="Correct", color="#4caf50", alpha=.8)
ax.bar(x+w/2, ci, w, label="Incorrect", color="#f44336", alpha=.8)
ax.set_xticks(x); ax.set_xticklabels(list(MODELS.keys()), rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Mean Confidence", fontsize=11)
ax.set_title("Confidence Calibration (Clean)", fontsize=13, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig(CFG.save_dir/"fig5_calibration.pdf", bbox_inches="tight", dpi=300)
plt.savefig(CFG.save_dir/"fig5_calibration.png", bbox_inches="tight", dpi=300)
plt.show()


# ═══════════════════════════════════════════════════════════════════════
# CELL 21 — LaTeX table
# ═══════════════════════════════════════════════════════════════════════
def make_latex(auc_table, model_names):
    short = {m: m.replace("Vanilla_","V-").replace("Singh2023_","S23-")
               .replace("Liu2023_","L23-") for m in model_names}
    n = len(model_names)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{AUC scores across all 12 scenarios. "
        r"\textbf{Bold}=best. \underline{Underline}=below deployment baseline.}",
        r"\label{tab:auc}",
        f"\\begin{{tabular}}{{l{'c'*n}}}",
        r"\toprule",
        "Scenario" + "".join(f" & {short[m]}" for m in model_names) + r" \\",
        r"\midrule",
    ]
    vb = DEPLOYMENT_BASELINE
    for scenario, scores in auc_table.items():
        best = max(scores.values()); vv = scores.get(vb, 0)
        row = scenario
        for m in model_names:
            v = scores[m]; cell = f"{v:.3f}"
            if abs(v-best)<1e-4: cell = r"\textbf{"+cell+"}"
            if not m.startswith("Vanilla") and v < vv: cell = r"\underline{"+cell+"}"
            row += f" & {cell}"
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

latex = make_latex(AUC, list(MODELS.keys()))
print(latex)
with open(CFG.save_dir/"table_auc.tex", "w") as f:
    f.write(latex)
print(f"\n✓ LaTeX table → {CFG.save_dir/'table_auc.tex'}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 22 — Save all results (with correct_mask for future analysis)
# ═══════════════════════════════════════════════════════════════════════
def to_json(obj):
    if isinstance(obj, EvaluationResult):
        return {"accuracy": obj.accuracy,
                "mean_confidence": obj.mean_confidence,
                "mean_loss": obj.mean_loss,
                "scenario": obj.scenario,
                "correct_mask": obj.correct_mask.tolist()}
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_json(i) for i in obj]
    if isinstance(obj, Path): return str(obj)
    return obj

all_results = {
    "config": {"n_samples": CFG.n_samples,
               "severity_levels": S.tolist(),
               "aa_version": CFG.aa_version,
               "aa_eps_benchmark": CFG.aa_eps_benchmark,
               "comp_eps_max": CFG.comp_eps_max,
               "models": list(MODELS.keys()),
               "seed": CFG.seed,
               "visprobe_source": VISPROBE_SOURCE},
    "auc_table": AUC,
    "protection_gaps": {k: {kk: vv if not isinstance(vv, np.floating)
                            else float(vv)
                            for kk, vv in v.items()}
                        for k, v in protection_gaps.items()},
    "crossovers": crossovers,
    "calibration": calibration,
    "vulnerability": {m: [(r.class_name, float(r.accuracy_drop)) for r in v]
                      for m, v in vuln.items()},
    "pure_adversarial":   to_json(pure_adv),
    "pure_environmental": to_json(pure_env),
    "compositional":      to_json(comp),
    "ablations":          to_json(abl),
    "sample_indices":     SAMPLE_IDX,
    "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
}

with open(CFG.save_dir/"results_full.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"✓ Full results → {CFG.save_dir/'results_full.json'}")

summary = {"auc_table": AUC, "protection_gaps": all_results["protection_gaps"],
           "crossovers": crossovers, "calibration": calibration,
           "n_samples": CFG.n_samples,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
with open(CFG.save_dir/"results_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"✓ Summary → {CFG.save_dir/'results_summary.json'}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 23 — Final report
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)

print(f"\nDataset:     ImageNet val, {CFG.n_samples} mutually correct (seed={CFG.seed})")
print(f"Attack:      AutoAttack ({CFG.aa_version})")
print(f"VisProbe:    {VISPROBE_SOURCE}")
print(f"Models:      {len(MODELS)}")
print(f"Scenarios:   {len(AUC)}")

print(f"\n{'─'*60}")
print("PURE ADVERSARIAL (AA ε=4/255)")
for m in MODELS:
    print(f"  {m:28s}  {pure_adv[m][-1].accuracy:.1%}")

print(f"\n{'─'*60}")
print("PROTECTION GAPS")
for rob, g in protection_gaps.items():
    print(f"  {rob:28s}  {g['gap_pct']:.1f}%  "
          f"[boot: {g['bootstrap'][0]:.3f}  "
          f"CI: {g['bootstrap'][1]:.3f}–{g['bootstrap'][2]:.3f}]")
valid = [g["gap_pct"] for g in protection_gaps.values() if not np.isnan(g["gap_pct"])]
print(f"  {'MEAN':28s}  {np.mean(valid):.1f}%")

print(f"\n{'─'*60}")
print("ROBUSTNESS REVERSALS (strong noise + AA, s=1.0)")
sc = "comp_strong_noise_aa"
va = comp[sc][van][-1].accuracy
for rob in ROBUST_NAMES:
    ra = comp[sc][rob][-1].accuracy
    tag = "REVERSED ✗" if va > ra else "Robust ✓"
    print(f"  vanilla={va:.1%}  {rob}={ra:.1%}  → {tag}")

print(f"\n{'─'*60}")
print("OUTPUT FILES")
for f in sorted(CFG.save_dir.glob("*")):
    sz = f.stat().st_size / 1024
    print(f"  {f.name:40s}  {sz:.0f} KB")
print("="*80)
