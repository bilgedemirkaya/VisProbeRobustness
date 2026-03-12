"""
EPSILON FIX FOR AUTOATTACK NOTEBOOK

Add this code to your notebook after Step 6 (Define Scenarios) to use
appropriate epsilon values for meaningful evaluation of vanilla models.
"""

import numpy as np

# Better epsilon ranges for vanilla models (which are very vulnerable)
scenarios_fixed = {
    # ==== BASELINES ====
    "Adversarial (AutoAttack)": {
        "env_fn_factory": lambda s: no_env,
        "eps_fn": lambda s: s,
        "severities": np.linspace(0.0, 0.004, 11),  # Max eps=0.004 (vanilla survives partially)
        "xlabel": r"$\varepsilon$ ($\ell_\infty$)",
        "group": "baseline",
        "description": "Pure adversarial — AutoAttack, no environmental degradation",
    },

    "Pure Low-Light (No Attack)": {
        "env_fn_factory": lambda s: (lambda imgs: apply_brightness(imgs, 1.0 - 0.7 * s)),
        "eps_fn": lambda s: 0.0,
        "severities": np.linspace(0.0, 1.0, 11),
        "xlabel": "Darkness severity $s$ (brightness = $1 - 0.7s$)",
        "group": "baseline",
        "description": "Pure environmental — no adversarial attack",
    },

    "Pure Blur (No Attack)": {
        "env_fn_factory": lambda s: (lambda imgs: apply_gaussian_blur(imgs, s * 3.0)),
        "eps_fn": lambda s: 0.0,
        "severities": np.linspace(0.0, 1.0, 11),
        "xlabel": r"Blur severity $s$ ($\sigma = 3s$)",
        "group": "baseline",
        "description": "Pure environmental — no adversarial attack",
    },

    "Pure Noise (No Attack)": {
        "env_fn_factory": lambda s: (lambda imgs: apply_gaussian_noise(imgs, s * 0.1)),
        "eps_fn": lambda s: 0.0,
        "severities": np.linspace(0.0, 1.0, 11),
        "xlabel": "Noise severity $s$ (std = $0.1s$)",
        "group": "baseline",
        "description": "Pure environmental — no adversarial attack",
    },

    # ==== COMPOSITIONAL (with smaller epsilons) ====
    "Low-Light + AutoAttack": {
        "env_fn_factory": lambda s: (lambda imgs: apply_brightness(imgs, 1.0 - 0.7 * s)),
        "eps_fn": lambda s: s * 0.003,  # Max eps=0.003 (vanilla can survive)
        "severities": np.linspace(0.0, 1.0, 11),
        "xlabel": r"Severity $s$ (brightness $1\to0.3$, $\varepsilon = 0.003s$)",
        "group": "compositional",
        "description": "Both degrade together — realistic co-occurrence",
    },

    "Blur + AutoAttack": {
        "env_fn_factory": lambda s: (lambda imgs: apply_gaussian_blur(imgs, s * 2.0)),
        "eps_fn": lambda s: s * 0.003,  # Max eps=0.003
        "severities": np.linspace(0.0, 1.0, 11),
        "xlabel": r"Severity $s$ ($\sigma = 2s$, $\varepsilon = 0.003s$)",
        "group": "compositional",
        "description": "Both degrade together — realistic co-occurrence",
    },

    "Noise + AutoAttack": {
        "env_fn_factory": lambda s: (lambda imgs: apply_gaussian_noise(imgs, s * 0.03)),
        "eps_fn": lambda s: s * 0.003,  # Max eps=0.003
        "severities": np.linspace(0.0, 1.0, 11),
        "xlabel": r"Severity $s$ (noise std = $0.03s$, $\varepsilon = 0.003s$)",
        "group": "compositional",
        "description": "Both degrade together",
    },

    "Strong Noise + AutoAttack": {
        "env_fn_factory": lambda s: (lambda imgs: apply_gaussian_noise(imgs, s * 0.15)),
        "eps_fn": lambda s: s * 0.002,  # Even smaller for strong noise
        "severities": np.linspace(0.0, 1.0, 11),
        "xlabel": r"Severity $s$ (noise std = $0.15s$, $\varepsilon = 0.002s$)",
        "group": "compositional",
        "description": "Strong noise — key reversal scenario",
    },

    # ==== ABLATIONS (with appropriate ranges) ====
    "Fixed Darkness (0.3) + AutoAttack Sweep": {
        "env_fn_factory": lambda eps: (lambda imgs: apply_brightness(imgs, 0.3)),
        "eps_fn": lambda eps: eps,
        "severities": np.linspace(0.0, 0.003, 11),  # Max eps=0.003
        "xlabel": r"$\varepsilon$ ($\ell_\infty$) at fixed brightness = 0.3",
        "group": "ablation",
        "description": "Robust starts broken from darkness — does adversarial defense help?",
    },

    "Full AutoAttack (eps=0.004) + Brightness Sweep": {
        "env_fn_factory": lambda s: (lambda imgs: apply_brightness(imgs, 1.0 - 0.7 * s)),
        "eps_fn": lambda s: 0.004,  # Fixed at threshold where vanilla barely survives
        "severities": np.linspace(0.0, 1.0, 11),
        "xlabel": r"Darkness $s$ at fixed $\varepsilon = 0.004$",
        "group": "ablation",
        "description": "Does robust advantage survive as environment degrades?",
    },

    "Full AutoAttack (eps=0.004) + Noise Sweep": {
        "env_fn_factory": lambda s: (lambda imgs: apply_gaussian_noise(imgs, s * 0.1)),
        "eps_fn": lambda s: 0.004,
        "severities": np.linspace(0.0, 1.0, 11),
        "xlabel": r"Noise std ($0.1s$) at fixed $\varepsilon = 0.004$",
        "group": "ablation",
        "description": "Does noise erode robust advantage at adversarial threshold?",
    },

    "Noise Sweep + Fixed AutoAttack (eps=0.002)": {
        "env_fn_factory": lambda s: (lambda imgs: apply_gaussian_noise(imgs, s * 0.1)),
        "eps_fn": lambda s: 0.002,  # Small enough for vanilla to partially survive
        "severities": np.linspace(0.0, 1.0, 11),
        "xlabel": r"Noise std ($0.1s$) at fixed $\varepsilon = 0.002$",
        "group": "ablation",
        "description": "With AutoAttack (incl. Square Attack), does noise still affect models?",
    }
}

# Usage in notebook:
print("="*80)
print("USING ADJUSTED EPSILON VALUES FOR MEANINGFUL EVALUATION")
print("="*80)
print("Original epsilon ranges would cause vanilla models to immediately collapse to 0%")
print("Adjusted ranges:")
print("  - Pure adversarial: eps up to 0.004 (instead of 0.03)")
print("  - Compositional: eps up to 0.003 (instead of 0.01)")
print("  - This allows us to see the gradual degradation")
print()
print("Expected results with adjusted epsilons:")
print("  - Vanilla ResNet50: 20-80% accuracy range")
print("  - Robust models: 60-95% accuracy range")
print("  - Clear separation between vanilla and robust performance")
print("="*80)