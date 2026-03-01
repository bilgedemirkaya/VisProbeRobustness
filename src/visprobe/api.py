"""
VisProbe unified API for robustness testing.

Provides one main function:
- search(): Find failure thresholds for single perturbation or preset of perturbations
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .report import Report
from .core.search_engine import SearchEngine
from .core.normalization import NormalizationHandler, NORMALIZATION_PRESETS
from .core.search_strategies import SearchStrategy
from .presets import load_preset, is_adversarial_preset
from .properties.classification import LabelConstant
from .strategies.base import CompositeStrategy, Strategy

logger = logging.getLogger(__name__)

# Type aliases
ModelLike = nn.Module
DataLike = Union[DataLoader, TensorDataset, List[tuple], torch.Tensor]

# Adversarial strategy availability
_ADVERSARIAL_AVAILABLE = False
try:
    from .strategies.adversarial import BIMStrategy, FGSMStrategy, PGDStrategy
    _ADVERSARIAL_AVAILABLE = True
except ImportError:
    FGSMStrategy = None
    PGDStrategy = None
    BIMStrategy = None


# =============================================================================
# Shared Utilities
# =============================================================================

def _auto_detect_device() -> torch.device:
    """
    Auto-detect the best available device.

    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _normalize_data(data: DataLike, device: torch.device) -> List[tuple]:
    """
    Normalize various data formats into a list of (image, label) tuples.

    Supports:
    - DataLoader
    - TensorDataset
    - List of (image, label) tuples
    - Single tensor (batch of images)
    """
    # Case 1: DataLoader
    if isinstance(data, DataLoader):
        samples = []
        for batch in data:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
            else:
                images = batch
                labels = torch.zeros(images.shape[0], dtype=torch.long)

            for img, lbl in zip(images, labels):
                samples.append((img.to(device), int(lbl.item())))
        return samples

    # Case 2: TensorDataset
    elif isinstance(data, TensorDataset):
        samples = []
        for item in data:
            if len(item) == 2:
                img, lbl = item
            else:
                img = item[0]
                lbl = 0
            samples.append((img.to(device), int(lbl)))
        return samples

    # Case 3: List of tuples
    elif isinstance(data, list):
        samples = []
        for item in data:
            if isinstance(item, tuple) and len(item) == 2:
                img, lbl = item
                samples.append((img.to(device), int(lbl)))
            else:
                raise ValueError(
                    f"List items must be (image, label) tuples, got {type(item)}"
                )
        return samples

    # Case 4: Single tensor (batch of images)
    elif isinstance(data, torch.Tensor):
        samples = []
        for img in data:
            samples.append((img.to(device), 0))
        return samples

    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. "
            "Expected DataLoader, TensorDataset, list of tuples, or tensor."
        )


def _check_adversarial_available(preset_name: str) -> None:
    """Check if adversarial strategies are available for a preset."""
    if is_adversarial_preset(preset_name) and not _ADVERSARIAL_AVAILABLE:
        raise ImportError(
            f"Preset '{preset_name}' requires adversarial strategies.\n"
            "Install the Adversarial Robustness Toolbox:\n"
            "  pip install adversarial-robustness-toolbox"
        )


def _create_strategy_factory(
    config: Dict[str, Any],
) -> Callable[[float], Strategy]:
    """
    Create a strategy factory function that instantiates strategies at a given level.

    This is crucial for search to work correctly - each level needs a fresh strategy
    configured with that level's parameters.

    Note: Normalization is now handled by SearchEngine, not by individual strategies.

    Args:
        config: Strategy configuration from preset

    Returns:
        Factory function: level -> Strategy
    """
    from .strategies.image import (
        BrightnessStrategy,
        ContrastStrategy,
        GammaStrategy,
        GaussianBlurStrategy,
        GaussianNoiseStrategy,
        JPEGCompressionStrategy,
        MotionBlurStrategy,
    )

    strategy_type = config["type"]

    # Compositional strategies - scale all components together
    if strategy_type == "compositional":
        # Get bounds for all components
        components = config["components"]
        component_factories = [_create_strategy_factory(c) for c in components]
        component_bounds = [_extract_level_bounds(c) for c in components]
        component_identities = [_get_identity_level(c) for c in components]

        # First component's range defines the search level scale
        first_lo, first_hi = component_bounds[0]

        def compositional_factory(level: float) -> Strategy:
            # Compute normalized progress (0 at identity/minimal, 1 at max perturbation)
            first_identity = component_identities[0]
            first_range = first_hi - first_lo

            if first_identity is not None and first_lo <= first_identity <= first_hi:
                # Identity is within range - use distance from identity
                if abs(first_identity - first_lo) >= abs(first_identity - first_hi):
                    # lo is further from identity = max perturbation
                    max_dist = abs(first_identity - first_lo)
                    progress = abs(level - first_identity) / max_dist if max_dist > 0 else 0.0
                else:
                    # hi is further from identity = max perturbation
                    max_dist = abs(first_hi - first_identity)
                    progress = abs(level - first_identity) / max_dist if max_dist > 0 else 0.0
            elif first_identity is not None and first_identity > first_hi:
                # Identity is above range (e.g., jpeg quality=100, range=[30,50])
                # Treat hi as minimal perturbation, lo as max perturbation
                progress = (first_hi - level) / first_range if first_range > 0 else 0.0
            elif first_identity is not None and first_identity < first_lo:
                # Identity is below range
                # Treat lo as minimal perturbation, hi as max perturbation
                progress = (level - first_lo) / first_range if first_range > 0 else 0.0
            else:
                # No identity, use linear interpolation from lo to hi
                progress = (level - first_lo) / first_range if first_range > 0 else 0.0
            progress = max(0.0, min(1.0, progress))

            strategies = []
            for i, (factory, (lo, hi), identity) in enumerate(
                zip(component_factories, component_bounds, component_identities)
            ):
                if i == 0:
                    # First component uses the actual level
                    strategies.append(factory(level))
                else:
                    # Other components scale from identity toward max based on progress
                    if identity is not None:
                        # Interpolate from identity toward the "far" end of range
                        # Determine which end is further from identity (max perturbation)
                        if abs(identity - lo) >= abs(identity - hi):
                            # lo is further from identity, so lo = max perturbation
                            comp_level = identity + progress * (lo - identity)
                        else:
                            # hi is further from identity, so hi = max perturbation
                            comp_level = identity + progress * (hi - identity)
                    else:
                        # No identity, interpolate from lo to hi
                        comp_level = lo + progress * (hi - lo)
                    strategies.append(factory(comp_level))

            return CompositeStrategy(strategies)

        return compositional_factory

    # Natural perturbations - level controls the parameter directly
    if strategy_type == "brightness":
        # level is the brightness_factor (e.g., 0.5 to 1.5)
        return lambda level: BrightnessStrategy(brightness_factor=level)

    elif strategy_type == "contrast":
        # level is the contrast_factor
        return lambda level: ContrastStrategy(contrast_factor=level)

    elif strategy_type == "gamma":
        # level is the gamma value
        return lambda level: GammaStrategy(gamma=level)

    elif strategy_type == "gaussian_blur":
        kernel_size = config.get("kernel_size", 5)
        # level is sigma
        return lambda level: GaussianBlurStrategy(kernel_size=kernel_size, sigma=level)

    elif strategy_type == "motion_blur":
        angle = config.get("angle", 0.0)
        # level is kernel_size (odd integer)
        return lambda level: MotionBlurStrategy(kernel_size=max(1, int(level) | 1), angle=angle)

    elif strategy_type == "jpeg_compression":
        # level is quality (10 to 100)
        return lambda level: JPEGCompressionStrategy(quality=int(level))

    elif strategy_type == "gaussian_noise":
        # level is std_dev (normalization handled by SearchEngine)
        return lambda level: GaussianNoiseStrategy(std_dev=level)

    # Adversarial attacks - level controls epsilon
    elif strategy_type == "fgsm":
        if not _ADVERSARIAL_AVAILABLE:
            raise ImportError(
                "FGSM strategy requires adversarial-robustness-toolbox. "
                "Install with: pip install adversarial-robustness-toolbox"
            )
        # level is eps
        return lambda level: FGSMStrategy(eps=level)

    elif strategy_type == "pgd":
        if not _ADVERSARIAL_AVAILABLE:
            raise ImportError(
                "PGD strategy requires adversarial-robustness-toolbox. "
                "Install with: pip install adversarial-robustness-toolbox"
            )
        max_iter = config.get("max_iter", 20)
        # level is eps, eps_step scales with it (min 1e-6 to avoid ART error)
        return lambda level: PGDStrategy(eps=level, eps_step=max(level / 10, 1e-6), max_iter=max_iter)

    elif strategy_type == "bim":
        if not _ADVERSARIAL_AVAILABLE:
            raise ImportError(
                "BIM strategy requires adversarial-robustness-toolbox. "
                "Install with: pip install adversarial-robustness-toolbox"
            )
        max_iter = config.get("max_iter", 10)
        # level is eps, eps_step scales with it (min 1e-6 to avoid ART error)
        return lambda level: BIMStrategy(eps=level, eps_step=max(level / max_iter, 1e-6), max_iter=max_iter)

    else:
        raise ValueError(f"Unknown strategy type in preset: {strategy_type}")


def _get_identity_level(config: Dict[str, Any]) -> Optional[float]:
    """
    Get the identity level (no perturbation) for a strategy.

    For perturbations like brightness/contrast where 1.0 = no change,
    the search should start at 1.0 and move outward.

    Returns None if the perturbation doesn't have a natural identity point.
    """
    strategy_type = config["type"]

    # Multiplicative factors: 1.0 = identity
    if strategy_type in ("brightness", "contrast", "gamma"):
        return 1.0

    # Additive/magnitude: 0.0 = identity
    elif strategy_type in ("gaussian_noise", "gaussian_blur", "fgsm", "pgd", "bim"):
        return 0.0

    # JPEG: 100 = no compression (identity)
    elif strategy_type == "jpeg_compression":
        return 100.0

    # Motion blur: kernel_size=1 = no blur
    elif strategy_type == "motion_blur":
        return 1.0

    # Compositional: identity is when first component is at identity
    # (all other components also scale to identity at this point)
    elif strategy_type == "compositional":
        first_comp = config["components"][0]
        return _get_identity_level(first_comp)

    return None


def _extract_level_bounds(config: Dict[str, Any]) -> tuple[float, float]:
    """Extract (min_level, max_level) from a strategy config."""
    strategy_type = config["type"]

    # First check for explicit min_level/max_level in config (preferred)
    if "min_level" in config and "max_level" in config:
        return (config["min_level"], config["max_level"])

    # Natural perturbations - fallback defaults
    if strategy_type == "brightness":
        return (config.get("min_factor", 0.5), config.get("max_factor", 1.5))
    elif strategy_type == "contrast":
        return (config.get("min_factor", 0.7), config.get("max_factor", 1.3))
    elif strategy_type == "gamma":
        return (config.get("min_gamma", 0.7), config.get("max_gamma", 1.3))
    elif strategy_type == "gaussian_blur":
        return (config.get("min_sigma", 0.0), config.get("max_sigma", 2.5))
    elif strategy_type == "motion_blur":
        return (config.get("min_kernel", 1), config.get("max_kernel", 25))
    elif strategy_type == "jpeg_compression":
        return (config.get("min_quality", 10), config.get("max_quality", 100))
    elif strategy_type == "gaussian_noise":
        return (config.get("min_std", 0.0), config.get("max_std", 0.05))

    # Adversarial attacks
    elif strategy_type in ("fgsm", "pgd", "bim"):
        return (config.get("min_eps", 0.0), config.get("max_eps", 8 / 255))

    elif strategy_type == "compositional":
        # Use first component's bounds
        first_comp = config["components"][0]
        return _extract_level_bounds(first_comp)

    else:
        return (0.0, 1.0)


def _is_adversarial_strategy(strategy_type: str) -> bool:
    """Check if a strategy type is adversarial."""
    return strategy_type in {"fgsm", "pgd", "bim", "apgd", "square_attack"}


def _compute_threat_model_scores(
    results: List[Dict[str, Any]],
    preset_config: Dict[str, Any],
) -> Dict[str, float]:
    """Compute scores broken down by threat model category."""
    strategies = preset_config.get("strategies", [])

    # Map strategy names to categories
    strategy_categories = {}
    for strat in strategies:
        name = strat.get("name", strat["type"])
        category = strat.get("category", "uncategorized")
        strategy_categories[name] = category

    # Group results by category
    category_scores = {
        "natural": [],
        "adversarial": [],
        "realistic_attack": [],
    }

    for result in results:
        strategy_name = result.get("strategy", "")
        strategy_type = result.get("strategy_type", "")

        # Find category
        category = strategy_categories.get(strategy_name)
        if category is None:
            # Infer from strategy type
            if _is_adversarial_strategy(strategy_type):
                category = "adversarial"
            elif strategy_type == "compositional":
                category = strategy_categories.get(strategy_name, "natural")
            else:
                category = "natural"

        if category in category_scores:
            category_scores[category].append(result.get("robustness_score", 0))

    # Compute averages
    threat_scores = {}
    for category, scores in category_scores.items():
        if scores:
            threat_scores[category] = sum(scores) / len(scores)

    return threat_scores


def _get_class_name(index: int, class_names: Optional[List[str]]) -> str:
    """Get human-readable class name from index."""
    if class_names is not None and 0 <= index < len(class_names):
        return class_names[index]
    return str(index)


def _enrich_failures_with_class_names(
    failures: List[Dict[str, Any]],
    class_names: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Add human-readable class names to failure records."""
    if class_names is None:
        return failures

    enriched = []
    for f in failures:
        enriched_f = dict(f)
        # Map indices to names
        if "original_pred" in f:
            enriched_f["original_pred_name"] = _get_class_name(f["original_pred"], class_names)
        if "perturbed_pred" in f:
            enriched_f["perturbed_pred_name"] = _get_class_name(f["perturbed_pred"], class_names)
        if "original_label" in f:
            enriched_f["original_label_name"] = _get_class_name(f["original_label"], class_names)
        enriched.append(enriched_f)
    return enriched


def _generate_sample_images(
    model: nn.Module,
    samples: List[tuple],
    strategy_factory: Callable[[float], Strategy],
    failure_threshold: float,
    class_names: Optional[List[str]] = None,
    normalization: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate sample images for dashboard visualization.

    Creates original, perturbed, and residual images at the failure threshold.

    Args:
        model: The model under test
        samples: List of (image, label) tuples (may be normalized)
        strategy_factory: Factory to create strategy at a given level
        failure_threshold: The perturbation level where model starts failing
        class_names: Optional class names for predictions
        normalization: Optional normalization handler to denormalize images for display

    Returns:
        Dict with 'original', 'perturbed', and 'residual' image data
    """
    import base64
    import io

    try:
        from PIL import Image as PILImage
    except ImportError:
        # PIL not available, skip image generation
        return {}

    if not samples:
        return {}

    def tensor_to_base64(tensor: torch.Tensor) -> str:
        """Convert a tensor image to base64 string."""
        img_np = tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype("uint8")
        pil_img = PILImage.fromarray(img_np)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Get first sample
    sample_img, sample_label = samples[0]
    sample_batch = sample_img.unsqueeze(0)

    # Denormalize for display if normalization is configured
    if normalization is not None:
        display_img = normalization.denormalize(sample_img)
    else:
        display_img = sample_img

    # Create strategy at failure level and generate perturbation
    try:
        strategy = strategy_factory(failure_threshold)

        # Use the same workflow as SearchEngine:
        # denormalize → perturb in pixel space → clamp → renormalize
        if normalization is not None:
            # Correct workflow: denormalize before perturbation
            perturbed_batch = normalization.denormalize_perturb_normalize(
                images=sample_batch,
                perturbation_fn=strategy,
                model=model,
                level=None,  # Strategy already configured with failure_threshold
                clamp=True,
            )
            # Get denormalized version for display
            display_perturbed = normalization.denormalize(perturbed_batch.squeeze(0))
        else:
            # No normalization - apply perturbation directly
            perturbed_batch = strategy.generate(sample_batch, model, level=failure_threshold)
            display_perturbed = perturbed_batch.squeeze(0)

    except Exception:
        # Strategy failed, skip image generation
        return {}

    # Compute residual on display images (amplified for visibility)
    residual = (display_perturbed - display_img).abs()
    residual = (residual * 5).clamp(0, 1)

    # Get predictions (using normalized images for model)
    with torch.no_grad():
        orig_out = model(sample_batch)
        pert_out = model(perturbed_batch)
        orig_pred = int(torch.argmax(orig_out, dim=-1).item())
        pert_pred = int(torch.argmax(pert_out, dim=-1).item())
        orig_conf = float(torch.softmax(orig_out, dim=-1).max().item())
        pert_conf = float(torch.softmax(pert_out, dim=-1).max().item())

    return {
        "original": {
            "image_b64": tensor_to_base64(display_img),  # Use denormalized display image
            "prediction": orig_pred,
            "prediction_name": _get_class_name(orig_pred, class_names),
            "confidence": orig_conf,
            "label": int(sample_label),
        },
        "perturbed": {
            "image_b64": tensor_to_base64(display_perturbed),  # Use denormalized display image
            "prediction": pert_pred,
            "prediction_name": _get_class_name(pert_pred, class_names),
            "confidence": pert_conf,
            "perturbation_level": failure_threshold,
        },
        "residual": {
            "image_b64": tensor_to_base64(residual),
            "prediction": "Difference (5x amplified)",
            "confidence": 0.0,
        },
    }


def _check_opportunistic_vulnerability(
    threat_scores: Dict[str, float],
    threshold: float = 0.1,
) -> Optional[str]:
    """
    Check if the model is vulnerable to opportunistic attacks.

    KEY INSIGHT: If realistic_attack score is significantly lower than
    both natural and adversarial scores, the model has a blind spot.
    """
    natural = threat_scores.get("natural")
    adversarial = threat_scores.get("adversarial")
    realistic = threat_scores.get("realistic_attack")

    if realistic is None:
        return None

    min_individual = None
    if natural is not None and adversarial is not None:
        min_individual = min(natural, adversarial)
    elif natural is not None:
        min_individual = natural
    elif adversarial is not None:
        min_individual = adversarial

    if min_individual is not None and (min_individual - realistic) > threshold:
        return (
            f"Model vulnerable to opportunistic attacks!\n"
            f"  Natural robustness:        {natural:.1%}\n"
            f"  Adversarial robustness:    {adversarial:.1%}\n"
            f"  Realistic attack:          {realistic:.1%} (gap: {min_individual - realistic:.1%})\n"
            f"  Implication: Attackers can exploit environmental conditions "
            f"to succeed with smaller perturbations."
        )

    return None


# =============================================================================
# Internal: Preset mode
# =============================================================================

def _run_preset_search(
    model: ModelLike,
    samples: List[tuple],
    preset: str,
    budget: int,
    device_obj: torch.device,
    norm_handler: Optional[NormalizationHandler],
    normalization: Union[str, Dict[str, Any], None],
    class_names: Optional[List[str]],
) -> Report:
    """Run multi-perturbation search using a preset."""
    start_time = time.time()

    # Check adversarial availability
    _check_adversarial_available(preset)

    # Load preset
    try:
        preset_config = load_preset(preset)
    except ValueError as e:
        raise ValueError(str(e))

    threat_model = preset_config.get("threat_model", "unknown")
    print(f"Loaded preset: {preset_config['name']}")
    print(f"   {preset_config['description']}")
    print(f"   Threat model: {threat_model}")

    if "novelty" in preset_config:
        print(f"   Key insight: {preset_config['novelty'][:100]}...")

    print(f"   Testing on {len(samples)} samples")
    if norm_handler is not None:
        print(f"   Normalization: {normalization if isinstance(normalization, str) else 'custom'}")
    else:
        print(f"   Normalization: None (data in [0,1] pixel space)")

    # Run search for each strategy
    print(f"\nRunning robustness tests...")
    property_fn = LabelConstant()

    results = []
    total_queries = 0
    all_failures = []

    strategies_config = preset_config["strategies"]
    queries_per_strategy = budget // len(strategies_config)

    for strat_config in strategies_config:
        strategy_factory = _create_strategy_factory(strat_config)
        level_min, level_max = _extract_level_bounds(strat_config)

        # Get identity level (where no perturbation occurs) for proper search start
        identity_level = _get_identity_level(strat_config)
        # Clamp identity to bounds, or use level_lo if no identity point
        if identity_level is not None:
            initial_level = max(level_min, min(identity_level, level_max))
        else:
            initial_level = level_min

        strategy_name = strat_config.get("name", strat_config["type"])
        print(f"\n  Testing: {strategy_name}")

        # Create SearchEngine with proper strategy factory
        engine = SearchEngine(
            model=model,
            strategy_factory=strategy_factory,
            property_fn=property_fn,
            samples=samples,
            search_method="adaptive",
            level_lo=level_min,
            level_hi=level_max,
            initial_level=initial_level,
            max_queries=queries_per_strategy,
            device=device_obj,
            normalization=norm_handler,
            verbose=False,
        )

        # Run with progress bar
        with tqdm(total=queries_per_strategy, desc=f"  {strategy_name}", leave=False) as pbar:
            result = engine.run(progress_bar=pbar)

        result["strategy"] = strategy_name
        result["strategy_type"] = strat_config["type"]
        result["category"] = strat_config.get("category", "uncategorized")
        # Enrich failures with class names
        result["failures"] = _enrich_failures_with_class_names(result["failures"], class_names)
        results.append(result)
        total_queries += result["queries"]
        all_failures.extend(result["failures"])

        print(f"    Failure threshold: {result['failure_threshold']:.3f}")
        print(f"    Robustness score: {result['robustness_score']:.2%}")

        # Show baseline accuracy warning if low
        baseline_acc = result.get("baseline_accuracy")
        valid_samples = result.get("valid_samples", len(samples))
        if baseline_acc is not None and baseline_acc < 1.0:
            print(f"    Baseline accuracy: {baseline_acc:.1%} ({valid_samples}/{result.get('total_samples', len(samples))} samples valid)")
        elif baseline_acc is None:
            print(f"    Mode: Prediction consistency (using model predictions as reference)")

    # Compute overall score
    overall_score = sum(r["robustness_score"] for r in results) / len(results)
    runtime = time.time() - start_time

    # Calculate unique failed samples
    unique_failed_indices = set()
    for failure in all_failures:
        unique_failed_indices.add(failure["index"])

    passed_samples = len(samples) - len(unique_failed_indices)

    # Threat model breakdown for comprehensive preset
    threat_model_scores = {}
    vulnerability_warning = None

    if preset_config.get("outputs_threat_breakdown") or threat_model == "all":
        threat_model_scores = _compute_threat_model_scores(results, preset_config)
        vulnerability_warning = _check_opportunistic_vulnerability(threat_model_scores)

    # Get baseline accuracy from first result (same for all strategies)
    baseline_accuracy = results[0].get("baseline_accuracy") if results else None
    valid_samples_count = results[0].get("valid_samples", len(samples)) if results else len(samples)
    using_model_preds = results[0].get("using_model_predictions", False) if results else False

    # Print summary
    print(f"\nTesting complete!")
    if baseline_accuracy is not None:
        print(f"   Baseline accuracy: {baseline_accuracy:.1%} ({valid_samples_count}/{len(samples)} samples)")
        if baseline_accuracy < 1.0:
            print(f"   NOTE: Only testing on {valid_samples_count} correctly classified samples")
    elif using_model_preds:
        print(f"   Mode: Prediction consistency (using model predictions as reference)")
    print(f"   Overall robustness score: {overall_score:.2%}")

    if threat_model_scores:
        print(f"\n   Threat Model Breakdown:")
        for tm, score in threat_model_scores.items():
            print(f"      {tm}: {score:.2%}")

    if vulnerability_warning:
        print(f"\n   CRITICAL WARNING:")
        print(f"   {vulnerability_warning}")

    print(f"\n   Total failures: {len(all_failures)}")
    print(f"   Unique failed samples: {len(unique_failed_indices)}")
    print(f"   Runtime: {runtime:.1f}s")
    print(f"   Model queries: {total_queries}")

    # Build report
    metrics = {
        "overall_robustness_score": overall_score,
        "baseline_accuracy": baseline_accuracy,
        "valid_samples": valid_samples_count,
        "total_failures": len(all_failures),
        "unique_failed_samples": len(unique_failed_indices),
        "strategies_tested": len(results),
        "threat_model": threat_model,
    }

    if threat_model_scores:
        metrics["threat_model_scores"] = threat_model_scores

    if vulnerability_warning:
        metrics["vulnerability_warning"] = vulnerability_warning

    report = Report(
        test_name=f"search_{preset}",
        test_type="preset",
        runtime=runtime,
        model_queries=total_queries,
        model_name=model.__class__.__name__,
        preset=preset,
        dataset=f"{len(samples)} samples",
        property_name="LabelConstant",
        strategy=preset,
        metrics=metrics,
        search={
            "preset": preset,
            "budget": budget,
            "threat_model": threat_model,
            "results": results,
        },
        total_samples=len(samples),
        passed_samples=passed_samples,
        failures=all_failures,
    )

    return report


# =============================================================================
# Internal: Single perturbation mode
# =============================================================================

def _run_single_search(
    model: ModelLike,
    samples: List[tuple],
    strategy: Union[Strategy, Callable[[float], Strategy]],
    perturbation: Optional[str],
    property_fn: Optional[Callable],
    search_method: Union[str, SearchStrategy, None],
    level_lo: float,
    level_hi: float,
    max_queries: int,
    device_obj: torch.device,
    normalization: Union[str, Dict[str, Any], None],
    verbose: bool,
    class_names: Optional[List[str]],
    strategy_name: Optional[str],
) -> Report:
    """Run single-perturbation threshold search."""
    start_time = time.time()

    if verbose:
        print(f"   Testing on {len(samples)} samples")

    # Default property
    if property_fn is None:
        property_fn = LabelConstant()

    # Convert strategy to factory if needed
    if callable(strategy) and not isinstance(strategy, Strategy):
        strategy_factory = strategy
        # Use provided name, or try to infer from first invocation
        if strategy_name is None:
            # Try to get class name from factory output
            try:
                test_strategy = strategy_factory(level_lo)
                inferred_name = test_strategy.__class__.__name__
                # Convert CamelCase to snake_case for filename compatibility
                import re
                strategy_name = re.sub(r'(?<!^)(?=[A-Z])', '_', inferred_name).lower()
                strategy_name = strategy_name.replace('_strategy', '')
            except Exception:
                strategy_name = "custom_strategy"
    else:
        fixed_strategy = strategy
        strategy_factory = lambda level: fixed_strategy
        if strategy_name is None:
            strategy_name = strategy.__class__.__name__

    # Create SearchEngine
    engine = SearchEngine(
        model=model,
        strategy_factory=strategy_factory,
        property_fn=property_fn,
        samples=samples,
        search_method=search_method,
        level_lo=level_lo,
        level_hi=level_hi,
        max_queries=max_queries,
        device=device_obj,
        normalization=normalization,
        verbose=verbose,
    )

    # Determine search method name for display
    search_method_name = type(engine.search_strategy).__name__

    # Run search
    if verbose:
        print(f"\nSearching for failure threshold ({search_method_name})...")
        with tqdm(total=max_queries, desc="Searching", leave=False) as pbar:
            result = engine.run(progress_bar=pbar)
    else:
        result = engine.run()

    runtime = time.time() - start_time

    # Get baseline accuracy info
    baseline_accuracy = result.get("baseline_accuracy")
    valid_samples = result.get("valid_samples", len(samples))
    using_model_preds = result.get("using_model_predictions", False)

    # Print results
    if verbose:
        print(f"\nSearch complete!")
        if using_model_preds:
            print(f"   Mode: Testing prediction consistency (model predictions as reference)")
            print(f"   NOTE: Labels don't match model's class space - using model's original predictions")
        elif baseline_accuracy is not None:
            print(f"   Baseline accuracy: {baseline_accuracy:.1%} ({valid_samples}/{len(samples)} samples)")
            if baseline_accuracy < 1.0:
                print(f"   NOTE: Only testing on {valid_samples} correctly classified samples")
        print(f"   Failure threshold: {result['failure_threshold']:.4f}")
        print(f"   Robustness score: {result['robustness_score']:.2%}")
        print(f"   Failures found: {len(result['failures'])}")
        print(f"   Queries used: {result['queries']}")
        print(f"   Runtime: {runtime:.1f}s")

    # Calculate unique failed samples
    unique_failed_indices = set()
    for failure in result["failures"]:
        unique_failed_indices.add(failure["index"])

    passed_samples = len(samples) - len(unique_failed_indices)

    # Enrich failures with class names
    enriched_failures = _enrich_failures_with_class_names(result["failures"], class_names)

    # Generate sample images for dashboard visualization
    sample_images = _generate_sample_images(
        model=model,
        samples=samples,
        strategy_factory=strategy_factory,
        failure_threshold=result["failure_threshold"],
        class_names=class_names,
        normalization=engine.normalization,
    )

    # Build report
    report = Report(
        test_name=f"search_{strategy_name}",
        test_type="search",
        runtime=runtime,
        model_queries=result["queries"] * len(samples),
        model_name=model.__class__.__name__,
        dataset=f"{len(samples)} samples",
        property_name="LabelConstant",
        strategy=strategy_name,
        metrics={
            "overall_robustness_score": result["robustness_score"],
            "baseline_accuracy": baseline_accuracy,
            "valid_samples": valid_samples,
            "failure_threshold": result["failure_threshold"],
            "last_pass_level": result["last_pass_level"],
            "total_failures": len(result["failures"]),
            "unique_failed_samples": len(unique_failed_indices),
        },
        search={
            "search_method": search_method_name,
            "level_lo": level_lo,
            "level_hi": level_hi,
            "max_queries": max_queries,
            "search_path": result["search_path"],
        },
        total_samples=len(samples),
        passed_samples=passed_samples,
        failures=enriched_failures,
        sample_images=sample_images,
    )

    return report


# =============================================================================
# Public API: search()
# =============================================================================

def search(
    model: ModelLike,
    data: DataLike,
    *,
    # Mode selection (mutually exclusive)
    preset: Optional[str] = None,
    perturbation: Optional[str] = None,
    strategy: Union[Strategy, Callable[[float], Strategy], None] = None,
    # Search parameters
    property_fn: Optional[Callable] = None,
    search_method: Union[str, SearchStrategy, None] = None,
    level_lo: Optional[float] = None,
    level_hi: Optional[float] = None,
    max_queries: int = 100,
    budget: int = 1000,
    # Environment
    device: str = "auto",
    normalization: Union[str, Dict[str, Any], None] = "imagenet",
    # Output options
    verbose: bool = True,
    class_names: Optional[List[str]] = None,
    strategy_name: Optional[str] = None,
) -> Report:
    """
    Find failure thresholds for model robustness testing.

    VisProbe tests **accuracy preservation**: models should maintain correct predictions
    when inputs are perturbed. Only correctly-classified samples are tested.

    Three ways to specify what to test (mutually exclusive):

    1. **Preset** - Test multiple perturbations at once:
        >>> report = search(model, data, preset="natural")
        >>> report = search(model, data, preset="comprehensive")

    2. **Perturbation** - Test a single perturbation (simple API):
        >>> report = search(model, data, perturbation="gaussian_blur")
        >>> report = search(model, data, perturbation=Perturbation.GAUSSIAN_NOISE)

    3. **Strategy** - Full control (advanced API):
        >>> report = search(model, data, strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
        ...                 level_lo=0.0, level_hi=0.15)

    Args:
        model: PyTorch model to test
        data: Test data (DataLoader, TensorDataset, list of tuples, or tensor)

        preset: Preset name for multi-perturbation testing.
                Options: "natural", "adversarial", "comprehensive", "lighting", etc.
                Use list_presets() to see all available presets.

        perturbation: Single perturbation name (e.g., "gaussian_blur", "gaussian_noise").
                      Use list_perturbations() to see available options.
                      Ranges auto-selected based on normalization parameter.

        strategy: (Advanced) Strategy instance or factory function.
                  Use when you need full control over perturbation parameters.

        property_fn: Property to test (default: LabelConstant - prediction unchanged)
        search_method: Search algorithm - "adaptive" (default), "binary", or "bayesian"
        level_lo: Lower bound for search range (auto-set for perturbation/preset)
        level_hi: Upper bound for search range (auto-set for perturbation/preset)
        max_queries: Max search iterations for single perturbation (default: 100)
        budget: Total query budget for preset mode (default: 1000)

        device: Device to use ("auto", "cuda", "cpu", or "mps")
        normalization: Dataset normalization. Options:
            - "imagenet": ImageNet normalization (default)
            - "cifar10", "cifar100", "mnist": Dataset-specific normalization
            - {"mean": [...], "std": [...]}: Custom normalization
            - None: No normalization (data in [0,1] pixel space)

        verbose: Print progress (default: True)
        class_names: Optional class names for human-readable output
        strategy_name: Name for custom strategy (auto-inferred if not provided)

    Returns:
        Report with robustness score, failure threshold, and detailed metrics.
        Call report.save() to persist for dashboard visualization.

    Examples:
        >>> # Test with preset (multiple perturbations)
        >>> report = search(model, data, preset="natural")
        >>> print(f"Score: {report.score:.1f}%")

        >>> # Test single perturbation
        >>> report = search(model, data, perturbation="gaussian_blur")
        >>> print(f"Fails at blur σ={report.metrics['failure_threshold']:.2f}")

        >>> # Advanced: custom strategy with explicit ranges
        >>> from visprobe.strategies.adversarial import FGSMStrategy
        >>> report = search(model, data,
        ...                 strategy=lambda eps: FGSMStrategy(eps=eps),
        ...                 level_lo=0.0, level_hi=0.03)
    """
    # Validate mutually exclusive parameters
    mode_count = sum([preset is not None, perturbation is not None, strategy is not None])
    if mode_count == 0:
        raise ValueError(
            "Must specify one of: preset, perturbation, or strategy.\n"
            "Examples:\n"
            "  search(model, data, preset='natural')  # Multi-perturbation\n"
            "  search(model, data, perturbation='gaussian_blur')  # Single perturbation"
        )
    if mode_count > 1:
        raise ValueError(
            "Specify only ONE of: preset, perturbation, or strategy.\n"
            "They are mutually exclusive modes."
        )

    # Auto-detect device
    if device == "auto":
        device_obj = _auto_detect_device()
    else:
        device_obj = torch.device(device) if isinstance(device, str) else device

    if verbose:
        print(f"Using device: {device_obj}")

    # Prepare model
    model = model.to(device_obj)
    model.eval()

    # Normalize data
    if verbose:
        print("Preparing data...")
    samples = _normalize_data(data, device_obj)

    # Parse normalization config
    norm_handler = NormalizationHandler.from_config(normalization)

    # Route to appropriate mode
    if preset is not None:
        # Multi-perturbation preset mode
        return _run_preset_search(
            model=model,
            samples=samples,
            preset=preset,
            budget=budget,
            device_obj=device_obj,
            norm_handler=norm_handler,
            normalization=normalization,
            class_names=class_names,
        )
    else:
        # Single perturbation mode
        if perturbation is not None:
            from .perturbations import get_perturbation

            # Infer dataset type from normalization parameter
            dataset_type = normalization if isinstance(normalization, str) else None

            # Get perturbation spec with appropriate ranges
            strategy_factory, auto_lo, auto_hi, auto_name = get_perturbation(
                perturbation, preset=dataset_type
            )

            # Use auto-detected ranges if not explicitly provided
            if level_lo is None:
                level_lo = auto_lo
            if level_hi is None:
                level_hi = auto_hi
            if strategy_name is None:
                strategy_name = perturbation

            strategy = strategy_factory

        # Set default ranges if still not provided
        if level_lo is None:
            level_lo = 0.0
        if level_hi is None:
            level_hi = 1.0

        return _run_single_search(
            model=model,
            samples=samples,
            strategy=strategy,
            perturbation=perturbation,
            property_fn=property_fn,
            search_method=search_method,
            level_lo=level_lo,
            level_hi=level_hi,
            max_queries=max_queries,
            device_obj=device_obj,
            normalization=normalization,
            verbose=verbose,
            class_names=class_names,
            strategy_name=strategy_name,
        )
