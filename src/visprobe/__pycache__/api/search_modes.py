"""
Search mode implementations (adaptive, grid, random, binary) for threshold discovery.

Provides efficient perturbation threshold search with:
- Exponential exploration for faster bound discovery
- Golden section search for efficient refinement
- Evaluation caching to avoid redundant model queries
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

from ..strategies.base import Strategy  # noqa: E402
from .model_wrap import _ModelWithIntermediateOutput  # noqa: E402
from .query_counter import QueryCounter  # noqa: E402


# ===== Helper Functions =====


def _build_search_path_entry(
    pert_out: torch.Tensor,
    level: float,
    batch_pass: bool,
    passed_mask: List[bool],
    passed_frac: float,
    class_names: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Build a standardized search path entry dictionary.

    Args:
        pert_out: Perturbed model output (logits)
        level: Current perturbation level
        batch_pass: Whether the batch passed the property test
        passed_mask: Per-sample pass/fail mask
        passed_frac: Fraction of samples that passed
        class_names: List of class names for predictions

    Returns:
        Dictionary containing level, pass status, predictions, and confidences
    """
    pert_pred_labels = torch.argmax(pert_out, dim=-1)
    pred_indices = pert_pred_labels.tolist()
    conf_list = torch.softmax(pert_out, dim=-1).max(dim=-1).values.tolist()

    return {
        "level": float(level),
        "passed": batch_pass,
        "passed_all": all(passed_mask),
        "passed_frac": float(passed_frac),
        "passed_mask": passed_mask,
        "prediction": (
            class_names[pred_indices[0]] if class_names else str(pred_indices[0])
        ),
        "confidence": conf_list[0],
        "predictions": [
            class_names[i] if class_names else str(i) for i in pred_indices
        ],
        "confidences": conf_list,
    }


def _configure_strategy_safe(perturb_obj: Any, runner: Any) -> None:
    """
    Safely configure a perturbation strategy with context from runner.

    Attempts to call configure() method with mean/std/seed. Falls back
    to direct attribute setting if configure() is not available.

    Args:
        perturb_obj: The perturbation strategy object to configure
        runner: TestRunner instance with context containing mean/std/seed
    """
    try:
        perturb_obj.configure(
            mean=runner.context["mean"],
            std=runner.context["std"],
            seed=runner.context.get("seed"),
        )
    except Exception:
        # Fallback: use internal _configure_strategy if available
        try:
            from .runner import TestRunner as _TR
            _TR._configure_strategy(
                perturb_obj,
                runner.context["mean"],
                runner.context["std"],
                None,
            )
        except Exception:
            # If both fail, strategy will use its defaults
            pass


# ===== Strategy Resolution =====


def resolve_strategy_for_level(runner, params: Dict[str, Any], level: float):
    """
    Resolve a strategy specification at a specific perturbation level.

    Handles both factory callables and static strategy objects, setting
    the appropriate level/eps attribute.

    Args:
        runner: TestRunner instance with context
        params: Search parameters containing strategy specification
        level: Perturbation level to use

    Returns:
        Configured Strategy instance
    """
    strategy_spec = params["strategy"]
    perturb_obj = None
    if callable(strategy_spec):
        try:
            candidate = strategy_spec(level)
            if callable(getattr(candidate, "generate", None)):
                perturb_obj = candidate
        except TypeError:
            perturb_obj = None
        if perturb_obj is None:
            perturb_obj = Strategy.resolve(strategy_spec, level=level)
    else:
        perturb_obj = Strategy.resolve(strategy_spec)
        if hasattr(perturb_obj, "eps"):
            perturb_obj.eps = level
        elif hasattr(perturb_obj, "level"):
            perturb_obj.level = level
    return perturb_obj


def _compute_batch_pass_from_mask(passed_mask: List[bool], reducer_spec: Any) -> Tuple[bool, float]:
    """Returns (batch_pass, passed_fraction) using the configured reducer."""
    if not passed_mask:
        return False, 0.0
    try:
        import torch as _torch

        passed_frac = _torch.tensor(passed_mask, dtype=_torch.float32).mean().item()
    except Exception:
        passed_frac = sum(1 for v in passed_mask if v) / max(1, len(passed_mask))

    reducer = reducer_spec or "all"
    if reducer == "all":
        return all(passed_mask), passed_frac
    if reducer == "any":
        return any(passed_mask), passed_frac
    if isinstance(reducer, str) and reducer.startswith("frac>="):
        try:
            thr = float(reducer.split(">=", 1)[1])
        except Exception:
            thr = 1.0
        return passed_frac >= thr, passed_frac
    # default
    return all(passed_mask), passed_frac


def perform_adaptive_search(  # noqa: C901
    runner, params: Dict[str, Any], clean_results: Tuple
) -> Dict:
    """
    Smart adaptive search with exponential exploration and binary refinement.

    Phase 1: Exponential exploration to quickly find failure region
    Phase 2: Binary search refinement within the failure bracket

    This is more efficient than step-halving when the failure threshold
    is unknown and could be anywhere in the range.
    """
    ctx = runner.context
    m, batch_tensor = ctx["model"], ctx["batch_tensor"]
    clean_logits, clean_feat = clean_results
    top_k = params.get("top_k")

    # Configuration with smarter defaults
    initial_step = float(params.get("initial_level", 0.01))
    min_step = float(params.get("min_step", 1e-5))
    max_queries = int(params.get("max_queries", 500))
    level_lo = float(params.get("level_lo", 0.0))
    level_hi = float(params.get("level_hi", 1.0))
    reducer_spec = params.get("reduce", "all")
    use_cache = params.get("cache_evaluations", True)

    path: List[Dict[str, Any]] = []
    top_k_path: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {"failure_threshold": None}
    best_fail: Optional[float] = None

    # Initialize evaluation cache for efficiency
    cache: Optional[Dict[float, Tuple[bool, float, List[bool]]]] = {} if use_cache else None
    cache_hits = 0

    # Per-image bracketing
    try:
        batch_size_local = int(clean_logits.shape[0])
    except Exception:
        batch_size_local = int(ctx["batch_size"])
    last_pass_levels: List[Optional[float]] = [None] * batch_size_local
    first_fail_levels: List[Optional[float]] = [None] * batch_size_local

    m.eval()
    first_failing_index: Optional[int] = None

    def _quantize_level(lvl: float) -> float:
        """Quantize level for cache key."""
        return round(lvl / 1e-6) * 1e-6

    def evaluate_at_level(level: float) -> Tuple[bool, float, List[bool], Any, Any, Any]:
        """Evaluate model at a level, using cache if available."""
        nonlocal cache_hits

        # Check cache
        cache_key = _quantize_level(level)
        if cache is not None and cache_key in cache:
            cached_batch_pass, cached_frac, cached_mask = cache[cache_key]
            cache_hits += 1
            # Still need to generate tensor for result
            perturb_obj = resolve_strategy_for_level(runner, params, level)
            _configure_strategy_safe(perturb_obj, runner)
            with QueryCounter(m) as qc:
                pert_tensor = perturb_obj.generate(batch_tensor, model=m, level=level)
                pert_model_output = m(pert_tensor)
            runner.query_count += qc.extra
            if isinstance(m, _ModelWithIntermediateOutput):
                pert_out_i, pert_feat_i = pert_model_output
            else:
                pert_out_i, pert_feat_i = pert_model_output, None
            return cached_batch_pass, cached_frac, cached_mask, pert_tensor, pert_out_i, pert_feat_i

        # Not cached - evaluate
        perturb_obj = resolve_strategy_for_level(runner, params, level)
        _configure_strategy_safe(perturb_obj, runner)

        with QueryCounter(m) as qc:
            pert_tensor = perturb_obj.generate(batch_tensor, model=m, level=level)
            pert_model_output = m(pert_tensor)
            if isinstance(m, _ModelWithIntermediateOutput):
                pert_out_i, pert_feat_i = pert_model_output
            else:
                pert_out_i, pert_feat_i = pert_model_output, None

            passed_mask: List[bool] = runner._evaluate_property_mask(
                clean_logits, runner._to_logits(pert_out_i)
            )
            batch_pass, passed_frac = _compute_batch_pass_from_mask(passed_mask, reducer_spec)

        runner.query_count += qc.extra

        # Store in cache
        if cache is not None:
            cache[cache_key] = (batch_pass, passed_frac, passed_mask)

        return batch_pass, passed_frac, passed_mask, pert_tensor, pert_out_i, pert_feat_i

    def log_iteration(level: float, batch_pass: bool, passed_mask: List[bool],
                      passed_frac: float, pert_out_i: Any, phase: str) -> None:
        """Log a search iteration to path."""
        entry = _build_search_path_entry(
            pert_out_i, level, batch_pass, passed_mask, passed_frac, ctx["class_names"]
        )
        entry["phase"] = phase
        path.append(entry)

        if top_k:
            from .analysis_utils import run_top_k_analysis as _run_tk
            overlap = _run_tk(runner, top_k, clean_logits, runner._to_logits(pert_out_i))
            if overlap is not None:
                top_k_path.append({"level": float(level), "overlap": overlap})

    def update_brackets(level: float, passed_mask: List[bool]) -> None:
        """Update per-sample bracket tracking."""
        nonlocal first_failing_index
        for si, ok in enumerate(passed_mask):
            if ok:
                last_pass_levels[si] = float(level)
            elif first_fail_levels[si] is None:
                first_fail_levels[si] = float(level)
                if first_failing_index is None:
                    first_failing_index = si

    # ========== PHASE 1: Exponential exploration ==========
    # Quickly find bounds by doubling step size until we hit a failure
    level = level_lo + initial_step
    step = initial_step
    last_pass = level_lo
    found_failure = False
    max_exploration_iters = max_queries // 2  # Reserve half for refinement

    logger.debug(f"Phase 1: Exponential exploration starting at {level}")

    exploration_count = 0
    while level <= level_hi and runner.query_count < max_exploration_iters:
        if runner.query_count >= max_queries:
            break

        level = min(level, level_hi)  # Clamp to upper bound

        batch_pass, passed_frac, passed_mask, pert_tensor, pert_out_i, pert_feat_i = \
            evaluate_at_level(level)

        log_iteration(level, batch_pass, passed_mask, passed_frac, pert_out_i, "exploration")
        update_brackets(level, passed_mask)
        exploration_count += 1

        if batch_pass:
            last_pass = level
            # Exponential increase for faster exploration
            step *= 2
            level = level + step
        else:
            # Found failure - record and switch to refinement
            found_failure = True
            if best_fail is None or level < best_fail:
                best_fail = float(level)
                result.update({
                    "failure_threshold": best_fail,
                    "perturbed_tensor": pert_tensor,
                    "perturbed_output": pert_out_i,
                    "perturbed_features": pert_feat_i,
                })
            break

    logger.debug(f"Phase 1 complete: {exploration_count} iterations, found_failure={found_failure}")

    # ========== PHASE 2: Binary search refinement ==========
    # Refine the failure threshold within the bracket [last_pass, best_fail]
    if found_failure and best_fail is not None:
        lo, hi = last_pass, best_fail

        logger.debug(f"Phase 2: Binary refinement between {lo} and {hi}")
        refinement_count = 0

        while (hi - lo) > min_step and runner.query_count < max_queries:
            mid = (lo + hi) / 2.0

            batch_pass, passed_frac, passed_mask, pert_tensor, pert_out_i, pert_feat_i = \
                evaluate_at_level(mid)

            log_iteration(mid, batch_pass, passed_mask, passed_frac, pert_out_i, "refinement")
            update_brackets(mid, passed_mask)
            refinement_count += 1

            if batch_pass:
                lo = mid
            else:
                hi = mid
                if mid < best_fail:
                    best_fail = float(mid)
                    result.update({
                        "failure_threshold": best_fail,
                        "perturbed_tensor": pert_tensor,
                        "perturbed_output": pert_out_i,
                        "perturbed_features": pert_feat_i,
                    })

        logger.debug(f"Phase 2 complete: {refinement_count} iterations")

    # If no failure found, try maximum level as final check
    if best_fail is None and runner.query_count < max_queries:
        batch_pass, passed_frac, passed_mask, pert_tensor, pert_out_i, pert_feat_i = \
            evaluate_at_level(level_hi)
        log_iteration(level_hi, batch_pass, passed_mask, passed_frac, pert_out_i, "final_check")
        update_brackets(level_hi, passed_mask)

        if not batch_pass:
            best_fail = float(level_hi)
            result.update({
                "failure_threshold": best_fail,
                "perturbed_tensor": pert_tensor,
                "perturbed_output": pert_out_i,
                "perturbed_features": pert_feat_i,
            })

    # Finalize output
    result["failure_threshold"] = best_fail
    result["path"] = path
    result["top_k_path"] = top_k_path
    result["cache_hits"] = cache_hits
    try:
        result["first_failing_index"] = int(
            first_failing_index if first_failing_index is not None else 0
        )
    except Exception:
        result["first_failing_index"] = 0

    # Midpoint estimate per image when bracketed
    per_sample_thresholds: List[Optional[float]] = [
        (0.5 * (float(lo) + float(hi))) if (lo is not None and hi is not None) else None
        for lo, hi in zip(last_pass_levels, first_fail_levels)
    ]
    result["per_sample_thresholds"] = per_sample_thresholds

    logger.info(f"Adaptive search complete: threshold={best_fail}, queries={runner.query_count}, cache_hits={cache_hits}")
    return result


def perform_grid_search(runner, params: Dict[str, Any], clean_results: Tuple) -> Dict:  # noqa: C901
    """
    Perform grid search by testing evenly-spaced perturbation levels.

    Tests a fixed number of levels uniformly distributed between level_lo and level_hi.
    Stops at the first failure point.

    Args:
        runner: TestRunner instance
        params: Search parameters (level_lo, level_hi, num_levels, etc.)
        clean_results: Tuple of (clean_logits, clean_features)

    Returns:
        Dictionary with failure_threshold, path, and top_k_path
    """
    ctx = runner.context
    m, batch_tensor = ctx["model"], ctx["batch_tensor"]
    clean_logits, _ = clean_results
    top_k = params.get("top_k")

    level_lo = float(params.get("level_lo", 0.0))
    level_hi = float(params.get("level_hi", 0.2))
    num_levels = int(params.get("num_levels", 21))
    max_queries = int(params.get("max_queries", 500))

    levels = np.linspace(level_lo, level_hi, num_levels)
    path, top_k_path = [], []
    result: Dict[str, Any] = {"failure_threshold": None}

    for level in levels:
        if runner.query_count >= max_queries:
            break
        level = float(max(level, 0.0))
        perturb_obj = resolve_strategy_for_level(runner, params, level)
        try:
            perturb_obj.configure(
                mean=runner.context["mean"],
                std=runner.context["std"],
                seed=runner.context.get("seed"),
            )
        except Exception:
            try:
                from .runner import TestRunner as _TR

                _TR._configure_strategy(
                    perturb_obj, runner.context["mean"], runner.context["std"], None
                )
            except Exception:
                pass

        m.eval()
        with QueryCounter(m) as qc:
            pert_tensor = perturb_obj.generate(batch_tensor, model=m, level=level)
            pert_model_output = m(pert_tensor)
            if isinstance(m, _ModelWithIntermediateOutput):
                pert_out_i, _ = pert_model_output
            else:
                pert_out_i = pert_model_output

            passed = (
                runner._evaluate_property(
                    clean_logits, runner._to_logits(pert_out_i), vectorized=True
                )
                > 0
            )

        runner.query_count += qc.extra

        pred_indices = torch.argmax(pert_out_i, dim=-1).tolist()
        conf_list = torch.softmax(pert_out_i, dim=-1).max(dim=-1).values.tolist()
        path.append(
            {
                "level": float(level),
                "passed": passed,
                "prediction": (
                    ctx["class_names"][pred_indices[0]]
                    if ctx["class_names"]
                    else str(pred_indices[0])
                ),
                "confidence": conf_list[0],
                "predictions": [
                    ctx["class_names"][i] if ctx["class_names"] else str(i) for i in pred_indices
                ],
                "confidences": conf_list,
            }
        )

        from .analysis_utils import run_top_k_analysis

        if top_k:
            overlap = run_top_k_analysis(runner, top_k, clean_logits, runner._to_logits(pert_out_i))
            if overlap is not None:
                top_k_path.append({"level": level, "overlap": overlap})

        if not passed and result["failure_threshold"] is None:
            result.update(
                {
                    "failure_threshold": float(level),
                    "perturbed_tensor": pert_tensor,
                    "perturbed_output": pert_out_i,
                }
            )
            break

    result["path"] = path
    result["top_k_path"] = top_k_path
    return result


def perform_random_search(  # noqa: C901
    runner, params: Dict[str, Any], clean_results: Tuple
) -> Dict:
    """
    Perform random search by sampling perturbation levels uniformly.

    Randomly samples num_samples levels between level_lo and level_hi,
    tracking the minimum failure level encountered.

    Args:
        runner: TestRunner instance
        params: Search parameters (level_lo, level_hi, num_samples, etc.)
        clean_results: Tuple of (clean_logits, clean_features)

    Returns:
        Dictionary with failure_threshold, path, top_k_path, and fail_levels
    """
    ctx = runner.context
    m, batch_tensor = ctx["model"], ctx["batch_tensor"]
    clean_logits, _ = clean_results
    top_k = params.get("top_k")

    level_lo = float(params.get("level_lo", 0.0))
    level_hi = float(params.get("level_hi", 0.2))
    num_samples = int(params.get("num_samples", 64))
    max_queries = int(params.get("max_queries", 500))

    rng = np.random.default_rng(params.get("random_seed", None))
    levels = rng.uniform(level_lo, level_hi, size=num_samples)
    path, top_k_path = [], []
    fail_levels: List[float] = []
    best_fail = None
    result: Dict[str, Any] = {"failure_threshold": None}

    for level in levels:
        if runner.query_count >= max_queries:
            break
        level = float(max(level, 0.0))
        perturb_obj = resolve_strategy_for_level(runner, params, level)
        try:
            perturb_obj.configure(
                mean=runner.context["mean"],
                std=runner.context["std"],
                seed=runner.context.get("seed"),
            )
        except Exception:
            try:
                from .runner import TestRunner as _TR

                _TR._configure_strategy(
                    perturb_obj, runner.context["mean"], runner.context["std"], None
                )
            except Exception:
                pass

        m.eval()
        with QueryCounter(m) as qc:
            pert_tensor = perturb_obj.generate(batch_tensor, model=m, level=level)
            pert_model_output = m(pert_tensor)
            if isinstance(m, _ModelWithIntermediateOutput):
                pert_out_i, _ = pert_model_output
            else:
                pert_out_i = pert_model_output

            passed = (
                runner._evaluate_property(
                    clean_logits, runner._to_logits(pert_out_i), vectorized=True
                )
                > 0
            )

        runner.query_count += qc.extra

        pred_indices = torch.argmax(pert_out_i, dim=-1).tolist()
        conf_list = torch.softmax(pert_out_i, dim=-1).max(dim=-1).values.tolist()
        path.append(
            {
                "level": float(level),
                "passed": passed,
                "prediction": (
                    ctx["class_names"][pred_indices[0]]
                    if ctx["class_names"]
                    else str(pred_indices[0])
                ),
                "confidence": conf_list[0],
                "predictions": [
                    ctx["class_names"][i] if ctx["class_names"] else str(i) for i in pred_indices
                ],
                "confidences": conf_list,
            }
        )

        from .analysis_utils import run_top_k_analysis

        if top_k:
            overlap = run_top_k_analysis(runner, top_k, clean_logits, runner._to_logits(pert_out_i))
            if overlap is not None:
                top_k_path.append({"level": level, "overlap": overlap})

        if not passed:
            fail_levels.append(float(level))
            if best_fail is None or level < best_fail:
                best_fail = float(level)
                result.update(
                    {
                        "failure_threshold": best_fail,
                        "perturbed_tensor": pert_tensor,
                        "perturbed_output": pert_out_i,
                    }
                )

    result["path"] = path
    result["top_k_path"] = top_k_path
    result["fail_levels"] = fail_levels
    return result


def perform_binary_search(  # noqa: C901
    runner, params: Dict[str, Any], clean_results: Tuple
) -> Dict:
    """
    Optimized binary search for finding failure threshold.

    More efficient than adaptive search for finding the exact failure point.
    Uses true binary search with O(log n) complexity and evaluation caching.
    """
    ctx = runner.context
    m, batch_tensor = ctx["model"], ctx["batch_tensor"]
    clean_logits, clean_feat = clean_results
    top_k = params.get("top_k")

    # Binary search bounds
    lo = float(params.get("level_lo", 0.0))
    hi = float(params.get("level_hi", 1.0))
    min_step = float(params.get("min_step", 1e-5))
    max_queries = int(params.get("max_queries", 500))
    reducer_spec = params.get("reduce", "all")
    use_cache = params.get("cache_evaluations", True)

    path: List[Dict[str, Any]] = []
    top_k_path: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {"failure_threshold": None}
    best_fail: Optional[float] = None

    # Initialize evaluation cache
    cache: Optional[Dict[float, Tuple[bool, float, List[bool]]]] = {} if use_cache else None
    cache_hits = 0

    # Per-image bracketing
    try:
        batch_size_local = int(clean_logits.shape[0])
    except Exception:
        batch_size_local = int(ctx["batch_size"])
    last_pass_levels: List[Optional[float]] = [None] * batch_size_local
    first_fail_levels: List[Optional[float]] = [None] * batch_size_local

    m.eval()
    first_failing_index: Optional[int] = None

    def _quantize_level(lvl: float) -> float:
        """Quantize level for cache key."""
        return round(lvl / 1e-6) * 1e-6

    logger.debug(f"Starting binary search between {lo} and {hi}")

    while (hi - lo) > min_step and runner.query_count < max_queries:
        # Binary search midpoint
        level = (lo + hi) / 2.0
        cache_key = _quantize_level(level)

        # Check cache first
        if cache is not None and cache_key in cache:
            cached_batch_pass, cached_frac, cached_mask = cache[cache_key]
            cache_hits += 1
            # Still need to generate tensor for result
            perturb_obj = resolve_strategy_for_level(runner, params, level)
            _configure_strategy_safe(perturb_obj, runner)
            with QueryCounter(m) as qc:
                pert_tensor = perturb_obj.generate(batch_tensor, model=m, level=level)
                pert_model_output = m(pert_tensor)
            runner.query_count += qc.extra
            if isinstance(m, _ModelWithIntermediateOutput):
                pert_out_i, pert_feat_i = pert_model_output
            else:
                pert_out_i, pert_feat_i = pert_model_output, None
            batch_pass, passed_frac, passed_mask = cached_batch_pass, cached_frac, cached_mask
        else:
            # Resolve and configure strategy for this level
            perturb_obj = resolve_strategy_for_level(runner, params, level)
            _configure_strategy_safe(perturb_obj, runner)

            with QueryCounter(m) as qc:
                pert_tensor = perturb_obj.generate(batch_tensor, model=m, level=level)
                pert_model_output = m(pert_tensor)
                if isinstance(m, _ModelWithIntermediateOutput):
                    pert_out_i, pert_feat_i = pert_model_output
                else:
                    pert_out_i, pert_feat_i = pert_model_output, None

                # Evaluate property mask
                passed_mask: List[bool] = runner._evaluate_property_mask(
                    clean_logits, runner._to_logits(pert_out_i)
                )
                batch_pass, passed_frac = _compute_batch_pass_from_mask(passed_mask, reducer_spec)

            runner.query_count += qc.extra

            # Store in cache
            if cache is not None:
                cache[cache_key] = (batch_pass, passed_frac, passed_mask)

        # Log this iteration using helper function
        path.append(
            _build_search_path_entry(
                pert_out_i, level, batch_pass, passed_mask, passed_frac, ctx["class_names"]
            )
        )

        if top_k:
            from .analysis_utils import run_top_k_analysis as _run_tk

            overlap = _run_tk(runner, top_k, clean_logits, runner._to_logits(pert_out_i))
            if overlap is not None:
                top_k_path.append({"level": float(level), "overlap": overlap})

        # Update per-sample brackets
        for si, ok in enumerate(passed_mask):
            if ok:
                last_pass_levels[si] = float(level)
            elif first_fail_levels[si] is None:
                first_fail_levels[si] = float(level)
                if first_failing_index is None:
                    first_failing_index = si

        # Binary search logic
        if batch_pass:
            # All passed, increase perturbation
            lo = level
            logger.debug(f"Level {level:.6f} passed, searching higher")
        else:
            # Failed, decrease perturbation
            hi = level
            if best_fail is None or level < best_fail:
                best_fail = float(level)
                result.update(
                    {
                        "failure_threshold": best_fail,
                        "perturbed_tensor": pert_tensor,
                        "perturbed_output": pert_out_i,
                        "perturbed_features": pert_feat_i,
                    }
                )
            logger.debug(f"Level {level:.6f} failed, searching lower")

    logger.info(
        f"Binary search completed in {len(path)} iterations, failure threshold: {best_fail}, cache_hits: {cache_hits}"
    )

    # Finalize output
    result["failure_threshold"] = best_fail
    result["path"] = path
    result["top_k_path"] = top_k_path
    result["cache_hits"] = cache_hits
    try:
        result["first_failing_index"] = int(
            first_failing_index if first_failing_index is not None else 0
        )
    except Exception:
        result["first_failing_index"] = 0

    # Midpoint estimate per image when bracketed
    per_sample_thresholds: List[Optional[float]] = [
        (0.5 * (float(lo) + float(hi))) if (lo is not None and hi is not None) else None
        for lo, hi in zip(last_pass_levels, first_fail_levels)
    ]
    result["per_sample_thresholds"] = per_sample_thresholds
    return result
