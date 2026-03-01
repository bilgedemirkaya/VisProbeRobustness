"""
Core test orchestration for VisProbe.

This module contains the TestRunner class that coordinates test execution,
delegating specific concerns to specialized modules for clarity.
"""

import logging
import os
import time
from typing import Any, Callable, Dict

import torch

from ..strategies.base import Strategy
from .analysis_utils import (
    run_corruption_sweep,
    run_ensemble_analysis,
    run_noise_sweep,
    run_resolution_impact,
    run_top_k_analysis,
)
from .context import TestContext
from .model_wrap import _ModelWithIntermediateOutput
from .property_evaluator import PropertyEvaluator
from .query_counter import QueryCounter
from .report import Report
from .report_builder import ReportBuilder
from .search_modes import (
    perform_adaptive_search,
    perform_binary_search,
    perform_grid_search,
    perform_random_search,
)
from .strategy_utils import StrategyConfig
from .utils import build_final_strategy, build_search_blocks, build_visuals, to_image_space, to_model_space

logger = logging.getLogger(__name__)


class TestRunner:
    """
    Orchestrates VisProbe test execution.

    This class coordinates the test workflow, delegating specific tasks to specialized
    modules for context initialization, property evaluation, and report building.
    """

    def __init__(self, user_func: Callable, test_type: str, params: Dict[str, Any]):
        """
        Initialize test runner.

        Args:
            user_func: User's decorated test function
            test_type: Type of test ("given" or "search")
            params: Test parameters from decorators
        """
        self.user_func = user_func
        self.test_type = test_type
        self.params = params
        self.query_count = 0
        self.start_time = time.perf_counter()
        self.context = TestContext.build(user_func)

    def run(self) -> Report:
        """
        Execute the test and return a report.

        Returns:
            Report object containing test results

        Raises:
            ValueError: If test_type is not recognized
        """
        if self.test_type == "given":
            report = self._run_given()
        elif self.test_type == "search":
            report = self._run_search()
        else:
            raise ValueError(f"Unknown test type: {self.test_type}")

        report.save()
        return report

    def _run_given(self) -> Report:
        """Execute a fixed perturbation test."""
        ctx, p = self.context, self.params
        m, batch_tensor, batch_size = ctx["model"], ctx["batch_tensor"], ctx["batch_size"]

        # Get clean model outputs
        m.eval()
        with QueryCounter(m) as qc0:
            model_output = m(batch_tensor)
        clean_out, clean_feat = self._unpack_model_output(model_output)
        self.query_count += 1 + qc0.extra

        # Configure and apply perturbation strategy
        perturb_obj = Strategy.resolve(p["strategy"])
        self._configure_strategy_safe(perturb_obj)

        with QueryCounter(m) as qc1:
            pert_tensor = perturb_obj.generate(batch_tensor, model=m)
            pert_model_output = m(pert_tensor)
        pert_out, pert_feat = self._unpack_model_output(pert_model_output)
        self.query_count += 1 + qc1.extra

        # Extract logits and evaluate property
        clean_logits = PropertyEvaluator.extract_logits(clean_out)
        pert_logits = PropertyEvaluator.extract_logits(pert_out)

        passed_samples = PropertyEvaluator.evaluate_count(
            self.user_func, clean_logits, pert_logits, batch_size, p.get("vectorized", False)
        )

        # Build report components
        original_img, perturbed_img = ReportBuilder.create_image_data_pair(
            batch_tensor, clean_logits, pert_tensor, pert_logits, ctx
        )

        ensemble = run_ensemble_analysis(self, clean_feat, pert_feat)
        run_meta = ReportBuilder.build_run_metadata(self.user_func, ctx, perturb_obj, None)
        per_sample = ReportBuilder.compute_per_sample_metrics(
            clean_logits, pert_logits, ctx["class_names"], p.get("top_k")
        )

        module_name = os.environ.get("VISPROBE_MODULE_NAME", self.user_func.__module__)

        return Report(
            test_name=f"{module_name}.{self.user_func.__name__}",
            test_type="given",
            runtime=time.perf_counter() - self.start_time,
            model_queries=self.query_count,
            total_samples=batch_size,
            passed_samples=passed_samples,
            original_image=original_img,
            perturbed_image=perturbed_img,
            ensemble_analysis=ensemble,
            resolution_impact=run_resolution_impact(
                self, p.get("resolutions"), perturb_obj, (clean_logits, clean_feat)
            ),
            noise_sweep_results=run_noise_sweep(self, p.get("noise_sweep"), (clean_logits, clean_feat)),
            corruption_sweep_results=run_corruption_sweep(
                self,
                p.get("corruptions", ["gaussian_noise", "brightness", "contrast"]),
                (clean_logits, clean_feat),
            ),
            top_k_analysis=run_top_k_analysis(self, p.get("top_k"), clean_logits, pert_logits),
            perturbation_info=ReportBuilder.create_perturbation_info(perturb_obj),
            model_name=self._get_model_name(ctx["model"]),
            property_name=getattr(self.user_func, "_visprobe_property_name", None),
            strategy=f"{perturb_obj.__class__.__name__}",
            metrics={
                "layer_cosine": ensemble,
                "topk_overlap": run_top_k_analysis(self, p.get("top_k"), clean_logits, pert_logits),
            },
            runtime_sec=time.perf_counter() - self.start_time,
            num_queries=self.query_count,
            seed=ctx.get("seed"),
            run_meta=run_meta,
            per_sample=per_sample,
        )

    def _run_search(self) -> Report:
        """Execute a search for minimal perturbation threshold."""
        ctx, p = self.context, self.params
        m, batch_tensor = ctx["model"], ctx["batch_tensor"]

        # Get clean model outputs
        m.eval()
        with QueryCounter(m) as qc0:
            model_output = m(batch_tensor)
        clean_out, clean_feat = self._unpack_model_output(model_output)
        self.query_count += 1 + qc0.extra
        clean_logits = PropertyEvaluator.extract_logits(clean_out)

        # Perform search
        mode = self.params.get("mode", "adaptive")
        if mode == "grid":
            search_results = perform_grid_search(self, p, (clean_logits, clean_feat))
        elif mode == "random":
            search_results = perform_random_search(self, p, (clean_logits, clean_feat))
        elif mode == "binary":
            search_results = perform_binary_search(self, p, (clean_logits, clean_feat))
        else:
            search_results = perform_adaptive_search(self, p, (clean_logits, clean_feat))

        fail_thresh = search_results.get("failure_threshold")

        # Build final strategy at failure threshold
        final_perturb_obj = (
            build_final_strategy(self, p["strategy"], fail_thresh) if fail_thresh is not None else None
        )
        if final_perturb_obj is not None:
            self._configure_strategy_safe(final_perturb_obj)

        # Build visualizations
        original_img, perturbed_img, residual_panel, residual_metrics, _ = build_visuals(
            self, batch_tensor, clean_logits, search_results
        )

        module_name = os.environ.get("VISPROBE_MODULE_NAME", self.user_func.__module__)

        ensemble = run_ensemble_analysis(self, clean_feat, search_results.get("perturbed_features"))

        # Build search blocks for report
        search_block, metrics_block, aggregates_block = build_search_blocks(
            self, mode, p, search_results, ensemble
        )

        run_meta = ReportBuilder.build_run_metadata(self.user_func, ctx, final_perturb_obj, fail_thresh)

        # Per-sample metrics for search
        per_sample = []
        if search_results.get("perturbed_output") is not None:
            per_sample = ReportBuilder.compute_per_sample_metrics(
                clean_logits, search_results.get("perturbed_output"), ctx["class_names"], p.get("top_k")
            )
            # Add search-specific fields
            for sample in per_sample:
                sample["threshold_estimate"] = fail_thresh
                sample["trace"] = search_results.get("path")

        return Report(
            test_name=f"{module_name}.{self.user_func.__name__}",
            test_type="search",
            runtime=time.perf_counter() - self.start_time,
            model_queries=self.query_count,
            failure_threshold=fail_thresh,
            search_path=search_results.get("path"),
            original_image=original_img,
            perturbed_image=perturbed_img,
            residual_image=residual_panel,
            residual_metrics=residual_metrics,
            ensemble_analysis=ensemble,
            resolution_impact=run_resolution_impact(
                self, p.get("resolutions"), final_perturb_obj, (clean_out, clean_feat)
            ),
            noise_sweep_results=run_noise_sweep(self, p.get("noise_sweep"), (clean_out, clean_feat)),
            corruption_sweep_results=run_corruption_sweep(
                self,
                p.get("corruptions", ["gaussian_noise", "brightness", "contrast"]),
                (clean_out, clean_feat),
            ),
            top_k_analysis=search_results.get("top_k_path"),
            perturbation_info=(
                ReportBuilder.create_perturbation_info(final_perturb_obj) if final_perturb_obj else None
            ),
            model_name=self._get_model_name(ctx["model"]),
            property_name=getattr(self.user_func, "_visprobe_property_name", None),
            strategy=(
                f"{final_perturb_obj.__class__.__name__}(eps)"
                if hasattr(final_perturb_obj, "eps")
                else f"{final_perturb_obj.__class__.__name__}"
            ),
            search=search_block,
            metrics=metrics_block,
            aggregates=aggregates_block,
            runtime_sec=time.perf_counter() - self.start_time,
            num_queries=self.query_count,
            seed=None,
            run_meta=run_meta,
            per_sample=per_sample,
        )

    # --- Helper methods ---

    def _unpack_model_output(self, model_output):
        """Unpack model output into (output, features) tuple."""
        if isinstance(model_output, tuple):
            return model_output
        return model_output, None

    def _configure_strategy_safe(self, strategy):
        """Configure strategy with error handling."""
        try:
            StrategyConfig.configure(
                strategy, self.context["mean"], self.context["std"], self.context["rng"]
            )
        except AttributeError:
            logger.debug(
                f"Strategy {strategy.__class__.__name__} doesn't have configure method, using fallback"
            )
            StrategyConfig.configure(strategy, self.context["mean"], self.context["std"], None)
        except TypeError as e:
            logger.warning(f"Strategy configure() has incompatible signature: {e}")
            StrategyConfig.configure(strategy, self.context["mean"], self.context["std"], None)
        except Exception as e:
            logger.error(f"Unexpected error configuring strategy: {e}")

    def _get_model_name(self, model) -> str:
        """Extract model name from model object."""
        if isinstance(model, _ModelWithIntermediateOutput):
            return type(model.__dict__.get("model", model)).__name__
        return type(model).__name__

    # --- Backward compatibility methods ---
    # These methods are kept for compatibility with existing code that calls them

    def _evaluate_property_mask(self, clean_logits: torch.Tensor, pert_logits: torch.Tensor):
        """Legacy method - delegates to PropertyEvaluator."""
        return PropertyEvaluator.evaluate_mask(self.user_func, clean_logits, pert_logits)

    def _evaluate_property_mask_from_results(self, clean_results, pert_results):
        """Legacy method - delegates to PropertyEvaluator."""
        return PropertyEvaluator.evaluate_mask_from_results(
            self.user_func, clean_results, pert_results, self.context["batch_size"]
        )

    @staticmethod
    def _to_logits(out):
        """Legacy method - delegates to PropertyEvaluator."""
        return PropertyEvaluator.extract_logits(out)

    @staticmethod
    def _configure_strategy(strategy, mean, std, rng):
        """Legacy method - delegates to StrategyConfig."""
        StrategyConfig.configure(strategy, mean, std, rng)

    def _denorm(self, tensor: torch.Tensor) -> torch.Tensor:
        """Converts a normalized tensor back to pixel space (0‒1)."""
        return to_image_space(tensor, self.context["mean"], self.context["std"])

    def _renorm(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalizes a pixel-space tensor using the context mean/std."""
        return to_model_space(tensor, self.context["mean"], self.context["std"])
