"""
Defines the data structures for VisProbe test reports.
"""

from __future__ import annotations

import base64
import csv
import html as html_module
import io
import json
import os
import re
import sys
import tempfile
import webbrowser
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
import torchvision.transforms as T

__all__ = ["ImageData", "Report", "PerturbationInfo", "Failure"]


def _sanitize_filename(name: str) -> str:
    """
    Sanitize a filename to prevent path traversal attacks.

    Removes directory separators and other potentially dangerous characters,
    keeping only alphanumeric characters, underscores, hyphens, and periods.

    Args:
        name: The filename to sanitize

    Returns:
        A safe filename string
    """
    # Remove any path components (e.g., "../", "/etc/", etc.)
    name = os.path.basename(name)
    # Replace any remaining potentially dangerous characters
    # Keep only alphanumeric, underscore, hyphen, and period
    name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)
    # Prevent empty strings or strings that are only periods
    if not name or name.strip('.') == '':
        name = 'unnamed_test'
    return name


def get_results_dir() -> str:
    """
    Get the platform-appropriate results directory.

    Priority:
    1. VISPROBE_RESULTS_DIR environment variable
    2. System temp directory + 'visprobe_results'

    Returns:
        str: Absolute path to the results directory
    """
    env_dir = os.environ.get("VISPROBE_RESULTS_DIR")
    if env_dir:
        return os.path.abspath(env_dir)
    return os.path.join(tempfile.gettempdir(), "visprobe_results")


def _detect_environment() -> str:
    """
    Detect the current execution environment.

    Returns:
        One of: "jupyter", "interactive", "script"
    """
    # Check for Jupyter/IPython
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            if "IPKernelApp" in ipython.config:
                return "jupyter"
            elif "TerminalInteractiveShell" in str(type(ipython)):
                return "interactive"
    except (ImportError, AttributeError):
        pass

    # Check for interactive Python session
    if hasattr(sys, "ps1"):
        return "interactive"

    # Check if stdin is a tty (interactive terminal)
    try:
        if sys.stdin.isatty() and sys.__stdin__.isatty():
            return "interactive"
    except (AttributeError, ValueError):
        pass

    return "script"


@dataclass
class Failure:
    """
    Represents a single failure case from robustness testing.

    Attributes:
        image_index: Index of the image in the test dataset
        original_prediction: Model's prediction on original image
        perturbed_prediction: Model's prediction on perturbed image
        perturbation_name: Name of the perturbation strategy
        perturbation_strength: Strength/intensity of the perturbation
        perturbation_params: Additional perturbation parameters
        confidence_drop: Drop in prediction confidence
        original_confidence: Confidence on original image
        perturbed_confidence: Confidence on perturbed image
        original_tensor: Original image tensor (optional, for export)
        perturbed_tensor: Perturbed image tensor (optional, for export)
    """
    image_index: int
    original_prediction: Union[str, int]
    perturbed_prediction: Union[str, int]
    perturbation_name: str
    perturbation_strength: float
    perturbation_params: Dict[str, Any] = field(default_factory=dict)
    confidence_drop: float = 0.0
    original_confidence: float = 0.0
    perturbed_confidence: float = 0.0
    original_tensor: Optional[torch.Tensor] = None
    perturbed_tensor: Optional[torch.Tensor] = None

    def to_dict(self, include_tensors: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "image_index": self.image_index,
            "original_prediction": self.original_prediction,
            "perturbed_prediction": self.perturbed_prediction,
            "perturbation_name": self.perturbation_name,
            "perturbation_strength": self.perturbation_strength,
            "perturbation_params": self.perturbation_params,
            "confidence_drop": self.confidence_drop,
            "original_confidence": self.original_confidence,
            "perturbed_confidence": self.perturbed_confidence,
        }
        if include_tensors:
            result["has_original_tensor"] = self.original_tensor is not None
            result["has_perturbed_tensor"] = self.perturbed_tensor is not None
        return result


@dataclass
class PanelImage:
    """Represents a generic image panel for diagnostics (no prediction needed)."""

    image_b64: str
    caption: str

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        caption: str = "",
    ) -> "PanelImage":
        """
        Create a PanelImage from a tensor.

        Args:
            tensor: Image tensor to convert
            mean: Optional channel means for denormalization
            std: Optional channel stds for denormalization
            caption: Caption text for the image

        Returns:
            PanelImage instance with base64-encoded image
        """
        if mean is not None and std is not None:
            mean_t = torch.tensor(mean).view(3, 1, 1)
            std_t = torch.tensor(std).view(3, 1, 1)
            tensor = tensor.detach().cpu() * std_t + mean_t
        else:
            tensor = tensor.detach().cpu()
        img_chw = tensor.squeeze(0).clamp(0, 1)
        pil_image = T.ToPILImage()(img_chw)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return cls(image_b64=image_b64, caption=caption)


@dataclass
class PerturbationInfo:
    """Contains metadata about the perturbation applied."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


from .utils import NumpyEncoder  # noqa: E402


@dataclass
class ImageData:
    """Represents an image and its model prediction for visualization."""

    image_b64: str
    prediction: str
    confidence: float

    @classmethod
    def from_tensors(
        cls,
        tensor: torch.Tensor,
        output: torch.Tensor,
        class_names: Optional[List[str]],
        mean: Optional[List[float]],
        std: Optional[List[float]],
    ) -> "ImageData":
        """Creates an ImageData object from tensors."""
        if mean and std:
            mean_t = torch.tensor(mean).view(3, 1, 1)
            std_t = torch.tensor(std).view(3, 1, 1)
            tensor = tensor.detach().cpu() * std_t + mean_t

        # Ensure shape [C,H,W] and avoid unnecessary resampling
        img_chw = tensor.squeeze(0).clamp(0, 1)
        pil_image = T.ToPILImage()(img_chw)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        probabilities = torch.nn.functional.softmax(output, dim=0)
        confidence, pred_idx = torch.max(probabilities, 0)
        prediction_label = (
            class_names[pred_idx.item()] if class_names else f"Class {pred_idx.item()}"
        )

        return cls(image_b64=image_b64, prediction=prediction_label, confidence=confidence.item())


@dataclass
class Report:
    """A unified report for all VisProbe test types."""

    test_name: str
    test_type: str
    runtime: float
    model_queries: int
    # Paper-aligned summary fields (optional and additive for compatibility)
    model_name: Optional[str] = None
    preset: Optional[str] = None
    dataset: Optional[str] = None
    property_name: Optional[str] = None
    strategy: Optional[str] = None
    search: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    runtime_sec: Optional[float] = None
    num_queries: Optional[int] = None
    seed: Optional[int] = None
    original_image: Optional[ImageData] = None
    perturbed_image: Optional[ImageData] = None
    ensemble_analysis: Optional[Dict[str, Any]] = None
    resolution_impact: Optional[Dict[str, float]] = None
    noise_sweep_results: Optional[List[Dict[str, Any]]] = None
    corruption_sweep_results: Optional[Dict[str, List[Dict[str, Any]]]] = None
    total_samples: Optional[int] = None
    passed_samples: Optional[int] = None
    failure_threshold: Optional[float] = None
    search_path: Optional[List[Dict[str, Any]]] = None
    perturbation_info: Optional[PerturbationInfo] = None
    top_k_analysis: Any = None  # Can be an int for 'given' or list for 'search'
    residual_image: Optional[PanelImage] = None
    residual_metrics: Optional[Dict[str, float]] = None
    # Extended, paper-aligned structures
    run_meta: Optional[Dict[str, Any]] = None
    per_sample: Optional[List[Dict[str, Any]]] = None
    aggregates: Optional[Dict[str, Any]] = None
    # Internal storage for Failure objects
    _failure_objects: List[Failure] = field(default_factory=list)
    # Normalization params for image export
    _mean: Optional[List[float]] = None
    _std: Optional[List[float]] = None

    # ===== Threat Model Aware Fields =====

    @property
    def threat_model(self) -> Optional[str]:
        """
        The threat model used for this test.

        Returns one of:
        - "passive": Natural perturbations only (no adversary)
        - "active": Adversarial attacks (white-box adversary)
        - "active_environmental": Combined natural + adversarial
        - "all": Comprehensive testing across all threat models
        """
        if self.metrics:
            return self.metrics.get("threat_model")
        if self.search and isinstance(self.search, dict):
            return self.search.get("threat_model")
        return None

    @property
    def threat_model_scores(self) -> Optional[Dict[str, float]]:
        """
        Per-threat-model robustness scores (for comprehensive tests).

        Returns dict like:
        {
            "natural": 0.75,
            "adversarial": 0.60,
            "realistic_attack": 0.45
        }

        Only available when using "comprehensive" preset or when
        outputs_threat_breakdown is True.
        """
        if self.metrics:
            return self.metrics.get("threat_model_scores")
        return None

    @property
    def threat_model_summary(self) -> Optional[Dict[str, Any]]:
        """
        Comprehensive threat model analysis including vulnerability detection.

        Returns dict like:
        {
            "threat_model": "all",
            "scores_by_threat": {"natural": 0.75, "adversarial": 0.60, "realistic_attack": 0.45},
            "overall_score": 0.60,
            "vulnerability_warning": "Model vulnerable to opportunistic attacks!..."
        }
        """
        if self.metrics:
            return self.metrics.get("threat_model_summary")
        return None

    @property
    def vulnerability_warning(self) -> Optional[str]:
        """
        Warning message if model is vulnerable to opportunistic attacks.

        This is the KEY INSIGHT: if realistic_attack score is significantly lower
        than both natural and adversarial scores, the model has a blind spot.

        Attackers can exploit environmental conditions (low-light, blur, compression)
        to succeed with smaller adversarial perturbations.
        """
        if self.metrics:
            return self.metrics.get("vulnerability_warning")
        return None

    def has_threat_breakdown(self) -> bool:
        """Check if this report includes threat model breakdown."""
        return self.threat_model_scores is not None

    @property
    def robust_accuracy(self) -> Optional[float]:
        if (
            self.test_type == "given"
            and self.total_samples is not None
            and self.passed_samples is not None
        ):
            if self.total_samples == 0:
                return 1.0
            return self.passed_samples / self.total_samples
        return None

    def to_json(self) -> str:
        """Serializes the report to a JSON string with images simplified (no base64)."""
        data = self._build_serializable_dict(results_dir=None)
        return json.dumps(data, cls=NumpyEncoder, indent=2)

    def save(self):
        """Saves the report to a JSON file and writes images to disk (no base64 in JSON)."""
        try:
            results_dir = get_results_dir()
            os.makedirs(results_dir, exist_ok=True)
            # Sanitize test name to prevent path traversal
            safe_name = _sanitize_filename(self.test_name)
            data = self._build_serializable_dict(results_dir=results_dir)
            file_path = os.path.join(results_dir, f"{safe_name}.json")
            with open(file_path, "w") as f:
                json.dump(data, f, cls=NumpyEncoder, indent=2)
            print(f"✅ Report saved to {file_path}")
            # Also save a CSV for quick plotting if per-sample exists
            try:
                if self.per_sample:
                    csv_path = os.path.join(results_dir, f"{safe_name}.csv")
                    fieldnames = sorted({k for row in self.per_sample for k in row.keys()})
                    with open(csv_path, "w", newline="") as cf:
                        writer = csv.DictWriter(cf, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in self.per_sample:
                            writer.writerow(row)
            except Exception as ce:
                print(f"⚠️  Could not save CSV for '{self.test_name}': {ce}")
        except Exception as e:
            print(f"⚠️  Could not save test report for '{self.test_name}': {e}")

    def _build_serializable_dict(self, results_dir: Optional[str]) -> Dict[str, Any]:  # noqa: C901
        """
        Build a JSON-serializable dictionary for this report, removing base64 image blobs.
        If results_dir is provided, write image files to disk and include file paths instead.
        """
        data = {k: v for k, v in asdict(self).items() if v is not None}
        if self.robust_accuracy is not None:
            data["robust_accuracy"] = self.robust_accuracy
        if "runtime_sec" not in data:
            data["runtime_sec"] = self.runtime
        if "num_queries" not in data:
            data["num_queries"] = self.model_queries
        if "property_name" in data:
            data["property"] = data.pop("property_name")

        def write_image(b64_str: str, suffix: str) -> Optional[str]:
            if not results_dir:
                return None
            try:
                img_bytes = base64.b64decode(b64_str)
                # Sanitize test name to prevent path traversal
                safe_name = _sanitize_filename(self.test_name)
                img_path = os.path.join(results_dir, f"{safe_name}.{suffix}.png")
                with open(img_path, "wb") as imf:
                    imf.write(img_bytes)
                return img_path
            except Exception:
                return None

        # Simplify image fields
        if "original_image" in data and data["original_image"]:
            oi = data["original_image"]
            img_path = write_image(oi.get("image_b64", ""), "original")
            data["original_image"] = {
                "image_path": img_path,
                "prediction": oi.get("prediction"),
                "confidence": oi.get("confidence"),
            }
        if "perturbed_image" in data and data["perturbed_image"]:
            pi = data["perturbed_image"]
            img_path = write_image(pi.get("image_b64", ""), "perturbed")
            data["perturbed_image"] = {
                "image_path": img_path,
                "prediction": pi.get("prediction"),
                "confidence": pi.get("confidence"),
            }
        if "residual_image" in data and data["residual_image"]:
            ri = data["residual_image"]
            img_path = write_image(ri.get("image_b64", ""), "residual")
            data["residual_image"] = {
                "image_path": img_path,
                "caption": ri.get("caption", "Residual"),
            }

        return data

    # ===== User-friendly API for quick_check() =====

    @property
    def score(self) -> Optional[float]:
        """
        Overall robustness score (0-1, higher is better).

        For quick_check tests, returns the average robustness score across all strategies.
        For given/search tests, returns robust_accuracy if available.

        Returns:
            Float between 0 and 1, or None if not applicable
        """
        # For quick_check tests
        if self.test_type == "quick_check" and self.metrics:
            return self.metrics.get("overall_robustness_score")

        # For traditional tests
        return self.robust_accuracy

    @property
    def failures(self) -> List[Dict[str, Any]]:
        """
        List of failure cases found during testing.

        Returns:
            List of dictionaries containing failure information
        """
        all_failures: List[Dict[str, Any]] = []

        # Include failures from search results (dict format)
        if self.search and isinstance(self.search, dict):
            results = self.search.get("results", [])
            for result in results:
                if "failures" in result:
                    all_failures.extend(result["failures"])

        # Include failures from _failure_objects (Failure format)
        for f in self._failure_objects:
            all_failures.append(f.to_dict())

        return all_failures

    @property
    def summary(self) -> Dict[str, Any]:
        """
        Key metrics summary dictionary.

        Returns:
            Dictionary with test_name, score, failures, runtime, queries,
            and threat model information if available
        """
        result = {
            "test_name": self.test_name,
            "test_type": self.test_type,
            "score": self.score,
            "total_failures": len(self.failures),
            "runtime_sec": self.runtime,
            "model_queries": self.model_queries,
            "total_samples": self.total_samples,
            "passed_samples": self.passed_samples,
        }

        # Add threat model information if available
        if self.threat_model:
            result["threat_model"] = self.threat_model

        if self.threat_model_scores:
            result["threat_model_scores"] = self.threat_model_scores

        if self.vulnerability_warning:
            result["has_vulnerability_warning"] = True

        return result

    def show(self, mode: Optional[str] = None) -> None:
        """
        Display the report in a context-appropriate way.

        Automatically detects the environment:
        - Jupyter: Inline HTML display
        - Interactive Python: Opens browser with HTML report
        - Script: Print concise summary

        Args:
            mode: Force a specific mode ("jupyter", "interactive", "script", or None for auto)
        """
        # Auto-detect mode if not specified
        if mode is None:
            mode = _detect_environment()

        if mode == "jupyter":
            self._show_jupyter()
        elif mode == "interactive":
            self._show_browser()
        else:
            self._show_text()

    def _show_text(self) -> None:
        """Print concise text summary."""
        print("\n" + "=" * 60)
        print(f"VisProbe Report: {self.test_name}")
        print("=" * 60)
        print(f"Test type: {self.test_type}")

        # Show threat model if available
        if self.threat_model:
            threat_labels = {
                "passive": "Passive (Natural Perturbations)",
                "active": "Active (Adversarial Attacks)",
                "active_environmental": "Active + Environmental (Realistic Attacks)",
                "all": "Comprehensive (All Threat Models)",
            }
            threat_label = threat_labels.get(self.threat_model, self.threat_model)
            print(f"Threat model: {threat_label}")

        if self.score is not None:
            print(f"Robustness score: {self.score:.2%}")

        # Show threat model breakdown if available
        if self.threat_model_scores:
            print("\n" + "-" * 60)
            print("Threat Model Breakdown:")
            print("-" * 60)
            for tm, tm_score in self.threat_model_scores.items():
                # Format threat model name nicely
                tm_display = tm.replace("_", " ").title()
                bar_len = int(tm_score * 20)
                bar = "#" * bar_len + "-" * (20 - bar_len)
                print(f"  {tm_display:20s} {tm_score:6.1%} [{bar}]")

        # Show vulnerability warning if present
        if self.vulnerability_warning:
            print("\n" + "!" * 60)
            print("CRITICAL SECURITY WARNING")
            print("!" * 60)
            print(self.vulnerability_warning)
            print("!" * 60)

        print(f"\nFailures found: {len(self.failures)}")
        print(f"Total samples: {self.total_samples}")
        print(f"Passed samples: {self.passed_samples}")
        print(f"Runtime: {self.runtime:.1f}s")
        print(f"Model queries: {self.model_queries}")

        # Show per-strategy results if available
        if self.search and isinstance(self.search, dict):
            results = self.search.get("results", [])
            if results:
                print("\n" + "-" * 60)
                print("Per-Strategy Results:")
                print("-" * 60)
                for result in results:
                    strategy_name = result.get("strategy", "Unknown")
                    score = result.get("robustness_score", 0)
                    threshold = result.get("failure_threshold", 0)
                    category = result.get("category", "")
                    cat_suffix = f" [{category}]" if category and category != "uncategorized" else ""
                    print(f"  {strategy_name:25s} Score: {score:.2%}  Threshold: {threshold:.3f}{cat_suffix}")

        print("=" * 60 + "\n")

    def _show_browser(self) -> None:
        """Open HTML report in browser."""
        html_content = self._generate_html_full()

        results_dir = get_results_dir()
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = _sanitize_filename(self.test_name)
        report_path = os.path.join(results_dir, f"{safe_name}_{timestamp}.html")

        with open(report_path, "w") as f:
            f.write(html_content)

        webbrowser.open(f"file://{report_path}")
        print(f"Report opened in browser: {report_path}")

    def _show_jupyter(self) -> None:
        """Display rich HTML in Jupyter notebook."""
        try:
            from IPython.display import HTML, display

            html = self._generate_html_summary()
            display(HTML(html))
        except ImportError:
            # Fallback to text if IPython not available
            self._show_text()

    def _generate_html_summary(self) -> str:
        """Generate HTML summary for Jupyter display."""
        score_pct = self.score * 100 if self.score else 0
        score_color = "#4CAF50" if score_pct > 70 else "#FF9800" if score_pct > 40 else "#F44336"

        # Threat model display
        threat_model_display = ""
        if self.threat_model:
            threat_labels = {
                "passive": "Passive (Natural)",
                "active": "Active (Adversarial)",
                "active_environmental": "Realistic Attack",
                "all": "Comprehensive",
            }
            threat_model_display = f"<p><strong>Threat Model:</strong> {threat_labels.get(self.threat_model, self.threat_model)}</p>"

        html = f"""
        <div style="border: 2px solid #ddd; padding: 20px; border-radius: 5px; font-family: Arial, sans-serif;">
            <h2 style="margin-top: 0;">VisProbe Report: {self.test_name}</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div>
                    <h3 style="color: {score_color};">Robustness Score: {score_pct:.1f}%</h3>
                    <p><strong>Test Type:</strong> {self.test_type}</p>
                    {threat_model_display}
                    <p><strong>Total Samples:</strong> {self.total_samples}</p>
                    <p><strong>Passed:</strong> {self.passed_samples}</p>
                </div>
                <div>
                    <p><strong>Failures Found:</strong> {len(self.failures)}</p>
                    <p><strong>Runtime:</strong> {self.runtime:.1f}s</p>
                    <p><strong>Model Queries:</strong> {self.model_queries}</p>
                </div>
            </div>
        """

        # Add threat model breakdown if available
        if self.threat_model_scores:
            html += "<h3>Threat Model Breakdown</h3>"
            html += "<div style='display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 15px;'>"
            for tm, tm_score in self.threat_model_scores.items():
                tm_pct = tm_score * 100
                tm_color = "#4CAF50" if tm_pct > 70 else "#FF9800" if tm_pct > 40 else "#F44336"
                tm_display = tm.replace("_", " ").title()
                html += f"""
                <div style="text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px; min-width: 120px;">
                    <div style="font-size: 24px; font-weight: bold; color: {tm_color};">{tm_pct:.1f}%</div>
                    <div style="font-size: 12px; color: #666;">{tm_display}</div>
                </div>
                """
            html += "</div>"

        # Add vulnerability warning if present
        if self.vulnerability_warning:
            html += f"""
            <div style="background-color: #ffebee; border: 2px solid #f44336; border-radius: 5px; padding: 15px; margin: 15px 0;">
                <h4 style="color: #c62828; margin-top: 0;">Critical Security Warning</h4>
                <pre style="white-space: pre-wrap; font-size: 12px; color: #333;">{html_module.escape(self.vulnerability_warning)}</pre>
            </div>
            """

        # Add per-strategy results if available
        if self.search and isinstance(self.search, dict):
            results = self.search.get("results", [])
            if results:
                html += "<h3>Per-Strategy Results</h3><table style='width:100%; border-collapse: collapse;'>"
                html += "<tr style='background-color: #f0f0f0;'><th style='padding: 8px; text-align: left;'>Strategy</th><th style='padding: 8px;'>Category</th><th style='padding: 8px;'>Score</th><th style='padding: 8px;'>Threshold</th></tr>"
                for result in results:
                    strategy = result.get("strategy", "Unknown")
                    category = result.get("category", "")
                    score = result.get("robustness_score", 0) * 100
                    threshold = result.get("failure_threshold", 0)
                    cat_display = category.replace("_", " ").title() if category and category != "uncategorized" else "-"
                    html += f"<tr><td style='padding: 8px;'>{strategy}</td><td style='padding: 8px; text-align: center;'>{cat_display}</td><td style='padding: 8px; text-align: center;'>{score:.1f}%</td><td style='padding: 8px; text-align: center;'>{threshold:.3f}</td></tr>"
                html += "</table>"

        html += "</div>"
        return html

    def _generate_html_full(self) -> str:
        """Generate full HTML page for browser display with modern styling."""
        score_pct = self.score * 100 if self.score else 0

        # Score color based on robustness
        if score_pct >= 70:
            score_color = "#4CAF50"  # Green
        elif score_pct >= 40:
            score_color = "#FF9800"  # Orange
        else:
            score_color = "#F44336"  # Red

        # Build failures table rows from both dict failures and Failure objects
        failures_rows = ""
        all_failures = self.failures + [f.to_dict() for f in self._failure_objects]
        for f in all_failures[:20]:
            if isinstance(f, dict):
                img_idx = f.get("image_index", f.get("index", "?"))
                orig = f.get("original_prediction", f.get("clean_label", "?"))
                pert = f.get("perturbed_prediction", f.get("pert_label", "?"))
                strat = f.get("perturbation_name", f.get("strategy", "?"))
                strength = f.get("perturbation_strength", f.get("threshold", 0))
                conf_drop = f.get("confidence_drop", 0)
            else:
                img_idx = f.image_index
                orig = f.original_prediction
                pert = f.perturbed_prediction
                strat = f.perturbation_name
                strength = f.perturbation_strength
                conf_drop = f.confidence_drop

            failures_rows += f"""
            <tr>
                <td>{img_idx}</td>
                <td>{html_module.escape(str(orig))}</td>
                <td>{html_module.escape(str(pert))}</td>
                <td>{html_module.escape(str(strat))}</td>
                <td>{strength:.4f}</td>
                <td>{conf_drop:.4f}</td>
            </tr>
            """

        preset_display = f"Preset: {html_module.escape(self.preset)} |" if self.preset else ""
        model_display = self.model_name or self.test_name

        content = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 900px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px;">
                <h1 style="margin: 0 0 10px 0;">VisProbe Report</h1>
                <h2 style="margin: 0; opacity: 0.9;">{html_module.escape(model_display)}</h2>
                <p style="margin: 10px 0 0 0; opacity: 0.8;">
                    {preset_display}
                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>

            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;">
                <div style="background: white; padding: 20px; border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;">
                    <div style="font-size: 36px; font-weight: bold; color: {score_color};">
                        {score_pct:.1f}%
                    </div>
                    <div style="color: #666; font-size: 14px;">Robustness Score</div>
                </div>
                <div style="background: white; padding: 20px; border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;">
                    <div style="font-size: 36px; font-weight: bold; color: #333;">
                        {self.total_samples or 0}
                    </div>
                    <div style="color: #666; font-size: 14px;">Total Samples</div>
                </div>
                <div style="background: white; padding: 20px; border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;">
                    <div style="font-size: 36px; font-weight: bold; color: #4CAF50;">
                        {self.passed_samples or 0}
                    </div>
                    <div style="color: #666; font-size: 14px;">Passed</div>
                </div>
                <div style="background: white; padding: 20px; border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;">
                    <div style="font-size: 36px; font-weight: bold; color: #F44336;">
                        {len(all_failures)}
                    </div>
                    <div style="color: #666; font-size: 14px;">Failures</div>
                </div>
            </div>

            <div style="background: white; padding: 20px; border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #333;">Performance</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <span style="color: #666;">Runtime:</span>
                        <strong>{self.runtime:.2f}s</strong>
                    </div>
                    <div>
                        <span style="color: #666;">Model Queries:</span>
                        <strong>{self.model_queries:,}</strong>
                    </div>
                </div>
            </div>
        """

        # Add threat model breakdown if available
        if self.threat_model_scores:
            content += """
            <div style="background: white; padding: 20px; border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #333;">Threat Model Breakdown</h3>
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            """
            for tm, tm_score in self.threat_model_scores.items():
                tm_pct = tm_score * 100
                if tm_pct >= 70:
                    tm_color = "#4CAF50"
                elif tm_pct >= 40:
                    tm_color = "#FF9800"
                else:
                    tm_color = "#F44336"
                tm_display = tm.replace("_", " ").title()
                content += f"""
                    <div style="flex: 1; min-width: 150px; text-align: center; padding: 15px;
                                border: 2px solid {tm_color}; border-radius: 8px;">
                        <div style="font-size: 32px; font-weight: bold; color: {tm_color};">
                            {tm_pct:.1f}%
                        </div>
                        <div style="color: #666; font-size: 14px; margin-top: 5px;">
                            {tm_display}
                        </div>
                    </div>
                """
            content += """
                </div>
            </div>
            """

        # Add vulnerability warning if present
        if self.vulnerability_warning:
            content += f"""
            <div style="background: #ffebee; padding: 20px; border-radius: 8px;
                        border: 2px solid #f44336; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #c62828;">
                    Critical Security Warning
                </h3>
                <pre style="white-space: pre-wrap; font-size: 13px; color: #333;
                            font-family: inherit; margin: 0;">{html_module.escape(self.vulnerability_warning)}</pre>
            </div>
            """

        # Add failures table if there are failures
        if all_failures:
            content += f"""
            <div style="background: white; padding: 20px; border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="margin-top: 0; color: #333;">
                    Failure Cases ({len(all_failures)} total, showing first 20)
                </h3>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        <thead>
                            <tr style="background: #f5f5f5;">
                                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Index</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Original</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Perturbed</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Strategy</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Strength</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Conf. Drop</th>
                            </tr>
                        </thead>
                        <tbody>
                            {failures_rows}
                        </tbody>
                    </table>
                </div>
            </div>
            """

        content += "</div>"

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>VisProbe Report - {html_module.escape(model_display)}</title>
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                    background: #f5f5f5;
                }}
                table tr:hover {{
                    background: #f9f9f9;
                }}
                td, th {{
                    border-bottom: 1px solid #eee;
                }}
            </style>
        </head>
        <body>
            {content}
        </body>
        </html>
        """

    def export_failures(self, n: int = 10, output_dir: Optional[str] = None) -> str:
        """
        Export the top N failure cases as a dataset.

        Args:
            n: Number of failures to export (default: 10)
            output_dir: Directory to save failures (default: visprobe_results/failures)

        Returns:
            Path to the exported directory
        """
        if output_dir is None:
            output_dir = os.path.join(get_results_dir(), "failures", self.test_name)

        os.makedirs(output_dir, exist_ok=True)

        failures_to_export = self.failures[:n]

        # Save failures metadata
        metadata_path = os.path.join(output_dir, "failures.json")
        with open(metadata_path, "w") as f:
            json.dump(failures_to_export, f, indent=2, cls=NumpyEncoder)

        print(f"✅ Exported {len(failures_to_export)} failures to {output_dir}")
        print(f"   Metadata: {metadata_path}")

        return output_dir

    def add_failure(self, failure: Failure) -> None:
        """
        Add a Failure object to the report.

        Args:
            failure: Failure object to add
        """
        self._failure_objects.append(failure)

    def add_metric(self, key: str, value: Any) -> None:
        """
        Add a custom metric to the report.

        Args:
            key: Metric name
            value: Metric value
        """
        if self.metrics is None:
            self.metrics = {}
        self.metrics[key] = value

    @classmethod
    def from_json(cls, path: str) -> "Report":
        """
        Load a Report from a JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            Report instance
        """
        with open(path) as f:
            data = json.load(f)

        report = cls(
            test_name=data.get("test_name", "unknown"),
            test_type=data.get("test_type", "unknown"),
            runtime=data.get("runtime", data.get("runtime_sec", 0.0)),
            model_queries=data.get("model_queries", data.get("num_queries", 0)),
            model_name=data.get("model_name"),
            preset=data.get("preset"),
            dataset=data.get("dataset"),
            total_samples=data.get("total_samples"),
            passed_samples=data.get("passed_samples"),
            metrics=data.get("metrics"),
            search=data.get("search"),
        )

        # Load failures as Failure objects if present
        for f_data in data.get("failures", []):
            if isinstance(f_data, dict) and "image_index" in f_data:
                failure = Failure(
                    image_index=f_data["image_index"],
                    original_prediction=f_data.get("original_prediction", "?"),
                    perturbed_prediction=f_data.get("perturbed_prediction", "?"),
                    perturbation_name=f_data.get("perturbation_name", "unknown"),
                    perturbation_strength=f_data.get("perturbation_strength", 0.0),
                    perturbation_params=f_data.get("perturbation_params", {}),
                    confidence_drop=f_data.get("confidence_drop", 0.0),
                    original_confidence=f_data.get("original_confidence", 0.0),
                    perturbed_confidence=f_data.get("perturbed_confidence", 0.0),
                )
                report.add_failure(failure)

        return report

    def __repr__(self) -> str:
        score_str = f"{self.score:.2%}" if self.score is not None else "N/A"
        return (
            f"Report(test_name={self.test_name!r}, test_type={self.test_type!r}, "
            f"score={score_str}, failures={len(self.failures) + len(self._failure_objects)})"
        )
