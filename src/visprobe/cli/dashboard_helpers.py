"""
Dashboard helper functions for VisProbe Streamlit UI.

Provides chart-based visualization with actionable insights.
"""

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image

from .utils import get_results_dir


# =============================================================================
# Core Utilities
# =============================================================================

def _get_image_bytes(image_dict: dict) -> Optional[bytes]:
    """Get image bytes from path or base64."""
    if not image_dict:
        return None

    img_path = image_dict.get("image_path")
    if img_path and os.path.exists(img_path):
        try:
            with open(img_path, "rb") as f:
                return f.read()
        except OSError:
            pass

    b64 = image_dict.get("image_b64")
    if b64:
        try:
            return base64.b64decode(b64)
        except (ValueError, TypeError):
            return None

    return None


def get_results(module_path: str) -> Dict[str, Any]:
    """Load all JSON reports for a test module."""
    results_dir = get_results_dir()
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    results = {}

    if not os.path.exists(results_dir):
        return results

    for f in sorted(os.listdir(results_dir)):
        if (f.startswith(f"{module_name}.") or f.startswith("__main__.")) and f.endswith(".json"):
            try:
                test_name = ".".join(f.split(".")[1:-1])
                with open(os.path.join(results_dir, f), "r") as file:
                    results[test_name] = json.load(file)
            except (IOError, json.JSONDecodeError):
                pass

    return results


def _normalize_score(score: Any) -> float:
    """Normalize score to 0-1 range (handles both percentage and fraction)."""
    if score is None:
        return 0.0
    score = float(score)
    # If score > 1, assume it's a percentage
    if score > 1:
        return score / 100.0
    return score


def _get_threat_level(score: float) -> tuple:
    """Get threat level indicator based on score."""
    if score >= 0.8:
        return ("success", "Low Risk", "Model handles perturbations well")
    elif score >= 0.5:
        return ("warning", "Medium Risk", "Some vulnerabilities detected")
    else:
        return ("error", "High Risk", "Significant vulnerabilities found")


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar(results: dict):
    """Render sidebar with summary and downloads."""
    st.sidebar.header("Test Results")

    # Quick summary
    total_tests = len(results)
    avg_score = 0
    if results:
        scores = [_normalize_score(r.get("score", r.get("robust_accuracy", 0))) for r in results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0

    st.sidebar.metric("Tests Run", total_tests)
    st.sidebar.metric("Avg Robustness", f"{avg_score:.1%}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Downloads")

    for test_name, report in results.items():
        st.sidebar.download_button(
            label=f"{test_name}",
            data=json.dumps(report, indent=2),
            file_name=f"{test_name}_report.json",
            mime="application/json",
            key=f"dl_{test_name}"
        )


# =============================================================================
# Image Rendering
# =============================================================================

def render_image(image_data: dict, header: str, width: int = 200):
    """Render an image with caption."""
    if not image_data:
        st.info(f"No {header.lower()} available")
        return

    data_bytes = _get_image_bytes(image_data)
    if not data_bytes:
        st.info(f"No {header.lower()} available")
        return

    try:
        pil = Image.open(io.BytesIO(data_bytes))
        w, h = pil.size
    except (OSError, ValueError):
        w, h = None, None

    pred = image_data.get('prediction', '?')
    conf = image_data.get('confidence', 0)

    # Format prediction display
    pred_display = str(pred)
    if isinstance(conf, (int, float)):
        caption = f"Predicted: {pred_display} ({conf:.1%} confidence)"
    else:
        caption = f"Predicted: {pred_display}"

    img_b64 = base64.b64encode(data_bytes).decode("utf-8")
    st.markdown(f"**{header}**")
    st.markdown(
        f'<img class="pixelated" src="data:image/png;base64,{img_b64}" width="{width}"/>',
        unsafe_allow_html=True
    )
    st.caption(caption)


# =============================================================================
# Key Metrics
# =============================================================================

def render_key_metrics(report: dict):
    """Render main summary metrics."""
    cols = st.columns(4)

    # Score
    score = _normalize_score(report.get("score", report.get("robust_accuracy", 0)))
    cols[0].metric("Robustness", f"{score:.1%}")

    # Samples
    passed = report.get("passed_samples", 0)
    total = report.get("total_samples", 0)
    cols[1].metric("Passed/Total", f"{passed}/{total}")

    # Queries
    cols[2].metric("Queries", report.get("model_queries", "N/A"))

    # Runtime
    runtime = report.get("runtime", 0)
    cols[3].metric("Runtime", f"{runtime:.1f}s")


# =============================================================================
# Strategies Section
# =============================================================================

def render_strategies_section(report: dict):
    """Render strategy comparison chart."""
    search = report.get("search", {})
    results = search.get("results", [])

    if not results:
        info = report.get("perturbation_info")
        if info:
            st.write(f"**Strategy:** {info.get('name', 'Unknown').replace('Strategy', '')}")
        return

    # Build strategy comparison data
    strategy_data = []
    for r in results:
        strategy = r.get("strategy", "Unknown")
        rob_score = r.get("robustness_score", 0)
        threshold = r.get("failure_threshold", 0)
        failures = len(r.get("failures", []))
        strategy_data.append({
            "Strategy": strategy,
            "Robustness": rob_score,
            "Threshold": threshold,
            "Failures": failures
        })

    if strategy_data:
        df = pd.DataFrame(strategy_data)

        # Bar chart of robustness by strategy
        st.bar_chart(df.set_index("Strategy")["Robustness"], use_container_width=True)

        # Compact table below
        st.dataframe(
            df.style.format({"Robustness": "{:.1%}", "Threshold": "{:.4f}"}),
            hide_index=True,
            use_container_width=True
        )


# =============================================================================
# Executive Summary
# =============================================================================

def render_executive_summary(report: dict):
    """Render executive summary with key findings."""
    st.header("Summary")

    # Get metrics
    score = _normalize_score(report.get("score", report.get("robust_accuracy", 0)))
    passed = report.get("passed_samples", 0)
    total = report.get("total_samples", 0)
    failed = report.get("failed_samples", total - passed)

    # Metrics from nested structure
    metrics = report.get("metrics", {})
    threat_model = metrics.get("threat_model", report.get("search", {}).get("threat_model", "unknown"))
    unique_failures = metrics.get("unique_failed_samples", failed)
    strategies_tested = metrics.get("strategies_tested", 1)

    # Baseline accuracy metrics
    baseline_accuracy = metrics.get("baseline_accuracy")
    valid_samples = metrics.get("valid_samples", total)

    # Check if using model predictions as reference (label mismatch case)
    # This happens when baseline_accuracy is None or from search results
    search_results = report.get("search", {}).get("results", [])
    using_model_preds = any(r.get("using_model_predictions", False) for r in search_results) if search_results else False

    # Show baseline accuracy info only when meaningful
    excluded_samples = total - valid_samples

    if using_model_preds or baseline_accuracy is None:
        st.info(
            f"**Prediction Consistency Mode** - Testing if perturbations change the model's predictions. "
            f"Using model's original predictions as reference (labels don't match model's class space)."
        )
    elif baseline_accuracy is not None and baseline_accuracy < 1.0 and excluded_samples > 0:
        # Only show if samples were actually excluded
        exclusion_pct = (excluded_samples / total) * 100
        st.warning(
            f"**{excluded_samples}/{total} samples excluded** ({exclusion_pct:.1f}%) - "
            f"Model initially misclassified these samples. "
            f"Robustness testing performed on {valid_samples} correctly-classified samples."
        )

    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)

    level, label, hint = _get_threat_level(score)
    # Get failure threshold for display
    failure_threshold = metrics.get("failure_threshold")

    col1.metric(
        "Robustness Score",
        f"{score:.1%}",
        help=f"Based on failure threshold: model tolerates {score:.0%} of the perturbation range before failing"
    )
    col2.metric(
        "Failures at Threshold",
        f"{unique_failures}",
        help=f"Samples that failed at threshold level {failure_threshold:.4f}" if failure_threshold else "Samples that failed"
    )

    # Show testing mode or samples tested
    if using_model_preds or baseline_accuracy is None:
        col3.metric("Testing Mode", "Consistency", help="Testing if perturbations change predictions")
    else:
        col3.metric("Samples Tested", f"{valid_samples}/{total}", help="Correctly-classified samples tested for robustness")

    col4.metric("Threat Model", threat_model.title())

    # Insight box
    # Show vulnerability level insights
    if level == "error":
        st.error(f"**{label}:** {hint}. Consider adversarial training or input preprocessing.")
    elif level == "warning":
        st.warning(f"**{label}:** {hint}. Review failure patterns below.")
    else:
        st.success(f"**{label}:** {hint}")


# =============================================================================
# Failure Triage
# =============================================================================

def render_failure_triage(report: dict):
    """Render failure analysis with charts."""
    st.header("Failure Analysis")

    # Get search results for per-strategy breakdown
    search = report.get("search", {})
    results = search.get("results", [])

    if results:
        # Strategy failure breakdown
        failure_by_strategy = []
        for r in results:
            strategy = r.get("strategy", "Unknown")
            failures = len(r.get("failures", []))
            rob = r.get("robustness_score", 0)
            threshold = r.get("failure_threshold", 0)
            failure_by_strategy.append({
                "Strategy": strategy,
                "Failures": failures,
                "Robustness": rob * 100,  # As percentage for chart
                "Threshold": threshold
            })

        if failure_by_strategy:
            df = pd.DataFrame(failure_by_strategy)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Failures by Strategy")
                # Horizontal bar chart
                chart_df = df.set_index("Strategy")[["Failures"]].sort_values("Failures", ascending=True)
                st.bar_chart(chart_df, horizontal=True, use_container_width=True)

            with col2:
                st.subheader("Robustness by Strategy")
                # Show robustness scores
                chart_df = df.set_index("Strategy")[["Robustness"]].sort_values("Robustness", ascending=False)
                st.bar_chart(chart_df, horizontal=True, use_container_width=True, color="#2ecc71")

            # Most vulnerable strategy insight
            worst = df.loc[df["Robustness"].idxmin()]
            best = df.loc[df["Robustness"].idxmax()]

            st.info(f"**Most vulnerable to:** {worst['Strategy']} ({worst['Robustness']:.1f}% robustness)  \n"
                   f"**Most robust against:** {best['Strategy']} ({best['Robustness']:.1f}% robustness)")

            # Also show individual failure details from all strategies
            all_failures = []
            for r in results:
                strategy = r.get("strategy", "Unknown")
                threshold = r.get("failure_threshold")
                for f in r.get("failures", []):
                    f_copy = dict(f)
                    f_copy["strategy"] = strategy
                    f_copy["_threshold"] = threshold  # Track threshold for display
                    all_failures.append(f_copy)

            if all_failures:
                st.subheader("Sample Failures at Threshold")
                # Get average/representative threshold for display
                thresholds = [f.get("_threshold") for f in all_failures if f.get("_threshold") is not None]
                avg_threshold = sum(thresholds) / len(thresholds) if thresholds else None
                _render_failure_table(all_failures, report.get("total_samples", len(all_failures)), avg_threshold)

            return

    # Fallback to failures list or per_sample_metrics
    failures_list = report.get("failures", [])
    per_sample = report.get("per_sample_metrics", [])
    failures = failures_list if failures_list else [s for s in per_sample if not s.get("passed", True)]

    if not failures:
        st.success("All samples passed - no failures detected")
        return

    total_samples = report.get("total_samples", len(per_sample) or len(failures))
    threshold = report.get("failure_threshold") or report.get("metrics", {}).get("failure_threshold")
    _render_failure_table(failures, total_samples, threshold)


def _render_failure_table(failures: list, total_samples: int, threshold: float = None):
    """Render failure details table."""
    if threshold is not None:
        st.write(f"**{len(failures)}** samples failed at threshold level {threshold:.4f}")
    else:
        st.write(f"**{len(failures)}** samples failed out of {total_samples}")

    # Show sample failure breakdown with descriptive headers
    if len(failures) <= 30:
        failure_data = []
        for f in failures:
            # Try multiple field names (new API uses original_pred/perturbed_pred)
            original = f.get("original_label", f.get("clean_label", "?"))
            original_name = f.get("original_label_name")
            predicted = f.get("perturbed_pred", f.get("pert_label", "?"))
            predicted_name = f.get("perturbed_pred_name")

            # Format display: prefer name if available, otherwise just show index
            if original_name:
                orig_display = f"{original_name} (#{original})"
            else:
                orig_display = str(original)

            if predicted_name:
                pred_display = f"{predicted_name} (#{predicted})"
            else:
                pred_display = str(predicted)

            row = {}

            # Add strategy if available (from multi-strategy tests)
            strategy = f.get("strategy")
            if strategy:
                row["Strategy"] = strategy

            row["Original Label"] = orig_display
            row["Predicted (After Perturbation)"] = pred_display

            # Add perturbation level if available
            level = f.get("level")
            if level is not None:
                # Use appropriate precision based on value magnitude
                if level == 0:
                    row["Perturbation Level"] = "0 (no perturbation)"
                elif level < 0.01:
                    row["Perturbation Level"] = f"{level:.4f}"
                elif level < 0.1:
                    row["Perturbation Level"] = f"{level:.3f}"
                else:
                    row["Perturbation Level"] = f"{level:.2f}"

            failure_data.append(row)

        st.dataframe(pd.DataFrame(failure_data), hide_index=True)

        st.caption(
            "**Threshold Level**: The minimum perturbation where the model starts failing (pass rate < 90%). "
            "**Original Label**: The ground truth class of the sample. "
            "**Predicted**: What the model incorrectly predicted after perturbation was applied."
        )


# =============================================================================
# Root Cause Analysis
# =============================================================================

def render_root_cause_analysis(report: dict):
    """Render root cause analysis with insights."""
    st.header("Root Cause Analysis")

    metrics = report.get("metrics", {})
    search = report.get("search", {})
    results = search.get("results", [])

    if not results:
        st.info("Run search-based tests for detailed root cause analysis")
        return

    # Analyze patterns
    low_threshold_strategies = []
    high_failure_strategies = []
    total_samples = report.get("total_samples", 1)

    for r in results:
        strategy = r.get("strategy", "Unknown")
        threshold = r.get("failure_threshold", 1)
        rob = r.get("robustness_score", 1)
        failures = len(r.get("failures", []))

        if threshold < 0.1:
            low_threshold_strategies.append((strategy, threshold))
        if failures > total_samples * 0.3:
            high_failure_strategies.append((strategy, failures))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Low Tolerance Perturbations")
        if low_threshold_strategies:
            for s, t in sorted(low_threshold_strategies, key=lambda x: x[1]):
                st.write(f"- **{s}**: fails at {t:.4f}")
            st.caption("These perturbations cause failures at very small magnitudes")
        else:
            st.write("No extremely sensitive perturbations found")

    with col2:
        st.subheader("High Impact Perturbations")
        if high_failure_strategies:
            for s, f in sorted(high_failure_strategies, key=lambda x: -x[1]):
                st.write(f"- **{s}**: {f} failures")
            st.caption("These perturbations affect many samples")
        else:
            st.write("No high-impact perturbations found")

    # Overall insight
    threat_model = metrics.get("threat_model", "unknown")
    overall_rob = metrics.get("overall_robustness_score", 0)

    if overall_rob < 0.1:
        st.error(f"**Critical vulnerability** under {threat_model} threats. Model needs robustness training.")
    elif overall_rob < 0.3:
        st.warning(f"**Significant sensitivity** to {threat_model} perturbations. Consider data augmentation.")
    else:
        st.info(f"Model shows reasonable resilience to {threat_model} perturbations.")


# =============================================================================
# Search Path Insights
# =============================================================================

def render_search_path_insights(report: dict):
    """Render search path with line chart."""
    st.header("Search Path Analysis")

    # Check multiple locations for search path
    search = report.get("search", {})
    search_path = report.get("search_path", []) or search.get("search_path", [])
    results = search.get("results", [])

    if results and not search_path:
        # Build combined search path from results (multi-strategy)
        combined_path = []
        for r in results:
            strategy = r.get("strategy", "Unknown")
            path = r.get("search_path", [])
            for step in path:
                # Create a copy to avoid mutating original
                step_copy = dict(step)
                step_copy["strategy"] = strategy
                combined_path.append(step_copy)
        search_path = combined_path

    if not search_path:
        st.info("No search path data available")
        return

    df = pd.DataFrame(search_path)

    if "level" not in df.columns:
        st.info("Search path data format not supported")
        return

    # Get threshold info from multiple locations
    threshold = report.get("failure_threshold") or report.get("metrics", {}).get("failure_threshold")

    # Group by strategy if available
    if "strategy" in df.columns:
        strategies = df["strategy"].unique()

        for strategy in strategies:
            strategy_df = df[df["strategy"] == strategy].copy()
            strategy_df = strategy_df.sort_values("level")

            if "pass_rate" in strategy_df.columns:
                st.subheader(f"{strategy}")

                # Build chart data with pass_rate and confidence if available
                chart_cols = ["pass_rate"]
                if "avg_confidence" in strategy_df.columns:
                    chart_cols.append("avg_confidence")

                chart_data = strategy_df.set_index("level")[chart_cols].copy()
                chart_data.columns = ["Pass Rate"] + (["Confidence"] if "avg_confidence" in strategy_df.columns else [])
                st.line_chart(chart_data, use_container_width=True)

                # Find threshold (where pass_rate drops below 90%)
                fail_threshold = strategy_df[strategy_df["pass_rate"] < 0.9]["level"].min()
                if pd.notna(fail_threshold):
                    st.caption(f"First failure at level {fail_threshold:.4f}")
    else:
        # Single strategy
        if "pass_rate" in df.columns:
            # Sort by level for the chart (shows pass rate across perturbation spectrum)
            df_by_level = df.sort_values("level").copy()

            # Build chart data with pass_rate and confidence if available
            st.subheader("Pass Rate & Confidence vs Perturbation Level")
            chart_cols = ["pass_rate"]
            if "avg_confidence" in df_by_level.columns:
                chart_cols.append("avg_confidence")

            chart_data = df_by_level.set_index("level")[chart_cols].copy()
            chart_data.columns = ["Pass Rate"] + (["Confidence"] if "avg_confidence" in df_by_level.columns else [])
            st.line_chart(chart_data, use_container_width=True)

            # Show search steps in chronological order (how search progressed)
            st.subheader("Search Progression")
            st.caption("Shows how the adaptive search explored the perturbation space (in order of execution)")

            # Sort by iteration (chronological order)
            df_chrono = df.sort_values("iteration").copy()

            # Build display columns based on available data
            base_cols = ["iteration", "level", "pass_rate"]
            display_df = df_chrono[base_cols].copy()
            display_df["pass_rate"] = display_df["pass_rate"].apply(lambda x: f"{x:.1%}")
            display_df["level"] = display_df["level"].apply(lambda x: f"{x:.5f}")

            # Add confidence columns if available
            if "avg_confidence" in df_chrono.columns:
                display_df["Confidence"] = df_chrono["avg_confidence"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            if "confidence_drop" in df_chrono.columns:
                display_df["Conf. Drop"] = df_chrono["confidence_drop"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

            display_df["Passed"] = df_chrono["passed"].apply(lambda x: "✓" if x else "✗")

            # Rename base columns
            display_df = display_df.rename(columns={"iteration": "Step", "level": "Level", "pass_rate": "Pass Rate"})
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.dataframe(df.head(15), hide_index=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Search Steps", len(search_path))

    if threshold is not None:
        col2.metric("Failure Threshold", f"{threshold:.5f}")

    # Calculate pass/fail ratio
    if "passed" in df.columns:
        pass_count = df["passed"].sum()
        fail_count = len(df) - pass_count
        col3.metric("Pass/Fail Steps", f"{pass_count}/{fail_count}")


# =============================================================================
# Actionable Recommendations
# =============================================================================

def render_actionable_recommendations(report: dict):
    """Render recommendations based on findings."""
    st.header("Recommendations")

    score = _normalize_score(report.get("score", report.get("robust_accuracy", 0)))
    metrics = report.get("metrics", {})
    threat_model = metrics.get("threat_model", "unknown")
    overall_rob = metrics.get("overall_robustness_score", 0)

    # Get vulnerable strategies
    search = report.get("search", {})
    results = search.get("results", [])
    vulnerable = [r.get("strategy") for r in results if r.get("robustness_score", 1) < 0.1]

    recommendations = []

    if score < 0.5:
        recommendations.append(("error", "High Priority", "Implement robustness improvements before deployment"))

    if "adversarial" in threat_model.lower() or any("fgsm" in str(v).lower() or "pgd" in str(v).lower() for v in vulnerable):
        recommendations.append(("warning", "Adversarial Training",
            "Consider adversarial training (PGD-AT) or certified defenses"))

    if any("brightness" in str(v).lower() or "contrast" in str(v).lower() for v in vulnerable):
        recommendations.append(("info", "Data Augmentation",
            "Add brightness/contrast jittering during training"))

    if any("blur" in str(v).lower() for v in vulnerable):
        recommendations.append(("info", "Blur Augmentation",
            "Include Gaussian blur in training augmentations"))

    if any("noise" in str(v).lower() for v in vulnerable):
        recommendations.append(("info", "Noise Robustness",
            "Add Gaussian noise augmentation or use denoising layers"))

    if any("jpeg" in str(v).lower() or "compression" in str(v).lower() for v in vulnerable):
        recommendations.append(("info", "Compression Artifacts",
            "Train with JPEG-compressed images"))

    if not recommendations:
        if score >= 0.8:
            st.success("Model shows good robustness. Ready for deployment.")
        else:
            st.info("Review failure patterns and consider targeted augmentations.")
        return

    for level, title, desc in recommendations:
        if level == "error":
            st.error(f"**{title}:** {desc}")
        elif level == "warning":
            st.warning(f"**{title}:** {desc}")
        else:
            st.info(f"**{title}:** {desc}")

    # Code example for top recommendation
    if score < 0.5:
        with st.expander("Example: Adding robustness training"):
            st.code("""
from torchvision import transforms

# Robustness-focused augmentation
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    # Add random noise
    transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x)),
])
""", language="python")


# =============================================================================
# Analysis Tabs
# =============================================================================

def render_analysis_tabs(report: dict):
    """Render detailed analysis in tabs."""

    tabs = st.tabs(["Strategy Details", "Raw Data"])

    with tabs[0]:
        search = report.get("search", {})
        results = search.get("results", [])

        if results:
            for r in results:
                strategy = r.get("strategy", "Unknown")
                with st.expander(f"{strategy}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Threshold", f"{r.get('failure_threshold', 0):.4f}")
                    col2.metric("Robustness", f"{r.get('robustness_score', 0):.1%}")
                    col3.metric("Failures", len(r.get("failures", [])))

                    # Search path for this strategy
                    path = r.get("search_path", [])
                    if path:
                        df = pd.DataFrame(path)
                        if "level" in df.columns and "pass_rate" in df.columns:
                            st.line_chart(df.set_index("level")["pass_rate"])
        else:
            st.info("No detailed strategy data available")

    with tabs[1]:
        st.json(report)


# =============================================================================
# AI-Style Insights
# =============================================================================

def render_insights(reports: list):
    """
    Render AI-generated insights comparing multiple reports.

    Args:
        reports: List of report dicts from JSON files
    """
    from ..report import Report, generate_insights

    st.header("🧠 Insights")

    if not reports or len(reports) < 2:
        st.info("Add more test results to see comparative insights.")
        return

    # Convert dicts to Report objects for insight generation
    report_objects = []
    for r in reports:
        try:
            # Create Report from dict
            report_obj = Report(
                test_name=r.get("test_name", "unknown"),
                test_type=r.get("test_type", "search"),
                timestamp=r.get("timestamp", ""),
                model_name=r.get("model_name", "Unknown"),
                dataset=r.get("dataset", "Unknown"),
                preset=r.get("preset"),
                property_name=r.get("property_name", "Unknown"),
                strategy=r.get("strategy", "Unknown"),
                score=r.get("score", 0),
                total_samples=r.get("total_samples", 0),
                passed_samples=r.get("passed_samples", 0),
                runtime=r.get("runtime", 0),
                model_queries=r.get("model_queries", 0),
                metrics=r.get("metrics", {}),
                search=r.get("search", {}),
                failures=r.get("failures", []),
            )
            report_objects.append(report_obj)
        except Exception:
            continue

    if len(report_objects) < 2:
        st.info("Need at least 2 valid reports for comparative analysis.")
        return

    # Generate insights
    insights = generate_insights(report_objects)

    for insight in insights:
        st.markdown(f"• {insight}")
