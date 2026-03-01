"""
VisProbe Dashboard - Visual robustness testing results.

Streamlit-based dashboard for analyzing model robustness test results.
"""

import os
import sys

import streamlit as st

from visprobe.cli.dashboard_helpers import (
    get_results,
    render_sidebar,
    render_executive_summary,
    render_failure_triage,
    render_root_cause_analysis,
    render_search_path_insights,
    render_actionable_recommendations,
    render_key_metrics,
    render_strategies_section,
    render_image,
    render_analysis_tabs,
    render_insights,
    _get_image_bytes,
)

# --- Page Configuration ---
st.set_page_config(page_title="VisProbe Dashboard", page_icon="🔬", layout="wide")


def render_report(test_name: str, report: dict):
    """Render a single test report with all analysis sections."""
    import io
    from PIL import Image as PILImage

    with st.expander(f"📋 **{test_name}**", expanded=True):
        # Section 1: Executive Summary
        render_executive_summary(report)

        # Section 2: Failure Analysis (with charts)
        render_failure_triage(report)

        # Section 3: Root Cause Analysis
        render_root_cause_analysis(report)

        # Section 4: Search Path (with line charts)
        search = report.get("search", {})
        results = search.get("results", [])
        search_path = report.get("search_path", []) or search.get("search_path", [])
        if results or search_path:
            render_search_path_insights(report)

        # Section 5: Recommendations
        render_actionable_recommendations(report)

        # Additional Details (collapsed by default)
        with st.expander("View Details", expanded=False):
            tab1, tab2 = st.tabs(["Images", "Raw Data"])

            with tab1:
                # Image comparison
                o = report.get("original_image")
                p = report.get("perturbed_image")
                r = report.get("residual_image")

                if o or p:
                    cols = st.columns(3 if r else 2)
                    with cols[0]:
                        render_image(o, "Original", width=180)
                    with cols[1]:
                        render_image(p, "Perturbed", width=180)
                    if r:
                        with cols[2]:
                            render_image(r, "Difference", width=180)
                else:
                    st.info("No sample images available")

            with tab2:
                render_analysis_tabs(report)


def main(module_path: str):
    """Main dashboard entry point."""
    st.title("🔬 VisProbe Robustness Dashboard")
    st.caption(f"Results for: `{os.path.basename(module_path)}`")

    # CSS for crisp image scaling
    st.markdown("""
        <style>
        img.pixelated {
          image-rendering: pixelated;
          image-rendering: crisp-edges;
        }
        </style>
    """, unsafe_allow_html=True)

    results = get_results(module_path)

    if not results:
        st.error(f"No results found for: `{os.path.basename(module_path)}`")
        st.info("Run your tests first, then visualize the results.")
        st.code(f"python {os.path.basename(module_path)}")
        st.stop()

    render_sidebar(results)

    # Render comparative insights at the top (if multiple reports)
    if len(results) > 1:
        render_insights(list(results.values()))
        st.divider()

    # Render each test report
    for test_name, report in results.items():
        render_report(test_name, report)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        st.error("No test module specified.")
        st.info("Usage: `visprobe visualize <test_file.py>`")
