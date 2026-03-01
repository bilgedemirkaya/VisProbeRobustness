"""
Shared utility functions for the VisProbe CLI.
"""

from __future__ import annotations

import os
import tempfile


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
