"""
Auto-initialization module for VisProbe.

Import this module at the top of your test files to automatically configure
device management, threading, and other stability settings.

Usage:
    import visprobe.auto_init  # Just import, no need to call anything

    # Your test code here - no manual device management needed
    from visprobe.api.decorators import given, model, data_source
    # ...
"""

import os
import warnings

# Suppress common noisy warnings
warnings.filterwarnings("ignore", message=".*antialias.*will change from None to True.*")

# Set environment variables early (before torch import)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"
if not os.environ.get("MKL_NUM_THREADS"):
    os.environ["MKL_NUM_THREADS"] = "1"

# Import after setting env vars
from .api.config import get_default_device  # noqa: E402

# Set default device if not already set
if not os.environ.get("VISPROBE_DEVICE"):
    default_device = get_default_device()
    os.environ["VISPROBE_DEVICE"] = str(default_device)

# Print configuration info for debugging
if os.environ.get("VISPROBE_DEBUG", "").lower() in ("1", "true", "yes"):
    print(f"[VisProbe Auto-Init] Device: {os.environ.get('VISPROBE_DEVICE')}")
    print(f"[VisProbe Auto-Init] OMP Threads: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"[VisProbe Auto-Init] MKL Threads: {os.environ.get('MKL_NUM_THREADS')}")
