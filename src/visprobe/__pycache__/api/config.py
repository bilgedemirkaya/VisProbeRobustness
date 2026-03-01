"""
Centralized configuration for the VisProbe API.
"""

import logging
import os
from typing import List

import torch

logger = logging.getLogger(__name__)

# Default normalization constants for ImageNet
IMAGENET_DEFAULT_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD: List[float] = [0.229, 0.224, 0.225]

# Default analysis configurations
DEFAULT_RESOLUTIONS = [(128, 128), (224, 224), (299, 299)]
DEFAULT_NOISE_SWEEP = {"levels": 15, "min_level": 0.0, "max_level": 0.5}


# Default device selection strategy
def get_default_device() -> torch.device:  # noqa: C901
    """
    Smart default device selection with broad hardware support.

    Priority order:
    1. Explicit VISPROBE_DEVICE environment variable
    2. If VISPROBE_PREFER_GPU=1, try available accelerators in order:
       a. CUDA (NVIDIA GPUs)
       b. ROCm (AMD GPUs)
       c. MPS (Apple Silicon)
    3. CPU (most stable, maximum compatibility)

    Returns:
        torch.device: The selected device

    Environment Variables:
        VISPROBE_DEVICE: Explicit device string (e.g., 'cuda', 'cpu', 'mps', 'hip')
        VISPROBE_PREFER_GPU: If '1', 'true', or 'yes', prefer GPU over CPU
        VISPROBE_DEBUG: If set, log device selection details
    """
    env_device = os.environ.get("VISPROBE_DEVICE", "").lower().strip()
    debug_mode = os.environ.get("VISPROBE_DEBUG", "").lower() in ("1", "true", "yes")

    # Explicit device override
    if env_device and env_device != "auto":
        if debug_mode:
            logger.info(f"Using explicit device from VISPROBE_DEVICE: {env_device}")
        return torch.device(env_device)

    # Auto-detect best available device
    prefer_gpu = os.environ.get("VISPROBE_PREFER_GPU", "").lower() in ("1", "true", "yes")

    if prefer_gpu:
        # Try CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            if debug_mode:
                logger.info(f"Selected CUDA device (GPU count: {torch.cuda.device_count()})")
            return torch.device("cuda")

        # Try ROCm (AMD GPUs)
        try:
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                # ROCm available
                if debug_mode:
                    logger.info("Selected ROCm/HIP device (AMD GPU)")
                return torch.device("hip")
        except Exception as e:
            if debug_mode:
                logger.debug(f"ROCm not available: {e}")

        # Try MPS (Apple Silicon)
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if hasattr(torch.backends.mps, "is_built") and torch.backends.mps.is_built():
                    if debug_mode:
                        logger.info("Selected MPS device (Apple Silicon)")
                    return torch.device("mps")
        except Exception as e:
            if debug_mode:
                logger.debug(f"MPS not available: {e}")

        if debug_mode:
            logger.info("No GPU found despite VISPROBE_PREFER_GPU=1, falling back to CPU")

    # Default to CPU for maximum compatibility
    if debug_mode:
        logger.info("Selected CPU device (default)")
    return torch.device("cpu")


# Set threading defaults for better stability
def configure_threading():  # noqa: C901
    """Configure threading for optimal performance and stability."""
    # Set environment variables before torch operations
    if not os.environ.get("OMP_NUM_THREADS"):
        os.environ["OMP_NUM_THREADS"] = "1"
    if not os.environ.get("MKL_NUM_THREADS"):
        os.environ["MKL_NUM_THREADS"] = "1"

    # Only set torch threading if not already configured
    try:
        # Check if we can safely set num_threads
        current_threads = torch.get_num_threads()
        desired_threads = max(1, int(os.environ.get("VF_THREADS", "1")))
        if current_threads != desired_threads:
            torch.set_num_threads(desired_threads)
            logger.debug(f"Set torch num_threads to {desired_threads}")
    except ValueError as e:
        logger.warning(f"Invalid VF_THREADS value: {e}")
    except RuntimeError as e:
        logger.debug(f"Could not set num_threads (may be already initialized): {e}")
    except Exception as e:
        logger.warning(f"Unexpected error setting num_threads: {e}")

    # Avoid setting interop threads if already initialized
    try:
        # Only set if we haven't done parallel work yet
        if not hasattr(torch, "_visprobe_interop_configured"):
            torch.set_num_interop_threads(1)
            torch._visprobe_interop_configured = True
            logger.debug("Set torch interop_threads to 1")
    except RuntimeError as e:
        # If it fails, parallel work has already started - this is expected
        logger.debug(f"Could not set interop_threads (parallel work already started): {e}")
    except Exception as e:
        logger.warning(f"Unexpected error setting interop_threads: {e}")
