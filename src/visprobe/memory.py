"""
GPU memory management for handling multiple models efficiently.
Automatically swaps models between CPU and GPU to prevent OOM.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import gc
import logging

logger = logging.getLogger(__name__)


class ModelMemoryManager:
    """
    Manages GPU memory by swapping models between CPU and GPU.

    Ensures only one model is on GPU at a time to prevent OOM errors.
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        device: str = "cuda",
        memory_threshold: float = 0.9
    ):
        """
        Initialize memory manager.

        Args:
            models: Dictionary of model name to model instance
            device: Target device (cuda/cpu)
            memory_threshold: Max GPU memory usage before forcing cleanup (0-1)
        """
        self.models = models
        self.device = device if torch.cuda.is_available() else "cpu"
        self.memory_threshold = memory_threshold
        self.current_model_name = None
        self.original_devices = {}

        # Store original device for each model
        for name, model in models.items():
            # Get first parameter's device as reference
            try:
                param = next(model.parameters())
                self.original_devices[name] = str(param.device)
            except StopIteration:
                self.original_devices[name] = "cpu"

        # Move all models to CPU initially
        self._move_all_to_cpu()

    def load_model(self, name: str) -> nn.Module:
        """
        Load a specific model to GPU, moving others to CPU.

        Args:
            name: Name of the model to load

        Returns:
            The model on the specified device, in eval mode
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self.models.keys())}")

        # If same model already loaded, just return it
        if self.current_model_name == name:
            return self.models[name]

        # Move current model to CPU if exists
        if self.current_model_name is not None:
            logger.info(f"Moving {self.current_model_name} to CPU")
            self.models[self.current_model_name].cpu()
            self._clear_cache()

        # Check memory before loading
        if self.device != "cpu":
            self._check_memory_available()

        # Load requested model to device
        logger.info(f"Loading {name} to {self.device}")
        model = self.models[name].to(self.device)
        model.eval()
        self.current_model_name = name

        # Log memory usage
        if self.device != "cpu":
            self._log_memory_usage(name)

        return model

    def get_current_model(self) -> Optional[nn.Module]:
        """Get the currently loaded model."""
        if self.current_model_name is None:
            return None
        return self.models[self.current_model_name]

    def release_all(self):
        """Move all models back to CPU and clear cache."""
        self._move_all_to_cpu()
        self.current_model_name = None

    def restore_original_devices(self):
        """Restore models to their original devices."""
        for name, model in self.models.items():
            original_device = self.original_devices.get(name, "cpu")
            model.to(original_device)
        self.current_model_name = None

    def estimate_model_memory(self, name: str) -> Dict[str, float]:
        """
        Estimate memory usage for a model.

        Args:
            name: Model name

        Returns:
            Dictionary with memory estimates in MB
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")

        model = self.models[name]
        param_memory = 0
        buffer_memory = 0

        # Calculate parameter memory
        for param in model.parameters():
            param_memory += param.numel() * param.element_size()

        # Calculate buffer memory
        for buffer in model.buffers():
            buffer_memory += buffer.numel() * buffer.element_size()

        return {
            'param_mb': param_memory / (1024 * 1024),
            'buffer_mb': buffer_memory / (1024 * 1024),
            'total_mb': (param_memory + buffer_memory) / (1024 * 1024)
        }

    def _move_all_to_cpu(self):
        """Move all models to CPU."""
        for name, model in self.models.items():
            model.cpu()
        self._clear_cache()

    def _clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def _check_memory_available(self):
        """Check if enough GPU memory is available."""
        if not torch.cuda.is_available():
            return

        # Get current memory usage
        allocated = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory

        if allocated > self.memory_threshold:
            logger.warning(f"GPU memory usage high: {allocated:.1%}")
            self._clear_cache()

            # Check again after clearing
            allocated = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            if allocated > self.memory_threshold:
                logger.warning(f"GPU memory still high after clearing: {allocated:.1%}")

    def _log_memory_usage(self, model_name: str):
        """Log current GPU memory usage."""
        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

        logger.info(
            f"GPU Memory after loading {model_name}: "
            f"Allocated: {allocated:.2f}GB, "
            f"Reserved: {reserved:.2f}GB, "
            f"Total: {total:.2f}GB"
        )


class BatchMemoryOptimizer:
    """
    Optimizes batch processing to prevent OOM errors.

    Automatically adjusts batch size based on available memory.
    """

    def __init__(
        self,
        initial_batch_size: int = 50,
        min_batch_size: int = 1,
        memory_threshold: float = 0.9
    ):
        """
        Initialize batch optimizer.

        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            memory_threshold: Max memory usage before reducing batch size
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = initial_batch_size
        self.memory_threshold = memory_threshold
        self.failure_count = 0

    def process_with_adaptive_batch(
        self,
        data: torch.Tensor,
        process_fn,
        device: str = "cuda"
    ) -> List[torch.Tensor]:
        """
        Process data with adaptive batch size.

        Args:
            data: Input tensor
            process_fn: Function to process a batch
            device: Device to use

        Returns:
            List of processed batches
        """
        results = []
        n_samples = len(data)
        start_idx = 0

        while start_idx < n_samples:
            batch_size = min(self.current_batch_size, n_samples - start_idx)
            batch = data[start_idx:start_idx + batch_size].to(device)

            try:
                # Try processing with current batch size
                result = process_fn(batch)
                results.append(result)
                start_idx += batch_size

                # Success - maybe increase batch size
                if self.failure_count > 0:
                    self.failure_count -= 1
                elif self.current_batch_size < self.max_batch_size:
                    self.current_batch_size = min(
                        self.current_batch_size + 10,
                        self.max_batch_size
                    )

            except RuntimeError as e:
                if "out of memory" in str(e):
                    # OOM - reduce batch size
                    logger.warning(f"OOM with batch size {batch_size}, reducing...")
                    torch.cuda.empty_cache()

                    self.failure_count += 1
                    self.current_batch_size = max(
                        self.current_batch_size // 2,
                        self.min_batch_size
                    )

                    if self.current_batch_size == self.min_batch_size and batch_size == 1:
                        logger.error("OOM even with batch size 1")
                        raise

                    # Don't increment start_idx, retry this batch
                else:
                    raise

        return results

    def reset(self):
        """Reset batch size to initial value."""
        self.current_batch_size = self.max_batch_size
        self.failure_count = 0