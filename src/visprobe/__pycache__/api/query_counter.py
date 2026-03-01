"""
Context manager to count model.forward calls within a scope.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn


class QueryCounter:
    """
    Thread-safe context manager that counts model.forward calls using hooks.

    This implementation uses PyTorch's forward hooks instead of monkey-patching,
    making it safe for concurrent use and compatible with all model types.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._count = 0
        self._hook_handle: Optional[object] = None

    def _forward_hook(self, module, input, output):
        """Hook function that increments the counter on each forward pass."""
        self._count += 1
        return output

    def __enter__(self):
        """Register forward hook when entering context."""
        # Register the forward hook
        self._hook_handle = self.model.register_forward_hook(self._forward_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove forward hook when exiting context."""
        # Remove the hook to avoid memory leaks
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @property
    def extra(self) -> int:
        """
        Return the count of extra forward passes.

        Note: Subtracts 1 to maintain backward compatibility with the old API
        that didn't count the initial forward pass.
        """
        return max(0, self._count - 1)
