"""
Lightweight wrapper to capture intermediate layer outputs during forward.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn


class _ModelWithIntermediateOutput(nn.Module):
    """A wrapper to capture intermediate layer outputs from a model."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str) -> Callable:
        """
        Creates a forward hook to capture layer outputs.

        Args:
            name: Name of the layer to capture

        Returns:
            Hook function that stores detached layer outputs
        """
        def hook(module, input, output):
            # Detach to avoid tracking gradients and reduce graph retention
            try:
                self._features[name] = output.detach()
            except Exception:
                # Fallback in case output is not a tensor (or tuple). Keep original behavior.
                self._features[name] = output

        return hook

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with intermediate layer output capture.

        Args:
            x: Input tensor

        Returns:
            Tuple of (model_output, dict_of_intermediate_features)
        """
        self._features.clear()
        output = self.model(x)
        return output, self._features.copy()

    def __del__(self):
        """Cleanup hooks on deletion to prevent memory leaks."""
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                pass
