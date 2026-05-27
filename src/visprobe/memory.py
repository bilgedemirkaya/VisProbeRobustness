"""
GPU memory management: keeps one model on GPU at a time, swaps the rest to CPU.
"""

import gc
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelMemoryManager:
    """One model on GPU, the rest on CPU; swap on demand."""

    def __init__(self, models: Dict[str, nn.Module], device: str = "cuda"):
        self.models = models
        self.device = device if torch.cuda.is_available() else "cpu"
        self.current_model_name: Optional[str] = None
        self._move_all_to_cpu()

    def load_model(self, name: str) -> nn.Module:
        """Move ``name`` to the GPU, move the previously-loaded model back to CPU."""
        if name not in self.models:
            raise ValueError(
                f"Model {name!r} not found. Available: {list(self.models.keys())}"
            )
        if self.current_model_name == name:
            return self.models[name]

        if self.current_model_name is not None:
            self.models[self.current_model_name].cpu()
            self._clear_cache()

        model = self.models[name].to(self.device).eval()
        self.current_model_name = name
        logger.info("Loaded %s to %s", name, self.device)
        return model

    def release_all(self) -> None:
        self._move_all_to_cpu()
        self.current_model_name = None

    def estimate_model_memory(self, name: str) -> Dict[str, float]:
        """Parameter + buffer size of a model, in MB."""
        if name not in self.models:
            raise ValueError(f"Model {name!r} not found")
        model = self.models[name]
        param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        buffer_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 ** 2)
        return {"param_mb": param_mb, "buffer_mb": buffer_mb, "total_mb": param_mb + buffer_mb}

    def _move_all_to_cpu(self) -> None:
        for model in self.models.values():
            model.cpu()
        self._clear_cache()

    def _clear_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
