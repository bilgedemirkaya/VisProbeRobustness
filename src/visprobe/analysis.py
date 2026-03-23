"""
Simplified analysis functions for model evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SampleResult:
    """Result for a single sample."""
    index: int
    predicted_class: int
    true_class: int
    confidence: float
    is_correct: bool
    top_5_classes: List[int]
    top_5_confidences: List[float]


@dataclass
class DetailedResults:
    """Detailed evaluation results."""
    model_name: str
    scenario: str
    samples: List[SampleResult]
    accuracy: float
    top_5_accuracy: float
    mean_confidence: float
    mean_loss: float
    correct_mask: np.ndarray
    total_samples: int
    metadata: Dict[str, Any]

    def get_failures(self) -> List[SampleResult]:
        """Get all failed samples."""
        return [s for s in self.samples if not s.is_correct]

    def get_successes(self) -> List[SampleResult]:
        """Get all successful samples."""
        return [s for s in self.samples if s.is_correct]

    def get_high_confidence_errors(self, threshold: float = 0.9) -> List[SampleResult]:
        """Get high confidence errors."""
        return [s for s in self.samples if not s.is_correct and s.confidence > threshold]


def evaluate_detailed(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    model_name: str = "model",
    scenario: str = "test",
    device: str = "cuda",
    batch_size: int = 50,
    top_k: int = 5,
    **kwargs
) -> DetailedResults:
    """
    Evaluate model with detailed per-sample tracking.

    Args:
        model: Model to evaluate
        images: Input images
        labels: Ground truth labels
        model_name: Name of the model
        scenario: Scenario name
        device: Device to use
        batch_size: Batch size for evaluation
        top_k: Number of top predictions to track
        **kwargs: Additional metadata to store

    Returns:
        DetailedResults with per-sample tracking
    """
    model = model.to(device)
    model.eval()

    samples = []
    all_correct = []
    all_top5_correct = []
    all_confidences = []
    total_loss = 0.0
    n_batches = 0

    n_samples = len(images)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            # Get batch
            batch_end = min(i + batch_size, n_samples)
            batch_images = images[i:batch_end].to(device)
            batch_labels = labels[i:batch_end].to(device)

            # Forward pass
            outputs = model(batch_images)
            loss = F.cross_entropy(outputs, batch_labels)

            # Get predictions
            probs = F.softmax(outputs, dim=1)
            top_k_probs, top_k_classes = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)

            # Process each sample
            for j in range(len(batch_images)):
                sample_idx = i + j
                predicted_class = top_k_classes[j, 0].item()
                true_class = batch_labels[j].item()
                confidence = top_k_probs[j, 0].item()
                is_correct = predicted_class == true_class
                is_top5_correct = true_class in top_k_classes[j].tolist()

                sample = SampleResult(
                    index=sample_idx,
                    predicted_class=predicted_class,
                    true_class=true_class,
                    confidence=confidence,
                    is_correct=is_correct,
                    top_5_classes=top_k_classes[j].tolist(),
                    top_5_confidences=top_k_probs[j].tolist()
                )
                samples.append(sample)
                all_correct.append(is_correct)
                all_top5_correct.append(is_top5_correct)
                all_confidences.append(confidence)

            total_loss += loss.item()
            n_batches += 1

    # Calculate metrics
    accuracy = float(np.mean(all_correct))
    top_5_accuracy = float(np.mean(all_top5_correct))
    mean_confidence = float(np.mean(all_confidences))
    mean_loss = total_loss / n_batches if n_batches > 0 else 0.0
    correct_mask = np.array(all_correct)

    # Store metadata
    metadata = {
        'device': device,
        'batch_size': batch_size,
        **kwargs
    }

    return DetailedResults(
        model_name=model_name,
        scenario=scenario,
        samples=samples,
        accuracy=accuracy,
        top_5_accuracy=top_5_accuracy,
        mean_confidence=mean_confidence,
        mean_loss=mean_loss,
        correct_mask=correct_mask,
        total_samples=n_samples,
        metadata=metadata
    )