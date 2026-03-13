"""
Detailed per-sample evaluation and tracking.

This module provides fine-grained analysis of model predictions,
tracking individual sample results for deeper insights.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class SampleResult:
    """Result for a single sample evaluation."""

    index: int
    label: int
    prediction: int
    correct: bool
    confidence: float
    logits: Optional[np.ndarray] = None
    top_k_predictions: Optional[List[int]] = None
    top_k_confidences: Optional[List[float]] = None


@dataclass
class DetailedResults:
    """Container for detailed evaluation results."""

    model_name: str
    scenario: str
    samples: List[SampleResult]
    accuracy: float
    correct_mask: np.ndarray
    total_samples: int
    metadata: Optional[Dict[str, Any]] = None

    def get_failures(self) -> List[SampleResult]:
        """Get all failed predictions."""
        return [s for s in self.samples if not s.correct]

    def get_successes(self) -> List[SampleResult]:
        """Get all successful predictions."""
        return [s for s in self.samples if s.correct]

    def get_high_confidence_errors(self, threshold: float = 0.8) -> List[SampleResult]:
        """Get errors where model was highly confident but wrong."""
        return [s for s in self.samples if not s.correct and s.confidence > threshold]

    def get_low_confidence_correct(self, threshold: float = 0.5) -> List[SampleResult]:
        """Get correct predictions with low confidence."""
        return [s for s in self.samples if s.correct and s.confidence < threshold]

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        failures = self.get_failures()
        high_conf_errors = self.get_high_confidence_errors()

        return {
            'model': self.model_name,
            'scenario': self.scenario,
            'total_samples': self.total_samples,
            'accuracy': self.accuracy,
            'num_failures': len(failures),
            'num_high_confidence_errors': len(high_conf_errors),
            'mean_confidence': np.mean([s.confidence for s in self.samples]),
            'mean_confidence_correct': np.mean([s.confidence for s in self.get_successes()]) if self.get_successes() else 0,
            'mean_confidence_incorrect': np.mean([s.confidence for s in failures]) if failures else 0,
        }


def evaluate_detailed(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    model_name: str = "Model",
    scenario: str = "default",
    device: Optional[str] = None,
    batch_size: int = 32,
    top_k: int = 5
) -> DetailedResults:
    """
    Perform detailed evaluation with per-sample tracking.

    Args:
        model: PyTorch model to evaluate
        images: Input images tensor
        labels: Ground truth labels
        model_name: Name identifier for the model
        scenario: Scenario description (e.g., "clean", "noisy")
        device: Device to run on (auto-detected if None)
        batch_size: Batch size for evaluation
        top_k: Number of top predictions to track

    Returns:
        DetailedResults containing per-sample results and statistics

    Example:
        >>> results = evaluate_detailed(resnet50, test_images, test_labels)
        >>> print(f"Accuracy: {results.accuracy:.1%}")
        >>> for failure in results.get_failures()[:5]:
        ...     print(f"Failed on sample {failure.index}")
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    samples = []
    correct_list = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            # Get model outputs
            outputs = model(batch_images)
            probs = F.softmax(outputs, dim=1)

            # Get predictions
            confidences, predictions = probs.max(dim=1)

            # Get top-k predictions
            top_k_probs, top_k_preds = probs.topk(min(top_k, probs.shape[1]), dim=1)

            # Process each sample
            for j in range(len(batch_images)):
                sample_idx = i + j
                correct = predictions[j].item() == batch_labels[j].item()

                sample = SampleResult(
                    index=sample_idx,
                    label=batch_labels[j].item(),
                    prediction=predictions[j].item(),
                    correct=correct,
                    confidence=confidences[j].item(),
                    logits=outputs[j].cpu().numpy(),
                    top_k_predictions=top_k_preds[j].cpu().tolist(),
                    top_k_confidences=top_k_probs[j].cpu().tolist(),
                )

                samples.append(sample)
                correct_list.append(correct)

    correct_mask = np.array(correct_list)
    accuracy = correct_mask.mean()

    return DetailedResults(
        model_name=model_name,
        scenario=scenario,
        samples=samples,
        accuracy=accuracy,
        correct_mask=correct_mask,
        total_samples=len(samples),
        metadata={
            'device': str(device),
            'batch_size': batch_size,
            'top_k': top_k,
        }
    )


def get_failures(results: DetailedResults, max_samples: Optional[int] = None) -> List[SampleResult]:
    """
    Get failed predictions from results.

    Args:
        results: Detailed evaluation results
        max_samples: Maximum number of failures to return

    Returns:
        List of failed sample results
    """
    failures = results.get_failures()
    if max_samples is not None:
        failures = failures[:max_samples]
    return failures


def get_successes(results: DetailedResults, max_samples: Optional[int] = None) -> List[SampleResult]:
    """
    Get successful predictions from results.

    Args:
        results: Detailed evaluation results
        max_samples: Maximum number of successes to return

    Returns:
        List of successful sample results
    """
    successes = results.get_successes()
    if max_samples is not None:
        successes = successes[:max_samples]
    return successes