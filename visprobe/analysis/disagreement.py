"""
Disagreement analysis for comparing model predictions.

This module analyzes where and how different models disagree,
useful for ensemble construction and understanding model diversity.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from .detailed_evaluation import DetailedResults, SampleResult


@dataclass
class DisagreementAnalysis:
    """Analysis of disagreement between two models."""

    model_a_name: str
    model_b_name: str
    total_samples: int
    both_correct: int
    both_wrong: int
    only_a_correct: int
    only_b_correct: int
    agreement_rate: float
    disagreement_rate: float
    complementarity_score: float
    disagreement_indices: List[int]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Disagreement Analysis: {self.model_a_name} vs {self.model_b_name}",
            f"Total samples: {self.total_samples}",
            f"Agreement rate: {self.agreement_rate:.1%}",
            f"",
            f"Both correct: {self.both_correct} ({self.both_correct/self.total_samples:.1%})",
            f"Both wrong: {self.both_wrong} ({self.both_wrong/self.total_samples:.1%})",
            f"Only {self.model_a_name} correct: {self.only_a_correct} ({self.only_a_correct/self.total_samples:.1%})",
            f"Only {self.model_b_name} correct: {self.only_b_correct} ({self.only_b_correct/self.total_samples:.1%})",
            f"",
            f"Complementarity score: {self.complementarity_score:.3f}",
        ]
        return "\n".join(lines)


def disagreement_analysis(
    results_a: DetailedResults,
    results_b: DetailedResults
) -> DisagreementAnalysis:
    """
    Analyze disagreement between two models.

    Args:
        results_a: Detailed results for model A
        results_b: Detailed results for model B

    Returns:
        DisagreementAnalysis with comprehensive statistics

    Example:
        >>> analysis = disagreement_analysis(vanilla_results, robust_results)
        >>> print(analysis.summary())
        >>> if analysis.complementarity_score > 0.5:
        ...     print("Models are complementary - good for ensemble!")
    """
    if results_a.total_samples != results_b.total_samples:
        raise ValueError("Results must have same number of samples")

    # Get correct masks
    correct_a = results_a.correct_mask
    correct_b = results_b.correct_mask

    # Calculate contingency table
    both_correct = np.sum(correct_a & correct_b)
    both_wrong = np.sum((~correct_a) & (~correct_b))
    only_a_correct = np.sum(correct_a & (~correct_b))
    only_b_correct = np.sum((~correct_a) & correct_b)

    total = len(correct_a)

    # Agreement metrics
    agreement_rate = (both_correct + both_wrong) / total
    disagreement_rate = 1 - agreement_rate

    # Complementarity score
    # High when models fail on different samples
    if only_a_correct + only_b_correct > 0:
        complementarity = min(only_a_correct, only_b_correct) / max(only_a_correct, only_b_correct)
    else:
        complementarity = 0.0

    # Find disagreement indices
    disagreement_indices = []
    for i in range(total):
        if correct_a[i] != correct_b[i]:
            disagreement_indices.append(i)

    return DisagreementAnalysis(
        model_a_name=results_a.model_name,
        model_b_name=results_b.model_name,
        total_samples=total,
        both_correct=int(both_correct),
        both_wrong=int(both_wrong),
        only_a_correct=int(only_a_correct),
        only_b_correct=int(only_b_correct),
        agreement_rate=float(agreement_rate),
        disagreement_rate=float(disagreement_rate),
        complementarity_score=float(complementarity),
        disagreement_indices=disagreement_indices
    )


def compute_complementarity_score(
    correct_masks: List[np.ndarray],
    method: str = "diversity"
) -> float:
    """
    Compute complementarity score for multiple models.

    Args:
        correct_masks: List of boolean arrays for each model
        method: Scoring method ('diversity' or 'coverage')

    Returns:
        Complementarity score (0-1, higher is better)

    Example:
        >>> masks = [model1.correct_mask, model2.correct_mask, model3.correct_mask]
        >>> score = compute_complementarity_score(masks)
        >>> print(f"Ensemble complementarity: {score:.2f}")
    """
    n_models = len(correct_masks)
    n_samples = len(correct_masks[0])

    if method == "diversity":
        # Diversity: models disagree on failures
        diversity_scores = []

        for i in range(n_samples):
            correct_count = sum(mask[i] for mask in correct_masks)
            # High score when some but not all models are correct
            if 0 < correct_count < n_models:
                diversity_scores.append(1.0 - abs(correct_count - n_models/2) / (n_models/2))
            else:
                diversity_scores.append(0.0)

        return float(np.mean(diversity_scores))

    elif method == "coverage":
        # Coverage: at least one model is correct
        coverage = np.zeros(n_samples, dtype=bool)
        for mask in correct_masks:
            coverage |= mask
        return float(coverage.mean())

    else:
        raise ValueError(f"Unknown method: {method}")


def pairwise_disagreement_matrix(
    results_list: List[DetailedResults]
) -> np.ndarray:
    """
    Compute pairwise disagreement rates between multiple models.

    Args:
        results_list: List of DetailedResults for each model

    Returns:
        Square matrix of disagreement rates

    Example:
        >>> matrix = pairwise_disagreement_matrix([results1, results2, results3])
        >>> print("Model 1 vs 2 disagreement:", matrix[0, 1])
    """
    n_models = len(results_list)
    matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                analysis = disagreement_analysis(results_list[i], results_list[j])
                matrix[i, j] = analysis.disagreement_rate

    return matrix


def find_diverse_subset(
    results_list: List[DetailedResults],
    subset_size: int,
    method: str = "greedy"
) -> List[int]:
    """
    Find the most diverse subset of models for ensemble.

    Args:
        results_list: List of DetailedResults for each model
        subset_size: Desired size of subset
        method: Selection method ('greedy' or 'exhaustive')

    Returns:
        Indices of selected models

    Example:
        >>> indices = find_diverse_subset(all_results, subset_size=3)
        >>> ensemble = [models[i] for i in indices]
    """
    n_models = len(results_list)

    if subset_size >= n_models:
        return list(range(n_models))

    if method == "greedy":
        # Greedy selection based on disagreement
        selected = []
        remaining = list(range(n_models))

        # Start with best individual model
        accuracies = [r.accuracy for r in results_list]
        best_idx = np.argmax(accuracies)
        selected.append(best_idx)
        remaining.remove(best_idx)

        # Add models that disagree most with selected set
        while len(selected) < subset_size:
            max_disagreement = -1
            best_candidate = None

            for idx in remaining:
                # Calculate average disagreement with selected models
                disagreements = []
                for sel_idx in selected:
                    analysis = disagreement_analysis(
                        results_list[idx],
                        results_list[sel_idx]
                    )
                    disagreements.append(analysis.disagreement_rate)

                avg_disagreement = np.mean(disagreements)

                if avg_disagreement > max_disagreement:
                    max_disagreement = avg_disagreement
                    best_candidate = idx

            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected

    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_failure_patterns(
    results_list: List[DetailedResults],
    min_models_failed: int = 2
) -> Dict[str, List[int]]:
    """
    Identify systematic failure patterns across models.

    Args:
        results_list: List of DetailedResults for each model
        min_models_failed: Minimum models that must fail for pattern

    Returns:
        Dictionary mapping failure patterns to sample indices

    Example:
        >>> patterns = analyze_failure_patterns([results1, results2, results3])
        >>> print(f"Samples where all models fail: {len(patterns['all_failed'])}")
    """
    n_models = len(results_list)
    n_samples = results_list[0].total_samples

    patterns = {
        'all_failed': [],
        'majority_failed': [],
        'all_succeeded': [],
    }

    # Also track specific combinations
    for i in range(n_samples):
        failed_models = []
        for j, results in enumerate(results_list):
            if not results.correct_mask[i]:
                failed_models.append(j)

        n_failed = len(failed_models)

        if n_failed == n_models:
            patterns['all_failed'].append(i)
        elif n_failed == 0:
            patterns['all_succeeded'].append(i)
        elif n_failed > n_models // 2:
            patterns['majority_failed'].append(i)

        # Track specific failure combinations
        if n_failed >= min_models_failed:
            pattern_key = f"models_{','.join(map(str, failed_models))}_failed"
            if pattern_key not in patterns:
                patterns[pattern_key] = []
            patterns[pattern_key].append(i)

    return patterns