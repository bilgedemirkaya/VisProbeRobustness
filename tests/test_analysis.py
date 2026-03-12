"""
Tests for the analysis module.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import List

from visprobe.analysis import (
    # Detailed evaluation
    SampleResult,
    DetailedResults,
    evaluate_detailed,

    # Statistical
    bootstrap_accuracy,
    bootstrap_delta,

    # Crossover
    CrossoverPoint,
    find_crossover,

    # Disagreement
    DisagreementAnalysis,
    disagreement_analysis,

    # Confidence
    ConfidenceProfile,
    confidence_profile,
    calibration_error,

    # Vulnerability
    ClassVulnerability,
    class_vulnerability,
    systematic_failures,
)


class DummyModel(nn.Module):
    """Dummy model for testing."""

    def __init__(self, num_classes: int = 10, always_correct: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.always_correct = always_correct
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.always_correct:
            # Return high confidence for correct class
            logits = torch.randn(batch_size, self.num_classes) - 5
            # Assume labels are sequential for testing
            for i in range(batch_size):
                logits[i, i % self.num_classes] = 5.0
        else:
            # Return random logits
            logits = torch.randn(batch_size, self.num_classes)
        return logits


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    return DummyModel(num_classes=10)


@pytest.fixture
def sample_data():
    """Create sample test data."""
    images = torch.randn(20, 10)  # 20 samples, 10 features
    labels = torch.arange(20) % 10  # Cycling through 10 classes
    return images, labels


@pytest.fixture
def sample_results() -> List[SampleResult]:
    """Create sample results for testing."""
    np.random.seed(42)
    results = []
    for i in range(100):
        correct = np.random.random() > 0.3  # 70% accuracy
        results.append(SampleResult(
            index=i,
            label=i % 10,
            prediction=(i % 10) if correct else ((i + 1) % 10),
            correct=correct,
            confidence=np.random.uniform(0.5, 1.0) if correct else np.random.uniform(0.1, 0.7),
            logits=np.random.randn(10),
            top_k_predictions=list(range(5)),
            top_k_confidences=list(np.random.random(5))
        ))
    return results


class TestDetailedEvaluation:
    """Tests for detailed evaluation functions."""

    def test_evaluate_detailed(self, dummy_model, sample_data):
        """Test detailed evaluation."""
        images, labels = sample_data
        results = evaluate_detailed(
            dummy_model,
            images,
            labels,
            model_name="TestModel",
            scenario="test"
        )

        assert isinstance(results, DetailedResults)
        assert results.model_name == "TestModel"
        assert results.scenario == "test"
        assert len(results.samples) == len(images)
        assert results.total_samples == len(images)
        assert 0 <= results.accuracy <= 1

    def test_sample_result_creation(self):
        """Test SampleResult dataclass."""
        sample = SampleResult(
            index=0,
            label=5,
            prediction=3,
            correct=False,
            confidence=0.75
        )

        assert sample.index == 0
        assert sample.label == 5
        assert sample.prediction == 3
        assert not sample.correct
        assert sample.confidence == 0.75

    def test_get_failures_and_successes(self, sample_results):
        """Test filtering failures and successes."""
        results = DetailedResults(
            model_name="Test",
            scenario="test",
            samples=sample_results,
            accuracy=0.7,
            correct_mask=np.array([s.correct for s in sample_results]),
            total_samples=len(sample_results)
        )

        failures = results.get_failures()
        successes = results.get_successes()

        assert len(failures) + len(successes) == len(sample_results)
        assert all(not s.correct for s in failures)
        assert all(s.correct for s in successes)

    def test_high_confidence_errors(self, sample_results):
        """Test finding high confidence errors."""
        results = DetailedResults(
            model_name="Test",
            scenario="test",
            samples=sample_results,
            accuracy=0.7,
            correct_mask=np.array([s.correct for s in sample_results]),
            total_samples=len(sample_results)
        )

        high_conf_errors = results.get_high_confidence_errors(threshold=0.6)
        assert all(not s.correct for s in high_conf_errors)
        assert all(s.confidence > 0.6 for s in high_conf_errors)


class TestStatistical:
    """Tests for statistical analysis functions."""

    def test_bootstrap_accuracy(self):
        """Test bootstrap confidence interval for accuracy."""
        np.random.seed(42)
        correct_mask = np.random.random(100) > 0.3  # ~70% accuracy

        mean_acc, lower, upper = bootstrap_accuracy(
            correct_mask,
            n_bootstrap=1000,
            confidence_level=0.95,
            random_state=42
        )

        assert 0 <= lower <= mean_acc <= upper <= 1
        assert upper - lower < 0.2  # Reasonable CI width

    def test_bootstrap_delta(self):
        """Test bootstrap confidence interval for model comparison."""
        np.random.seed(42)
        correct_a = np.random.random(100) > 0.3  # ~70% accuracy
        correct_b = np.random.random(100) > 0.4  # ~60% accuracy

        delta, lower, upper = bootstrap_delta(
            correct_a,
            correct_b,
            n_bootstrap=1000,
            paired=True,
            random_state=42
        )

        assert lower <= delta <= upper
        # Model A should be better
        assert delta > 0

    def test_bootstrap_delta_unpaired(self):
        """Test unpaired bootstrap comparison."""
        np.random.seed(42)
        correct_a = np.random.random(50) > 0.3
        correct_b = np.random.random(75) > 0.4

        delta, lower, upper = bootstrap_delta(
            correct_a,
            correct_b,
            n_bootstrap=1000,
            paired=False,
            random_state=42
        )

        assert lower <= delta <= upper


class TestCrossover:
    """Tests for crossover detection."""

    def test_find_crossover(self):
        """Test finding crossover point."""
        severities = np.linspace(0, 1, 11)
        perf_a = 1.0 - severities  # Decreasing
        perf_b = 0.5 - 0.3 * severities  # Slower decrease

        crossover = find_crossover(
            severities,
            perf_a,
            perf_b,
            model_a_name="ModelA",
            model_b_name="ModelB"
        )

        assert isinstance(crossover, CrossoverPoint)
        assert 0 <= crossover.severity <= 1
        assert crossover.model_a_name == "ModelA"
        assert crossover.model_b_name == "ModelB"

    def test_no_crossover(self):
        """Test when no crossover exists."""
        severities = np.linspace(0, 1, 11)
        perf_a = 1.0 - 0.3 * severities  # Always better
        perf_b = 0.5 - 0.3 * severities  # Parallel, always worse

        crossover = find_crossover(severities, perf_a, perf_b)
        assert crossover is None


class TestDisagreement:
    """Tests for disagreement analysis."""

    def test_disagreement_analysis(self, sample_results):
        """Test disagreement analysis between models."""
        # Create two sets of results with some disagreement
        results_a = DetailedResults(
            model_name="ModelA",
            scenario="test",
            samples=sample_results,
            accuracy=0.7,
            correct_mask=np.array([s.correct for s in sample_results]),
            total_samples=len(sample_results)
        )

        # Flip some predictions for model B
        samples_b = sample_results.copy()
        for i in range(0, len(samples_b), 3):
            samples_b[i] = SampleResult(
                index=samples_b[i].index,
                label=samples_b[i].label,
                prediction=samples_b[i].prediction,
                correct=not samples_b[i].correct,
                confidence=samples_b[i].confidence
            )

        results_b = DetailedResults(
            model_name="ModelB",
            scenario="test",
            samples=samples_b,
            accuracy=0.6,
            correct_mask=np.array([s.correct for s in samples_b]),
            total_samples=len(samples_b)
        )

        analysis = disagreement_analysis(results_a, results_b)

        assert isinstance(analysis, DisagreementAnalysis)
        assert analysis.model_a_name == "ModelA"
        assert analysis.model_b_name == "ModelB"
        assert 0 <= analysis.agreement_rate <= 1
        assert analysis.disagreement_rate == 1 - analysis.agreement_rate
        assert analysis.both_correct + analysis.both_wrong + analysis.only_a_correct + analysis.only_b_correct == len(sample_results)


class TestConfidence:
    """Tests for confidence analysis."""

    def test_confidence_profile(self, sample_results):
        """Test confidence profile generation."""
        profile = confidence_profile(sample_results)

        assert isinstance(profile, ConfidenceProfile)
        assert 0 <= profile.mean_confidence <= 1
        assert 0 <= profile.mean_confidence_correct <= 1
        assert 0 <= profile.mean_confidence_incorrect <= 1
        assert 0 <= profile.pct_high_confidence_errors <= 100
        assert 0 <= profile.calibration_error <= 1

    def test_calibration_error(self, sample_results):
        """Test calibration error calculation."""
        ece = calibration_error(sample_results, n_bins=10, method="ECE")
        mce = calibration_error(sample_results, n_bins=10, method="MCE")

        assert 0 <= ece <= 1
        assert 0 <= mce <= 1
        assert mce >= ece  # MCE should be at least as large as ECE


class TestVulnerability:
    """Tests for vulnerability analysis."""

    def test_class_vulnerability(self, sample_results):
        """Test class vulnerability analysis."""
        # Create perturbed results with lower accuracy
        perturbed = []
        for s in sample_results:
            # Make some correct predictions incorrect
            if s.correct and np.random.random() > 0.5:
                perturbed.append(SampleResult(
                    index=s.index,
                    label=s.label,
                    prediction=(s.prediction + 1) % 10,
                    correct=False,
                    confidence=s.confidence * 0.8
                ))
            else:
                perturbed.append(s)

        vulnerabilities = class_vulnerability(
            sample_results,
            perturbed,
            top_k=5
        )

        assert len(vulnerabilities) <= 5
        for vuln in vulnerabilities:
            assert isinstance(vuln, ClassVulnerability)
            assert vuln.clean_accuracy >= vuln.perturbed_accuracy
            assert vuln.accuracy_drop >= 0

    def test_systematic_failures(self, sample_results):
        """Test systematic failure detection."""
        # Create multiple scenarios
        scenarios = [sample_results]

        # Add two more scenarios with some overlapping failures
        for _ in range(2):
            scenario = []
            for s in sample_results:
                # Keep some failures consistent
                if not s.correct:
                    scenario.append(s)
                else:
                    # Randomly flip some successes
                    if np.random.random() > 0.7:
                        scenario.append(SampleResult(
                            index=s.index,
                            label=s.label,
                            prediction=(s.prediction + 1) % 10,
                            correct=False,
                            confidence=s.confidence * 0.8
                        ))
                    else:
                        scenario.append(s)
            scenarios.append(scenario)

        failures = systematic_failures(
            scenarios,
            min_scenarios_failed=2,
            scenario_names=["Clean", "Noise", "Blur"]
        )

        assert isinstance(failures, list)
        for failure in failures:
            assert failure['n_failures'] >= 2
            assert 0 <= failure['failure_rate'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])