"""
Tests for the visprobe.properties module.
"""

import pytest
import torch

from visprobe.properties import (
    CompositeProperty,
    ConfidenceDrop,
    L2Distance,
    LabelConstant,
    TopKStability,
    get_top_prediction,
    get_topk_predictions,
)
from visprobe.properties.helpers import extract_logits


class TestLabelConstant:
    """Tests for LabelConstant property."""

    def test_same_label_passes(self, sample_logits):
        """Test that identical logits pass."""
        prop = LabelConstant()
        result = prop(sample_logits, sample_logits)
        assert result.all()

    def test_different_label_fails(self, sample_logits):
        """Test that different labels fail."""
        prop = LabelConstant()
        perturbed = sample_logits.clone()
        # Flip the top prediction for first sample
        perturbed[0, 0] = 100.0
        perturbed[0, 1:] = -100.0
        result = prop(sample_logits, perturbed)
        assert not result[0]  # First sample should fail

    def test_tuple_input(self, sample_logits):
        """Test that tuple (logits, features) input works."""
        prop = LabelConstant()
        original = (sample_logits, torch.randn(4, 128))
        perturbed = (sample_logits, torch.randn(4, 128))
        result = prop(original, perturbed)
        assert result.all()


class TestTopKStability:
    """Tests for TopKStability property."""

    def test_overlap_mode(self, sample_logits):
        """Test overlap mode."""
        prop = TopKStability(k=5, mode="overlap", min_overlap=3)
        result = prop(sample_logits, sample_logits)
        assert result.all()

    def test_containment_mode(self, sample_logits):
        """Test containment mode."""
        prop = TopKStability(k=5, mode="containment")
        result = prop(sample_logits, sample_logits)
        assert result.all()

    def test_jaccard_mode(self, sample_logits):
        """Test jaccard mode."""
        prop = TopKStability(k=5, mode="jaccard", min_jaccard=0.4)
        result = prop(sample_logits, sample_logits)
        assert result.all()

    def test_invalid_k_raises(self):
        """Test that k < 1 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            TopKStability(k=0)

    def test_invalid_overlap_raises(self):
        """Test that invalid min_overlap raises ValueError."""
        with pytest.raises(ValueError, match="min_overlap must be between 1 and k"):
            TopKStability(k=5, mode="overlap", min_overlap=10)

    def test_invalid_jaccard_raises(self):
        """Test that invalid min_jaccard raises ValueError."""
        with pytest.raises(ValueError, match="min_jaccard must be in"):
            TopKStability(k=5, mode="jaccard", min_jaccard=1.5)


class TestConfidenceDrop:
    """Tests for ConfidenceDrop property."""

    def test_no_drop_passes(self, sample_logits):
        """Test that identical logits pass."""
        prop = ConfidenceDrop(max_drop=0.3)
        result = prop(sample_logits, sample_logits)
        assert result.all()

    def test_small_drop_passes(self, sample_logits):
        """Test that small confidence drops pass."""
        prop = ConfidenceDrop(max_drop=0.3)
        perturbed = sample_logits * 0.95  # Slight reduction
        result = prop(sample_logits, perturbed)
        assert result.all()

    def test_large_drop_fails(self, sample_logits):
        """Test that large confidence drops fail."""
        prop = ConfidenceDrop(max_drop=0.1)
        perturbed = sample_logits * 0.1  # Major reduction
        result = prop(sample_logits, perturbed)
        assert not result.all()  # At least one should fail

    def test_invalid_max_drop_raises(self):
        """Test that invalid max_drop raises ValueError."""
        with pytest.raises(ValueError, match="max_drop must be between 0.0 and 1.0"):
            ConfidenceDrop(max_drop=1.5)


class TestL2Distance:
    """Tests for L2Distance property."""

    def test_zero_distance_passes(self, sample_logits):
        """Test that identical logits pass."""
        prop = L2Distance(max_delta=1.0)
        result = prop(sample_logits, sample_logits)
        assert result.all()

    def test_small_distance_passes(self, sample_logits):
        """Test that small perturbations pass."""
        prop = L2Distance(max_delta=10.0)
        perturbed = sample_logits + torch.randn_like(sample_logits) * 0.01
        result = prop(sample_logits, perturbed)
        assert result.all()

    def test_large_distance_fails(self, sample_logits):
        """Test that large perturbations fail."""
        prop = L2Distance(max_delta=0.1)
        perturbed = sample_logits + torch.randn_like(sample_logits) * 100.0
        result = prop(sample_logits, perturbed)
        assert not result.any()  # All should fail


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_top_prediction(self, sample_logits):
        """Test get_top_prediction function."""
        idx, conf = get_top_prediction(sample_logits)
        assert isinstance(idx, int)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_get_topk_predictions(self, sample_logits):
        """Test get_topk_predictions function."""
        indices = get_topk_predictions(sample_logits, k=5)
        assert isinstance(indices, torch.Tensor)
        assert indices.shape[-1] == 5

    def test_get_topk_invalid_k_raises(self, sample_logits):
        """Test that k < 1 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            get_topk_predictions(sample_logits, k=0)

    def test_get_topk_empty_tensor_raises(self):
        """Test that empty tensor raises ValueError."""
        empty = torch.tensor([])
        with pytest.raises(ValueError, match="Cannot get top-k predictions from empty tensor"):
            get_topk_predictions(empty, k=5)

    def test_extract_logits_from_tensor(self, sample_logits):
        """Test extracting logits from raw tensor."""
        result = extract_logits(sample_logits)
        assert torch.equal(result, sample_logits)

    def test_extract_logits_from_tuple(self, sample_logits):
        """Test extracting logits from tuple."""
        features = torch.randn(4, 128)
        result = extract_logits((sample_logits, features))
        assert torch.equal(result, sample_logits)

    def test_extract_logits_from_dict(self, sample_logits):
        """Test extracting logits from dict."""
        result = extract_logits({"output": sample_logits})
        assert torch.equal(result, sample_logits)

    def test_extract_logits_invalid_dict_raises(self):
        """Test that dict without 'output' key raises TypeError."""
        with pytest.raises(TypeError, match="Dict must contain 'output' key"):
            extract_logits({"wrong_key": torch.randn(4, 10)})

    def test_extract_logits_invalid_type_raises(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            extract_logits([1, 2, 3])


class TestBatchedProperties:
    """Tests for batched property evaluation."""

    @pytest.fixture
    def batch_logits(self):
        """Batched logits [batch_size=8, num_classes=10]."""
        return torch.randn(8, 10)

    @pytest.fixture
    def single_logits(self):
        """Single sample logits [num_classes=10]."""
        return torch.randn(10)

    def test_label_constant_batched_returns_tensor(self, batch_logits):
        """LabelConstant should return tensor for batched input."""
        prop = LabelConstant()
        result = prop(batch_logits, batch_logits)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (8,)
        assert result.all()  # All should pass (same input)

    def test_label_constant_batched_per_sample(self, batch_logits):
        """LabelConstant should return per-sample results."""
        prop = LabelConstant()
        perturbed = batch_logits.clone()
        # Flip prediction for samples 0, 2, 4
        for i in [0, 2, 4]:
            perturbed[i] = -perturbed[i]

        result = prop(batch_logits, perturbed)
        assert isinstance(result, torch.Tensor)
        # Samples 0, 2, 4 should fail; 1, 3, 5, 6, 7 should pass
        assert not result[0] and not result[2] and not result[4]
        assert result[1] and result[3] and result[5]

    def test_label_constant_single_returns_bool(self, single_logits):
        """LabelConstant should return bool for single input."""
        prop = LabelConstant()
        result = prop(single_logits, single_logits)
        assert isinstance(result, bool)
        assert result is True

    def test_topk_stability_batched_returns_tensor(self, batch_logits):
        """TopKStability should return tensor for batched input."""
        prop = TopKStability(k=5, mode="overlap", min_overlap=3)
        result = prop(batch_logits, batch_logits)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (8,)
        assert result.all()

    def test_confidence_drop_batched_returns_tensor(self, batch_logits):
        """ConfidenceDrop should return tensor for batched input."""
        prop = ConfidenceDrop(max_drop=0.3)
        result = prop(batch_logits, batch_logits)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (8,)
        assert result.all()

    def test_l2_distance_batched_returns_tensor(self, batch_logits):
        """L2Distance should return tensor for batched input."""
        prop = L2Distance(max_delta=1.0)
        result = prop(batch_logits, batch_logits)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (8,)
        assert result.all()

    def test_l2_distance_batched_per_sample(self, batch_logits):
        """L2Distance should compute per-sample distances."""
        prop = L2Distance(max_delta=5.0)
        perturbed = batch_logits.clone()
        # Add large perturbation to sample 0 only
        perturbed[0] += 100.0

        result = prop(batch_logits, perturbed)
        assert not result[0]  # Sample 0 should fail
        assert result[1:].all()  # Others should pass


class TestCompositePropertyBatched:
    """Tests for CompositeProperty with batched inputs."""

    @pytest.fixture
    def batch_logits(self):
        """Batched logits [batch_size=4, num_classes=10]."""
        return torch.randn(4, 10)

    def test_composite_and_batched(self, batch_logits):
        """CompositeProperty AND mode with batched input."""
        prop = LabelConstant() & ConfidenceDrop(max_drop=0.5)
        result = prop(batch_logits, batch_logits)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4,)
        assert result.all()

    def test_composite_or_batched(self, batch_logits):
        """CompositeProperty OR mode with batched input."""
        prop = LabelConstant() | L2Distance(max_delta=0.001)
        result = prop(batch_logits, batch_logits)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4,)
        assert result.all()  # LabelConstant passes even if L2 fails

    def test_composite_mixed_results(self, batch_logits):
        """CompositeProperty with mixed per-sample results."""
        perturbed = batch_logits.clone()
        # Flip label for sample 0
        perturbed[0] = -perturbed[0]

        # AND: both must pass
        prop_and = LabelConstant() & ConfidenceDrop(max_drop=0.9)
        result = prop_and(batch_logits, perturbed)
        assert not result[0]  # Label changed, so AND fails
        assert result[1:].all()  # Others pass

        # OR: at least one must pass
        prop_or = LabelConstant() | ConfidenceDrop(max_drop=0.9)
        result = prop_or(batch_logits, perturbed)
        assert result.all()  # ConfidenceDrop passes for all
