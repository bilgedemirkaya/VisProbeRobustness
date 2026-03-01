"""
This module defines the base class and result type for all properties in VisProbe.
"""

from __future__ import annotations

from typing import Any, List, Literal, Union

import torch


class Property:
    """
    Base class for all properties in VisProbe.

    A property is a function or class that asserts a specific behavior
    of a model, typically by comparing its output on an original and a
    perturbed input.
    """

    def __call__(self, original: Any, perturbed: Any) -> bool:
        """
        Executes the property check.

        Args:
            original: The output of the model on the original input.
            perturbed: The output of the model on the perturbed input.

        Returns:
            True if the property holds, False otherwise.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__

    def __and__(self, other: Property) -> CompositeProperty:
        """
        Combine properties with AND logic using & operator.

        Example:
            combined = LabelConstant() & ConfidenceDrop(0.2)
        """
        if isinstance(self, CompositeProperty) and self.mode == "and":
            return CompositeProperty([*self.properties, other], mode="and")
        return CompositeProperty([self, other], mode="and")

    def __or__(self, other: Property) -> CompositeProperty:
        """
        Combine properties with OR logic using | operator.

        Example:
            combined = TopKStability(k=3) | TopKStability(k=5)
        """
        if isinstance(self, CompositeProperty) and self.mode == "or":
            return CompositeProperty([*self.properties, other], mode="or")
        return CompositeProperty([self, other], mode="or")


class CompositeProperty(Property):
    """
    Combines multiple properties with AND/OR logic.

    Supports short-circuit evaluation for performance optimization:
    - AND mode: stops on first False
    - OR mode: stops on first True

    Examples:
        # Both must pass (strict robustness)
        CompositeProperty([LabelConstant(), ConfidenceDrop(0.2)], mode="and")

        # At least one must pass (relaxed robustness)
        CompositeProperty([TopKStability(k=3), TopKStability(k=5)], mode="or")

        # Pythonic syntax
        prop1 & prop2  # AND
        prop1 | prop2  # OR
    """

    def __init__(
        self,
        properties: List[Property],
        mode: Literal["and", "or"] = "and",
    ):
        if not properties:
            raise ValueError("properties list cannot be empty")
        if mode not in ("and", "or"):
            raise ValueError("mode must be 'and' or 'or'")

        self.properties = properties
        self.mode = mode

    def __call__(self, original: Any, perturbed: Any) -> Union[bool, torch.Tensor]:
        """
        Evaluate all properties with short-circuit logic.

        Supports both single-sample and batched inputs. For batched inputs,
        returns a tensor of per-sample boolean results.

        Args:
            original: The output of the model on the original input.
            perturbed: The output of the model on the perturbed input.

        Returns:
            For AND mode: True/tensor where all properties pass
            For OR mode: True/tensor where at least one property passes
        """
        results = [prop(original, perturbed) for prop in self.properties]

        # Check if we're dealing with batched results (tensors)
        if any(isinstance(r, torch.Tensor) for r in results):
            # Convert all to tensors for consistent handling
            tensor_results = []
            for r in results:
                if isinstance(r, torch.Tensor):
                    tensor_results.append(r)
                else:
                    # Broadcast scalar to match batch size
                    ref = next(t for t in results if isinstance(t, torch.Tensor))
                    tensor_results.append(
                        torch.full_like(ref, r, dtype=torch.bool)
                    )

            # Stack and reduce
            stacked = torch.stack(tensor_results)  # [n_props, batch_size]
            if self.mode == "and":
                return stacked.all(dim=0)  # All props must pass per sample
            else:
                return stacked.any(dim=0)  # Any prop passes per sample

        # Single sample case: original short-circuit logic
        if self.mode == "and":
            return all(results)
        else:
            return any(results)

    def __str__(self) -> str:
        props_str = ", ".join(str(p) for p in self.properties)
        return f"CompositeProperty([{props_str}], mode='{self.mode}')"
