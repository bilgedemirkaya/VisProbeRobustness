from __future__ import annotations

"""Thread-safe registries for VisProbe test functions.
"""

import threading
from typing import Callable, List


class TestRegistry:
    """Thread-safe test registry using thread-local storage."""

    _local = threading.local()

    @classmethod
    def get_given_tests(cls) -> List[Callable]:
        """Get the list of registered @given tests for the current thread."""
        if not hasattr(cls._local, "given_tests"):
            cls._local.given_tests = []
        return cls._local.given_tests

    @classmethod
    def get_search_tests(cls) -> List[Callable]:
        """Get the list of registered @search tests for the current thread."""
        if not hasattr(cls._local, "search_tests"):
            cls._local.search_tests = []
        return cls._local.search_tests

    @classmethod
    def register_given(cls, test: Callable) -> None:
        """Register a @given test."""
        cls.get_given_tests().append(test)

    @classmethod
    def register_search(cls, test: Callable) -> None:
        """Register a @search test."""
        cls.get_search_tests().append(test)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tests in the current thread."""
        if hasattr(cls._local, "given_tests"):
            cls._local.given_tests.clear()
        if hasattr(cls._local, "search_tests"):
            cls._local.search_tests.clear()


# Backward compatibility: expose as module-level attributes
# These are now properties that access thread-local storage
GIVEN_TESTS = TestRegistry.get_given_tests()
SEARCH_TESTS = TestRegistry.get_search_tests()
