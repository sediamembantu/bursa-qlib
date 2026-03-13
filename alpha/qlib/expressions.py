"""Feature definitions for the repository's qlib-style workflow.

This module intentionally does not depend on `pyqlib`. The project currently
uses a lightweight local workflow that borrows qlib concepts while running
entirely from source.
"""

from alpha.factors.combiner import get_factor_columns

ALPHA158_SUBSET = [
    "close / Ref(close, 1) - 1",
    "Mean(close, 5) / close",
    "Mean(close, 10) / close",
    "Std(close / Ref(close, 1) - 1, 20)",
    "volume / Mean(volume, 20)",
]

MALAYSIA_FACTOR_COLUMNS = get_factor_columns()


def describe_feature_sets() -> dict[str, list[str]]:
    """Return the feature groups used by the qlib-style pipeline."""
    return {
        "alpha158_subset": ALPHA158_SUBSET,
        "malaysia_specific": MALAYSIA_FACTOR_COLUMNS,
    }
