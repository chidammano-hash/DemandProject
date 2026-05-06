"""Clustering library — feature engineering, training, labeling, and scenario management.

Only lightweight constants are re-exported at the package level. Heavy
training / labeling / scenario helpers must be imported from their
submodules directly so that consumers needing only the constants (e.g. the
``/clustering/core-features`` API endpoint) don't pay the cost of importing
matplotlib + sklearn + scipy at app boot.
"""

from common.ml.clustering.constants import CORE_FEATURES, LOG_TRANSFORM_FEATURES

__all__ = [
    "CORE_FEATURES",
    "LOG_TRANSFORM_FEATURES",
]
