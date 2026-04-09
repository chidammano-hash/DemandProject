"""SKU feature computation — unified orchestration package.

Re-exports the public API from submodules:
  - compute: load_sales_from_db, compute_all_sku_features
  - classifiers: classify_seasonality_profile, classify_variability_class
  - persistence: write_features_to_dim_sku
"""

from common.ml.sku_features.classifiers import (
    classify_seasonality_profile,
    classify_variability_class,
)
from common.ml.sku_features.compute import (
    compute_all_sku_features,
    load_sales_from_db,
)
from common.ml.sku_features.persistence import write_features_to_dim_sku

__all__ = [
    "classify_seasonality_profile",
    "classify_variability_class",
    "compute_all_sku_features",
    "load_sales_from_db",
    "write_features_to_dim_sku",
]
