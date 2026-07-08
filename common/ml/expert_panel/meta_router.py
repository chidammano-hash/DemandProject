"""Per-DFU meta-router: LightGBM classifier that predicts the best algorithm.

Trains on per-DFU historical accuracy data (from ``build_dfu_accuracy_matrix``)
and DFU attributes (demand shape, volume tier, cluster, etc.).  At inference
time, returns the predicted best algorithm and a confidence score for each DFU.
Low-confidence DFUs are routed to the inverse-WAPE blend instead.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from common.ml.model_registry import build_tree_classifier, fit_tree_classifier

logger = logging.getLogger(__name__)

# Categorical features sourced from dfu_attrs (dim_sku)
_ATTRS_CATS = ["ml_cluster", "variability_class", "seasonality_profile", "abc_xyz_segment"]

# Categorical features sourced from classification_df
_CLS_CATS = ["segment", "volume_tier"]

# Numeric features sourced from classification_df
_CLS_NUMS = ["adi", "cv2", "mean_demand", "std_demand", "n_periods", "n_nonzero"]

# All categorical features (union)
_ALL_CATS = _ATTRS_CATS + _CLS_CATS

_REQUIRED_META_ROUTER_KEYS = {
    "n_estimators": "meta_n_estimators",
    "learning_rate": "meta_learning_rate",
    "num_leaves": "meta_num_leaves",
    "min_child_samples": "meta_min_child_samples",
    "subsample": "meta_subsample",
    "colsample_bytree": "meta_colsample_bytree",
    "reg_lambda": "meta_reg_lambda",
    "class_weight": "meta_class_weight",
    "random_state": "meta_random_state",
    "verbose": "meta_verbose",
    "min_n_months_filter": "meta_min_n_months_filter",
}


@dataclass
class MetaRouterModel:
    """Container for a trained meta-router classifier and its encoding metadata.

    Attributes:
        model: Fitted tree classifier.
        feature_cols: Ordered list of feature column names used during training.
        cat_feature_idx: Indices into ``feature_cols`` that are categorical.
        cat_categories: Mapping from categorical column name → sorted category list,
            used to produce consistent integer codes at prediction time.
        label_to_algorithm: Integer class label → algorithm_id string.
        algorithm_to_label: Reverse of label_to_algorithm (populated post-init).
    """

    model: Any
    feature_cols: list[str]
    cat_feature_idx: list[int]
    cat_categories: dict[str, list]
    label_to_algorithm: dict[int, str]
    algorithm_to_label: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.algorithm_to_label = {v: k for k, v in self.label_to_algorithm.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_default_hybrid_config() -> dict[str, Any]:
    cfg_path = Path(__file__).with_name("expert_panel_config.yaml")
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    hybrid_cfg = cfg.get("hybrid_ensemble", {})
    if not isinstance(hybrid_cfg, dict):
        raise ValueError("expert_panel_config.yaml hybrid_ensemble must be a mapping")
    return hybrid_cfg


def _resolve_meta_router_params(
    hybrid_config: Mapping[str, Any] | None = None,
    *,
    n_estimators: int | None = None,
    learning_rate: float | None = None,
    num_leaves: int | None = None,
    min_n_months_filter: int | None = None,
) -> tuple[dict[str, Any], int]:
    """Resolve meta-router classifier params from YAML-backed config."""
    cfg = dict(_load_default_hybrid_config() if hybrid_config is None else hybrid_config)
    overrides = {
        "meta_n_estimators": n_estimators,
        "meta_learning_rate": learning_rate,
        "meta_num_leaves": num_leaves,
        "meta_min_n_months_filter": min_n_months_filter,
    }
    cfg.update({key: value for key, value in overrides.items() if value is not None})

    missing = [
        config_key
        for config_key in _REQUIRED_META_ROUTER_KEYS.values()
        if config_key not in cfg or cfg[config_key] is None
    ]
    if missing:
        raise ValueError(
            "meta-router LightGBM params missing required YAML keys: " + ", ".join(sorted(missing))
        )

    model_params = {
        param_key: cfg[config_key]
        for param_key, config_key in _REQUIRED_META_ROUTER_KEYS.items()
        if param_key != "min_n_months_filter"
    }
    return model_params, int(cfg["meta_min_n_months_filter"])


def _join_features(
    dfu_accuracy_matrix: pd.DataFrame | None,
    dfu_attrs: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join DFU attributes and classification stats into one feature frame.

    If ``dfu_accuracy_matrix`` is provided, also attaches the best-algorithm
    target column.  Otherwise produces a features-only frame for prediction.

    Returns:
        DataFrame with sku_ck, feature columns, and optionally
        a ``best_algorithm`` column.
    """
    # Attrs subset (only columns that exist)
    attrs_cols = ["sku_ck"] + [c for c in _ATTRS_CATS if c in dfu_attrs.columns]
    feat = dfu_attrs[attrs_cols].copy()

    # Classification subset
    cls_cols = ["sku_ck"] + [c for c in (_CLS_CATS + _CLS_NUMS) if c in classification_df.columns]
    feat = feat.merge(classification_df[cls_cols], on="sku_ck", how="inner")

    if dfu_accuracy_matrix is not None:
        # Best algorithm = row with minimum WAPE per DFU
        best = (
            dfu_accuracy_matrix.sort_values("wape")
            .groupby("sku_ck", sort=False)
            .first()
            .reset_index()[["sku_ck", "algorithm_id"]]
            .rename(columns={"algorithm_id": "best_algorithm"})
        )
        feat = feat.merge(best, on="sku_ck", how="inner")

    return feat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_meta_router(
    dfu_accuracy_matrix: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
    classification_df: pd.DataFrame,
    n_estimators: int | None = None,
    learning_rate: float | None = None,
    num_leaves: int | None = None,
    min_n_months_filter: int | None = None,
    *,
    hybrid_config: Mapping[str, Any] | None = None,
) -> MetaRouterModel:
    """Train a LightGBM multiclass classifier to predict best algorithm per DFU.

    The training label is the ``algorithm_id`` with the lowest WAPE for each
    DFU in ``dfu_accuracy_matrix``.  Features are DFU attributes (ml_cluster,
    variability_class, seasonality_profile, abc_xyz_segment) joined with
    demand classification statistics (ADI, CV², mean demand, etc.).

    Args:
        dfu_accuracy_matrix: Output of ``build_dfu_accuracy_matrix``.
            Required columns: sku_ck, algorithm_id, wape, n_months.
        dfu_attrs: DFU attribute table (from dim_sku / golden set load).
            Required columns: sku_ck, plus any subset of _ATTRS_CATS.
        classification_df: Output of ``classify_demand``.
            Required columns: sku_ck, plus any subset of _CLS_CATS + _CLS_NUMS.
        n_estimators: Optional compatibility override for YAML meta_n_estimators.
        learning_rate: Optional compatibility override for YAML meta_learning_rate.
        num_leaves: Optional compatibility override for YAML meta_num_leaves.
        min_n_months_filter: Exclude DFUs where the winning algorithm has
            fewer matched months than this (noisy labels).
        hybrid_config: Optional ``hybrid_ensemble`` config mapping. When omitted,
            values are loaded from ``expert_panel_config.yaml``.

    Returns:
        Fitted ``MetaRouterModel``.

    Raises:
        ValueError: If the resulting training set is too small (< 10 DFUs)
            or has only 1 unique class.
    """
    model_params, resolved_min_n_months_filter = _resolve_meta_router_params(
        hybrid_config,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_n_months_filter=min_n_months_filter,
    )

    reliable = dfu_accuracy_matrix[dfu_accuracy_matrix["n_months"] >= resolved_min_n_months_filter]
    if reliable.empty:
        raise ValueError(
            f"train_meta_router: dfu_accuracy_matrix has no rows with "
            f"n_months >= {resolved_min_n_months_filter}. Cannot train."
        )

    feat = _join_features(reliable, dfu_attrs, classification_df)

    if len(feat) < 10:
        raise ValueError(
            f"train_meta_router: only {len(feat)} DFUs have all required "
            "features after joining. Cannot train a reliable classifier."
        )

    n_classes = feat["best_algorithm"].nunique()
    if n_classes < 2:
        raise ValueError(
            f"train_meta_router: only {n_classes} class(es) in target "
            f"({feat['best_algorithm'].unique()}). Need >= 2 to discriminate."
        )

    # Integer-encode target
    algorithms = sorted(feat["best_algorithm"].unique())
    alg_to_label = {a: i for i, a in enumerate(algorithms)}
    label_to_alg = {i: a for i, a in enumerate(algorithms)}
    feat = feat.copy()
    feat["_label"] = feat["best_algorithm"].map(alg_to_label)

    # Feature columns = everything except identifier, target, encoded target
    exclude = {"sku_ck", "best_algorithm", "_label"}
    feature_cols = [c for c in feat.columns if c not in exclude]

    # Identify categorical column indices for LightGBM
    cat_cols_present = [c for c in _ALL_CATS if c in feature_cols]
    cat_feature_idx = [feature_cols.index(c) for c in cat_cols_present]

    # Encode categorical columns as integer codes for LightGBM.
    # Store the category lists so prediction uses identical encodings.
    feat = feat.copy()
    cat_categories: dict[str, list] = {}
    for c in cat_cols_present:
        cats = sorted(feat[c].dropna().unique().tolist())
        cat_categories[c] = cats
        feat[c] = pd.Categorical(feat[c], categories=cats).codes  # -1 for unseen → treated as NaN

    X = feat[feature_cols].to_numpy(dtype=float, na_value=np.nan)
    y = feat["_label"].values

    clf = build_tree_classifier("lgbm", model_params)
    fit_tree_classifier(
        clf,
        "lgbm",
        X,
        y,
        categorical_feature=cat_feature_idx if cat_feature_idx else "auto",
    )

    logger.info(
        "Meta-router trained: %d DFUs, %d classes (%s), %d features",
        len(feat),
        n_classes,
        ", ".join(algorithms),
        len(feature_cols),
    )

    return MetaRouterModel(
        model=clf,
        feature_cols=feature_cols,
        cat_feature_idx=cat_feature_idx,
        cat_categories=cat_categories,
        label_to_algorithm=label_to_alg,
    )


def predict_meta_router(
    meta_model: MetaRouterModel,
    dfu_attrs: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict the best algorithm and routing confidence for each DFU.

    Uses the same feature construction as training.  DFUs missing from either
    ``dfu_attrs`` or ``classification_df`` are silently excluded — the caller
    is responsible for routing them to the blend fallback.

    Args:
        meta_model: Fitted ``MetaRouterModel`` from ``train_meta_router``.
        dfu_attrs: DFU attribute table (same schema as at training time).
        classification_df: Demand classification output.

    Returns:
        DataFrame with columns:
            sku_ck              — DFU identifier
            predicted_algorithm — algorithm_id string (best predicted algorithm)
            confidence          — max softmax probability (0–1)
    """
    feat = _join_features(None, dfu_attrs, classification_df)

    if feat.empty:
        logger.warning("predict_meta_router: no DFUs after joining dfu_attrs + classification_df")
        return pd.DataFrame(columns=["sku_ck", "predicted_algorithm", "confidence"])

    sku_cks = feat["sku_ck"].values

    # Encode categoricals using the same category lists as training
    feat = feat.copy()
    cat_cols_present = [meta_model.feature_cols[i] for i in meta_model.cat_feature_idx]
    for c in cat_cols_present:
        if c in feat.columns:
            cats = meta_model.cat_categories.get(c, sorted(feat[c].dropna().unique().tolist()))
            feat[c] = pd.Categorical(feat[c], categories=cats).codes

    # Fill any feature columns absent in this prediction set
    for c in meta_model.feature_cols:
        if c not in feat.columns:
            feat[c] = 0

    X = feat[meta_model.feature_cols].to_numpy(dtype=float, na_value=np.nan)
    proba: np.ndarray = meta_model.model.predict_proba(X)

    predicted_labels: np.ndarray = proba.argmax(axis=1)
    confidences: np.ndarray = proba.max(axis=1)

    predicted_algorithms = [
        meta_model.label_to_algorithm.get(int(lbl), "seasonal_naive") for lbl in predicted_labels
    ]

    result = pd.DataFrame(
        {
            "sku_ck": sku_cks,
            "predicted_algorithm": predicted_algorithms,
            "confidence": confidences,
        }
    )

    logger.info(
        "Meta-router predictions: %d DFUs | algorithm distribution: %s",
        len(result),
        result["predicted_algorithm"].value_counts().to_dict(),
    )
    return result
