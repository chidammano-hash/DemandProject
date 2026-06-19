"""Cluster- and segment-based champion strategies.

Includes:
  - ``per_segment``           — Syntetos-Boylan demand classification routing
  - ``per_cluster``           — ML cluster -> best model lookup
  - ``cluster_regime_hybrid`` — Per-cluster strategy with regime switching
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.ml.champion.basic import strategy_expanding
from common.ml.champion.helpers import _expanding_stats, select_output_cols
from common.ml.champion.registry import (
    _DFU_COLS,
    _DFU_MONTH_COLS,
    _OUTPUT_COLS,
    STRATEGY_REGISTRY,
    register_strategy,
)


# ---------------------------------------------------------------------------
# Per-segment (Syntetos-Boylan) helpers + strategy
# ---------------------------------------------------------------------------

_DEFAULT_SEGMENT_STRATEGY_MAP: dict[str, dict[str, Any]] = {
    "smooth": {"strategy": "expanding"},
    "erratic": {"strategy": "ensemble", "top_k": 5},
    "intermittent": {"strategy": "rolling", "window_months": 6},
    "lumpy": {"strategy": "rolling", "window_months": 3, "min_prior": 2},
}


def _classify_demand_segments(
    df: pd.DataFrame,
    adi_threshold: float,
    cv2_threshold: float,
) -> dict[tuple[str, str, str], str]:
    """Classify each DFU into a Syntetos-Boylan demand segment.

    Uses only causally-available history: months with startdate < T - exec_lag
    where T is the latest month per DFU.  Classification is computed once per
    DFU using all available causal history (not per-month).

    Returns a mapping of (item_id, customer_group, loc) -> segment name.

    Segments:
        smooth:       ADI < threshold AND CV^2 < threshold
        erratic:      ADI < threshold AND CV^2 >= threshold
        intermittent: ADI >= threshold AND CV^2 < threshold
        lumpy:        ADI >= threshold AND CV^2 >= threshold

    DFUs with all-zero demand are assigned to "lumpy" (most conservative).
    """
    df_sorted = df.sort_values(_DFU_COLS + ["startdate"]).copy()

    # Deduplicate to one row per DFU-month (demand is same across models)
    dfu_demand = df_sorted.drop_duplicates(
        subset=_DFU_MONTH_COLS, keep="first",
    )[_DFU_COLS + ["startdate", "tothist_dmd"]].copy()

    if "execution_lag" in df.columns:
        lag_map: dict[tuple, int] = {
            k: int(v) if pd.notna(v) else 0
            for k, v in df.groupby(_DFU_COLS)["execution_lag"].first().items()
        }
    else:
        lag_map = {}

    segment_map: dict[tuple[str, str, str], str] = {}

    for dfu_key, group in dfu_demand.groupby(_DFU_COLS, sort=False):
        g = group.sort_values("startdate")
        exec_lag = lag_map.get(dfu_key, 0)

        # Causal window: exclude the last (exec_lag) months
        if exec_lag > 0 and len(g) > exec_lag:
            g_causal = g.iloc[:-exec_lag]
        else:
            g_causal = g

        demands = g_causal["tothist_dmd"].values.astype(float)
        n_periods = len(demands)
        n_nonzero = int(np.count_nonzero(demands))

        # All-zero demand -> lumpy (most conservative / short-memory routing)
        if n_nonzero == 0 or n_periods == 0:
            segment_map[dfu_key] = "lumpy"
            continue

        # ADI = average demand interval = n_periods / n_nonzero
        adi = n_periods / n_nonzero

        # CV^2 of non-zero demands
        nonzero_vals = demands[demands != 0]
        mean_nz = float(nonzero_vals.mean())
        std_nz = float(nonzero_vals.std(ddof=1)) if len(nonzero_vals) > 1 else 0.0
        cv2 = (std_nz / mean_nz) ** 2 if mean_nz > 0 else 0.0

        # Classify using Syntetos-Boylan quadrant
        if adi < adi_threshold and cv2 < cv2_threshold:
            segment_map[dfu_key] = "smooth"
        elif adi < adi_threshold and cv2 >= cv2_threshold:
            segment_map[dfu_key] = "erratic"
        elif adi >= adi_threshold and cv2 < cv2_threshold:
            segment_map[dfu_key] = "intermittent"
        else:
            segment_map[dfu_key] = "lumpy"

    return segment_map


@register_strategy("per_segment")
def strategy_per_segment(
    df: pd.DataFrame,
    *,
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
    segment_strategy_map: dict[str, dict[str, Any]] | None = None,
    min_prior_months: int = 3,
    **kwargs: Any,
) -> pd.DataFrame:
    """Per-segment champion selection using Syntetos-Boylan demand classification.

    Classifies each DFU into one of four demand archetypes (smooth, erratic,
    intermittent, lumpy) and routes to different selection strategies:

        smooth:       expanding (stable demand, long history helps)
        erratic:      ensemble top_k=5 (high variance, hedge more)
        intermittent: rolling window=6 (recent patterns matter)
        lumpy:        rolling window=3, min_prior=2 (very short memory)

    Classification is strictly causal -- only months with startdate < T - exec_lag
    are used for computing ADI and CV^2.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with per-DFU per-month per-model errors.
    adi_threshold : float
        ADI threshold separating smooth/erratic from intermittent/lumpy.
        Default 1.32 (Syntetos-Boylan standard).
    cv2_threshold : float
        CV^2 threshold separating smooth/intermittent from erratic/lumpy.
        Default 0.49 (Syntetos-Boylan standard).
    segment_strategy_map : dict[str, dict] | None
        Optional override for segment-to-strategy mapping.  Keys are segment
        names, values are dicts with ``"strategy"`` plus any strategy-specific
        kwargs.  If None, uses the default SBA mapping.
    min_prior_months : int
        Minimum prior months passed to each sub-strategy (can be overridden
        per-segment via segment_strategy_map).
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # Resolve segment -> strategy mapping
    seg_map = dict(_DEFAULT_SEGMENT_STRATEGY_MAP)
    if segment_strategy_map is not None:
        seg_map.update(segment_strategy_map)

    # Classify each DFU
    segment_assignments = _classify_demand_segments(
        df, adi_threshold, cv2_threshold,
    )

    # Group DFU keys by segment
    segment_to_dfus: dict[str, list[tuple[str, str, str]]] = {}
    for dfu_key, segment in segment_assignments.items():
        segment_to_dfus.setdefault(segment, []).append(dfu_key)

    # Run each segment's sub-strategy on its subset of data
    all_winners: list[pd.DataFrame] = []

    for segment, dfu_keys in segment_to_dfus.items():
        # Build filter for this segment's DFUs
        key_df = pd.DataFrame(dfu_keys, columns=_DFU_COLS)
        segment_data = df.merge(key_df, on=_DFU_COLS, how="inner")

        if segment_data.empty:
            continue

        # Resolve strategy name and kwargs for this segment
        seg_config = seg_map.get(segment, {"strategy": "expanding"})
        strategy_name = seg_config.get("strategy", "expanding")
        seg_kwargs: dict[str, Any] = {
            k: v for k, v in seg_config.items() if k != "strategy"
        }

        # Use segment-specific min_prior if provided, else the global default
        if "min_prior" in seg_kwargs:
            seg_kwargs["min_prior_months"] = seg_kwargs.pop("min_prior")
        elif "min_prior_months" not in seg_kwargs:
            seg_kwargs["min_prior_months"] = min_prior_months

        # Look up the sub-strategy from the registry
        strategy_fn = STRATEGY_REGISTRY.get(strategy_name)
        if strategy_fn is None:
            strategy_fn = STRATEGY_REGISTRY["expanding"]

        # Execute the sub-strategy
        winners = strategy_fn(segment_data, **seg_kwargs)
        if not winners.empty:
            all_winners.append(winners)

    if not all_winners:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    combined = pd.concat(all_winners, ignore_index=True)
    # Deduplicate -- a DFU-month should only appear once
    combined = combined.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")
    return combined[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: per_cluster
# ---------------------------------------------------------------------------

@register_strategy("per_cluster")
def strategy_per_cluster(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    dfu_features: pd.DataFrame | None = None,
    cluster_col: str = "ml_cluster",
    **kwargs: Any,
) -> pd.DataFrame:
    """Per-cluster champion: best model per ML cluster applied to all DFUs.

    Different ML clusters represent distinct demand profiles (e.g. smooth
    high-volume vs intermittent low-volume).  This strategy learns which
    model performs best *within each cluster* and assigns that cluster-level
    champion to every DFU in the cluster.

    Steps:
      1. Join ``dfu_features`` (must contain ``ml_cluster``) onto ``df``.
      2. Compute expanding WAPE per (cluster, model) across all DFUs in the
         cluster, strictly causal (exec-lag-aware via ``_expanding_stats``).
      3. For each cluster, rank models by aggregate WAPE and pick the best.
      4. For each DFU-month, look up the DFU's cluster and select that
         cluster's champion model's forecast.

    Falls back to ``strategy_expanding`` when:
      - ``dfu_features`` is None or missing the cluster column.
      - A DFU has no cluster assignment (unmapped DFUs use the global best).

    Parameters
    ----------
    dfu_features : pd.DataFrame | None
        Must contain item_id, customer_group, loc, and ``cluster_col``.
    cluster_col : str
        Column name in ``dfu_features`` holding the cluster label.
    min_prior_months : int
        Minimum causally-available prior months for a model to qualify.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # No features → pure expanding fallback
    if dfu_features is None or cluster_col not in dfu_features.columns:
        return strategy_expanding(df, min_prior_months=min_prior_months)

    # ── Merge cluster labels onto monthly errors ──────────────────────────
    cluster_map = dfu_features[_DFU_COLS + [cluster_col]].drop_duplicates(
        subset=_DFU_COLS, keep="first",
    )
    df_c = df.merge(cluster_map, on=_DFU_COLS, how="left")

    # DFUs without cluster assignment → will use global best later
    has_cluster = df_c[df_c[cluster_col].notna()]
    no_cluster = df_c[df_c[cluster_col].isna()]

    # ── Compute expanding stats (causal, exec-lag-aware) ──────────────────
    if has_cluster.empty:
        return strategy_expanding(df, min_prior_months=min_prior_months)

    has_cluster = _expanding_stats(has_cluster)

    qualified = has_cluster[has_cluster["prior_count"] >= min_prior_months].copy()
    qualified["prior_wape"] = (
        qualified["cum_abs_err"] / qualified["cum_actual"].abs()
    )
    qualified = qualified[qualified["prior_wape"].notna()]

    if qualified.empty:
        return strategy_expanding(df, min_prior_months=min_prior_months)

    # ── Per-cluster: aggregate WAPE per (cluster, model) ──────────────────
    cluster_model_stats = (
        qualified.groupby([cluster_col, "model_id"], sort=False)
        .agg(
            total_abs_err=("cum_abs_err", "sum"),
            total_actual=("cum_actual", "sum"),
        )
        .reset_index()
    )
    cluster_model_stats["agg_wape"] = (
        cluster_model_stats["total_abs_err"]
        / cluster_model_stats["total_actual"].abs().clip(lower=1e-6)
    )

    # Best model per cluster
    cluster_champions: dict[Any, str] = {}
    for cluster_id, group in cluster_model_stats.groupby(cluster_col, sort=False):
        best = group.loc[group["agg_wape"].idxmin()]
        cluster_champions[cluster_id] = best["model_id"]

    # Global best model (fallback for unmapped DFUs)
    global_stats = cluster_model_stats.groupby("model_id", sort=False).agg(
        total_abs_err=("total_abs_err", "sum"),
        total_actual=("total_actual", "sum"),
    ).reset_index()
    global_stats["agg_wape"] = (
        global_stats["total_abs_err"]
        / global_stats["total_actual"].abs().clip(lower=1e-6)
    )
    global_best = global_stats.loc[global_stats["agg_wape"].idxmin(), "model_id"]

    # ── Assign cluster champion to each DFU-month ─────────────────────────
    # Map each qualified row to its cluster champion
    qualified["_cluster_champion"] = qualified[cluster_col].map(cluster_champions)

    # Filter to rows where model_id matches the cluster champion
    winners = qualified[
        qualified["model_id"] == qualified["_cluster_champion"]
    ].copy()
    winners = winners.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")

    parts: list[pd.DataFrame] = [select_output_cols(winners)]

    # ── Handle unmapped DFUs (no cluster) with global best ────────────────
    if not no_cluster.empty:
        no_cluster_subset = no_cluster.drop(columns=[cluster_col])
        global_winners = no_cluster_subset[
            no_cluster_subset["model_id"] == global_best
        ].copy()
        if not global_winners.empty:
            global_winners["prior_wape"] = 0.0
            for col in _OUTPUT_COLS:
                if col not in global_winners.columns and col != "source_mix":
                    global_winners[col] = 0.0
            parts.append(select_output_cols(global_winners))

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")
    combined = combined[_OUTPUT_COLS].reset_index(drop=True)

    # Build cluster diagnostics
    cluster_diag = {}
    for cid, group in cluster_model_stats.groupby(cluster_col, sort=False):
        best_model = cluster_champions.get(cid, "unknown")
        best_wape = float(group.loc[group["agg_wape"].idxmin(), "agg_wape"])
        cluster_diag[str(cid)] = {"champion": best_model, "agg_wape": round(best_wape * 100, 2)}

    # Count DFU-months per champion in the output
    model_counts = combined["model_id"].value_counts().to_dict()
    total = sum(model_counts.values())
    model_pct = {m: round(c / total * 100, 2) for m, c in sorted(model_counts.items(), key=lambda x: -x[1])}

    combined.attrs["weight_diagnostics"] = {
        "type": "per_cluster",
        "cluster_champions": {str(k): v for k, v in cluster_champions.items()},
        "global_fallback": global_best,
        "model_share_pct": model_pct,
        "cluster_details": cluster_diag,
    }
    return combined


# ---------------------------------------------------------------------------
# Strategy: cluster_regime_hybrid
# ---------------------------------------------------------------------------

@register_strategy("cluster_regime_hybrid")
def strategy_cluster_regime_hybrid(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    dfu_features: pd.DataFrame | None = None,
    cluster_col: str = "ml_cluster",
    variance_window: int = 4,
    variance_threshold: float = 2.0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Cluster + regime hybrid: per-cluster strategy with regime switching.

    Combines per_cluster structural grouping with regime_adaptive temporal
    adaptation. Within each cluster:
      - Stable regime DFU-months: use expanding (long-memory, stable)
      - Shift regime DFU-months: use rolling (short-memory, adaptive)

    Falls back to regime_adaptive when cluster features unavailable.
    """
    # Deferred import to avoid a circular dependency with regime.py
    from common.ml.champion.regime import strategy_regime_adaptive

    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    if dfu_features is None or cluster_col not in dfu_features.columns:
        return strategy_regime_adaptive(
            df, min_prior_months=min_prior_months,
            variance_window=variance_window,
            variance_threshold=variance_threshold,
        )

    # Merge cluster labels
    cluster_map = dfu_features[_DFU_COLS + [cluster_col]].drop_duplicates(
        subset=_DFU_COLS, keep="first",
    )
    df_c = df.merge(cluster_map, on=_DFU_COLS, how="left")

    # Run regime_adaptive per cluster
    parts: list[pd.DataFrame] = []
    for cluster_id, cluster_df in df_c.groupby(cluster_col, sort=False):
        if cluster_df.empty:
            continue
        # Drop cluster column before passing to regime_adaptive
        cluster_data = cluster_df.drop(columns=[cluster_col])
        winners = strategy_regime_adaptive(
            cluster_data,
            min_prior_months=min_prior_months,
            variance_window=variance_window,
            variance_threshold=variance_threshold,
        )
        if not winners.empty:
            parts.append(winners)

    # Handle DFUs without cluster
    no_cluster = df_c[df_c[cluster_col].isna()].drop(columns=[cluster_col])
    if not no_cluster.empty:
        winners = strategy_regime_adaptive(
            no_cluster, min_prior_months=min_prior_months,
        )
        if not winners.empty:
            parts.append(winners)

    if not parts:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")
    return combined[_OUTPUT_COLS].reset_index(drop=True)
