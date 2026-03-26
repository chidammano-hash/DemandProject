"""Centralized row builder library for API test mock data.

Each function returns a tuple matching the exact column order expected by
the corresponding SQL query / router. Named parameters with sensible
defaults reduce boilerplate across test files.

Usage:
    from tests.api.row_builders import build_safety_stock_detail_row
    cursor.fetchall.return_value = [build_safety_stock_detail_row(item_id="X")]
"""
from __future__ import annotations

import datetime


# ---------------------------------------------------------------------------
# Generic dimension row (dim_item, dim_location, dim_customer, etc.)
# ---------------------------------------------------------------------------

def build_domain_row(
    sk: int = 1,
    ck: str = "CK1",
    *,
    col1: str = "val1",
    col2: str = "val2",
    col3: str | None = None,
    load_ts: str = "2025-01-01T00:00:00",
    modified_ts: str = "2025-01-01T00:00:00",
) -> tuple:
    """Generic dimension table row.

    Column order: sk, ck, col1, col2, col3, load_ts, modified_ts

    Adapt col1/col2/col3 to the actual domain columns as needed.
    For example, dim_item has item_id, description, brand, category, etc.
    """
    return (sk, ck, col1, col2, col3, load_ts, modified_ts)


# ---------------------------------------------------------------------------
# fact_sales_monthly
# ---------------------------------------------------------------------------

def build_sales_row(
    item_id: str = "100320",
    customer_group: str = "GRP1",
    loc: str = "1401-BULK",
    month: str = "2025-01-01",
    type_: int = 1,
    qty: float = 1000.0,
    revenue: float = 25000.0,
    load_ts: str = "2025-01-01T00:00:00",
) -> tuple:
    """fact_sales_monthly row.

    Column order: item_id, customer_group, loc, month, type, qty, revenue, load_ts
    """
    return (item_id, customer_group, loc, month, type_, qty, revenue, load_ts)


# ---------------------------------------------------------------------------
# fact_external_forecast_monthly
# ---------------------------------------------------------------------------

def build_forecast_row(
    item_id: str = "100320",
    loc: str = "1401-BULK",
    forecast_date: str = "2025-01-01",
    actual_month: str = "2025-02-01",
    basefcst_pref: float = 950.0,
    model_id: str = "external",
    lag: int = 0,
    load_ts: str = "2025-01-01T00:00:00",
) -> tuple:
    """fact_external_forecast_monthly row.

    Column order: item_id, loc, forecast_date, actual_month,
                  basefcst_pref, model_id, lag, load_ts
    """
    return (item_id, loc, forecast_date, actual_month, basefcst_pref, model_id, lag, load_ts)


# ---------------------------------------------------------------------------
# Inventory position (from /inventory/position endpoint)
# ---------------------------------------------------------------------------

def build_inventory_row(
    item_id: str = "100320",
    loc: str = "1401-BULK",
    snapshot_date: str = "2025-06-01",
    dos: float = 30.0,
    qty_on_hand: float = 100.0,
    qty_on_order: float = 150.0,
    safety_stock: float = 50.0,
    reorder_point: float = 25.0,
) -> tuple:
    """Inventory position row (as returned by /inventory/position).

    Column order: item_id, loc, snapshot_date, dos, qty_on_hand,
                  qty_on_order, safety_stock, reorder_point
    """
    return (item_id, loc, snapshot_date, dos, qty_on_hand, qty_on_order, safety_stock, reorder_point)


# ---------------------------------------------------------------------------
# Safety stock summary — tagged union rows (S=summary, C=class, G=gap)
# ---------------------------------------------------------------------------

def build_safety_stock_summary_row(
    tag: str = "S",
    val1: str = "500",
    val2: str = "85",
    val3: str = "1.12",
    val4: str = "14.0",
    val5: str = "-4200.0",
    val6: str | None = None,
    val7: str | None = None,
    val8: str | None = None,
) -> tuple:
    """Safety stock summary tagged row (single combined query).

    Tag values:
      - "S": summary — (tag, total_dfus, below_ss_count, avg_ss_coverage,
                         target_dos, total_ss_gap_units, None, None, None)
      - "C": class   — (tag, abc_vol, count, below_ss_count,
                         avg_ss_combined, avg_coverage, None, None, None)
      - "G": gap     — (tag, item_id, loc, current_qty, ss_combined,
                         ss_gap, ss_coverage, None, None)

    Column order: tag, val1..val8 (semantics depend on tag)
    """
    return (tag, val1, val2, val3, val4, val5, val6, val7, val8)


# ---------------------------------------------------------------------------
# Safety stock detail
# ---------------------------------------------------------------------------

def build_safety_stock_detail_row(
    item_id: str = "ITEM001",
    loc: str = "LOC1",
    abc_vol: str = "A",
    service_level_target: float = 0.98,
    z_score: float = 2.054,
    ss_combined: float = 180.0,
    reorder_point: float = 340.0,
    current_qty_on_hand: float = 120.0,
    current_dos: float = 8.0,
    ss_gap: float = -60.0,
    ss_coverage: float = 0.667,
    is_below_ss: bool = True,
    target_dos_min: float = 18.0,
) -> tuple:
    """Safety stock detail row.

    Column order: item_id, loc, abc_vol, service_level_target, z_score,
                  ss_combined, reorder_point, current_qty_on_hand,
                  current_dos, ss_gap, ss_coverage, is_below_ss,
                  target_dos_min
    """
    return (
        item_id, loc, abc_vol, service_level_target, z_score,
        ss_combined, reorder_point, current_qty_on_hand,
        current_dos, ss_gap, ss_coverage, is_below_ss,
        target_dos_min,
    )


# ---------------------------------------------------------------------------
# EOQ summary — by-ABC row
# ---------------------------------------------------------------------------

def build_eoq_summary_abc_row(
    abc_vol: str = "A",
    count: int = 30,
    avg_eoq: float = 310.0,
    total_cycle_stock: float = 500.0,
    total_annual_cost: float = 25000.0,
    avg_order_frequency: float = 3.9,
) -> tuple:
    """EOQ summary by-ABC row.

    Column order: abc_vol, count, avg_eoq, total_cycle_stock,
                  total_annual_cost, avg_order_frequency
    """
    return (abc_vol, count, avg_eoq, total_cycle_stock, total_annual_cost, avg_order_frequency)


# ---------------------------------------------------------------------------
# EOQ detail
# ---------------------------------------------------------------------------

def build_eoq_detail_row(
    item_id: str = "ITEM001",
    loc: str = "LOC1",
    abc_vol: str = "A",
    unit_cost: float = 100.0,
    annual_demand: float = 1200.0,
    ordering_cost: float = 50.0,
    holding_rate: float = 0.25,
    lead_time_days: float = 10.0,
    moq: float = 1.0,
    raw_eoq: float = 219.09,
    effective_eoq: float = 219.09,
    cycle_stock: float = 109.54,
    order_frequency: float = 5.48,
    ordering_annual_cost: float = 273.86,
    holding_annual_cost: float = 273.86,
    total_annual_cost: float = 547.72,
    notes: str | None = None,
) -> tuple:
    """EOQ detail row (17 columns).

    Column order: item_id, loc, abc_vol, unit_cost, annual_demand,
                  ordering_cost, holding_rate, lead_time_days, moq,
                  raw_eoq, effective_eoq, cycle_stock, order_frequency,
                  ordering_annual_cost, holding_annual_cost,
                  total_annual_cost, notes
    """
    return (
        item_id, loc, abc_vol,
        unit_cost, annual_demand,
        ordering_cost, holding_rate, lead_time_days, moq,
        raw_eoq, effective_eoq, cycle_stock, order_frequency,
        ordering_annual_cost, holding_annual_cost, total_annual_cost,
        notes,
    )


# ---------------------------------------------------------------------------
# Insight row (11 columns — used by update_insight_status RETURNING)
# ---------------------------------------------------------------------------

def build_insight_row(
    insight_id: int = 1,
    status: str = "acknowledged",
    insight_type: str = "stockout_risk",
    item_id: str = "100320",
    loc: str = "1401-BULK",
    abc_vol: str = "A",
    financial_impact_est: float = 8500.0,
    dos: float = 18.0,
    total_lt_days: int = 14,
    champion_wape: float = 0.41,
    forecast_bias_pct: float = 0.20,
) -> tuple:
    """AI Planner insight row (11 columns — INSERT/UPDATE RETURNING).

    Column order: insight_id, status, insight_type, item_id, loc, abc_vol,
                  financial_impact_est, dos, total_lt_days, champion_wape,
                  forecast_bias_pct
    """
    return (
        insight_id, status, insight_type, item_id, loc, abc_vol,
        financial_impact_est, dos, total_lt_days, champion_wape,
        forecast_bias_pct,
    )


# ---------------------------------------------------------------------------
# Insight list row (6 columns — used by GET /ai-planner/insights)
# ---------------------------------------------------------------------------

def build_insight_list_row(
    insight_id: int = 1,
    insight_type: str = "stockout_risk",
    severity: str = "critical",
    item_id: str = "100320",
    loc: str = "1401-BULK",
    summary: str = "Low DOS for item 100320",
) -> tuple:
    """AI Planner insight list row (6 columns).

    Column order: insight_id, insight_type, severity, item_id, loc, summary
    """
    return (insight_id, insight_type, severity, item_id, loc, summary)


# ---------------------------------------------------------------------------
# LGBM tuning run list row (14 columns)
# ---------------------------------------------------------------------------

def build_tuning_run_list_row(
    run_id: int = 1,
    run_label: str = "baseline",
    model_id: str = "lgbm_cluster",
    started_at: str = "2026-03-22T10:00:00",
    completed_at: str = "2026-03-22T11:00:00",
    status: str = "completed",
    accuracy_pct: float = 69.34,
    wape: float = 30.66,
    bias: float = -0.0132,
    n_predictions: int = 2725140,
    n_dfus: int = 50602,
    notes: str | None = None,
    is_promoted: bool = True,
    promoted_at: str | None = "2026-03-23T08:00:00",
) -> tuple:
    """LGBM tuning run list row (14 columns, from /lgbm-tuning/runs).

    Column order: run_id, run_label, model_id, started_at, completed_at,
                  status, accuracy_pct, wape, bias, n_predictions,
                  n_dfus, notes, is_promoted, promoted_at
    """
    return (
        run_id, run_label, model_id, started_at, completed_at,
        status, accuracy_pct, wape, bias, n_predictions,
        n_dfus, notes, is_promoted, promoted_at,
    )


# ---------------------------------------------------------------------------
# LGBM tuning run detail row (17 columns)
# ---------------------------------------------------------------------------

def build_tuning_run_detail_row(
    run_id: int = 1,
    run_label: str = "baseline",
    model_id: str = "lgbm_cluster",
    started_at: str = "2026-03-22T10:00:00",
    completed_at: str = "2026-03-22T11:00:00",
    status: str = "completed",
    params: str = "{}",
    feature_count: int = 37,
    features: str = "[]",
    accuracy_pct: float = 69.34,
    wape: float = 30.66,
    bias: float = -0.0132,
    n_predictions: int = 2725140,
    n_dfus: int = 50602,
    metadata: str = "{}",
    notes: str | None = None,
    backup_path: str | None = None,
) -> tuple:
    """LGBM tuning run detail row (17 columns, from /lgbm-tuning/runs/{id}).

    Column order: run_id, run_label, model_id, started_at, completed_at,
                  status, params, feature_count, features,
                  accuracy_pct, wape, bias, n_predictions, n_dfus,
                  metadata, notes, backup_path
    """
    return (
        run_id, run_label, model_id, started_at, completed_at,
        status, params, feature_count, features,
        accuracy_pct, wape, bias, n_predictions, n_dfus,
        metadata, notes, backup_path,
    )


# ---------------------------------------------------------------------------
# Unified model tuning — experiment list row (22 columns)
# ---------------------------------------------------------------------------

def build_experiment_list_row(
    run_id: int = 1,
    run_label: str = "baseline_v1",
    model_id: str = "lgbm_cluster",
    started_at: str = "2026-03-20T10:00:00",
    completed_at: str = "2026-03-20T11:00:00",
    status: str = "completed",
    accuracy_pct: float = 72.5,
    wape: float = 27.5,
    bias: float = 0.032,
    n_predictions: int = 580000,
    n_dfus: int = 3200,
    notes: str | None = None,
    is_promoted: bool = False,
    promoted_at: str | None = None,
    job_id: str | None = None,
    template_id: str | None = None,
    is_results_promoted: bool = False,
    results_promoted_at: str | None = None,
    results_promote_job_id: str | None = None,
    cluster_source: str = "production",
    cluster_experiment_id: int | None = None,
    cluster_experiment_label: str | None = None,
) -> tuple:
    """Unified model tuning experiment list row (22 columns).

    Column order: run_id, run_label, model_id, started_at, completed_at,
                  status, accuracy_pct, wape, bias, n_predictions, n_dfus,
                  notes, is_promoted, promoted_at, job_id, template_id,
                  is_results_promoted, results_promoted_at,
                  results_promote_job_id, cluster_source,
                  cluster_experiment_id, cluster_experiment_label
    """
    return (
        run_id, run_label, model_id, started_at, completed_at,
        status, accuracy_pct, wape, bias, n_predictions, n_dfus,
        notes, is_promoted, promoted_at, job_id, template_id,
        is_results_promoted, results_promoted_at, results_promote_job_id,
        cluster_source, cluster_experiment_id, cluster_experiment_label,
    )


# ---------------------------------------------------------------------------
# Unified model tuning — experiment detail row (24 columns)
# ---------------------------------------------------------------------------

def build_experiment_detail_row(
    run_id: int = 1,
    run_label: str = "baseline_v1",
    model_id: str = "lgbm_cluster",
    started_at: str = "2026-03-20T10:00:00",
    completed_at: str = "2026-03-20T11:00:00",
    status: str = "completed",
    params: str | None = '{"n_estimators": 1500}',
    feature_count: int = 17,
    features: str | None = '["lag_1", "lag_2"]',
    accuracy_pct: float = 72.5,
    wape: float = 27.5,
    bias: float = 0.032,
    n_predictions: int = 580000,
    n_dfus: int = 3200,
    metadata: str | None = "{}",
    notes: str | None = None,
    backup_path: str | None = None,
    job_id: str | None = None,
    template_id: str | None = None,
    is_promoted: bool = False,
    promoted_at: str | None = None,
    is_results_promoted: bool = False,
    results_promoted_at: str | None = None,
    results_promote_job_id: str | None = None,
) -> tuple:
    """Unified model tuning experiment detail row (24 columns).

    Column order: run_id, run_label, model_id, started_at, completed_at,
                  status, params, feature_count, features,
                  accuracy_pct, wape, bias, n_predictions, n_dfus,
                  metadata, notes, backup_path, job_id, template_id,
                  is_promoted, promoted_at, is_results_promoted,
                  results_promoted_at, results_promote_job_id
    """
    return (
        run_id, run_label, model_id, started_at, completed_at,
        status, params, feature_count, features,
        accuracy_pct, wape, bias, n_predictions, n_dfus,
        metadata, notes, backup_path, job_id, template_id,
        is_promoted, promoted_at,
        is_results_promoted, results_promoted_at, results_promote_job_id,
    )


# ---------------------------------------------------------------------------
# Unified model tuning — timeframe row (10 columns)
# ---------------------------------------------------------------------------

def build_timeframe_row(
    tf_id: int = 1,
    run_id: int = 1,
    timeframe: str = "A",
    train_end: str = "2025-04-01",
    predict_start: str = "2025-05-01",
    predict_end: str = "2026-02-01",
    n_predictions: int = 58000,
    accuracy_pct: float = 65.5,
    wape: float = 34.5,
    bias: float = 0.05,
) -> tuple:
    """Unified model tuning timeframe row (10 columns).

    Column order: tf_id, run_id, timeframe, train_end, predict_start,
                  predict_end, n_predictions, accuracy_pct, wape, bias
    """
    return (tf_id, run_id, timeframe, train_end, predict_start, predict_end,
            n_predictions, accuracy_pct, wape, bias)


# ---------------------------------------------------------------------------
# Execution-lag row (used by both unified tuning and champion experiments)
# ---------------------------------------------------------------------------

def build_lag_row(
    exec_lag: int = 0,
    n_predictions: int = 116000,
    n_dfus: int = 3200,
    accuracy_pct: float = 79.5,
    wape: float = 20.5,
    bias: float = 0.028,
) -> tuple:
    """Execution-lag breakdown row (6 columns).

    Column order: exec_lag, n_predictions, n_dfus, accuracy_pct, wape, bias
    """
    return (exec_lag, n_predictions, n_dfus, accuracy_pct, wape, bias)


# ---------------------------------------------------------------------------
# Champion experiment row (28 columns)
# ---------------------------------------------------------------------------

def build_champion_experiment_row(
    experiment_id: int = 1,
    label: str = "Test Champion",
    notes: str | None = None,
    template_id: str | None = "expanding_conservative",
    status: str = "completed",
    created_at: str = "2026-03-25T10:00:00+00:00",
    started_at: str = "2026-03-25T10:00:05+00:00",
    completed_at: str = "2026-03-25T10:05:00+00:00",
    runtime_seconds: float = 295.0,
    job_id: str | None = "job-champ-123",
    strategy: str = "expanding",
    strategy_params: str | None = '{"min_prior_months": 3}',
    meta_learner_params: str | None = None,
    models: str = '["lgbm_cluster", "catboost_cluster", "xgboost_cluster"]',
    metric: str = "accuracy_pct",
    lag_mode: str = "execution",
    min_sku_rows: int = 3,
    champion_accuracy: float | None = 71.50,
    ceiling_accuracy: float | None = 78.20,
    gap_bps: float | None = 670.0,
    n_champions: int | None = 5000,
    n_dfu_months: int | None = 60000,
    model_distribution: str | None = '{"lgbm_cluster": 45.2, "catboost_cluster": 30.1, "xgboost_cluster": 24.7}',
    is_promoted: bool = False,
    promoted_at: str | None = None,
    is_results_promoted: bool = False,
    results_promoted_at: str | None = None,
    results_promote_job_id: str | None = None,
) -> tuple:
    """Champion experiment row (28 columns).

    Column order: experiment_id, label, notes, template_id, status,
                  created_at, started_at, completed_at, runtime_seconds,
                  job_id, strategy, strategy_params, meta_learner_params,
                  models, metric, lag_mode, min_sku_rows,
                  champion_accuracy, ceiling_accuracy, gap_bps,
                  n_champions, n_dfu_months, model_distribution,
                  is_promoted, promoted_at, is_results_promoted,
                  results_promoted_at, results_promote_job_id
    """
    return (
        experiment_id, label, notes, template_id, status,
        created_at, started_at, completed_at, runtime_seconds, job_id,
        strategy, strategy_params, meta_learner_params, models,
        metric, lag_mode, min_sku_rows,
        champion_accuracy, ceiling_accuracy, gap_bps,
        n_champions, n_dfu_months, model_distribution,
        is_promoted, promoted_at, is_results_promoted,
        results_promoted_at, results_promote_job_id,
    )


# ---------------------------------------------------------------------------
# Champion experiment — per-lag row (6 columns)
# ---------------------------------------------------------------------------

def build_champion_lag_row(
    exec_lag: int = 0,
    champion_accuracy: float = 71.5,
    ceiling_accuracy: float = 78.2,
    gap_bps: float = 670.0,
    n_dfu_months: int = 12000,
    model_distribution: str = "{}",
) -> tuple:
    """Champion experiment per-lag breakdown row (6 columns).

    Column order: exec_lag, champion_accuracy, ceiling_accuracy,
                  gap_bps, n_dfu_months, model_distribution
    """
    return (exec_lag, champion_accuracy, ceiling_accuracy, gap_bps,
            n_dfu_months, model_distribution)


# ---------------------------------------------------------------------------
# Champion experiment — per-month row (6 columns)
# ---------------------------------------------------------------------------

def build_champion_month_row(
    month_start: str = "2025-01-01",
    champion_accuracy: float = 71.5,
    ceiling_accuracy: float = 78.2,
    gap_bps: float = 670.0,
    n_champions: int = 1000,
    model_distribution: str = "{}",
) -> tuple:
    """Champion experiment per-month breakdown row (6 columns).

    Column order: month_start, champion_accuracy, ceiling_accuracy,
                  gap_bps, n_champions, model_distribution
    """
    return (month_start, champion_accuracy, ceiling_accuracy, gap_bps,
            n_champions, model_distribution)


# ---------------------------------------------------------------------------
# Cluster experiment row (25 columns)
# ---------------------------------------------------------------------------

def build_cluster_experiment_row(
    experiment_id: int = 1,
    scenario_id: str = "sc_20260320_100000_a1b2",
    label: str = "Test Experiment",
    notes: str | None = None,
    template_id: str | None = "production_baseline",
    status: str = "completed",
    created_at: str = "2026-03-20T10:00:00+00:00",
    started_at: str = "2026-03-20T10:00:05+00:00",
    completed_at: str = "2026-03-20T10:05:00+00:00",
    runtime_seconds: float = 295.0,
    job_id: str | None = "job-123",
    feature_params: str | None = '{"time_window_months": 24, "min_months_history": 1}',
    model_params: str | None = '{"k_range": [3, 12], "min_cluster_size_pct": 2.0}',
    label_params: str | None = '{"volume_high": 0.75, "volume_low": 0.25}',
    optimal_k: int | None = 8,
    silhouette_score: float | None = 0.342,
    inertia: float | None = 150000.0,
    total_dfus: int | None = 12000,
    n_clusters: int | None = 8,
    cluster_sizes: str | None = '{"0": 3000, "1": 4000, "2": 5000}',
    profiles: str | None = '[{"label": "high_volume_steady", "count": 3000}]',
    k_selection_results: str | None = '{"k_values": [3,4,5], "inertias": [1000,800,600], "silhouette_scores": [0.3,0.35,0.32]}',
    is_promoted: bool = False,
    promoted_at: str | None = None,
    artifacts_path: str | None = "/tmp/clustering_scenarios/sc_test",
) -> tuple:
    """Cluster experiment row (25 columns).

    Column order: experiment_id, scenario_id, label, notes, template_id,
                  status, created_at, started_at, completed_at,
                  runtime_seconds, job_id, feature_params, model_params,
                  label_params, optimal_k, silhouette_score, inertia,
                  total_dfus, n_clusters, cluster_sizes, profiles,
                  k_selection_results, is_promoted, promoted_at,
                  artifacts_path
    """
    return (
        experiment_id, scenario_id, label, notes, template_id,
        status, created_at, started_at, completed_at, runtime_seconds,
        job_id, feature_params, model_params, label_params,
        optimal_k, silhouette_score, inertia, total_dfus, n_clusters,
        cluster_sizes, profiles, k_selection_results,
        is_promoted, promoted_at, artifacts_path,
    )


# ---------------------------------------------------------------------------
# Exception row (18 columns)
# ---------------------------------------------------------------------------

def build_exception_row(
    exception_id: str = "exc-001",
    item_id: str = "ITEM001",
    loc: str = "LOC1",
    exception_date: datetime.date = datetime.date(2026, 3, 4),
    exception_type: str = "below_rop",
    severity: str = "high",
    current_qty_on_hand: float = 150.0,
    current_dos: float = 30.0,
    ss_combined: float = 200.0,
    reorder_point: float = 180.0,
    recommended_order_qty: float = 100.0,
    recommended_order_by: datetime.date = datetime.date(2026, 3, 11),
    expected_receipt_date: datetime.date = datetime.date(2026, 3, 16),
    estimated_order_value: float = 1500.0,
    policy_id: str = "A_continuous_v1",
    status: str = "open",
    acknowledged_by: str | None = None,
    notes: str | None = None,
) -> tuple:
    """Exception queue row (18 columns).

    Column order: exception_id, item_id, loc, exception_date,
                  exception_type, severity, current_qty_on_hand,
                  current_dos, ss_combined, reorder_point,
                  recommended_order_qty, recommended_order_by,
                  expected_receipt_date, estimated_order_value,
                  policy_id, status, acknowledged_by, notes
    """
    return (
        exception_id, item_id, loc, exception_date,
        exception_type, severity, current_qty_on_hand,
        current_dos, ss_combined, reorder_point,
        recommended_order_qty, recommended_order_by,
        expected_receipt_date, estimated_order_value,
        policy_id, status, acknowledged_by, notes,
    )


# ---------------------------------------------------------------------------
# Compare row (unified model tuning, 16 columns)
# ---------------------------------------------------------------------------

def build_compare_row(
    run_id: int = 1,
    run_label: str = "baseline_v1",
    model_id: str = "lgbm_cluster",
    accuracy_pct: float = 72.22,
    wape: float = 27.78,
    bias: float = 0.032,
    n_predictions: int = 580000,
    n_dfus: int = 3200,
    status: str = "completed",
    params: str = '{"n_estimators": 1500}',
    features: str | None = '["lag_1"]',
    feature_count: int = 17,
    metadata: str = "{}",
    cluster_source: str = "production",
    cluster_experiment_id: int | None = None,
    cluster_experiment_label: str | None = None,
) -> tuple:
    """Compare endpoint row (16 columns).

    Column order: run_id, run_label, model_id, accuracy_pct, wape, bias,
                  n_predictions, n_dfus, status, params, features,
                  feature_count, metadata, cluster_source,
                  cluster_experiment_id, cluster_experiment_label
    """
    return (
        run_id, run_label, model_id, accuracy_pct, wape, bias,
        n_predictions, n_dfus, status, params, features, feature_count,
        metadata, cluster_source, cluster_experiment_id,
        cluster_experiment_label,
    )
