export type DomainMeta = {
  name: string;
  plural: string;
  default_sort: string;
  columns: string[];
  numeric_fields: string[];
  date_fields: string[];
  category_fields: string[];
};

export type DomainPage = {
  total: number;
  total_approximate?: boolean;
  limit: number;
  offset: number;
  [key: string]: unknown;
};

export type SuggestPayload = {
  values?: string[];
};

export type SamplePairPayload = {
  item?: string | null;
  location?: string | null;
};

export type ClusterInfo = {
  cluster_id: string;
  label: string;
  count: number;
  pct_of_total: number;
  avg_demand: number;
  cv_demand: number;
};

export type SkuClustersPayload = {
  domain: string;
  total_assigned: number;
  clusters: ClusterInfo[];
};

export type ClusterProfilesPayload = {
  profiles: ClusterProfile[];
  metadata: {
    optimal_k: number | null;
    silhouette_score: number | null;
    inertia: number | null;
  };
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sql?: string | null;
  data?: Record<string, unknown>[] | null;
  columns?: string[];
  row_count?: number | null;
  error?: string | null;
};

export type AccuracyKpis = {
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  sum_forecast: number;
  sum_actual: number;
  sku_count: number;
};

export type AccuracySliceRow = {
  bucket: string;
  n_rows: number;
  by_model: Record<string, AccuracyKpis>;
};

export type AccuracySlicePayload = {
  group_by: string;
  rows: AccuracySliceRow[];
  common_sku_count?: number;
  sku_counts?: Record<string, number>;
};

export type LagPoint = {
  lag: number;
  by_model: Record<string, AccuracyKpis>;
};

export type LagCurvePayload = {
  by_lag: LagPoint[];
};

// --- Per-DFU accuracy decomposition (diagnostic layer, /forecast/accuracy/decomposition) ---
export type UnweightedAccuracy = {
  n_dfus: number;
  n_undefined: number;
  mean_accuracy_pct: number | null;
  median_accuracy_pct: number | null;
};

// MASE is naive-relative: <1 beats the naive baseline, ≈1 on par, >1 worse.
// n_undefined = DFUs with no usable in-sample naive baseline (cold-start / flat
// history); they are excluded from mean/median rather than counted as a miss.
export type UnweightedMase = {
  n_dfus: number;
  n_undefined: number;
  mean_mase: number | null;
  median_mase: number | null;
};

export type DecompositionModelEntry = {
  volume_weighted: AccuracyKpis;
  unweighted: UnweightedAccuracy;
  mase: UnweightedMase;
  error_contribution_pct: number | null;
  n_dfus: number;
};

export type DecompositionRow = {
  bucket: string;
  by_model: Record<string, DecompositionModelEntry>;
};

export type AccuracyDecompositionPayload = {
  group_by: string;
  lag_filter: number;
  models: string[] | null;
  rows: DecompositionRow[];
  source: string;
  mase_seasonal_period_rule: string;
};

export type ErrorContributor = {
  item_id: string;
  customer_group: string;
  loc: string;
  cluster_assignment: string;
  region: string;
  abc_vol: string;
  seasonality_profile: string;
  sum_actual: number;
  sum_abs_error: number;
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  bias_direction: string;
  error_contribution_pct: number | null;
  cumulative_contribution_pct: number | null;
};

export type ErrorContributorsPayload = {
  models: string[] | null;
  lag_filter: number;
  limit: number;
  total_abs_error: number;
  total_dfus: number;
  contributors: ErrorContributor[];
  source: string;
};

export type MarketIntelPayload = {
  item_id: string;
  location_id: string;
  item_desc: string | null;
  brand_name: string | null;
  category: string | null;
  state_id: string | null;
  site_desc: string | null;
  search_results: MarketIntelSearchResult[];
  narrative: string;
  generated_at: string;
};

export type SkuAnalysisMode = "item_location" | "all_items_at_location" | "item_at_all_locations";

export type SkuAnalysisKpis = {
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  sum_forecast: number;
  sum_actual: number;
  months_covered: number;
};

export type SkuAnalysisPayload = {
  mode: SkuAnalysisMode;
  item: string;
  location: string;
  points: number;
  models: string[];
  // Values are numbers/strings, except champion_mix which is a {model,weight}[] array.
  series: Record<string, number | string | { model: string; weight: number }[]>[];
  model_monthly: Record<string, SkuModelMonthly[]>;
  dfu_attributes: Record<string, string | null>[];
  // Human-readable item description (dim_item.item_desc) for the breadcrumb (U3.5).
  item_desc?: string | null;
  scope_count?: number;
  // The champion's winning source model per month (item_location mode only) and
  // the dominant one across months — lets the chart label the champion line
  // "champion (N-BEATS)". Empty/null when source_model_id isn't populated.
  champion_source_by_month?: Record<string, string>;
  champion_dominant_source?: string | null;
  // Per-month blend composition for blended champions, e.g.
  // {"2025-01-01": [{model, weight}, ...]}. Drives the champion tooltip mix.
  champion_mix_by_month?: Record<string, { model: string; weight: number }[]>;
};

export type InventoryPosition = {
  item_id: string;
  loc: string;
  snapshot_date: string;
  lead_time_days: number | null;
  qty_on_hand: number;
  qty_on_hand_on_order: number;
  qty_on_order: number;
  mtd_sales: number;
};

export type InventoryKpis = {
  total_on_hand: number;
  total_on_order: number;
  avg_lead_time_days: number | null;
  dos: number | null;
  dos_delta?: number | null;
  dos_prev_month?: number | null;
  woc: number | null;
  inventory_turns: number | null;
  lt_coverage: number | null;
  distinct_items: number;
  distinct_locations: number;
  months_covered: number;
};

export type InventoryTrendPoint = {
  month: string;
  total_on_hand: number;
  total_on_order: number;
  monthly_sales: number;
  avg_lead_time: number;
  dos: number | null;
  safety_stock?: number | null;
  cycle_stock?: number | null;
};

export type InventoryTrendParams = {
  safety_stock?: number | null;
  reorder_point_units?: number | null;
  service_level_target?: number | null;
  z_score?: number | null;
  demand_cv?: number | null;
  eoq?: number | null;
  eoq_cycle_stock?: number | null;
  order_frequency?: number | null;
  order_policy?: string | null;
  policy_type?: string | null;
};

export type InventoryPositionPayload = {
  total: number;
  limit: number;
  offset: number;
  positions: InventoryPosition[];
};

export type InventoryTrendPayload = {
  trend: InventoryTrendPoint[];
  params?: InventoryTrendParams;
};

export type InventoryItemDetailPayload = {
  item: string;
  location: string;
  snapshots: InventoryPosition[];
};

export type Theme = "light" | "soft" | "dark";

// Feature 37: Inventory Backtest types

export type InvBacktestModelMetrics = {
  sku_months: number;
  stockout_count: number;
  stockout_rate: number;
  excess_count: number;
  excess_rate: number;
  cycle_service_level: number;
  avg_dos: number | null;
  wape: number | null;
  bias: number | null;
};

export type InvBacktestSummaryPayload = {
  models: string[];
  excess_dos_threshold: number;
  by_model: Record<string, InvBacktestModelMetrics>;
};

export type InvBacktestTrendPoint = {
  month: string;
  by_model: Record<
    string,
    {
      stockout_rate: number;
      excess_rate: number;
      avg_dos: number | null;
      wape: number | null;
    }
  >;
};

export type InvBacktestTrendPayload = {
  trend: InvBacktestTrendPoint[];
};

export type InvBacktestRootCausePayload = {
  model_id: string;
  stockout_total: number;
  stockout_under_forecast: number;
  stockout_over_forecast: number;
  stockout_exact: number;
  excess_total: number;
  excess_over_forecast: number;
  excess_under_forecast: number;
  excess_exact: number;
};

export type InvBacktestDetailRow = {
  item_id: string;
  loc: string;
  month: string;
  model_id: string;
  forecast: number;
  actual_demand: number;
  eom_qty_on_hand: number;
  dos: number | null;
  event_type: string;
  forecast_error: number;
  pct_error: number | null;
  bias_direction: string;
  seasonality_profile: string;
  zero_velocity_flag: boolean;
};

export type InvBacktestDetailPayload = {
  total: number;
  limit: number;
  offset: number;
  rows: InvBacktestDetailRow[];
};
