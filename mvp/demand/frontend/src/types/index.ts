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

export type DfuClustersPayload = {
  domain: string;
  total_assigned: number;
  clusters: ClusterInfo[];
};

export type ClusterProfile = {
  cluster_id: number;
  label: string;
  mean_demand: number;
  cv_demand: number;
  seasonality_strength: number;
  trend_slope: number;
  growth_rate: number;
  zero_demand_pct: number;
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
  dfu_count: number;
};

export type AccuracySliceRow = {
  bucket: string;
  n_rows: number;
  by_model: Record<string, AccuracyKpis>;
};

export type AccuracySlicePayload = {
  group_by: string;
  rows: AccuracySliceRow[];
  common_dfu_count?: number;
  dfu_counts?: Record<string, number>;
};

export type LagPoint = {
  lag: number;
  by_model: Record<string, AccuracyKpis>;
};

export type LagCurvePayload = {
  by_lag: LagPoint[];
};

export type MarketIntelSearchResult = {
  title: string;
  link: string;
  snippet: string;
};

export type MarketIntelPayload = {
  item_no: string;
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

export type DfuAnalysisMode = "item_location" | "all_items_at_location" | "item_at_all_locations";

export type DfuAnalysisKpis = {
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  sum_forecast: number;
  sum_actual: number;
  months_covered: number;
};

export type DfuModelMonthly = {
  month: string;
  forecast: number;
  actual: number;
};

export type DfuAnalysisPayload = {
  mode: DfuAnalysisMode;
  item: string;
  location: string;
  points: number;
  models: string[];
  series: Record<string, number | string>[];
  model_monthly: Record<string, DfuModelMonthly[]>;
  dfu_attributes: Record<string, string | null>[];
  scope_count?: number;
};

export type InventoryPosition = {
  item_no: string;
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
};

export type InventoryPositionPayload = {
  total: number;
  limit: number;
  offset: number;
  positions: InventoryPosition[];
};

export type InventoryTrendPayload = {
  trend: InventoryTrendPoint[];
};

export type InventoryItemDetailPayload = {
  item: string;
  location: string;
  snapshots: InventoryPosition[];
};

export type Theme = "light" | "soft" | "dark";

export type ElementConfig = {
  symbol: string;
  number: number;
  name: string;
  color: string;
  activeColor: string;
  glow: string;
};

// Feature 37: Inventory Backtest types

export type InvBacktestModelMetrics = {
  dfu_months: number;
  stockout_count: number;
  stockout_rate: number;
  excess_count: number;
  excess_rate: number;
  service_level: number;
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
  item_no: string;
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
};

export type InvBacktestDetailPayload = {
  total: number;
  limit: number;
  offset: number;
  rows: InvBacktestDetailRow[];
};
