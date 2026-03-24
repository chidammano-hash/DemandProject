import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature6: Inventory Health Score
// ---------------------------------------------------------------------------

export interface HealthSummaryFilters {
  abc_vol?: string;
  cluster_assignment?: string;
  region?: string;
  variability_class?: string;
}

export interface HealthTierBreakdown {
  healthy: number;
  monitor: number;
  at_risk: number;
  critical: number;
}

export interface HealthComponentAvgs {
  ss_coverage: number | null;
  dos_target: number | null;
  stockout_risk: number | null;
  forecast_accuracy: number | null;
}

export interface HealthHistogramBucket {
  bucket: string;
  count: number;
}

export interface HealthSummaryPayload {
  total_skus: number;
  by_tier: HealthTierBreakdown;
  avg_health_score: number | null;
  component_avgs: HealthComponentAvgs;
  score_histogram: HealthHistogramBucket[];
}

export interface HealthDetailParams {
  item?: string;
  location?: string;
  health_tier?: string;
  abc_vol?: string;
  cluster_assignment?: string;
  variability_class?: string;
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: string;
}

export interface HealthDetailRow {
  item_id: string;
  loc: string;
  abc_vol: string | null;
  variability_class: string | null;
  cluster_assignment: string | null;
  health_score: number;
  health_tier: string;
  score_ss_coverage: number;
  score_dos_target: number;
  score_stockout_risk: number;
  score_forecast_accuracy: number;
  ss_coverage: number | null;
  current_dos: number | null;
  target_dos_min: number | null;
  target_dos_max: number | null;
  is_below_ss: boolean | null;
  recent_wape: number | null;
  stockout_count_3m: number | null;
}

export interface HealthDetailPayload {
  total: number;
  rows: HealthDetailRow[];
}

export interface HealthHeatmapCell {
  x: string;
  y: string;
  avg_health_score: number | null;
  count: number;
  critical_count: number;
}

export interface HealthHeatmapPayload {
  x_labels: string[];
  y_labels: string[];
  cells: HealthHeatmapCell[];
}

export const healthKeys = {
  summary: (filters?: Record<string, unknown>) => ["health-summary", filters ?? {}] as const,
  detail:  (params?: Record<string, unknown>) => ["health-detail",  params ?? {}]   as const,
  heatmap: (groupX?: string, groupY?: string) => ["health-heatmap", groupX ?? "abc_vol", groupY ?? "variability_class"] as const,
};

export async function fetchHealthSummary(
  filters: HealthSummaryFilters = {},
): Promise<HealthSummaryPayload> {
  const qs = new URLSearchParams();
  if (filters.abc_vol) qs.set("abc_vol", filters.abc_vol);
  if (filters.cluster_assignment) qs.set("cluster_assignment", filters.cluster_assignment);
  if (filters.region) qs.set("region", filters.region);
  if (filters.variability_class) qs.set("variability_class", filters.variability_class);
  const q = qs.toString();
  return fetchJson(`/inv-planning/health/summary${q ? "?" + q : ""}`);
}

export async function fetchHealthDetail(
  params: HealthDetailParams = {},
): Promise<HealthDetailPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 100),
    offset: String(params.offset ?? 0),
    sort_by: params.sort_by ?? "health_score",
    sort_dir: params.sort_dir ?? "asc",
  });
  if (params.item) qs.set("item", params.item);
  if (params.location) qs.set("location", params.location);
  if (params.health_tier) qs.set("health_tier", params.health_tier);
  if (params.abc_vol) qs.set("abc_vol", params.abc_vol);
  if (params.cluster_assignment) qs.set("cluster_assignment", params.cluster_assignment);
  if (params.variability_class) qs.set("variability_class", params.variability_class);
  return fetchJson(`/inv-planning/health/detail?${qs}`);
}

export async function fetchHealthHeatmap(
  group_x = "abc_vol",
  group_y = "variability_class",
): Promise<HealthHeatmapPayload> {
  return fetchJson(
    `/inv-planning/health/heatmap?group_x=${encodeURIComponent(group_x)}&group_y=${encodeURIComponent(group_y)}`,
  );
}
