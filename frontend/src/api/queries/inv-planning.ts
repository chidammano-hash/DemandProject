import { fetchJson } from "./request";

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature1: Demand Variability
// ---------------------------------------------------------------------------

export interface VariabilitySummaryPayload {
  total_skus: number;
  by_class: { low: number; medium: number; high: number; lumpy: number };
  cv_percentiles: { p25: number | null; p50: number | null; p75: number | null; p95: number | null };
  avg_cv: number | null;
  avg_intermittency_ratio: number | null;
  top_volatile: {
    item_id: string;
    loc: string;
    abc_vol: string | null;
    cluster_assignment: string | null;
    demand_mean: number | null;
    demand_std: number | null;
    demand_cv: number | null;
    demand_mad: number | null;
    intermittency_ratio: number | null;
    variability_class: string | null;
  }[];
}

export interface VariabilityDetailRow {
  item_id: string;
  loc: string;
  abc_vol: string | null;
  cluster_assignment: string | null;
  demand_mean: number | null;
  demand_std: number | null;
  demand_cv: number | null;
  demand_mad: number | null;
  demand_p50: number | null;
  demand_p90: number | null;
  demand_skewness: number | null;
  demand_kurtosis: number | null;
  zero_demand_months: number | null;
  total_demand_months: number | null;
  intermittency_ratio: number | null;
  variability_class: string | null;
  demand_profile_ts: string | null;
}

export interface VariabilityDetailPayload {
  total: number;
  rows: VariabilityDetailRow[];
}

export async function fetchVariabilitySummary(params: {
  abc_vol?: string;
  cluster_assignment?: string;
}): Promise<VariabilitySummaryPayload> {
  const qs = new URLSearchParams();
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  if (params.cluster_assignment?.trim()) qs.set("cluster_assignment", params.cluster_assignment.trim());
  return fetchJson(`/inv-planning/variability/summary?${qs}`);
}

export async function fetchVariabilityDetail(params: {
  item?: string;
  location?: string;
  abc_vol?: string;
  variability_class?: string;
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: string;
}): Promise<VariabilityDetailPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 50),
    offset: String(params.offset ?? 0),
    sort_by: params.sort_by ?? "demand_cv",
    sort_dir: params.sort_dir ?? "desc",
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  if (params.variability_class?.trim()) qs.set("variability_class", params.variability_class.trim());
  return fetchJson(`/inv-planning/variability/detail?${qs}`);
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature2: Lead Time Variability
// ---------------------------------------------------------------------------

export interface LtSummaryPayload {
  total_profiles: number;
  by_class: { stable: number; moderate: number; volatile: number };
  avg_lt_cv: number | null;
  avg_lt_mean_days: number | null;
  lt_cv_p50: number | null;
  lt_cv_p95: number | null;
  top_volatile: {
    item_id: string;
    loc: string;
    lt_mean_days: number | null;
    lt_std_days: number | null;
    lt_cv: number | null;
    lt_min_days: number | null;
    lt_max_days: number | null;
    observation_count: number | null;
    lt_variability_class: string | null;
  }[];
}

export interface LtProfileRow {
  item_id: string;
  loc: string;
  lt_mean_days: number | null;
  lt_std_days: number | null;
  lt_cv: number | null;
  lt_min_days: number | null;
  lt_max_days: number | null;
  lt_p25_days: number | null;
  lt_p50_days: number | null;
  lt_p75_days: number | null;
  lt_p95_days: number | null;
  observation_count: number | null;
  observation_months: number | null;
  lt_variability_class: string | null;
  computed_at: string | null;
}

export interface LtProfilePayload {
  total: number;
  rows: LtProfileRow[];
}

export async function fetchLtSummary(params: {
  abc_vol?: string;
}): Promise<LtSummaryPayload> {
  const qs = new URLSearchParams();
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  return fetchJson(`/inv-planning/lead-time/summary?${qs}`);
}

export async function fetchLtProfile(params: {
  item?: string;
  location?: string;
  lt_variability_class?: string;
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: string;
}): Promise<LtProfilePayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 50),
    offset: String(params.offset ?? 0),
    sort_by: params.sort_by ?? "lt_cv",
    sort_dir: params.sort_dir ?? "desc",
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  if (params.lt_variability_class?.trim()) qs.set("lt_variability_class", params.lt_variability_class.trim());
  return fetchJson(`/inv-planning/lead-time/profile?${qs}`);
}

// ---------------------------------------------------------------------------
// Re-exports from feature sub-modules
// ---------------------------------------------------------------------------
export * from "./inv-planning-eoq";
export * from "./inv-planning-policy";
export * from "./inv-planning-health";
export * from "./inv-planning-exceptions";
export * from "./inv-planning-safety-stock";
export * from "./inv-planning-signals";
export * from "./inv-planning-abc";
export * from "./inv-planning-supplier";
export * from "./inv-planning-intramonth";
export * from "./production-forecast";
export * from "./supply";
export * from "./inv-planning-projection";
export * from "./inv-planning-replenishment";
