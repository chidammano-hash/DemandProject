/**
 * SKU Features API — SKU Features tab
 *
 * Query module for fetching SKU-level computed features (seasonality profiles,
 * variability classes, trend direction, CV, amplitude, etc.).
 *
 * Endpoints:
 *   GET /sku-features/summary        — aggregate summary + categorical distributions
 *   GET /sku-features/list           — paginated, sortable, filterable SKU rows
 *   GET /sku-features/distributions  — histogram bins for continuous features
 */

import { buildQuerySuffix } from "./helpers";
import { fetchJson } from "./request";

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

export interface SkuFeaturesSummary {
  total_skus: number;
  last_computed: string | null;
  distributions: {
    seasonality_profile: Record<string, number>;
    variability_class: Record<string, number>;
    trend_direction: Record<string, number>;
  };
  averages: Record<string, number>;
}

export interface SkuFeatureRow {
  sku_ck: string;
  item_id: string;
  loc: string;
  ml_cluster: string | null;
  seasonality_profile: string | null;
  variability_class: string | null;
  trend_direction: number | null;
  cv_demand: number | null;
  seasonal_amplitude: number | null;
  trend_r2: number | null;
  zero_demand_pct: number | null;
  adi: number | null;
  cagr: number | null;
  recency_ratio: number | null;
  features_computed_ts: string | null;
}

export interface FeatureDistribution {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface FeatureDistributions {
  features: Record<string, FeatureDistribution[]>;
}

// ---------------------------------------------------------------------------
// Request params
// ---------------------------------------------------------------------------

export interface SkuFeaturesListParams {
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: "asc" | "desc";
  search?: string;
  seasonality_profile?: string;
  variability_class?: string;
  trend_direction?: string;
}

// ---------------------------------------------------------------------------
// Query key factory
// ---------------------------------------------------------------------------

export const skuFeatureKeys = {
  summary: ["sku-features", "summary"] as const,
  list: (params: Record<string, unknown>) =>
    ["sku-features", "list", params] as const,
  distributions: ["sku-features", "distributions"] as const,
};

// Backward-compat alias (old name used in some components)
export const skuFeaturesKeys = skuFeatureKeys;

// ---------------------------------------------------------------------------
// Stale times (ms)
// ---------------------------------------------------------------------------

export const STALE_SKU_FEATURES = {
  SUMMARY: 300_000,       // 5min — summary rarely changes
  LIST: 60_000,           // 1min — list may be actively browsed
  DISTRIBUTIONS: 300_000, // 5min — histogram data is expensive, cache longer
} as const;

// ---------------------------------------------------------------------------
// Fetch functions
// ---------------------------------------------------------------------------

const BASE = "/sku-features";

/** Fetch aggregate summary: totals, categorical distributions, averages. */
export async function fetchSkuFeaturesSummary(): Promise<SkuFeaturesSummary> {
  return fetchJson<SkuFeaturesSummary>(`${BASE}/summary`);
}

/** Fetch paginated, sortable, filterable list of SKU feature rows. */
export async function fetchSkuFeaturesList(
  params?: SkuFeaturesListParams,
): Promise<{ rows: SkuFeatureRow[]; total: number }> {
  const qs = buildQuerySuffix({
    limit: params?.limit,
    offset: params?.offset,
    sort_by: params?.sort_by,
    sort_dir: params?.sort_dir,
    search: params?.search,
    seasonality_profile: params?.seasonality_profile,
    variability_class: params?.variability_class,
    trend_direction: params?.trend_direction,
  });
  return fetchJson<{ rows: SkuFeatureRow[]; total: number }>(`${BASE}/list${qs}`);
}

/** Fetch histogram distributions for continuous features. */
export async function fetchSkuFeaturesDistributions(
  bins?: number,
): Promise<FeatureDistributions> {
  const qs = buildQuerySuffix({ bins });
  return fetchJson<FeatureDistributions>(`${BASE}/distributions${qs}`);
}

/** Trigger background computation of all SKU features. Returns job_id. */
export async function triggerComputeSkuFeatures(
  timeWindowMonths = 36,
): Promise<{ job_id: string; status: string }> {
  return fetchJson<{ job_id: string; status: string }>(`${BASE}/compute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ time_window_months: timeWindowMonths }),
  });
}
