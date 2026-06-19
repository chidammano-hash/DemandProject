/**
 * Cluster EDA (Exploratory Data Analysis) — API types, query keys, and fetchers.
 *
 * Fetchers normalise backend response shapes into the frontend contract types.
 * The `Raw*` interfaces mirror the actual FastAPI response shapes from
 * `api/routers/forecasting/cluster_eda.py`.
 */

import { fetchJson } from "./core";

// --- Types (frontend contract) ---

export interface ClusterProfileRow {
  cluster: number;
  n_dfus: number;
  mean_demand: number;
  cv: number;
  zero_pct: number;
  seasonal_amplitude: number;
  accuracy_pct: number | null;
}

export interface ErrorConcentration {
  top_10_pct_share: number;
  worst_months: Array<{ month: string; error_share: number }>;
  worst_clusters: Array<{ cluster: number; error_share: number; n_dfus: number }>;
}

export interface ClusterDistributionBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface ResidualPoint {
  cluster: number;
  mean_residual: number;
  std_residual: number;
  skew: number;
  n_dfus: number;
}

export interface SeasonalityHeatmapRow {
  cluster: number;
  values: number[];
}

export interface SeasonalityHeatmapResponse {
  months: string[];
  rows: SeasonalityHeatmapRow[];
}

// --- Raw wire shapes (mirror api/routers/forecasting/cluster_eda.py) ---

/** One row from GET /cluster-eda/profile `clusters[]`. */
interface RawProfileCluster {
  ml_cluster: number;
  n_dfus: number;
  mean_demand: number | null;
  cv_demand: number | null;
  zero_pct: number | null;
  overall_mean: number | null;
  demand_std: number | null;
  accuracy_pct: number | null;
  wape: number | null;
}

interface RawProfileResponse {
  clusters: RawProfileCluster[];
  warning: string | null;
}

/** GET /cluster-eda/error-concentration. */
interface RawErrorConcentrationResponse {
  top_error_dfus: { top_10pct_share: number | null; top_20pct_share: number | null };
  error_by_month: Array<{ month: number; wape: number | null; bias: number | null }>;
  error_by_cluster: Array<{
    cluster: number;
    wape: number | null;
    bias: number | null;
    share_of_total_error: number | null;
  }>;
  error_by_abc: Array<{ abc_class: string; wape: number | null; bias: number | null }>;
  warning: string | null;
}

/** GET /cluster-eda/demand-distribution/{cluster_id}. */
interface RawDistributionResponse {
  cluster_id: number;
  n_dfus: number;
  histogram: Array<{ bucket: string; count: number }>;
  percentiles: Record<string, number | null>;
  top_dfus: Array<{ sku_ck: string; mean_demand: number | null; cv: number | null }>;
  warning: string | null;
}

/** GET /cluster-eda/residual-analysis. */
interface RawResidualResponse {
  residual_stats: {
    mean: number | null;
    std: number | null;
    skew: number | null;
    kurtosis: number | null;
  };
  residual_by_horizon: Array<{ lag: number; mean_error: number | null; rmse: number | null }>;
  worst_dfus: Array<{
    sku_ck: string;
    mean_abs_error: number | null;
    bias: number | null;
    cluster: number;
  }>;
  bias_by_cluster: Array<{ cluster: number; bias: number | null; direction: string }>;
  warning: string | null;
}

/** GET /cluster-eda/seasonality-heatmap — month numbers, cluster ids, and a values matrix. */
interface RawSeasonalityResponse {
  clusters: number[];
  months: number[];
  values: Array<Array<number | null>>;
  warning?: string;
}

// --- Query keys ---

export const clusterEdaKeys = {
  all: ["cluster-eda"] as const,
  profile: () => [...clusterEdaKeys.all, "profile"] as const,
  errorConcentration: () => [...clusterEdaKeys.all, "error-concentration"] as const,
  distribution: (id: number) => [...clusterEdaKeys.all, "distribution", id] as const,
  residuals: (modelId: string) => [...clusterEdaKeys.all, "residuals", modelId] as const,
  seasonalityHeatmap: () => [...clusterEdaKeys.all, "seasonality-heatmap"] as const,
};

// --- Helpers ---

const MONTH_LABELS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
] as const;

function monthLabel(monthNum: number): string {
  return MONTH_LABELS[monthNum - 1] ?? String(monthNum);
}

/** Parse a backend histogram bucket label ("0", "1-10", "1000+") into numeric bounds. */
function parseBucket(bucket: string): { start: number; end: number } {
  if (bucket.endsWith("+")) {
    const start = Number(bucket.slice(0, -1));
    return { start, end: start };
  }
  const dash = bucket.indexOf("-");
  if (dash === -1) {
    const value = Number(bucket);
    return { start: value, end: value };
  }
  return { start: Number(bucket.slice(0, dash)), end: Number(bucket.slice(dash + 1)) };
}

// --- Fetchers ---

export async function fetchClusterProfile(): Promise<{ rows: ClusterProfileRow[] }> {
  const raw = await fetchJson<RawProfileResponse>("/cluster-eda/profile");
  const rows: ClusterProfileRow[] = (raw.clusters ?? []).map((r) => ({
    cluster: r.ml_cluster ?? 0,
    n_dfus: r.n_dfus ?? 0,
    mean_demand: r.mean_demand ?? 0,
    cv: r.cv_demand ?? 0,
    zero_pct: r.zero_pct ?? 0,
    // The profile endpoint does not expose seasonal amplitude; surfaced as 0
    // until a dedicated seasonality column is added to the response.
    seasonal_amplitude: 0,
    accuracy_pct: r.accuracy_pct,
  }));
  return { rows };
}

export async function fetchErrorConcentration(): Promise<ErrorConcentration> {
  const raw = await fetchJson<RawErrorConcentrationResponse>("/cluster-eda/error-concentration");

  const topShare = raw.top_error_dfus?.top_10pct_share ?? 0;

  // The backend reports WAPE per month/cluster; rank descending so "worst" first.
  const worst_months = (raw.error_by_month ?? [])
    .map((m) => ({ month: monthLabel(m.month), error_share: m.wape ?? 0 }))
    .sort((a, b) => b.error_share - a.error_share);

  const worst_clusters = (raw.error_by_cluster ?? [])
    .map((c) => ({
      cluster: c.cluster ?? 0,
      error_share: c.wape ?? 0,
      // The error-by-cluster aggregation does not return a DFU count.
      n_dfus: 0,
    }))
    .sort((a, b) => b.error_share - a.error_share);

  return { top_10_pct_share: topShare, worst_months, worst_clusters };
}

export async function fetchClusterDistribution(
  id: number,
): Promise<{ cluster: number; bins: ClusterDistributionBin[] }> {
  const raw = await fetchJson<RawDistributionResponse>(`/cluster-eda/demand-distribution/${id}`);
  const bins: ClusterDistributionBin[] = (raw.histogram ?? []).map((h) => {
    const { start, end } = parseBucket(h.bucket);
    return { bin_start: start, bin_end: end, count: h.count };
  });
  return { cluster: raw.cluster_id ?? id, bins };
}

export async function fetchResidualAnalysis(
  modelId = "lgbm_cluster",
): Promise<{ model_id: string; clusters: ResidualPoint[] }> {
  const raw = await fetchJson<RawResidualResponse>(
    `/cluster-eda/residual-analysis?model_id=${modelId}`,
  );
  // The endpoint returns per-cluster bias and a single residual-stats summary;
  // project them onto the per-cluster residual contract.
  const stats = raw.residual_stats ?? { mean: null, std: null, skew: null, kurtosis: null };
  const clusters: ResidualPoint[] = (raw.bias_by_cluster ?? []).map((c) => ({
    cluster: c.cluster ?? 0,
    mean_residual: c.bias ?? 0,
    std_residual: stats.std ?? 0,
    skew: stats.skew ?? 0,
    n_dfus: 0,
  }));
  return { model_id: modelId, clusters };
}

export async function fetchSeasonalityHeatmap(): Promise<SeasonalityHeatmapResponse> {
  const raw = await fetchJson<RawSeasonalityResponse>("/cluster-eda/seasonality-heatmap");

  const months: string[] = (raw.months ?? []).map((m) => monthLabel(m));
  const rows: SeasonalityHeatmapRow[] = (raw.clusters ?? []).map((clusterId, idx) => ({
    cluster: clusterId,
    values: (raw.values[idx] ?? []).map((v) => v ?? 0),
  }));

  return { months, rows };
}
