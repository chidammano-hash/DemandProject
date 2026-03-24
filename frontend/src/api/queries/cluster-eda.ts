/**
 * Cluster EDA — API types, query keys, and fetchers for cluster exploratory data analysis.
 *
 * NOTE: Fetchers normalise backend response shapes into the types below.
 */

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

// --- Query keys ---

export const clusterEdaKeys = {
  all: ["cluster-eda"] as const,
  profile: () => [...clusterEdaKeys.all, "profile"] as const,
  errorConcentration: () => [...clusterEdaKeys.all, "error-concentration"] as const,
  distribution: (id: number) => [...clusterEdaKeys.all, "distribution", id] as const,
  residuals: (modelId: string) => [...clusterEdaKeys.all, "residuals", modelId] as const,
  seasonalityHeatmap: () => [...clusterEdaKeys.all, "seasonality-heatmap"] as const,
};

// --- Fetchers ---

export async function fetchClusterProfile(): Promise<{ rows: ClusterProfileRow[] }> {
  const res = await fetch("/cluster-eda/profile");
  if (!res.ok) throw new Error(`fetchClusterProfile: ${res.status}`);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await res.json();
  // API returns {clusters: [...]} or {rows: [...]}
  const rows: ClusterProfileRow[] = (raw.rows ?? raw.clusters ?? []).map(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (r: any) => ({
      cluster: r.cluster ?? r.ml_cluster ?? 0,
      n_dfus: r.n_dfus ?? 0,
      mean_demand: r.mean_demand ?? 0,
      cv: r.cv ?? r.cv_demand ?? 0,
      zero_pct: r.zero_pct ?? r.zero_demand_pct ?? 0,
      seasonal_amplitude: r.seasonal_amplitude ?? 0,
      accuracy_pct: r.accuracy_pct ?? null,
    }),
  );
  return { rows };
}

export async function fetchErrorConcentration(): Promise<ErrorConcentration> {
  const res = await fetch("/cluster-eda/error-concentration");
  if (!res.ok) throw new Error(`fetchErrorConcentration: ${res.status}`);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await res.json();

  // API returns {top_error_dfus: {top_10pct_share}, error_by_month: [...], error_by_cluster: [...]}
  const topShare =
    raw.top_10_pct_share ??
    raw.top_error_dfus?.top_10pct_share ??
    raw.top_error_dfus?.top_10_pct_share ??
    0;

  const worst_months = (raw.worst_months ?? raw.error_by_month ?? []).map(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (m: any) => ({
      month: m.month ?? "",
      error_share: m.error_share ?? m.wape ?? 0,
    }),
  );

  const worst_clusters = (raw.worst_clusters ?? raw.error_by_cluster ?? []).map(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (c: any) => ({
      cluster: c.cluster ?? c.ml_cluster ?? 0,
      error_share: c.error_share ?? c.wape ?? 0,
      n_dfus: c.n_dfus ?? 0,
    }),
  );

  return { top_10_pct_share: topShare, worst_months, worst_clusters };
}

export async function fetchClusterDistribution(
  id: number,
): Promise<{ cluster: number; bins: ClusterDistributionBin[] }> {
  const res = await fetch(`/cluster-eda/demand-distribution/${id}`);
  if (!res.ok) throw new Error(`fetchClusterDistribution: ${res.status}`);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await res.json();
  return {
    cluster: raw.cluster ?? id,
    bins: raw.bins ?? [],
  };
}

export async function fetchResidualAnalysis(
  modelId = "lgbm_cluster",
): Promise<{ model_id: string; clusters: ResidualPoint[] }> {
  const res = await fetch(`/cluster-eda/residual-analysis?model_id=${modelId}`);
  if (!res.ok) throw new Error(`fetchResidualAnalysis: ${res.status}`);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await res.json();
  return {
    model_id: raw.model_id ?? modelId,
    clusters: raw.clusters ?? [],
  };
}

export async function fetchSeasonalityHeatmap(): Promise<SeasonalityHeatmapResponse> {
  const res = await fetch("/cluster-eda/seasonality-heatmap");
  if (!res.ok) throw new Error(`fetchSeasonalityHeatmap: ${res.status}`);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await res.json();

  // API returns {clusters: number[], months: string[], values: number[][]}
  const months: string[] = raw.months ?? [];
  let rows: SeasonalityHeatmapRow[] = [];

  if (Array.isArray(raw.rows)) {
    rows = raw.rows;
  } else if (Array.isArray(raw.clusters) && Array.isArray(raw.values)) {
    rows = raw.clusters.map((clusterId: number, idx: number) => ({
      cluster: clusterId,
      values: raw.values[idx] ?? [],
    }));
  }

  return { months, rows };
}
