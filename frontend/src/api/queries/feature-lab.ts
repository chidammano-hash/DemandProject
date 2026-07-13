/**
 * Feature Lab — API types, query keys, and fetchers for feature importance and stability analysis.
 *
 * NOTE: The fetchers normalise backend response shapes into the types below,
 * so component code can rely on a stable contract. The `Raw*` interfaces below
 * mirror the actual FastAPI response shapes from
 * `api/routers/forecasting/feature_lab.py`.
 */

import { fetchJson } from "./request";

// --- Types (frontend contract) ---

export interface FeatureImportanceRow {
  feature: string;
  shap_value: number;
  category: string;
  rank: number;
}

export interface FeatureStabilityRow {
  feature: string;
  mean_rank: number;
  rank_std: number;
  stability: "stable" | "moderate" | "unstable";
  n_folds: number;
}

export interface FeatureCorrelationCell {
  feature_a: string;
  feature_b: string;
  correlation: number;
}

export interface ClusterFeatureImportance {
  cluster: number;
  features: Array<{ feature: string; shap_value: number; rank: number }>;
}

export interface FeatureCategory {
  category: string;
  color: string;
  count: number;
}

// --- Raw wire shapes (mirror api/routers/forecasting/feature_lab.py) ---

/** One feature entry from GET /feature-lab/importance. */
interface RawImportanceFeature {
  name: string;
  mean_abs_shap: number;
  rank: number;
  selected_count: number;
  n_timeframes: number;
  category: string;
}

interface RawImportanceResponse {
  available: boolean;
  model_id: string;
  features: RawImportanceFeature[];
  total_features: number;
  selected_features: number;
}

/** One feature entry from GET /feature-lab/stability. */
interface RawStabilityFeature {
  name: string;
  ranks_by_timeframe: number[];
  mean_rank: number;
  rank_std: number;
  stability: "high" | "medium" | "unstable";
  min_rank: number;
  max_rank: number;
}

interface RawStabilityResponse {
  available: boolean;
  features: RawStabilityFeature[];
}

/** GET /feature-lab/correlation — features list plus a square correlation matrix. */
interface RawCorrelationResponse {
  available: boolean;
  features: string[];
  matrix: number[][];
  high_correlation_pairs: Array<{
    feature_a: string;
    feature_b: string;
    correlation: number;
    recommendation: string;
  }>;
}

/** GET /feature-lab/per-cluster-importance — cluster ids are returned as strings. */
interface RawPerClusterResponse {
  available: boolean;
  clusters: string[];
  features: string[];
  importance_matrix: number[][];
  cluster_specific_features: Array<{
    cluster: string;
    top_unique_feature: string;
    note: string;
  }>;
  note?: string;
}

/** One category entry from GET /feature-lab/categories. */
interface RawCategory {
  name: string;
  features: string[];
  description: string;
  count: number;
}

interface RawCategoriesResponse {
  available: boolean;
  model_id: string;
  categories: RawCategory[];
  total_features: number;
}

// --- Query keys ---

export const featureLabKeys = {
  all: ["feature-lab"] as const,
  importance: (modelId?: string) => [...featureLabKeys.all, "importance", modelId ?? "default"] as const,
  stability: () => [...featureLabKeys.all, "stability"] as const,
  correlation: (topN?: number) => [...featureLabKeys.all, "correlation", topN ?? 20] as const,
  clusterImportance: (cluster: number) => [...featureLabKeys.all, "cluster-importance", cluster] as const,
  categories: () => [...featureLabKeys.all, "categories"] as const,
};

// --- Default category colours (used when API doesn't provide colour) ---

const DEFAULT_CATEGORY_COLORS: Record<string, string> = {
  lag: "#2563EB",
  rolling: "#0D9488",
  seasonal: "#D97706",
  calendar: "#0891B2",
  cluster: "#EC4899",
  external: "#84CC16",
  demand: "#f97316",
  profile: "#8B5CF6",
  derived: "#06B6D4",
  categorical: "#F43F5E",
  fourier: "#D97706",
  croston: "#10B981",
  cross_dfu: "#6366F1",
  other: "#64748B",
};

// --- Stability mapping (API uses "high"/"medium"/"unstable"; frontend uses "stable"/"moderate"/"unstable") ---

function mapStability(s: string): "stable" | "moderate" | "unstable" {
  const lower = s.toLowerCase();
  if (lower === "high" || lower === "stable") return "stable";
  if (lower === "medium" || lower === "moderate") return "moderate";
  return "unstable";
}

// --- Fetchers ---

export async function fetchFeatureImportance(
  modelId?: string,
): Promise<{ features: FeatureImportanceRow[]; model_id: string }> {
  const sp = new URLSearchParams();
  if (modelId) sp.set("model_id", modelId);
  const raw = await fetchJson<RawImportanceResponse>(
    `/feature-lab/importance${sp.toString() ? `?${sp}` : ""}`,
  );
  const features: FeatureImportanceRow[] = (raw.features ?? []).map((f) => ({
    feature: f.name,
    shap_value: f.mean_abs_shap,
    category: f.category,
    rank: f.rank,
  }));
  return { features, model_id: raw.model_id ?? modelId ?? "lgbm_cluster" };
}

export async function fetchFeatureStability(): Promise<{ features: FeatureStabilityRow[] }> {
  const raw = await fetchJson<RawStabilityResponse>("/feature-lab/stability");
  const features: FeatureStabilityRow[] = (raw.features ?? []).map((f) => ({
    feature: f.name,
    mean_rank: f.mean_rank,
    rank_std: f.rank_std,
    stability: mapStability(f.stability),
    n_folds: Array.isArray(f.ranks_by_timeframe) ? f.ranks_by_timeframe.length : 0,
  }));
  return { features };
}

export async function fetchFeatureCorrelation(
  topN = 20,
): Promise<{ top_n: number; features: string[]; cells: FeatureCorrelationCell[] }> {
  const raw = await fetchJson<RawCorrelationResponse>(`/feature-lab/correlation?top_n=${topN}`);
  const features: string[] = raw.features ?? [];

  // Backend returns a square correlation matrix; flatten the upper triangle to non-zero cells.
  const matrix = raw.matrix ?? [];
  const cells: FeatureCorrelationCell[] = [];
  for (let i = 0; i < features.length; i++) {
    for (let j = i + 1; j < features.length; j++) {
      const corr = matrix[i]?.[j] ?? 0;
      if (corr !== 0) {
        cells.push({ feature_a: features[i], feature_b: features[j], correlation: corr });
      }
    }
  }

  return { top_n: features.length, features, cells };
}

export async function fetchClusterFeatureImportance(
  cluster: number,
): Promise<ClusterFeatureImportance> {
  const raw = await fetchJson<RawPerClusterResponse>("/feature-lab/per-cluster-importance");

  // Backend returns {clusters, features, importance_matrix} with string cluster ids —
  // extract the row for the requested cluster.
  const clusterStr = String(cluster);
  const clusterIdx = (raw.clusters ?? []).indexOf(clusterStr);

  if (clusterIdx === -1 || !raw.importance_matrix?.[clusterIdx]) {
    return { cluster, features: [] };
  }

  const featureNames: string[] = raw.features ?? [];
  const importanceRow: number[] = raw.importance_matrix[clusterIdx];

  const features = featureNames
    .map((name, i) => ({
      feature: name,
      shap_value: importanceRow[i] ?? 0,
      rank: 0,
    }))
    .filter((f) => f.shap_value > 0)
    .sort((a, b) => b.shap_value - a.shap_value);

  features.forEach((f, i) => {
    f.rank = i + 1;
  });

  return { cluster, features };
}

export async function fetchFeatureCategories(): Promise<{ categories: FeatureCategory[] }> {
  const raw = await fetchJson<RawCategoriesResponse>("/feature-lab/categories");
  const categories: FeatureCategory[] = (raw.categories ?? []).map((c) => ({
    category: c.name,
    color: DEFAULT_CATEGORY_COLORS[c.name] ?? "#64748B",
    count: c.count,
  }));
  return { categories };
}
