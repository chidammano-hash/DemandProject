/**
 * Feature Lab — API types, query keys, and fetchers for feature importance and stability analysis.
 *
 * NOTE: The fetchers normalise backend response shapes into the types below,
 * so component code can rely on a stable contract.
 */

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

// --- Stability mapping (API uses "high"/"medium"/"low"; frontend uses "stable"/"moderate"/"unstable") ---

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
  const res = await fetch(`/feature-lab/importance${sp.toString() ? `?${sp}` : ""}`);
  if (!res.ok) throw new Error(`fetchFeatureImportance: ${res.status}`);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await res.json();
  const features: FeatureImportanceRow[] = (raw.features ?? []).map(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (f: any) => ({
      feature: f.feature ?? f.name ?? "",
      shap_value: f.shap_value ?? f.mean_abs_shap ?? 0,
      category: f.category ?? "other",
      rank: f.rank ?? 0,
    }),
  );
  return { features, model_id: raw.model_id ?? modelId ?? "lgbm_cluster" };
}

export async function fetchFeatureStability(): Promise<{ features: FeatureStabilityRow[] }> {
  const res = await fetch("/feature-lab/stability");
  if (!res.ok) throw new Error(`fetchFeatureStability: ${res.status}`);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await res.json();
  const features: FeatureStabilityRow[] = (raw.features ?? []).map(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (f: any) => ({
      feature: f.feature ?? f.name ?? "",
      mean_rank: f.mean_rank ?? 0,
      rank_std: f.rank_std ?? 0,
      stability: mapStability(f.stability ?? "low"),
      n_folds: f.n_folds ?? (Array.isArray(f.ranks_by_timeframe) ? f.ranks_by_timeframe.length : 0),
    }),
  );
  return { features };
}

export async function fetchFeatureCorrelation(
  topN = 20,
): Promise<{ top_n: number; features: string[]; cells: FeatureCorrelationCell[] }> {
  const res = await fetch(`/feature-lab/correlation?top_n=${topN}`);
  if (!res.ok) throw new Error(`fetchFeatureCorrelation: ${res.status}`);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await res.json();
  const features: string[] = raw.features ?? [];

  // Convert matrix (2D array) → cells array
  let cells: FeatureCorrelationCell[] = [];
  if (Array.isArray(raw.cells)) {
    cells = raw.cells;
  } else if (Array.isArray(raw.matrix)) {
    for (let i = 0; i < features.length; i++) {
      for (let j = i + 1; j < features.length; j++) {
        const corr = raw.matrix[i]?.[j] ?? 0;
        if (corr !== 0) {
          cells.push({ feature_a: features[i], feature_b: features[j], correlation: corr });
        }
      }
    }
  }

  return { top_n: features.length, features, cells };
}

export async function fetchClusterFeatureImportance(
  cluster: number,
): Promise<ClusterFeatureImportance> {
  const res = await fetch("/feature-lab/per-cluster-importance");
  if (!res.ok) throw new Error(`fetchClusterFeatureImportance: ${res.status}`);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await res.json();

  // Backend returns {clusters, features, importance_matrix} — extract the requested cluster row
  const clusterStr = String(cluster);
  const clusterIdx = (raw.clusters ?? []).indexOf(clusterStr);

  if (clusterIdx === -1 || !raw.importance_matrix?.[clusterIdx]) {
    return { cluster, features: [] };
  }

  const featureNames: string[] = raw.features ?? [];
  const importanceRow: number[] = raw.importance_matrix[clusterIdx];

  const features = featureNames
    .map((name: string, i: number) => ({
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
  const res = await fetch("/feature-lab/categories");
  if (!res.ok) throw new Error(`fetchFeatureCategories: ${res.status}`);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raw: any = await res.json();
  const categories: FeatureCategory[] = (raw.categories ?? []).map(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (c: any) => ({
      category: c.category ?? c.name ?? "other",
      color: c.color ?? DEFAULT_CATEGORY_COLORS[c.category ?? c.name ?? "other"] ?? "#64748B",
      count: c.count ?? 0,
    }),
  );
  return { categories };
}
