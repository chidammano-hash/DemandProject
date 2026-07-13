import { fetchJson } from "./request";
import type {
  SkuClustersPayload,
  ClusterProfilesPayload,
} from "@/types";

// ---------------------------------------------------------------------------
// Clustering queries
// ---------------------------------------------------------------------------
export async function fetchSkuClusters(source: string): Promise<SkuClustersPayload> {
  return fetchJson(`/domains/sku/clusters?source=${source}`);
}

export async function fetchClusterProfiles(): Promise<ClusterProfilesPayload> {
  return fetchJson("/domains/sku/clusters/profiles");
}

export interface ClusteringDefaultsPayload {
  feature_params: { time_window_months: number; min_months_history: number };
  model_params: {
    k_range: number[];
    min_cluster_size_pct: number;
    use_pca: boolean;
    pca_components: number | null;
    all_features: boolean;
  };
  label_params: {
    volume_high: number;
    volume_low: number;
    cv_steady: number;
    cv_volatile: number;
    seasonality_threshold: number;
    zero_demand_threshold: number;
  };
}

export async function fetchClusteringDefaults(): Promise<ClusteringDefaultsPayload> {
  return fetchJson("/clustering/defaults");
}

/** The canonical clustering feature set (mirrors `GET /clustering/core-features`). U6.10. */
export interface ClusterCoreFeaturesPayload {
  features: string[];
}

export async function fetchClusterCoreFeatures(): Promise<ClusterCoreFeaturesPayload> {
  return fetchJson("/clustering/core-features");
}

export interface ClusteringScenarioParams {
  feature_params?: ClusteringDefaultsPayload["feature_params"];
  model_params?: ClusteringDefaultsPayload["model_params"];
  label_params?: ClusteringDefaultsPayload["label_params"];
  relabel_only?: boolean;
  previous_scenario_id?: string;
}

export interface ScenarioProfile {
  label: string;
  count: number;
  pct_of_total: number;
  mean_demand: number;
  cv_demand: number;
  seasonality_strength: number;
  trend_slope: number;
  growth_rate: number;
  zero_demand_pct: number;
}

export interface PCAScatterPoint {
  pc1: number;
  pc2: number;
  cluster: number;
}

export interface PCAScatterData {
  pc1_variance: number;
  pc2_variance: number;
  points: PCAScatterPoint[];
}

export interface ClusteringScenarioResult {
  scenario_id: string;
  status: "completed" | "failed";
  runtime_seconds: number;
  params: Record<string, unknown>;
  result: {
    optimal_k: number;
    silhouette_score: number;
    inertia: number;
    total_dfus: number;
    total_skus?: number;
    k_selection_results: {
      k_values: number[];
      inertias: number[];
      silhouette_scores: number[];
      ch_scores?: number[];
      combined_scores?: number[];
      feasible_mask?: boolean[];
    };
    profiles: ScenarioProfile[];
    feature_importance?: { feature: string; variance_ratio: number }[];
    pca_scatter?: PCAScatterData;
  } | null;
  error?: string | null;
}

export interface ScenarioEstimate {
  estimated_seconds: number;
  sku_count: number;
  training_sample: number;
  sampled: boolean;
  k_range: number;
}

export interface ScenarioStatusResponse {
  scenario_id: string;
  status: "running" | "completed" | "failed";
  elapsed_seconds?: number;
  runtime_seconds?: number;
  result?: ClusteringScenarioResult;
  error?: string;
}

export async function fetchScenarioEstimate(params: {
  k_min: number;
  k_max: number;
}): Promise<ScenarioEstimate> {
  const qs = new URLSearchParams({
    k_min: String(params.k_min),
    k_max: String(params.k_max),
  });
  return fetchJson(`/clustering/scenario/estimate?${qs}`);
}

export async function fetchScenarioStatus(scenarioId: string): Promise<ScenarioStatusResponse> {
  return fetchJson(`/clustering/scenario/${encodeURIComponent(scenarioId)}/status`);
}

export async function runClusteringScenario(
  params: ClusteringScenarioParams,
): Promise<{ scenario_id: string; status: string; job_id?: string }> {
  return fetchJson("/clustering/scenario", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export async function promoteScenario(scenarioId: string): Promise<unknown> {
  return fetchJson(`/clustering/scenario/${encodeURIComponent(scenarioId)}/promote`, {
    method: "POST",
  });
}
