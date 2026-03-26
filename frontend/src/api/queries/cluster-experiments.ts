/**
 * Cluster Experiments API — Cluster Experimentation Studio
 *
 * Query module for cluster experiment CRUD, comparison, and promotion.
 * Follows the same pattern as unified-model-tuning.ts.
 */

import { buildSearchParams } from "./helpers";
import type { PCAScatterData } from "./core";

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

export interface FeatureParams {
  time_window_months: number;
  min_months_history: number;
}

export interface ModelParams {
  k_range: [number, number];
  min_cluster_size_pct: number;
  use_pca: boolean;
  pca_components: number | null;
  all_features?: boolean;
}

export interface LabelParams {
  volume_high?: number;
  volume_low?: number;
  cv_steady?: number;
  cv_volatile?: number;
  seasonality_threshold?: number;
  zero_demand_threshold?: number;
}

export interface ClusterProfile {
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

export interface KSelectionResults {
  k_values: number[];
  inertias: number[];
  silhouette_scores: number[];
  ch_scores?: number[];
  combined_scores?: number[];
  feasible_mask?: boolean[];
  pca_scatter?: PCAScatterData;
}

export type ClusterExperimentStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface ClusterExperiment {
  experiment_id: number;
  scenario_id: string;
  label: string;
  notes: string | null;
  template_id: string | null;
  status: ClusterExperimentStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  runtime_seconds: number | null;
  job_id: string | null;
  feature_params: FeatureParams | null;
  model_params: ModelParams | null;
  label_params: LabelParams | null;
  optimal_k: number | null;
  silhouette_score: number | null;
  inertia: number | null;
  total_dfus: number | null;
  n_clusters: number | null;
  cluster_sizes: Record<string, number> | null;
  profiles: ClusterProfile[] | null;
  k_selection_results: KSelectionResults | null;
  is_promoted: boolean;
  promoted_at: string | null;
  artifacts_path: string | null;
}

// ---------------------------------------------------------------------------
// Comparison types
// ---------------------------------------------------------------------------

export interface ClusterExperimentComparison {
  experiment_a: ClusterExperiment;
  experiment_b: ClusterExperiment;
  quality_comparison: {
    silhouette_delta: number;
    inertia_delta: number;
    k_delta: number;
    verdict: "improved" | "degraded" | "mixed" | "neutral";
  };
  profile_comparison: {
    clusters_only_in_a: string[];
    clusters_only_in_b: string[];
    common_clusters: Array<{
      label: string;
      count_a: number;
      count_b: number;
      delta_count: number;
    }>;
  };
  migration_matrix: Record<string, Record<string, number>>;
  total_dfus_migrated: number;
  total_dfus_unchanged: number;
}

// ---------------------------------------------------------------------------
// Template types
// ---------------------------------------------------------------------------

export interface ClusterExperimentTemplate {
  id: string;
  label: string;
  description: string;
  source?: string;
  feature_params?: Partial<FeatureParams>;
  model_params?: Partial<ModelParams>;
  label_params?: Partial<LabelParams>;
}

// ---------------------------------------------------------------------------
// Request / Response payloads
// ---------------------------------------------------------------------------

export interface CreateClusterExperimentPayload {
  label: string;
  notes?: string;
  template?: string;
  feature_params?: Partial<FeatureParams>;
  model_params?: Partial<ModelParams>;
  label_params?: Partial<LabelParams>;
}

// ---------------------------------------------------------------------------
// Query key factory
// ---------------------------------------------------------------------------

export const clusterExperimentKeys = {
  all: ["cluster-experiments"] as const,
  experiments: (params?: Record<string, unknown>) =>
    ["cluster-experiments", "list", params] as const,
  experiment: (id: number) =>
    ["cluster-experiments", "detail", id] as const,
  compare: (aId: number, bId: number) =>
    ["cluster-experiments", "compare", aId, bId] as const,
  templates: () =>
    ["cluster-experiments", "templates"] as const,
  completed: () =>
    ["cluster-experiments", "completed"] as const,
  usedBy: (id: number) =>
    ["cluster-experiments", "used-by", id] as const,
};

// ---------------------------------------------------------------------------
// Stale times (ms)
// ---------------------------------------------------------------------------

export const CLUSTER_EXP_STALE = {
  EXPERIMENTS: 10_000,    // 10s — active experiments refresh frequently
  EXPERIMENT: 30_000,     // 30s — single experiment detail
  COMPARE: 120_000,       // 2min — comparison data is expensive, cache longer
  TEMPLATES: 600_000,     // 10min — templates rarely change
  COMPLETED: 300_000,     // 5min — completed list for algorithm tuning dropdown
  USED_BY: 60_000,        // 1min — algorithm experiments referencing this cluster
} as const;

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

const BASE = "/cluster-experiments";

async function fetchOrThrow<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Fetchers — READ
// ---------------------------------------------------------------------------

/** List cluster experiments with optional filters. */
export async function fetchClusterExperiments(
  params?: {
    status?: string;
    page?: number;
    page_size?: number;
  },
): Promise<{ experiments: ClusterExperiment[]; total: number }> {
  const sp = buildSearchParams({
    status: params?.status,
    page: params?.page,
    page_size: params?.page_size,
  });
  const qs = sp.toString();
  return fetchOrThrow(`${BASE}${qs ? `?${qs}` : ""}`, {
    cache: "no-cache",
  });
}

/** Get a single cluster experiment with full detail. */
export async function fetchClusterExperiment(
  id: number,
): Promise<ClusterExperiment> {
  return fetchOrThrow(`${BASE}/${id}`);
}

/** Compare two cluster experiments (quality, profiles, migration matrix). */
export async function fetchClusterComparison(
  aId: number,
  bId: number,
): Promise<ClusterExperimentComparison> {
  const sp = buildSearchParams({ a_id: aId, b_id: bId });
  return fetchOrThrow(`${BASE}/compare?${sp}`);
}

/** Get available cluster experiment templates. */
export async function fetchClusterTemplates(): Promise<{
  templates: ClusterExperimentTemplate[];
}> {
  return fetchOrThrow(`${BASE}/templates`);
}

/** List only completed experiments (for algorithm tuning cluster source dropdown). */
export async function fetchCompletedClusterExperiments(): Promise<{
  experiments: ClusterExperiment[];
}> {
  return fetchOrThrow(`${BASE}/completed`, { cache: "no-cache" });
}

/** List algorithm tuning experiments that reference this cluster experiment. */
export async function fetchClusterExperimentUsedBy(
  id: number,
): Promise<{ runs: Array<{ run_id: number; run_label: string; model_id: string; status: string }> }> {
  return fetchOrThrow(`${BASE}/${id}/used-by`);
}

// ---------------------------------------------------------------------------
// Fetchers — WRITE
// ---------------------------------------------------------------------------

/** Create and launch a new cluster experiment. */
export async function createClusterExperiment(
  payload: CreateClusterExperimentPayload,
): Promise<{
  experiment_id: number;
  scenario_id: string;
  status: string;
  job_id: string;
}> {
  return fetchOrThrow(BASE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/** Delete a completed, failed, or cancelled cluster experiment. */
export async function deleteClusterExperiment(
  id: number,
): Promise<{ deleted: boolean }> {
  return fetchOrThrow(`${BASE}/${id}`, {
    method: "DELETE",
  });
}

/** Promote a completed cluster experiment to production (writes to dim_sku.ml_cluster). */
export async function promoteClusterExperiment(
  id: number,
): Promise<{ status: string; dfus_updated: number }> {
  return fetchOrThrow(`${BASE}/${id}/promote`, {
    method: "POST",
  });
}
