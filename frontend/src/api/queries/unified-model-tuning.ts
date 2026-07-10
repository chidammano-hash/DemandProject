/**
 * Unified Model Tuning API — Feature 46
 *
 * Query module for LightGBM tuning experiments.
 * Replaces the split between lgbm-tuning.ts and model-tuning.ts with a
 * parametrized `/model-tuning/{model}/` prefix.
 */

import type { TuningComparison } from "./lgbm-tuning";
import { fetchJson } from "./core";
import { buildSearchParams } from "./helpers";

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

export type ModelType = "lgbm";

/** API path prefix per tree-model family (e.g. "/model-tuning/lgbm"). */
export const MODEL_PREFIX: Record<ModelType, string> = {
  lgbm: "/model-tuning/lgbm",
};

export interface TuningExperiment {
  run_id: number;
  run_label: string;
  model_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  n_predictions: number | null;
  n_dfus: number | null;
  params: Record<string, unknown> | null;
  features: string[] | null;
  feature_count: number | null;
  notes: string | null;
  started_at: string | null;
  completed_at: string | null;
  is_promoted: boolean;
  promoted_at: string | null;
  is_results_promoted: boolean;
  results_promoted_at: string | null;
  results_promote_job_id: string | null;
  job_id: string | null;
  template_id: string | null;
  metadata: Record<string, unknown> | null;
  cluster_source: "production" | "experimental";
  cluster_experiment_id: number | null;
  cluster_experiment_label: string | null;
}

export interface TuningLag {
  exec_lag: number;
  n_predictions: number;
  n_dfus: number;
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
}

// ---------------------------------------------------------------------------
// Comparison types
// ---------------------------------------------------------------------------

export interface TuningLagComparison {
  exec_lag: number;
  baseline_acc: number;
  candidate_acc: number;
  delta_acc: number;
  baseline_wape: number;
  candidate_wape: number;
  delta_wape: number;
  baseline_bias: number;
  candidate_bias: number;
  delta_bias: number;
}

interface UnifiedClusterComparison {
  cluster: string;
  baseline_accuracy: number | null;
  candidate_accuracy: number | null;
  delta_accuracy: number | null;
  baseline_wape: number | null;
  candidate_wape: number | null;
  baseline_n_dfus: number | null;
  candidate_n_dfus: number | null;
}

interface UnifiedMonthComparison {
  month: string;
  baseline_accuracy: number | null;
  candidate_accuracy: number | null;
  delta_accuracy: number | null;
  baseline_wape: number | null;
  candidate_wape: number | null;
}

interface UnifiedTimeframeComparison {
  timeframe: string;
  baseline_accuracy: number | null;
  candidate_accuracy: number | null;
  delta_accuracy: number | null;
}

interface UnifiedParamDiff {
  param: string;
  baseline: unknown;
  candidate: unknown;
}

interface UnifiedFeatureDiffs {
  baseline_count: number;
  candidate_count: number;
  added: string[];
  removed: string[];
  common_count: number;
}

interface UnifiedConfigDiff {
  setting: string;
  baseline: unknown;
  candidate: unknown;
}

export interface UnifiedTuningComparison {
  baseline: TuningExperiment;
  candidate: TuningExperiment;
  delta_accuracy: number;
  delta_wape: number;
  delta_bias: number;
  verdict: "improved" | "degraded" | "neutral";
  per_lag: TuningLagComparison[];
  per_cluster: {
    ml_cluster: UnifiedClusterComparison[];
    business_cluster: UnifiedClusterComparison[];
  };
  per_month: UnifiedMonthComparison[];
  per_timeframe: UnifiedTimeframeComparison[];
  param_diffs: UnifiedParamDiff[];
  param_commons: UnifiedParamDiff[];
  feature_diffs: UnifiedFeatureDiffs;
  config_diffs: UnifiedConfigDiff[];
}

// ---------------------------------------------------------------------------
// Template types
// ---------------------------------------------------------------------------

export interface ExperimentTemplate {
  id: string;
  label: string;
  description: string;
  params: Record<string, unknown>;
  config: Record<string, unknown>;
  source: "algorithm_config" | "expert" | "custom";
}

// ---------------------------------------------------------------------------
// Request / Response payloads
// ---------------------------------------------------------------------------

export interface CreateExperimentPayload {
  run_label: string;
  notes?: string;
  template?: string;
  params: Record<string, unknown>;
  config: Record<string, unknown>;
}

export interface CreateExperimentResponse {
  run_id: number;
  job_id: string;
  status: string;
  model: string;
  run_label: string;
  message: string;
}

export interface ExperimentLogsResponse {
  run_id: number;
  model?: string;
  log: string;
  offset: number;
  next_offset: number;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "unknown";
  has_more?: boolean;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface ModelLagAccuracy {
  exec_lag: number;
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
}

export interface ModelLagComparison {
  exec_lag: number;
  baseline_accuracy: number | null;
  candidate_accuracy: number | null;
  delta_accuracy: number | null;
  baseline_wape: number | null;
  candidate_wape: number | null;
}

export interface PromotionLogEntry {
  id: number;
  run_id: number;
  model_id: string;
  promoted_at: string;
  promoted_by: string | null;
  previous_run_id: number | null;
  params_written: Record<string, unknown>;
  accuracy_pct: number | null;
  notes: string | null;
}

// ---------------------------------------------------------------------------
// Query key factory
// ---------------------------------------------------------------------------

export const modelTuningKeys = {
  all: ["model-tuning"] as const,
  experiments: (model: ModelType, params?: Record<string, unknown>) =>
    ["model-tuning", model, "experiments", params] as const,
  experiment: (model: ModelType, runId: number) =>
    ["model-tuning", model, "experiment", runId] as const,
  // Aliases used by the LightGBM tuning tab.
  runs: (model: ModelType, params?: Record<string, unknown>) =>
    ["model-tuning", model, "experiments", params] as const,
  run: (model: ModelType, runId: number) => ["model-tuning", model, "experiment", runId] as const,
  lags: (model: ModelType, runId: number) => ["model-tuning", model, "lags", runId] as const,
  clusters: (model: ModelType, runId: number, execLag?: number) =>
    ["model-tuning", model, "clusters", runId, execLag] as const,
  months: (model: ModelType, runId: number) => ["model-tuning", model, "months", runId] as const,
  logs: (model: ModelType, runId: number, offset?: number) =>
    ["model-tuning", model, "logs", runId, offset] as const,
  compare: (model: ModelType, baselineId: number, candidateId: number, execLag?: number) =>
    ["model-tuning", model, "compare", baselineId, candidateId, execLag] as const,
  promoted: (model: ModelType) => ["model-tuning", model, "promoted"] as const,
  templates: (model: ModelType) => ["model-tuning", model, "templates"] as const,
  promotions: (model: ModelType) => ["model-tuning", model, "promotions"] as const,
  promoteResultsStatus: (model: ModelType, runId: number) =>
    ["model-tuning", model, "promote-results-status", runId] as const,
};

// ---------------------------------------------------------------------------
// Stale times (ms)
// ---------------------------------------------------------------------------

export const MODEL_TUNING_STALE = {
  EXPERIMENTS: 10_000, // 10s — active runs refresh frequently
  EXPERIMENT: 30_000, // 30s — single experiment detail
  LAGS: 60_000, // 1min — lag data is static after completion
  COMPARE: 60_000, // 1min — comparison data
  TEMPLATES: 600_000, // 10min — templates rarely change
  LOGS: 2_000, // 2s — log polling interval
  PROMOTED: 60_000, // 1min — promoted run
} as const;

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

function prefix(model: ModelType): string {
  return `/model-tuning/${model}`;
}

async function fetchOrThrow<T>(url: string, init?: RequestInit): Promise<T> {
  return fetchJson<T>(url, init);
}

// ---------------------------------------------------------------------------
// Fetchers — READ
// ---------------------------------------------------------------------------

/** List experiments for a model with optional filters. */
export async function fetchModelExperiments(
  model: ModelType,
  params?: {
    status?: string;
    exec_lag?: number;
    page?: number;
    page_size?: number;
  }
): Promise<{ experiments: TuningExperiment[]; total: number }> {
  const sp = buildSearchParams({
    status: params?.status,
    exec_lag: params?.exec_lag,
    page: params?.page,
    page_size: params?.page_size,
  });
  const qs = sp.toString();
  return fetchOrThrow(`${prefix(model)}/experiments${qs ? `?${qs}` : ""}`, {
    cache: "no-cache",
  });
}

/** Get a single experiment with full detail. */
export async function fetchModelExperiment(
  model: ModelType,
  runId: number
): Promise<TuningExperiment> {
  return fetchOrThrow(`${prefix(model)}/experiments/${runId}`);
}

/** Per-execution-lag accuracy breakdown for a run. */
export async function fetchModelExperimentLags(
  model: ModelType,
  runId: number
): Promise<{ run_id: number; model: string; lags: TuningLag[] }> {
  return fetchOrThrow(`${prefix(model)}/experiments/${runId}/lags`);
}

/** Per-cluster accuracy for a run, optionally filtered by exec_lag. */
export async function fetchModelExperimentClusters(
  model: ModelType,
  runId: number,
  execLag?: number
): Promise<{ run_id: number; clusters: UnifiedClusterComparison[] }> {
  const sp = buildSearchParams({ exec_lag: execLag });
  const qs = sp.toString();
  return fetchOrThrow(`${prefix(model)}/experiments/${runId}/clusters${qs ? `?${qs}` : ""}`);
}

/** Per-month accuracy for a run, optionally filtered by exec_lag. */
export async function fetchModelExperimentMonths(
  model: ModelType,
  runId: number
): Promise<{ run_id: number; months: UnifiedMonthComparison[] }> {
  return fetchOrThrow(`${prefix(model)}/experiments/${runId}/months`);
}

/** Incremental log streaming (offset-based polling). */
export async function fetchModelExperimentLogs(
  model: ModelType,
  runId: number,
  offset?: number
): Promise<ExperimentLogsResponse> {
  const sp = buildSearchParams({ offset });
  const qs = sp.toString();
  return fetchOrThrow(`${prefix(model)}/experiments/${runId}/logs${qs ? `?${qs}` : ""}`, {
    cache: "no-cache",
  });
}

/** Compare two runs with lag-level deltas, optionally for a specific exec_lag. */
export async function fetchTuningComparison2(
  model: ModelType,
  baselineId: number,
  candidateId: number,
  execLag?: number
): Promise<UnifiedTuningComparison> {
  const sp = buildSearchParams({
    baseline_id: baselineId,
    candidate_id: candidateId,
    exec_lag: execLag,
  });
  return fetchOrThrow(`${prefix(model)}/compare?${sp}`);
}

/** Compare runs with the lag shape expected by the model-tuning panels. */
export async function fetchUnifiedModelTuningComparison(
  model: ModelType,
  baselineId: number,
  candidateId: number,
  execLag?: number
): Promise<TuningComparison & { per_lag?: ModelLagComparison[] }> {
  const data = await fetchTuningComparison2(model, baselineId, candidateId, execLag);
  return {
    ...data,
    per_lag: data.per_lag?.map((lag) => ({
      exec_lag: lag.exec_lag,
      baseline_accuracy: lag.baseline_acc,
      candidate_accuracy: lag.candidate_acc,
      delta_accuracy: lag.delta_acc,
      baseline_wape: lag.baseline_wape,
      candidate_wape: lag.candidate_wape,
    })),
  } as unknown as TuningComparison & { per_lag?: ModelLagComparison[] };
}

/** Convenience fetcher for panels that only need lag accuracy rows. */
export async function fetchModelLagAccuracy(
  model: ModelType,
  runId: number
): Promise<ModelLagAccuracy[]> {
  const data = await fetchModelExperimentLags(model, runId);
  return data.lags;
}

/** Get available experiment templates for a model. */
export async function fetchModelTemplates(
  model: ModelType
): Promise<{ model: string; templates: ExperimentTemplate[] }> {
  return fetchOrThrow(`${prefix(model)}/templates`);
}

/** Get the currently promoted (champion) run for a model. */
export async function fetchModelPromoted(
  model: ModelType
): Promise<{ promoted: TuningExperiment | null }> {
  return fetchOrThrow(`${prefix(model)}/promoted`, { cache: "no-cache" });
}

/** List the promotion audit trail for a model. */
export async function fetchModelPromotions(
  model: ModelType
): Promise<{ promotions: PromotionLogEntry[] }> {
  return fetchOrThrow(`${prefix(model)}/promotions`);
}

// ---------------------------------------------------------------------------
// Fetchers — WRITE
// ---------------------------------------------------------------------------

/** Create and launch a new tuning experiment. */
export async function submitModelExperiment(
  model: ModelType,
  payload: CreateExperimentPayload
): Promise<CreateExperimentResponse> {
  return fetchOrThrow(`${prefix(model)}/experiments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/** Promote a completed run to production champion. */
export async function promoteModelExperiment(
  model: ModelType,
  runId: number
): Promise<{ promoted: boolean; run_id: number; params_written: Record<string, unknown> }> {
  return fetchOrThrow(`${prefix(model)}/experiments/${runId}/promote`, {
    method: "POST",
  });
}

/** Cancel a running or queued experiment. */
export async function cancelModelExperiment(
  model: ModelType,
  runId: number
): Promise<{ cancelled: boolean; run_id: number; status: string }> {
  return fetchOrThrow(`${prefix(model)}/experiments/${runId}/cancel`, {
    method: "POST",
  });
}

/** Delete a completed, failed, or cancelled experiment. */
export async function deleteModelExperiment(
  model: ModelType,
  runId: number
): Promise<{ deleted: boolean; run_id: number }> {
  return fetchOrThrow(`${prefix(model)}/experiments/${runId}`, {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// Results promotion
// ---------------------------------------------------------------------------

export interface PromoteResultsResponse {
  job_id: string;
  run_id: number;
  model: string;
  message: string;
}

export interface PromoteResultsStatus {
  status: string;
  is_results_promoted: boolean;
  results_promoted_at?: string | null;
  progress_pct?: number;
  progress_msg?: string;
  error?: string | null;
}

/** Promote results — load backtest predictions into DB. */
export async function promoteModelResults(
  model: ModelType,
  runId: number
): Promise<PromoteResultsResponse> {
  return fetchOrThrow(`${prefix(model)}/experiments/${runId}/promote-results`, {
    method: "POST",
  });
}

/** Poll the status of a results promotion job. */
export async function fetchPromoteResultsStatus(
  model: ModelType,
  runId: number
): Promise<PromoteResultsStatus> {
  return fetchOrThrow(`${prefix(model)}/experiments/${runId}/promote-results/status`);
}

// ---------------------------------------------------------------------------
// Pipeline Config
// ---------------------------------------------------------------------------

export interface PipelineAlgorithm {
  type: string;
  enabled: boolean;
  tune: boolean;
  backtest: boolean;
  compete: boolean;
  forecast: boolean;
  expert: boolean;
  cluster_strategy?: string;
  output_dir: string;
  params?: Record<string, unknown>;
  notes?: string;
}

export interface PipelineConfig {
  algorithms: Record<string, PipelineAlgorithm>;
  clustering: {
    enabled: boolean;
    config_ref: string;
    tuning_profiles_ref: string;
    steps: Record<string, boolean>;
  };
  backtest: {
    n_timeframes: number;
    embargo_months: number;
    forecast_horizon: number;
    early_stop_pct: number;
  };
  tuning: {
    n_trials: number;
    gap_months: number;
    n_splits: number;
  };
  champion: {
    strategy: string;
    fallback_model_id: string;
    metric: string;
    strategy_params: Record<string, unknown>;
  };
  production_forecast: {
    horizon_months: number;
    min_history_months: number;
    cold_start_model_id: string;
    cold_start_min_months: number;
  };
  backtest_sampling: {
    enabled: boolean;
    default_target_n: number;
    default_method: string;
  };
  pipeline: {
    stages: Array<{
      name: string;
      description: string;
      makefile_target: string;
      depends_on: string[];
    }>;
  };
}

export const pipelineConfigKeys = {
  config: ["pipeline-config"] as const,
};

export async function fetchPipelineConfig(): Promise<PipelineConfig> {
  const data = await fetchJson<{ raw?: PipelineConfig; values?: PipelineConfig } & PipelineConfig>(
    "/config/forecast_pipeline_config"
  );
  return data.raw || data.values || data;
}

export async function updatePipelineConfig(values: Record<string, unknown>): Promise<void> {
  await fetchJson<unknown>("/config/forecast_pipeline_config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values }),
  });
}
