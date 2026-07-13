/**
 * Backtest Management API — run/load/version management for all models.
 *
 * Endpoints under `/backtest-management/` provide backtest run tracking,
 * submission, and DB-loading for both tunable and non-tunable models.
 */

import { fetchJson } from "./request";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BacktestRun {
  id: number;
  model_id: string;
  job_id: string | null;
  status: "queued" | "running" | "completed" | "failed";
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  n_predictions: number | null;
  n_dfus: number | null;
  is_loaded_to_db: boolean;
  loaded_at: string | null;
  load_job_id: string | null;
  created_at: string;
  completed_at: string | null;
  is_loaded_to_candidate?: boolean;
  candidate_loaded_at?: string | null;
}

export interface BacktestModelSummary {
  latest_run: BacktestRun | null;
  has_predictions: boolean;
  current_accuracy: number | null;
  current_wape: number | null;
  has_job_type?: boolean;
  enabled?: boolean;
  type?: string;
}

export interface BacktestSummary {
  [model_id: string]: BacktestModelSummary;
}

// Training status — returned by /backtest-management/training-status
export interface ModelTrainingStatus {
  model_id: string;
  type: string;
  trained: boolean;
  /** True only when the server validated this model against current inputs. */
  ready: boolean;
  trained_at: string | null;
  training_mode: string | null;
  n_dfus: number | null;
  planning_date: string | null;
  artifact_id?: string | null;
  stale_reason?: string | null;
}

export interface TrainingStatusMap {
  [model_id: string]: ModelTrainingStatus;
}

export interface SnapshotContenderReadiness {
  model_id: string;
  rank: number;
  ready: boolean;
  stale_reason: string | null;
}

export interface SnapshotRosterReadiness {
  planning_month: string;
  ready: boolean;
  champion_ready: boolean;
  roster_model_count: number;
  ready_contender_count: number;
  required_contender_count: number;
  contenders: SnapshotContenderReadiness[];
  stale_reason: string | null;
  action_pipeline: "model-refresh" | "forecast-publish" | null;
}

/** Promotion status — the currently active promoted model. */
export interface PromotionStatus {
  id: number;
  model_id: string;
  promotion_type: "single" | "champion";
  champion_experiment_id: number | null;
  plan_version: string;
  promoted_at: string | null;
  dfu_count: number | null;
  total_rows: number | null;
  promoted_by: string;
  notes: string | null;
  source_run_id: string | null;
  production_run_id: string | null;
  candidate_checksum: string | null;
  production_checksum: string | null;
  archive_checksum: string | null;
  archived_at: string | null;
}

/** Candidate forecast summary per model. */
export interface CandidateSummary {
  model_id: string;
  row_count: number;
  dfu_count: number;
  last_loaded_at: string | null;
  avg_accuracy: number | null;
}

export type CandidateSummaryMap = Record<string, CandidateSummary>;

/** Staging forecast summary per model. */
export interface StagingSummary {
  model_id: string;
  source_run_id: string;
  run_status: "ready" | "promoted";
  promotion_eligible: boolean;
  generation_purpose: "release_candidate";
  row_count: number;
  dfu_count: number;
  source_model_count: number;
  forecast_month_generated: string | null;
  last_generated_at: string | null;
  min_forecast_month: string | null;
  max_forecast_month: string | null;
}
export type StagingSummaryMap = Record<string, StagingSummary>;

/** Promote response. */
export interface BacktestPromotionResponse {
  model_id: string;
  promotion_type: string;
  plan_version: string;
  source_run_id: string;
  production_run_id: string;
  candidate_checksum: string;
  outgoing_archive_checksum: string | null;
  rows_promoted: number;
  dfu_count: number;
}

// ---------------------------------------------------------------------------
// Query key factory
// ---------------------------------------------------------------------------

export const backtestMgmtKeys = {
  summary: ["backtest-management", "summary"] as const,
  trainingStatus: ["backtest-management", "training-status"] as const,
  snapshotRosterReadiness: ["backtest-management", "snapshot-roster-readiness"] as const,
  promotionStatus: ["backtest-management", "promotion-status"] as const,
  candidateSummary: ["backtest-management", "candidate-summary"] as const,
  stagingSummary: ["backtest-management", "staging-summary"] as const,
  runs: (modelId: string) => ["backtest-management", "runs", modelId] as const,
  current: (modelId: string) => ["backtest-management", "current", modelId] as const,
};

// ---------------------------------------------------------------------------
// Stale times (ms)
// ---------------------------------------------------------------------------

export const BACKTEST_MGMT_STALE = {
  SUMMARY: 30_000, // 30s — summary refreshes frequently for active runs
  RUNS: 15_000, // 15s — run list for a model
  CURRENT: 60_000, // 1min — current metadata from disk
  PROMOTION: 30_000, // 30s — promotion status
  CANDIDATES: 30_000, // 30s — candidate summary
} as const;

// ---------------------------------------------------------------------------
// Fetchers — READ
// ---------------------------------------------------------------------------

/** Fetch summary for all models (latest run, accuracy, loaded status). */
export async function fetchBacktestSummary(): Promise<BacktestSummary> {
  return fetchJson<BacktestSummary>("/backtest-management/summary");
}

/** Fetch run history for a specific model. */
export async function fetchBacktestRuns(modelId: string): Promise<BacktestRun[]> {
  return fetchJson<BacktestRun[]>(`/backtest-management/${modelId}/runs`);
}

/** Fetch current metadata from disk for a model. */
export async function fetchBacktestCurrent(modelId: string): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>(`/backtest-management/${modelId}/current`);
}

/** Fetch training status for all forecastable models. */
export async function fetchTrainingStatus(): Promise<TrainingStatusMap> {
  return fetchJson<TrainingStatusMap>("/backtest-management/training-status");
}

/** Validate the current champion plus exact top-three publish evidence. */
export async function fetchSnapshotRosterReadiness(): Promise<SnapshotRosterReadiness> {
  return fetchJson<SnapshotRosterReadiness>("/backtest-management/snapshot-roster-readiness");
}

/** Fetch staging forecast summary per model. */
export async function fetchStagingSummary(): Promise<StagingSummaryMap> {
  return fetchJson<StagingSummaryMap>("/backtest-management/staging-summary");
}

// ---------------------------------------------------------------------------
// Fetchers — WRITE
// ---------------------------------------------------------------------------

/** Result of submitting a backtest run. `status` is "queued" when a new run was
 *  created (it runs now or queues behind active backtests) or "already_running"
 *  when this model already has a run in flight (no duplicate is started). */
export interface SubmitBacktestRunResult {
  run_id: number | null; // null when status === "already_running"
  job_id: string;
  model_id: string;
  status: "queued" | "already_running";
  message?: string;
}

/** Submit a new backtest run for a model. Submission never fails on concurrency:
 *
 * - `parallel=false` (default) runs backtests one at a time — extra submissions
 *   queue automatically and run sequentially.
 * - `parallel=true` lets different model families run concurrently.
 * - Re-running a model that already has a run in flight is a no-op; the result
 *   comes back with `status: "already_running"` instead of an error.
 */
export async function submitBacktestRun(
  modelId: string,
  parallel = false
): Promise<SubmitBacktestRunResult> {
  const qs = parallel ? "?parallel=true" : "";
  return fetchJson<SubmitBacktestRunResult>(`/backtest-management/${modelId}/run${qs}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
}

/** Load backtest predictions into the database. */
export async function submitBacktestLoad(
  modelId: string,
  runId?: number
): Promise<{ job_id: string }> {
  const body = runId != null ? JSON.stringify({ run_id: runId }) : undefined;
  return fetchJson<{ job_id: string }>(`/backtest-management/${modelId}/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    ...(body ? { body } : {}),
  });
}

/** Submit a training job for a model (train on full history for production). */
export async function submitTraining(modelId: string): Promise<{ job_id: string }> {
  return fetchJson<{ job_id: string }>(`/backtest-management/${modelId}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
}

/** Submit a generate-forecast job for a model (produces staging forecast).
 *
 * `horizon` and `confidenceIntervals` are threaded to the backend as query
 * params so the Forecast panel's controls actually take effect — previously
 * they were dropped for single-model generation. Omit either to fall back to
 * the pipeline config default.
 */
export async function submitGenerateForecast(
  modelId: string,
  opts?: { horizon?: number; confidenceIntervals?: boolean }
): Promise<{ job_id: string; model_id: string; source_run_id: string }> {
  const qs = new URLSearchParams();
  if (opts?.horizon != null) qs.set("horizon", String(opts.horizon));
  if (opts?.confidenceIntervals != null) {
    qs.set("confidence_intervals", String(opts.confidenceIntervals));
  }
  const suffix = qs.toString() ? `?${qs}` : "";
  return fetchJson<{ job_id: string; model_id: string; source_run_id: string }>(
    `/backtest-management/${modelId}/generate${suffix}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    }
  );
}

// ---------------------------------------------------------------------------
// Fetchers — PROMOTION WORKFLOW
// ---------------------------------------------------------------------------

/** Fetch current promotion status (which model is in production). */
export async function fetchPromotionStatus(): Promise<{ promoted: PromotionStatus | null }> {
  return fetchJson<{ promoted: PromotionStatus | null }>("/backtest-management/promotion-status");
}

/** Fetch candidate forecast summary per model. */
export async function fetchCandidateSummary(): Promise<CandidateSummaryMap> {
  return fetchJson<CandidateSummaryMap>("/backtest-management/candidate-summary");
}

/** Promote a model (or 'champion') to production. Copies candidates → fact_production_forecast. */
export async function submitPromote(
  modelId: string,
  sourceRunId: string
): Promise<BacktestPromotionResponse> {
  const qs = new URLSearchParams({ source_run_id: sourceRunId });
  return fetchJson<BacktestPromotionResponse>(`/backtest-management/${modelId}/promote?${qs}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
}
