/**
 * Backtest Management API — run/load/version management for all models.
 *
 * Endpoints under `/backtest-management/` provide backtest run tracking,
 * submission, and DB-loading for both tunable and non-tunable models.
 */

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
  trained_at: string | null;
  training_mode: string | null;
  n_dfus: number | null;
  planning_date: string | null;
}

export interface TrainingStatusMap {
  [model_id: string]: ModelTrainingStatus;
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
  row_count: number;
  dfu_count: number;
  forecast_month_generated: string | null;
  last_generated_at: string | null;
  min_forecast_month: string | null;
  max_forecast_month: string | null;
}
export type StagingSummaryMap = Record<string, StagingSummary>;

/** Promote response. */
export interface PromoteResponse {
  model_id: string;
  promotion_type: string;
  plan_version: string;
  rows_promoted: number;
  dfu_count: number;
}

// ---------------------------------------------------------------------------
// Query key factory
// ---------------------------------------------------------------------------

export const backtestMgmtKeys = {
  summary: ["backtest-management", "summary"] as const,
  trainingStatus: ["backtest-management", "training-status"] as const,
  promotionStatus: ["backtest-management", "promotion-status"] as const,
  candidateSummary: ["backtest-management", "candidate-summary"] as const,
  stagingSummary: ["backtest-management", "staging-summary"] as const,
  runs: (modelId: string) =>
    ["backtest-management", "runs", modelId] as const,
  current: (modelId: string) =>
    ["backtest-management", "current", modelId] as const,
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
// Fetch helper
// ---------------------------------------------------------------------------

async function fetchOrThrow<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res
      .json()
      .catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Fetchers — READ
// ---------------------------------------------------------------------------

/** Fetch summary for all models (latest run, accuracy, loaded status). */
export async function fetchBacktestSummary(): Promise<BacktestSummary> {
  return fetchOrThrow("/backtest-management/summary", { cache: "no-cache" });
}

/** Fetch run history for a specific model. */
export async function fetchBacktestRuns(
  modelId: string,
): Promise<BacktestRun[]> {
  return fetchOrThrow(`/backtest-management/${modelId}/runs`, {
    cache: "no-cache",
  });
}

/** Fetch current metadata from disk for a model. */
export async function fetchBacktestCurrent(
  modelId: string,
): Promise<Record<string, unknown>> {
  return fetchOrThrow(`/backtest-management/${modelId}/current`);
}

/** Fetch training status for all forecastable models. */
export async function fetchTrainingStatus(): Promise<TrainingStatusMap> {
  return fetchOrThrow("/backtest-management/training-status", {
    cache: "no-cache",
  });
}

/** Fetch staging forecast summary per model. */
export async function fetchStagingSummary(): Promise<StagingSummaryMap> {
  return fetchOrThrow("/backtest-management/staging-summary", { cache: "no-cache" });
}

// ---------------------------------------------------------------------------
// Fetchers — WRITE
// ---------------------------------------------------------------------------

/** Submit a new backtest run for a model. */
export async function submitBacktestRun(
  modelId: string,
): Promise<{ run_id: number; job_id: string }> {
  return fetchOrThrow(`/backtest-management/${modelId}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
}

/** Load backtest predictions into the database. */
export async function submitBacktestLoad(
  modelId: string,
  runId?: number,
): Promise<{ job_id: string }> {
  const body = runId != null ? JSON.stringify({ run_id: runId }) : undefined;
  return fetchOrThrow(`/backtest-management/${modelId}/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    ...(body ? { body } : {}),
  });
}

/** Submit a training job for a model (train on full history for production). */
export async function submitTraining(
  modelId: string,
): Promise<{ job_id: string }> {
  return fetchOrThrow(`/backtest-management/${modelId}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
}

/** Submit a generate-forecast job for a model (produces staging forecast). */
export async function submitGenerateForecast(
  modelId: string,
): Promise<{ job_id: string; model_id: string }> {
  return fetchOrThrow(`/backtest-management/${modelId}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
}

// ---------------------------------------------------------------------------
// Fetchers — PROMOTION WORKFLOW
// ---------------------------------------------------------------------------

/** Fetch current promotion status (which model is in production). */
export async function fetchPromotionStatus(): Promise<{ promoted: PromotionStatus | null }> {
  return fetchOrThrow("/backtest-management/promotion-status", { cache: "no-cache" });
}

/** Fetch candidate forecast summary per model. */
export async function fetchCandidateSummary(): Promise<CandidateSummaryMap> {
  return fetchOrThrow("/backtest-management/candidate-summary", { cache: "no-cache" });
}

/** Promote a model (or 'champion') to production. Copies candidates → fact_production_forecast. */
export async function submitPromote(modelId: string): Promise<PromoteResponse> {
  return fetchOrThrow(`/backtest-management/${modelId}/promote`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
}
