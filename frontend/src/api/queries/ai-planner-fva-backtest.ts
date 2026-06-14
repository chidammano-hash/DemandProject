// AI Planner FVA Backtest queries
// Spec: docs/specs/PRD/PRD-ai-planner-fva-backtest.md (§6 API Surface)
// Vite proxy: /ai-planner/* prefix already configured.
import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Backend response types — mirror Pydantic models in
// api/routers/forecasting/ai_fva_backtest.py
// ---------------------------------------------------------------------------

export type BacktestStatus =
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export type Provider =
  | "ollama"
  | "anthropic"
  | "openai"
  | "openai_compat";

export interface RunMetadata {
  run_id: string;
  status: BacktestStatus;
  started_at: string | null;
  completed_at: string | null;
  window_months: number;
  as_of_date: string;
  horizon_months: number;
  provider: Provider;
  ai_model: string;
  n_dfus_sampled: number | null;
  n_recommendations: number | null;
  estimated_cost_usd: number | null;
  actual_cost_usd: number | null;
  error_message: string | null;
}

export interface RunSummary {
  run_id: string;
  baseline_wape_pct: number | null;
  ai_wape_pct: number | null;
  lift_pct: number | null;
  n_dfus: number;
  n_winners: number;
  n_losers: number;
  n_ties: number;
  win_rate_pct: number | null;
}

export interface RecommendationRollupRow {
  recommendation_code: string;
  baseline_wape_pct: number | null;
  ai_wape_pct: number | null;
  lift_pct: number | null;
  n_obs: number;
  avg_confidence: number | null;
  avg_pct_change: number | null;
}

export interface ByMonthRow {
  forecast_run_month: string;
  baseline_wape_pct: number | null;
  ai_wape_pct: number | null;
  n_dfus: number;
}

export interface DfuRow {
  item_id: string;
  loc: string;
  sae_baseline: number;
  sae_ai: number;
  abs_error_reduction: number;
  n_obs: number;
}

export interface DfuDetailLag {
  forecast_run_month: string;
  target_month: string;
  lag: number;
  baseline_qty: number | null;
  ai_qty: number | null;
  actual_qty: number | null;
}

export interface DfuDetailRecommendation {
  forecast_run_month: string;
  recommendation_code: string;
  pct_change: number | null;
  confidence: number | null;
  rationale: string | null;
  evidence_keys: string[] | null;
}

export interface DfuDetailSummary {
  n_obs: number;
  baseline_wape_pct: number | null;
  ai_wape_pct: number | null;
  lift_pp: number | null;
}

export interface DfuDetail {
  run_id: string;
  item_id: string;
  loc: string;
  summary: DfuDetailSummary;
  lags: DfuDetailLag[];
  recommendations: DfuDetailRecommendation[];
}

export interface StartRunRequest {
  window_months?: number;
  as_of_date?: string;
  horizon_months?: number;
  provider?: Provider;
  limit_dfus?: number;
  notes?: string;
}

export interface StartRunResponse {
  status: "accepted";
  message: string;
}

// ---------------------------------------------------------------------------
// React Query keys
// ---------------------------------------------------------------------------

export const aiFvaBacktestKeys = {
  root: ["ai-planner", "fva-backtest"] as const,
  list: (status?: BacktestStatus) =>
    ["ai-planner", "fva-backtest", "list", status ?? null] as const,
  detail: (runId: string) =>
    ["ai-planner", "fva-backtest", "detail", runId] as const,
  summary: (runId: string) =>
    ["ai-planner", "fva-backtest", "summary", runId] as const,
  byRecommendation: (runId: string) =>
    ["ai-planner", "fva-backtest", "by-recommendation", runId] as const,
  byMonth: (runId: string) =>
    ["ai-planner", "fva-backtest", "by-month", runId] as const,
  dfus: (runId: string, sort: string, limit: number) =>
    ["ai-planner", "fva-backtest", "dfus", runId, sort, limit] as const,
  dfuDetail: (runId: string, itemId: string, loc: string) =>
    ["ai-planner", "fva-backtest", "dfu-detail", runId, itemId, loc] as const,
};

// ---------------------------------------------------------------------------
// Fetchers
// ---------------------------------------------------------------------------

const BASE = "/ai-planner/fva-backtest";

export async function listFvaBacktestRuns(
  status?: BacktestStatus,
  limit = 50,
): Promise<{ runs: RunMetadata[]; count: number }> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (status) params.set("status", status);
  return fetchJson(`${BASE}/runs?${params.toString()}`);
}

export async function getFvaBacktestRun(runId: string): Promise<RunMetadata> {
  return fetchJson(`${BASE}/runs/${runId}`);
}

export async function getFvaBacktestSummary(runId: string): Promise<RunSummary> {
  return fetchJson(`${BASE}/runs/${runId}/summary`);
}

export async function getFvaBacktestByRecommendation(
  runId: string,
): Promise<{ run_id: string; rows: RecommendationRollupRow[] }> {
  return fetchJson(`${BASE}/runs/${runId}/by-recommendation`);
}

export async function getFvaBacktestByMonth(
  runId: string,
): Promise<{ run_id: string; rows: ByMonthRow[] }> {
  return fetchJson(`${BASE}/runs/${runId}/by-month`);
}

export async function getFvaBacktestDfus(
  runId: string,
  opts: { limit?: number; sort?: "error_reduction" | "item_id" } = {},
): Promise<{ run_id: string; rows: DfuRow[]; count: number }> {
  const params = new URLSearchParams({
    limit: String(opts.limit ?? 100),
    sort: opts.sort ?? "error_reduction",
  });
  return fetchJson(`${BASE}/runs/${runId}/dfus?${params.toString()}`);
}

export async function getFvaBacktestDfuDetail(
  runId: string,
  itemId: string,
  loc: string,
): Promise<DfuDetail> {
  const params = new URLSearchParams({ item_id: itemId, loc });
  return fetchJson(`${BASE}/runs/${runId}/dfu-detail?${params.toString()}`);
}

export async function startFvaBacktestRun(
  body: StartRunRequest,
): Promise<StartRunResponse> {
  return fetchJson(`${BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
