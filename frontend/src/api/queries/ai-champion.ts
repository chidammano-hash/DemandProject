/** AI Champion forward adjuster — spec 02-forecasting/27-ai-champion-forecast.md */
import { fetchJson } from "./core";

export interface AiChampionRun {
  run_id: string;
  plan_version: string;
  provider: string;
  ai_model: string;
  status: string;
  n_dfus: number;
  n_adjusted: number;
  est_cost_usd: number | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface AiChampionRecommendationRollup {
  recommendation_code: string;
  dfus: number;
}

export interface AiChampionLatestResponse {
  run: AiChampionRun | null;
  by_recommendation: AiChampionRecommendationRollup[];
}

export interface AiChampionForecastRow {
  item_id: string;
  loc: string;
  forecast_month: string | null;
  horizon_months: number | null;
  champion_qty: number | null;
  ai_qty: number | null;
  recommendation_code: string;
  pct_change: number | null;
  confidence: number | null;
  rationale: string | null;
}

export interface AiChampionForecastResponse {
  total: number;
  rows: AiChampionForecastRow[];
}

export interface AiChampionGenerateResponse {
  job_id: string;
  status: string;
}

export interface AiChampionGenerateParams {
  provider?: string;
  limit_dfus?: number;
}

export const aiChampionKeys = {
  latest: () => ["ai-champion", "latest"] as const,
  forecast: (params: { item_id?: string; adjusted_only?: boolean; limit?: number }) =>
    ["ai-champion", "forecast", params] as const,
};

export async function fetchAiChampionLatest(): Promise<AiChampionLatestResponse> {
  return fetchJson("/ai-champion/latest");
}

export async function fetchAiChampionForecast(params?: {
  item_id?: string;
  adjusted_only?: boolean;
  limit?: number;
  offset?: number;
}): Promise<AiChampionForecastResponse> {
  const sp = new URLSearchParams();
  if (params?.item_id) sp.set("item_id", params.item_id);
  if (params?.adjusted_only) sp.set("adjusted_only", "true");
  if (params?.limit != null) sp.set("limit", String(params.limit));
  if (params?.offset != null) sp.set("offset", String(params.offset));
  const qs = sp.toString();
  return fetchJson(`/ai-champion/forecast${qs ? `?${qs}` : ""}`);
}

export async function triggerAiChampionGenerate(
  params?: AiChampionGenerateParams,
): Promise<AiChampionGenerateResponse> {
  return fetchJson("/ai-champion/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params ?? {}),
  });
}
