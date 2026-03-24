/**
 * Generic model tuning API — CatBoost & XGBoost.
 * Reuses the same types as lgbm-tuning but points at model-specific prefixes.
 */
import type {
  TuningRun,
  TuningComparison,
  TuningComparisonSummary,
  PromoteResponse,
  PromotedRun,
  TuningTimeframe,
} from "./lgbm-tuning";

export type ModelType = "lgbm" | "catboost" | "xgboost";

// Re-export types for convenience
export type { TuningRun, TuningComparison, TuningComparisonSummary, PromoteResponse, PromotedRun, TuningTimeframe };

// --- URL prefix map ---
const MODEL_PREFIX: Record<ModelType, string> = {
  lgbm: "/lgbm-tuning",
  catboost: "/catboost-tuning",
  xgboost: "/xgboost-tuning",
};

// --- Query keys ---
export const modelTuningKeys = {
  runs: (model: ModelType, params?: { limit?: number; status?: string }) =>
    [`${model}-tuning-runs`, params] as const,
  run: (model: ModelType, runId: number) =>
    [`${model}-tuning-run`, runId] as const,
  compare: (model: ModelType, baselineId: number, candidateId: number) =>
    [`${model}-tuning-compare`, baselineId, candidateId] as const,
  comparisons: (model: ModelType, limit?: number) =>
    [`${model}-tuning-comparisons`, limit] as const,
  promoted: (model: ModelType) =>
    [`${model}-tuning-promoted`] as const,
};

// --- Fetchers ---
export async function fetchModelTuningRuns(
  model: ModelType,
  params?: { limit?: number; status?: string },
): Promise<{ runs: TuningRun[]; total: number }> {
  const sp = new URLSearchParams();
  if (params?.limit) sp.set("limit", String(params.limit));
  if (params?.status) sp.set("status", params.status);
  const res = await fetch(`${MODEL_PREFIX[model]}/runs?${sp}`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`Failed to fetch ${model} tuning runs: ${res.status}`);
  return res.json();
}

export async function fetchModelTuningRun(
  model: ModelType,
  runId: number,
): Promise<TuningRun & { timeframes: TuningTimeframe[] }> {
  const res = await fetch(`${MODEL_PREFIX[model]}/runs/${runId}`);
  if (!res.ok) throw new Error(`Failed to fetch ${model} tuning run: ${res.status}`);
  return res.json();
}

export async function fetchModelTuningComparison(
  model: ModelType,
  baselineId: number,
  candidateId: number,
): Promise<TuningComparison> {
  const res = await fetch(`${MODEL_PREFIX[model]}/compare?baseline_id=${baselineId}&candidate_id=${candidateId}`);
  if (!res.ok) throw new Error(`Failed to fetch ${model} comparison: ${res.status}`);
  return res.json();
}

export async function fetchModelTuningComparisons(
  model: ModelType,
  limit?: number,
): Promise<{ comparisons: TuningComparisonSummary[] }> {
  const sp = new URLSearchParams();
  if (limit) sp.set("limit", String(limit));
  const res = await fetch(`${MODEL_PREFIX[model]}/comparisons?${sp}`);
  if (!res.ok) throw new Error(`Failed to fetch ${model} comparisons: ${res.status}`);
  return res.json();
}

export async function promoteModelRun(model: ModelType, runId: number): Promise<PromoteResponse> {
  const res = await fetch(`${MODEL_PREFIX[model]}/runs/${runId}/promote`, { method: "POST" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(body.detail ?? `Promote failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchModelPromotedRun(model: ModelType): Promise<{ promoted: PromotedRun | null }> {
  const res = await fetch(`${MODEL_PREFIX[model]}/promoted`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`Failed to fetch ${model} promoted run: ${res.status}`);
  return res.json();
}
