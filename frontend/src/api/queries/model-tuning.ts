/**
 * LightGBM model-tuning API.
 * Reuses the same types as lgbm-tuning but points at model-specific prefixes.
 */
import { fetchJson } from "./core";
import type {
  TuningRun,
  TuningComparison,
  TuningComparisonSummary,
  PromoteResponse,
  PromotedRun,
  TuningTimeframe,
} from "./lgbm-tuning";

// ModelType is now exported from unified-model-tuning.ts — do not re-export here.
import type { ModelType } from "./unified-model-tuning";

// Types (TuningRun, TuningComparison, etc.) are already exported from lgbm-tuning.ts via the barrel.

// --- URL prefix map ---
const MODEL_PREFIX: Record<ModelType, string> = {
  lgbm: "/lgbm-tuning",
};

// Prefix for the unified /model-tuning/<model>/experiments endpoints used by ModelTuningTab.
const MODEL_TUNING_EXPERIMENTS_PREFIX: Record<ModelType, string> = {
  lgbm: "/model-tuning/lgbm",
};

export interface ModelExperimentsResponse {
  experiments: TuningRun[];
  total: number;
}

export interface ModelSummary {
  best: number | null;
  runs: number;
  active: number;
  promoted: number | null;
}

/** Fetch the experiments list for a tunable model with optional filters. */
export async function fetchModelExperiments(
  model: ModelType,
  opts?: { limit?: number; status?: string; exec_lag?: number }
): Promise<ModelExperimentsResponse> {
  const sp = new URLSearchParams();
  if (opts?.limit) sp.set("limit", String(opts.limit));
  if (opts?.status && opts.status !== "all") sp.set("status", opts.status);
  if (opts?.exec_lag !== undefined) sp.set("exec_lag", String(opts.exec_lag));
  return fetchJson<ModelExperimentsResponse>(
    `${MODEL_TUNING_EXPERIMENTS_PREFIX[model]}/experiments?${sp}`,
    { cache: "no-cache" }
  );
}

/** Compute a summary (best, runs, active, promoted) for a tunable model. */
export async function fetchModelSummary(model: ModelType): Promise<ModelSummary> {
  try {
    const data = await fetchJson<ModelExperimentsResponse>(
      `${MODEL_TUNING_EXPERIMENTS_PREFIX[model]}/experiments?limit=100`,
      { cache: "no-cache" }
    );
    const exps: TuningRun[] = data.experiments ?? [];
    const completed = exps.filter((e) => e.status === "completed" && e.accuracy_pct != null);
    const best = completed.reduce<number | null>(
      (acc, e) => (!acc || (e.accuracy_pct ?? 0) > acc ? (e.accuracy_pct ?? 0) : acc),
      null
    );
    const promoted = exps.find((e) => e.is_promoted)?.accuracy_pct ?? null;
    return {
      best,
      runs: exps.length,
      active: exps.filter((e) => e.status === "running").length,
      promoted,
    };
  } catch {
    return { best: null, runs: 0, active: 0, promoted: null };
  }
}

// Query keys are now in unified-model-tuning.ts — modelTuningKeys removed to avoid conflict.

// --- Fetchers ---
export async function fetchModelTuningRuns(
  model: ModelType,
  params?: { limit?: number; status?: string }
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
  runId: number
): Promise<TuningRun & { timeframes: TuningTimeframe[] }> {
  const res = await fetch(`${MODEL_PREFIX[model]}/runs/${runId}`);
  if (!res.ok) throw new Error(`Failed to fetch ${model} tuning run: ${res.status}`);
  return res.json();
}

export async function fetchModelTuningComparison(
  model: ModelType,
  baselineId: number,
  candidateId: number
): Promise<TuningComparison> {
  const res = await fetch(
    `${MODEL_PREFIX[model]}/compare?baseline_id=${baselineId}&candidate_id=${candidateId}`
  );
  if (!res.ok) throw new Error(`Failed to fetch ${model} comparison: ${res.status}`);
  return res.json();
}

export async function fetchModelTuningComparisons(
  model: ModelType,
  limit?: number
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

export async function fetchModelPromotedRun(
  model: ModelType
): Promise<{ promoted: PromotedRun | null }> {
  const res = await fetch(`${MODEL_PREFIX[model]}/promoted`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`Failed to fetch ${model} promoted run: ${res.status}`);
  return res.json();
}
