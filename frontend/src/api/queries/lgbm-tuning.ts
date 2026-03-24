import { STALE } from "./core";

// --- Query keys ---
export const lgbmTuningKeys = {
  runs: (params?: { limit?: number; status?: string }) => ["lgbm-tuning-runs", params] as const,
  run: (runId: number) => ["lgbm-tuning-run", runId] as const,
  compare: (baselineId: number, candidateId: number) => ["lgbm-tuning-compare", baselineId, candidateId] as const,
  comparisons: (limit?: number) => ["lgbm-tuning-comparisons", limit] as const,
  promoted: () => ["lgbm-tuning-promoted"] as const,
};

// --- Types ---
export interface TuningRun {
  run_id: number;
  run_label: string;
  model_id: string;
  started_at: string;
  completed_at: string | null;
  status: "running" | "completed" | "failed";
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  n_predictions: number | null;
  n_dfus: number | null;
  feature_count: number | null;
  params: Record<string, unknown> | null;
  notes: string | null;
  is_promoted?: boolean;
  promoted_at?: string | null;
}

export interface PromoteResponse {
  promoted: boolean;
  run_id: number;
  run_label: string;
  accuracy_pct: number | null;
  params_written: Record<string, unknown>;
  old_params: Record<string, unknown>;
}

export interface PromotedRun {
  run_id: number;
  run_label: string;
  model_id: string;
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  promoted_at: string | null;
  params: Record<string, unknown> | null;
}

export interface TuningTimeframe {
  timeframe: string;
  train_end: string;
  predict_start: string;
  predict_end: string;
  n_predictions: number | null;
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
}

export interface TuningRunDetail {
  run: TuningRun;
  timeframes: TuningTimeframe[];
}

export interface ClusterComparison {
  cluster: string;
  baseline_accuracy: number | null;
  candidate_accuracy: number | null;
  delta_accuracy: number | null;
  baseline_wape: number | null;
  candidate_wape: number | null;
  baseline_n_dfus: number | null;
  candidate_n_dfus: number | null;
}

export interface MonthComparison {
  month: string;
  baseline_accuracy: number | null;
  candidate_accuracy: number | null;
  delta_accuracy: number | null;
  baseline_wape: number | null;
  candidate_wape: number | null;
}

export interface ParamDiff {
  param: string;
  baseline: unknown;
  candidate: unknown;
}

export interface ParamCommon {
  param: string;
  value: unknown;
}

export interface FeatureDiffs {
  baseline_count: number;
  candidate_count: number;
  added: string[];
  removed: string[];
  common_count: number;
}

export interface ConfigDiff {
  setting: string;
  baseline: unknown;
  candidate: unknown;
}

export interface ConfigCommon {
  setting: string;
  value: unknown;
}

export interface TuningComparison {
  baseline: TuningRun;
  candidate: TuningRun;
  delta_accuracy: number;
  delta_wape: number;
  delta_bias: number;
  verdict: "improved" | "degraded" | "neutral";
  param_diffs?: ParamDiff[];
  param_common?: ParamCommon[];
  feature_diffs?: FeatureDiffs;
  config_diffs?: ConfigDiff[];
  config_common?: ConfigCommon[];
  per_timeframe?: Array<{
    timeframe: string;
    baseline_accuracy: number | null;
    candidate_accuracy: number | null;
    delta_accuracy: number | null;
  }>;
  per_cluster?: {
    ml_cluster: ClusterComparison[];
    business_cluster: ClusterComparison[];
  };
  per_month?: MonthComparison[];
  baseline_has_breakdowns?: boolean;
  candidate_has_breakdowns?: boolean;
}

export interface TuningComparisonSummary {
  id: number;
  baseline_label: string;
  baseline_id: number;
  candidate_label: string;
  candidate_id: number;
  delta_accuracy: number;
  delta_wape: number;
  verdict: string;
  created_at: string;
}

// --- Fetchers ---
export async function fetchTuningRuns(params?: { limit?: number; status?: string }): Promise<{ runs: TuningRun[]; total: number }> {
  const sp = new URLSearchParams();
  if (params?.limit) sp.set("limit", String(params.limit));
  if (params?.status) sp.set("status", params.status);
  const res = await fetch(`/lgbm-tuning/runs?${sp}`);
  if (!res.ok) throw new Error(`Failed to fetch tuning runs: ${res.status}`);
  return res.json();
}

export async function fetchTuningRun(runId: number): Promise<TuningRunDetail> {
  const res = await fetch(`/lgbm-tuning/runs/${runId}`);
  if (!res.ok) throw new Error(`Failed to fetch tuning run: ${res.status}`);
  return res.json();
}

export async function fetchTuningComparison(baselineId: number, candidateId: number): Promise<TuningComparison> {
  const res = await fetch(`/lgbm-tuning/compare?baseline_id=${baselineId}&candidate_id=${candidateId}`);
  if (!res.ok) throw new Error(`Failed to fetch comparison: ${res.status}`);
  return res.json();
}

export async function fetchTuningComparisons(limit?: number): Promise<{ comparisons: TuningComparisonSummary[] }> {
  const sp = new URLSearchParams();
  if (limit) sp.set("limit", String(limit));
  const res = await fetch(`/lgbm-tuning/comparisons?${sp}`);
  if (!res.ok) throw new Error(`Failed to fetch comparisons: ${res.status}`);
  return res.json();
}

export async function promoteRun(runId: number): Promise<PromoteResponse> {
  const res = await fetch(`/lgbm-tuning/runs/${runId}/promote`, { method: "POST" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(body.detail ?? `Promote failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchPromotedRun(): Promise<{ promoted: PromotedRun | null }> {
  const res = await fetch("/lgbm-tuning/promoted");
  if (!res.ok) throw new Error(`Failed to fetch promoted run: ${res.status}`);
  return res.json();
}
