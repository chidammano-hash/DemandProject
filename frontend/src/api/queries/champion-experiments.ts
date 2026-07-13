/**
 * Champion Experiments API — Champion Selection Experimentation Studio
 *
 * Query module for champion strategy experiment CRUD, comparison, and promotion.
 * Follows the same pattern as cluster-experiments.ts and unified-model-tuning.ts.
 */

import { buildSearchParams } from "./helpers";
import { fetchJson } from "./request";

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

export interface ChampionExperiment {
  experiment_id: number;
  label: string;
  notes: string | null;
  template_id: string | null;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  runtime_seconds: number | null;
  job_id: string | null;
  strategy: string;
  strategy_params: Record<string, unknown> | null;
  meta_learner_params: Record<string, unknown> | null;
  models: string[];
  metric: string;
  lag_mode: string;
  min_sku_rows: number;
  champion_accuracy: number | null;
  ceiling_accuracy: number | null;
  gap_bps: number | null;
  n_champions: number | null;
  n_dfu_months: number | null;
  model_distribution: Record<string, number> | null;
  is_promoted: boolean;
  promoted_at: string | null;
  is_results_promoted: boolean;
  results_promoted_at: string | null;
  results_promote_job_id: string | null;
}

export interface ChampionExperimentLag {
  exec_lag: number;
  champion_accuracy: number | null;
  ceiling_accuracy: number | null;
  gap_bps: number | null;
  n_dfu_months: number | null;
  model_distribution: Record<string, number> | null;
}

export interface ChampionExperimentMonth {
  month_start: string;
  champion_accuracy: number | null;
  ceiling_accuracy: number | null;
  gap_bps: number | null;
  n_champions: number | null;
  model_distribution: Record<string, number> | null;
}

export interface ChampionOverallComparison {
  experiment_a: {
    champion_accuracy: number | null;
    ceiling_accuracy: number | null;
    gap_bps: number | null;
    n_dfu_months: number | null;
  };
  experiment_b: {
    champion_accuracy: number | null;
    ceiling_accuracy: number | null;
    gap_bps: number | null;
    n_dfu_months: number | null;
  };
  delta_champion_accuracy: number | null;
  delta_ceiling_accuracy: number | null;
  delta_gap_bps: number | null;
  verdict: "a_better" | "b_better" | "mixed";
}

export interface ChampionLagComparison {
  exec_lag: number;
  a_champion_accuracy: number | null;
  b_champion_accuracy: number | null;
  delta_accuracy: number | null;
  a_gap_bps: number | null;
  b_gap_bps: number | null;
}

export interface ChampionMonthComparison {
  month_start: string;
  a_champion_accuracy: number | null;
  b_champion_accuracy: number | null;
  delta_accuracy: number | null;
}

export interface ChampionModelDistComparison {
  model_id: string;
  a_pct: number;
  b_pct: number;
  delta_pct: number;
}

export interface ChampionConfigDiff {
  key: string;
  a: unknown;
  b: unknown;
}

export interface ChampionExperimentComparison {
  experiment_a_id: number;
  experiment_b_id: number;
  overall_comparison: ChampionOverallComparison;
  per_lag_comparison: ChampionLagComparison[];
  per_month_comparison: ChampionMonthComparison[];
  model_dist_comparison: ChampionModelDistComparison[];
  config_diffs: ChampionConfigDiff[];
  source: "cache" | "computed";
}

export interface ChampionExperimentTemplate {
  id: string;
  label: string;
  description: string;
  source?: string;
  strategy?: string;
  strategy_params?: Record<string, unknown>;
  meta_learner_params?: Record<string, unknown>;
  models?: string[];
  metric?: string;
  lag_mode?: string;
  min_sku_rows?: number;
}

export interface ChampionPromotionLogEntry {
  id: number;
  experiment_id: number;
  promoted_at: string | null;
  promoted_by: string | null;
  previous_experiment_id: number | null;
  strategy: string | null;
  champion_accuracy: number | null;
  config_snapshot: Record<string, unknown> | null;
}

// ---------------------------------------------------------------------------
// Query keys
// ---------------------------------------------------------------------------

export const championExperimentKeys = {
  all: ["champion-experiments"] as const,
  experiments: (params?: Record<string, unknown>) =>
    ["champion-experiments", "list", params] as const,
  experiment: (id: number) =>
    ["champion-experiments", "detail", id] as const,
  lags: (id: number) =>
    ["champion-experiments", "lags", id] as const,
  months: (id: number) =>
    ["champion-experiments", "months", id] as const,
  logs: (id: number, offset?: number) =>
    ["champion-experiments", "logs", id, offset] as const,
  compare: (aId: number, bId: number, execLag?: number) =>
    ["champion-experiments", "compare", aId, bId, execLag] as const,
  templates: () => ["champion-experiments", "templates"] as const,
  promoted: () => ["champion-experiments", "promoted"] as const,
  promotions: () => ["champion-experiments", "promotions"] as const,
};

// ---------------------------------------------------------------------------
// Stale times (ms)
// ---------------------------------------------------------------------------

export const CHAMPION_EXP_STALE = {
  EXPERIMENTS: 10_000,
  EXPERIMENT: 30_000,
  COMPARE: 120_000,
  TEMPLATES: 600_000,
  PROMOTED: 30_000,
  PROMOTIONS: 60_000,
  LAGS: 30_000,
  MONTHS: 30_000,
};

// ---------------------------------------------------------------------------
// Base URL
// ---------------------------------------------------------------------------

const BASE = "/champion-experiments";

// ---------------------------------------------------------------------------
// Read fetchers
// ---------------------------------------------------------------------------

export async function fetchChampionExperiments(
  opts?: { status?: string; limit?: number; offset?: number; exec_lag?: number },
): Promise<{ experiments: ChampionExperiment[]; total: number }> {
  const sp = buildSearchParams(opts ?? {});
  const res = await fetch(`${BASE}?${sp}`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`Failed to fetch champion experiments: ${res.status}`);
  return res.json();
}

export async function fetchChampionExperiment(
  id: number,
): Promise<ChampionExperiment> {
  const res = await fetch(`${BASE}/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch champion experiment ${id}: ${res.status}`);
  return res.json();
}

export async function fetchChampionExperimentLags(
  id: number,
): Promise<{ experiment_id: number; lags: ChampionExperimentLag[] }> {
  const res = await fetch(`${BASE}/${id}/lags`);
  if (!res.ok) throw new Error(`Failed to fetch champion experiment lags: ${res.status}`);
  return res.json();
}

export async function fetchChampionExperimentMonths(
  id: number,
): Promise<{ experiment_id: number; months: ChampionExperimentMonth[] }> {
  const res = await fetch(`${BASE}/${id}/months`);
  if (!res.ok) throw new Error(`Failed to fetch champion experiment months: ${res.status}`);
  return res.json();
}

export async function fetchChampionExperimentLogs(
  id: number,
  offset = 0,
): Promise<{
  experiment_id: number;
  log: string;
  offset: number;
  next_offset: number;
  status: string;
  has_more: boolean;
}> {
  const res = await fetch(`${BASE}/${id}/logs?offset=${offset}`);
  if (!res.ok) throw new Error(`Failed to fetch champion experiment logs: ${res.status}`);
  return res.json();
}

export async function fetchChampionTemplates(): Promise<{
  templates: ChampionExperimentTemplate[];
}> {
  const res = await fetch(`${BASE}/templates`);
  if (!res.ok) throw new Error(`Failed to fetch champion templates: ${res.status}`);
  return res.json();
}

export async function compareChampionExperiments(
  aId: number,
  bId: number,
  execLag?: number,
): Promise<ChampionExperimentComparison> {
  const sp = new URLSearchParams({ a_id: String(aId), b_id: String(bId) });
  if (execLag !== undefined) sp.set("exec_lag", String(execLag));
  const res = await fetch(`${BASE}/compare?${sp}`);
  if (!res.ok) throw new Error(`Failed to compare champion experiments: ${res.status}`);
  return res.json();
}

export async function fetchPromotedChampionExperiment(): Promise<{
  promoted: ChampionExperiment | null;
}> {
  const res = await fetch(`${BASE}/promoted`);
  if (!res.ok) throw new Error(`Failed to fetch promoted champion experiment: ${res.status}`);
  return res.json();
}

export async function fetchChampionPromotions(): Promise<{
  promotions: ChampionPromotionLogEntry[];
}> {
  const res = await fetch(`${BASE}/promotions`);
  if (!res.ok) throw new Error(`Failed to fetch champion promotions: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Write fetchers
// ---------------------------------------------------------------------------

export async function createChampionExperiment(body: {
  label: string;
  notes?: string;
  template?: string;
  strategy: string;
  strategy_params?: Record<string, unknown>;
  meta_learner_params?: Record<string, unknown>;
  models?: string[];
  metric?: string;
  lag_mode?: string;
  min_sku_rows?: number;
}): Promise<{
  experiment_id: number;
  job_id: string;
  status: string;
  strategy: string;
  label: string;
}> {
  const res = await fetch(BASE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to create champion experiment: ${text}`);
  }
  return res.json();
}

export async function assignChampionExperiment(
  id: number,
): Promise<{
  source_experiment_id: number;
  job_id: string;
  status: string;
  message: string;
}> {
  return fetchJson(`${BASE}/${id}/assign`, { method: "POST" });
}

export async function fetchChampionResultsStatus(
  id: number,
): Promise<{
  experiment_id: number;
  is_results_promoted: boolean;
  results_promoted_at: string | null;
  status: string;
  progress_pct?: number;
  progress_msg?: string;
  error?: string;
}> {
  const res = await fetch(`${BASE}/${id}/promote-results/status`);
  if (!res.ok) throw new Error(`Failed to fetch results status: ${res.status}`);
  return res.json();
}

export async function cancelChampionExperiment(
  id: number,
): Promise<{ cancelled: boolean; experiment_id: number; previous_status: string }> {
  const res = await fetch(`${BASE}/${id}/cancel`, { method: "POST" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to cancel champion experiment: ${text}`);
  }
  return res.json();
}

export async function deleteChampionExperiment(
  id: number,
): Promise<{ deleted: boolean; experiment_id: number }> {
  const res = await fetch(`${BASE}/${id}`, { method: "DELETE" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to delete champion experiment: ${text}`);
  }
  return res.json();
}
