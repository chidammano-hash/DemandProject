/**
 * Champion Sweeps API — Champion Strategy Sweep (Tournament).
 *
 * Query module for creating/listing sweeps and reading the global leaderboard +
 * per-segment winner map. Mirrors champion-experiments.ts. See spec 30.
 */

import { buildSearchParams } from "./helpers";

const BASE = "/champion-sweeps";

// ---------------------------------------------------------------------------
// Types (mirror the backend Pydantic / DB shapes)
// ---------------------------------------------------------------------------

export type SweepMode = "global" | "per_segment" | "both";
export type SweepAxis = "demand_class" | "ml_cluster" | "abc_xyz";
export type SweepObjective = "accuracy" | "gap_to_ceiling" | "robust";
export type SweepStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface ChampionSweep {
  sweep_id: number;
  label: string;
  notes: string | null;
  mode: SweepMode;
  segment_axis: SweepAxis;
  objective: SweepObjective;
  grid_spec: Record<string, unknown>;
  parallel: boolean;
  baseline_experiment_id: number | null;
  status: SweepStatus;
  candidate_count: number | null;
  completed_count: number;
  job_id: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  runtime_seconds: number | null;
  best_global_experiment_id: number | null;
  composite_experiment_id: number | null;
  recommended_experiment_id: number | null;
  recommended_score: number | null;
  recommended_gate_eligible: boolean | null;
}

export interface SweepLeaderboardMember {
  experiment_id: number;
  global_rank: number | null;
  global_score: number | null;
  gate_eligible: boolean | null;
  is_composite: boolean;
  skipped_duplicate: boolean;
  label: string;
  strategy: string;
  strategy_params: Record<string, unknown> | null;
  models: string[] | null;
  metric: string;
  champion_accuracy: number | null;
  ceiling_accuracy: number | null;
  gap_bps: number | null;
  status: SweepStatus;
}

export interface SweepSegmentScore {
  segment: string;
  experiment_id: number;
  n_dfus: number | null;
  accuracy: number | null;
  score: number | null;
  segment_rank: number | null;
  label: string;
  strategy: string;
}

export interface SweepSegment {
  segment: string;
  winner: SweepSegmentScore | null;
  candidates: SweepSegmentScore[];
}

export interface CreateSweepBody {
  label: string;
  notes?: string;
  mode?: SweepMode;
  segment_axis?: SweepAxis;
  objective?: SweepObjective;
  grid_spec: Record<string, unknown>;
  parallel?: boolean;
  baseline_experiment_id?: number | null;
}

// ---------------------------------------------------------------------------
// Query keys
// ---------------------------------------------------------------------------

export const championSweepKeys = {
  all: ["champion-sweeps"] as const,
  list: (status?: string) => ["champion-sweeps", "list", status ?? "all"] as const,
  detail: (id: number) => ["champion-sweeps", "detail", id] as const,
  leaderboard: (id: number) => ["champion-sweeps", "leaderboard", id] as const,
  segments: (id: number) => ["champion-sweeps", "segments", id] as const,
};

// ---------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------

export async function fetchChampionSweeps(
  opts?: { status?: string; limit?: number; offset?: number },
): Promise<{ sweeps: ChampionSweep[]; offset: number; limit: number }> {
  const sp = buildSearchParams(opts ?? {});
  const res = await fetch(`${BASE}?${sp}`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`Failed to fetch champion sweeps: ${res.status}`);
  return res.json();
}

export async function fetchChampionSweep(id: number): Promise<ChampionSweep> {
  const res = await fetch(`${BASE}/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch champion sweep ${id}: ${res.status}`);
  return res.json();
}

export async function fetchSweepLeaderboard(
  id: number,
): Promise<{ sweep_id: number; members: SweepLeaderboardMember[] }> {
  const res = await fetch(`${BASE}/${id}/leaderboard`);
  if (!res.ok) throw new Error(`Failed to fetch sweep leaderboard ${id}: ${res.status}`);
  return res.json();
}

export async function fetchSweepSegments(
  id: number,
): Promise<{ sweep_id: number; segments: SweepSegment[] }> {
  const res = await fetch(`${BASE}/${id}/segments`);
  if (!res.ok) throw new Error(`Failed to fetch sweep segments ${id}: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Writes
// ---------------------------------------------------------------------------

export async function createChampionSweep(
  body: CreateSweepBody,
): Promise<{
  sweep_id: number;
  job_id: string;
  status: string;
  candidate_count: number;
  label: string;
}> {
  const res = await fetch(BASE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to create champion sweep: ${text}`);
  }
  return res.json();
}

export async function cancelChampionSweep(
  id: number,
): Promise<{ sweep_id: number; status: string }> {
  const res = await fetch(`${BASE}/${id}/cancel`, { method: "POST" });
  if (!res.ok) throw new Error(`Failed to cancel sweep ${id}: ${res.status}`);
  return res.json();
}

export async function promoteSweepWinner(
  id: number,
): Promise<{ sweep_id: number; promoted_experiment_id: number; promoted: boolean }> {
  const res = await fetch(`${BASE}/${id}/promote-winner`, { method: "POST" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to promote sweep winner: ${text}`);
  }
  return res.json();
}

export async function deleteChampionSweep(
  id: number,
): Promise<{ sweep_id: number; deleted: boolean }> {
  const res = await fetch(`${BASE}/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Failed to delete sweep ${id}: ${res.status}`);
  return res.json();
}
