/**
 * Smart Change Detection + Chain Submission query layer.
 *
 * Wraps `/integration/scan` and `/integration/chains` FastAPI endpoints
 * (api/routers/platform/integration.py) for scanning `data/input/` for changed
 * source files and composing/submitting a sequential chain of domain load jobs
 * (masters first, then details).
 *
 * Mirrors the conventions of `integration.ts` — uses the shared `fetchJson`
 * wrapper, `buildSearchParams` helper, and a centralized `chainKeys` factory.
 */

import { fetchJson } from "./request";
import { buildSearchParams } from "./helpers";

// ---------------------------------------------------------------------------
// Types — mirror the backend Pydantic schemas exactly
// ---------------------------------------------------------------------------

/** Whether a domain is a dimension (master) or a fact (detail). */
export type DomainKind = "dim" | "fact";

/** Lifecycle status of a submitted chain. */
export type ChainStatus = "queued" | "running" | "success" | "failed" | "halted";

/** Load mode for a single chain step / job. */
export type LoadMode = "onetime" | "delta" | "file";

/** Per-file diff entry as produced by the scanner. */
export type FileChange = {
  path: string;
  current_hash: string | null;
  last_hash: string | null;
  changed: boolean;
};

/** Per-domain change summary returned by the scanner. */
export type DomainChange = {
  domain: string;
  kind: DomainKind;
  changed: boolean;
  reason: string;
  proposed_mode: LoadMode;
  proposed_slice: string | null;
  source_files: FileChange[];
};

/** A single ordered step in the proposed chain. */
export type ChainStep = {
  step: number;
  domain: string;
  mode: LoadMode;
  slice: string | null;
};

/** One answer sent back to the planner. */
export type PlannerAnswer = {
  question_id: string;
  answer: string;
};

/** One planner question shown in the UI. */
export type PlannerQuestion = {
  id: string;
  prompt: string;
  answer_type: "text" | "choice" | "boolean";
  options: string[];
  required: boolean;
  reason: string | null;
};

/** Evidence item exposed by the planner. */
export type PlannerEvidence = {
  kind: "scan" | "job" | "batch";
  label: string;
  value: string;
};

/** Full result of a scan operation. */
export type ScanResult = {
  scanned_at: string;
  changes: DomainChange[];
  proposed_chain: ChainStep[];
};

/** Planner result returned by POST /integration/scan/plan. */
export type ScanPlanResult = ScanResult & {
  plan_id: string;
  provider: string;
  model: string;
  status: "questions" | "planned" | "fallback";
  confidence: number;
  explanation: string;
  risk_flags: string[];
  questions: PlannerQuestion[];
  recommended_chain: ChainStep[];
  evidence: PlannerEvidence[];
};

/** Body for POST /integration/scan/plan. */
export type ScanPlanRequest = {
  answers?: PlannerAnswer[];
};

/** High-level summary row for a chain (used in list views). */
export type ChainSummary = {
  id: string;
  status: ChainStatus;
  total_steps: number;
  completed_steps: number;
  failed_step: number | null;
  started_at: string;
  completed_at: string | null;
  duration_ms: number | null;
  triggered_by: string | null;
};

/** Per-step job inside a chain detail response. */
export type ChainJob = {
  step: number;
  job_id: string;
  domain: string;
  mode: LoadMode;
  slice: string | null;
  status: "queued" | "running" | "success" | "failed" | "skipped";
  rows_loaded: number;
  rows_inserted: number | null;
  rows_updated: number | null;
  rows_deleted: number | null;
  error_message: string | null;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
};

/** Chain detail response — summary plus full per-step job list. */
export type ChainDetail = ChainSummary & { jobs: ChainJob[] };

/** Body for POST /integration/chains. */
export type SubmitChainRequest = {
  jobs: { domain: string; mode: LoadMode; slice?: string; file?: string }[];
  triggered_by?: string;
};

/** Response from POST /integration/chains (202 accepted). */
export type SubmitChainResponse = {
  chain_id: string;
  jobs: { job_id: string; step: number; domain: string; mode: LoadMode }[];
  status: "queued";
};

// ---------------------------------------------------------------------------
// React Query key factory
// ---------------------------------------------------------------------------

/** Centralized TanStack Query keys for the integration chain module. */
export const chainKeys = {
  all: ["integration_chain"] as const,
  scan: ["integration_chain", "scan"] as const,
  chains: (filter?: { limit?: number }) =>
    ["integration_chain", "chains", filter ?? {}] as const,
  chain: (id: string) => ["integration_chain", "chain", id] as const,
};

// ---------------------------------------------------------------------------
// Fetchers
// ---------------------------------------------------------------------------

/** Internal envelope returned by list endpoints. */
type ListEnvelope<T> = { items: T[] };

/** GET /integration/scan — scan `data/input/` for changed source files. */
export async function scanInputs(): Promise<ScanResult> {
  return fetchJson<ScanResult>("/integration/scan");
}

/** POST /integration/scan/plan — ask the AI planner for the safest sequence. */
export async function planScan(
  req: ScanPlanRequest = {},
): Promise<ScanPlanResult> {
  return fetchJson<ScanPlanResult>("/integration/scan/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

/** GET /integration/chains — list recent chains, optionally limited. */
export async function listChains(limit?: number): Promise<ChainSummary[]> {
  const qs = buildSearchParams({ limit }).toString();
  const path = qs ? `/integration/chains?${qs}` : "/integration/chains";
  const data = await fetchJson<ListEnvelope<ChainSummary>>(path);
  return data.items;
}

/** GET /integration/chains/{chain_id} — fetch a single chain (404 throws). */
export async function getChain(id: string): Promise<ChainDetail> {
  return fetchJson<ChainDetail>(
    `/integration/chains/${encodeURIComponent(id)}`,
  );
}

/** POST /integration/chains — enqueue a sequential chain of load jobs. */
export async function submitChain(
  req: SubmitChainRequest,
): Promise<SubmitChainResponse> {
  return fetchJson<SubmitChainResponse>("/integration/chains", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}
