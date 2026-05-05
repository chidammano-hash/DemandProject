/**
 * Data Integration query layer.
 *
 * Wraps `/integration/*` FastAPI endpoints (api/routers/platform/integration.py)
 * for submitting and inspecting domain load jobs (one-time, delta, file).
 * Uses the shared `fetchJson` wrapper and `buildSearchParams` helper to keep
 * error handling and query-string conventions consistent with the rest of the
 * `src/api/queries/` modules.
 */

import { fetchJson } from "./core";
import { buildSearchParams } from "./helpers";

// ---------------------------------------------------------------------------
// Types — mirror the backend Pydantic schemas exactly
// ---------------------------------------------------------------------------

/** Load mode for a Data Integration job. */
export type LoadMode = "onetime" | "delta" | "file";

/** Lifecycle status of a Data Integration job. */
export type JobStatus = "queued" | "running" | "success" | "failed" | "skipped";

/** A single Data Integration job row as returned by the backend. */
export type Job = {
  id: string;
  domain: string;
  mode: LoadMode;
  slice: string | null;
  file_path: string | null;
  status: JobStatus;
  /** Headline row count: inserted+updated for delta; full count for onetime. */
  rows_loaded: number;
  /** Net inserts (delta path; null for legacy paths). */
  rows_inserted: number | null;
  /** Net updates in place (delta path; null for legacy). */
  rows_updated: number | null;
  /** Net deletes (file-slice / onetime; null for delta). */
  rows_deleted: number | null;
  error_message: string | null;
  started_at: string;
  completed_at: string | null;
  duration_ms: number | null;
  triggered_by: string | null;
};

/** Metadata about a registered domain available for ingestion. */
export type DomainInfo = {
  name: string;
  partitioned: boolean;
  partition_format: string | null;
  partition_field: string | null;
  /** True when TRUNCATE on this domain's table would CASCADE to fact tables. */
  onetime_cascades: boolean;
  /** Tables that would be wiped if onetime is run on this domain. */
  cascade_targets: string[];
};

/** Pool / table health snapshot for the integration subsystem. */
export type HealthStatus = { pool: string; table: string };

/** Body for POST /integration/jobs. */
export type SubmitJobRequest = {
  domain: string;
  mode: LoadMode;
  slice?: string;
  file?: string;
  /** Required to run onetime on a domain whose TRUNCATE would CASCADE. */
  confirm_destructive?: boolean;
  /** Run REINDEX TABLE after a successful upsert (slow; opt-in). */
  reindex?: boolean;
};

/** Response from POST /integration/jobs (202 accepted). */
export type SubmitJobResponse = { job_id: string; status: string };

// ---------------------------------------------------------------------------
// React Query key factory
// ---------------------------------------------------------------------------

/** Centralized TanStack Query keys for the integration module. */
export const integrationKeys = {
  all: ["integration"] as const,
  domains: ["integration", "domains"] as const,
  health: ["integration", "health"] as const,
  jobs: (filter?: { domain?: string; limit?: number }) =>
    ["integration", "jobs", filter ?? {}] as const,
  job: (id: string) => ["integration", "job", id] as const,
};

// ---------------------------------------------------------------------------
// Fetchers
// ---------------------------------------------------------------------------

/** Internal envelope returned by list endpoints. */
type ListEnvelope<T> = { items: T[] };

/** GET /integration/domains — list registered domains, unwrapping `items`. */
export async function listDomains(): Promise<DomainInfo[]> {
  const data = await fetchJson<ListEnvelope<DomainInfo>>("/integration/domains");
  return data.items;
}

/** GET /integration/jobs — list recent jobs, optionally filtered by domain/limit. */
export async function listJobs(
  filter?: { domain?: string; limit?: number },
): Promise<Job[]> {
  const qs = buildSearchParams({
    domain: filter?.domain,
    limit: filter?.limit,
  }).toString();
  const path = qs ? `/integration/jobs?${qs}` : "/integration/jobs";
  const data = await fetchJson<ListEnvelope<Job>>(path);
  return data.items;
}

/** GET /integration/jobs/{id} — fetch a single job (404 throws). */
export async function getJob(id: string): Promise<Job> {
  return fetchJson<Job>(`/integration/jobs/${encodeURIComponent(id)}`);
}

/** POST /integration/jobs — enqueue a new load job; returns the job id + status. */
export async function submitJob(req: SubmitJobRequest): Promise<SubmitJobResponse> {
  return fetchJson<SubmitJobResponse>("/integration/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

/** GET /integration/health — pool + table health snapshot. */
export async function getHealth(): Promise<HealthStatus> {
  return fetchJson<HealthStatus>("/integration/health");
}

/** Filter for purgeJobs — all keys optional. */
export type PurgeFilter = {
  older_than_hours?: number;
  status?: JobStatus;
  domain?: string;
};

/**
 * DELETE /integration/jobs — purge terminal jobs (queued/running are always
 * preserved). Returns the count actually deleted.
 */
export async function purgeJobs(filter: PurgeFilter = {}): Promise<{ deleted: number }> {
  const qs = buildSearchParams({
    older_than_hours: filter.older_than_hours,
    status: filter.status,
    domain: filter.domain,
  }).toString();
  const path = qs ? `/integration/jobs?${qs}` : "/integration/jobs";
  return fetchJson<{ deleted: number }>(path, { method: "DELETE" });
}
