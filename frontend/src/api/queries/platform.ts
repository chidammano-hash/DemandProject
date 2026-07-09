/**
 * Platform query functions for Specs 08-01 through 08-10.
 * Covers: auth, data quality, notifications, collaboration, FVA, reports, webhooks.
 */

import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Auth (08-02)
// ---------------------------------------------------------------------------
export interface LoginPayload { email: string; password: string; }
export interface TokenResponse { access_token: string; refresh_token: string; token_type: string; user: UserProfile; }
export interface UserProfile { user_id: string; email: string; display_name: string; role: string; is_active?: boolean; created_at?: string; last_login_at?: string; }
export interface AuditEntry { audit_id: number; user_id: string | null; email: string | null; display_name: string | null; action: string; resource_type: string; resource_id: string; old_value: Record<string, unknown> | null; new_value: Record<string, unknown> | null; created_at: string | null; }

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem("ds_access_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export const fetchLogin = async (payload: LoginPayload): Promise<TokenResponse> =>
  fetchJson("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const fetchMe = async (): Promise<UserProfile> =>
  fetchJson("/auth/me", { headers: authHeaders() });

export const fetchUsers = async (limit = 50, offset = 0) =>
  fetchJson(`/users?limit=${limit}&offset=${offset}`, { headers: authHeaders() });

export const fetchAuditLog = async (limit = 50, offset = 0) =>
  fetchJson(`/users/audit-log?limit=${limit}&offset=${offset}`, { headers: authHeaders() });

// ---------------------------------------------------------------------------
// Data Quality (08-01)
// ---------------------------------------------------------------------------
export interface DQDomainScore { domain: string; score: number | null; passed: number; failed: number; warnings: number; skipped: number; info_fails: number; warning_fails: number; total: number; }
export interface DQCheck { check_id: number; check_name: string; check_type: string; domain: string; table_name: string; severity: string; enabled: boolean; last_status: string | null; last_value: number | null; last_run: string | null; }
export interface DQHistoryEntry { check_id: number; check_name: string; check_type?: string; domain: string; table_name: string; severity: string; status: string; metric_value: number | null; details: Record<string, unknown> | string | null; run_ts: string | null; }
export interface DQFixItem { id: number; fix_type: string; description: string; affected_rows: number; recommendation: string | null; status: string; }
export interface DQFixApplyResult { applied: DQFixItem[]; skipped: DQFixItem[]; total_applied: number; total_skipped: number; total_rows_fixed: number; }

export const dqKeys = { dashboard: ["dq", "dashboard"], checks: ["dq", "checks"], history: (domain?: string) => ["dq", "history", domain ?? ""], fixPreview: ["dq", "fix", "preview"] } as const;

export const fetchDQDashboard = async (): Promise<{ domains: DQDomainScore[] }> =>
  fetchJson("/data-quality/dashboard");

export const fetchDQChecks = async (): Promise<{ checks: DQCheck[] }> =>
  fetchJson("/data-quality/checks");

export const fetchDQHistory = async (domain?: string, days = 7, limit = 200): Promise<{ entries: DQHistoryEntry[] }> => {
  const params = new URLSearchParams();
  if (domain) params.set("domain", domain);
  params.set("days", String(days));
  params.set("limit", String(limit));
  return fetchJson(`/data-quality/history?${params.toString()}`);
};

export const runDQChecks = async (domain?: string): Promise<{ triggered: number; message: string }> => {
  const params = domain ? `?domain=${domain}` : "";
  return fetchJson(`/data-quality/run${params}`, { method: "POST" });
};

export const fetchDQFixPreview = async (): Promise<{ items: DQFixItem[]; total: number }> =>
  fetchJson("/data-quality/fix/preview");

export const applyDQFixes = async (fixIds: number[]): Promise<DQFixApplyResult> =>
  fetchJson("/data-quality/fix/apply", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ fix_ids: fixIds }) });

// ---------------------------------------------------------------------------
// Pipeline Lineage
// ---------------------------------------------------------------------------
export interface LoadBatch {
  batch_id: number; domain: string;
  source_file: string | null; source_hash: string | null;
  row_count_in: number | null; row_count_out: number | null;
  status: string; started_at: string | null; completed_at: string | null;
  error_message: string | null;
}
export interface DQCorrection {
  correction_id: number; domain: string; table_name: string;
  item_id: string | null; loc: string | null;
  period: string | null; column_name: string;
  old_value: number | null; new_value: number | null;
  fix_type: string; fix_strategy: string | null;
  threshold: number | null;
  lower_bound: number | null; upper_bound: number | null;
  applied_at: string | null;
}
export const lineageKeys = {
  batches: ["lineage", "batches"],
  batchDetail: (id: number) => ["lineage", "batch", id],
  row: (domain: string, key: string) => ["lineage", "row", domain, key],
  corrections: ["lineage", "corrections"],
} as const;

export interface DQCorrectionSummary {
  item_id: string; loc: string;
  correction_count: number;
  domains: string[]; tables: string[];
  columns: string[]; fix_types: string[];
  strategies: string[];
  earliest_period: string | null;
  latest_period: string | null;
  latest_at: string | null;
}

export const correctionKeys = {
  byItem: (item_id: string, loc: string) => ["dq", "corrections", item_id, loc],
  summary: (domain?: string, fixType?: string) => ["dq", "corrections", "summary", domain ?? "", fixType ?? ""],
} as const;

export const STALE_LINEAGE = 30_000;

export const fetchBatches = async (domain?: string, status?: string, limit = 50): Promise<{ batches: LoadBatch[]; total: number }> => {
  const params = new URLSearchParams();
  if (domain) params.set("domain", domain);
  if (status) params.set("status", status);
  params.set("limit", String(limit));
  return fetchJson(`/data-quality/batches?${params}`);
};

export const fetchBatchDetail = async (batchId: number): Promise<LoadBatch> =>
  fetchJson(`/data-quality/batches/${batchId}`);

export const fetchRowLineage = async (domain: string, businessKey: string) =>
  fetchJson(`/data-quality/row/${domain}/${encodeURIComponent(businessKey)}`);

export const fetchCorrections = async (domain?: string, fixType?: string, batchId?: number, limit = 50): Promise<{ corrections: DQCorrection[]; total: number }> => {
  const params = new URLSearchParams();
  if (domain) params.set("domain", domain);
  if (fixType) params.set("fix_type", fixType);
  if (batchId) params.set("batch_id", String(batchId));
  params.set("limit", String(limit));
  return fetchJson(`/data-quality/corrections?${params}`);
};

export const fetchCorrectionsByItem = async (
  item_id: string,
  loc: string,
  limit = 500,
): Promise<{ corrections: DQCorrection[]; total: number }> => {
  const params = new URLSearchParams({ item_id, loc, limit: String(limit) });
  return fetchJson(`/data-quality/corrections?${params}`);
};

export const fetchCorrectionsSummary = async (
  domain?: string,
  fixType?: string,
  limit = 200,
  offset = 0,
): Promise<{ skus: DQCorrectionSummary[]; total: number }> => {
  const params = new URLSearchParams();
  if (domain) params.set("domain", domain);
  if (fixType) params.set("fix_type", fixType);
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  return fetchJson(`/data-quality/corrections/summary?${params}`);
};

// ---------------------------------------------------------------------------
// Notifications (08-04)
// ---------------------------------------------------------------------------
export const notifKeys = { history: ["notifications", "history"], channels: ["notifications", "channels"] } as const;

export const fetchNotificationHistory = async (limit = 50) =>
  fetchJson(`/notifications/history?limit=${limit}`);

export const fetchNotificationChannels = async () =>
  fetchJson("/notifications/channels");

// ---------------------------------------------------------------------------
// Collaboration (08-05)
// ---------------------------------------------------------------------------
export interface Annotation { annotation_id: number; user_id: string | null; email: string | null; display_name: string | null; parent_id: number | null; body: string; mentions: string[] | null; is_resolved: boolean; created_at: string | null; updated_at: string | null; }
export interface SharedView { view_id: number; user_id: string | null; title: string; tab: string; filters: Record<string, unknown>; layout: Record<string, unknown>; is_public: boolean; created_at: string | null; }

export const collabKeys = { annotations: (rt: string, rid: string) => ["annotations", rt, rid], mentions: ["mentions", "me"], sharedViews: ["shared-views"] } as const;

export const fetchAnnotations = async (resourceType: string, resourceId: string): Promise<{ annotations: Annotation[] }> =>
  fetchJson(`/collaboration/annotations?resource_type=${resourceType}&resource_id=${resourceId}`);

export const fetchMyMentions = async (limit = 20) =>
  fetchJson(`/collaboration/mentions/me?limit=${limit}`);

export const fetchSharedViews = async (): Promise<{ views: SharedView[] }> =>
  fetchJson("/collaboration/shared-views");

// ---------------------------------------------------------------------------
// FVA Tracking (08-07)
// ---------------------------------------------------------------------------
export interface FVASnapshotAccuracyRow {
  model_id: string;
  snapshot_role: "champion" | "contender";
  contender_rank: number | null;
  lag: number;
  forecast_month: string;
  n_dfus: number;
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  fva_vs_champion_pts: number | null;
  n_dfus_common: number;
}

export interface FVASnapshotMonth {
  record_month: string;
  closed_lag_count: number;
  latest_closed_forecast_month: string | null;
  last_refresh_at: string | null;
}

export const fvaKeys = {
  waterfall: (m: number) => ["fva", "waterfall", m],
  interventions: ["fva", "interventions"],
  roi: (m: number) => ["fva", "roi", m],
  snapshotMonths: ["fva", "snapshot-months"],
  snapshotAccuracy: (recordMonth: string) => ["fva", "snapshot", recordMonth],
} as const;

export const fetchFVAWaterfall = async (months = 12) =>
  fetchJson(`/fva/waterfall?months=${months}`);

export const fetchFVAInterventions = async (limit = 50, offset = 0) =>
  fetchJson(`/fva/interventions?limit=${limit}&offset=${offset}`);

export const fetchFVAROI = async (months = 12) =>
  fetchJson(`/fva/roi-summary?months=${months}`);

export const fetchFVASnapshotMonths = async (): Promise<{ months: FVASnapshotMonth[] }> =>
  fetchJson("/fva/snapshot-months");

export const fetchFVASnapshotAccuracy = async (recordMonth: string): Promise<{ record_month: string; rows: FVASnapshotAccuracyRow[] }> =>
  fetchJson(`/fva/snapshot-accuracy?record_month=${encodeURIComponent(recordMonth)}`);

// ---------------------------------------------------------------------------
// Reports (08-08)
// ---------------------------------------------------------------------------
export const reportKeys = { templates: ["reports", "templates"], schedules: ["reports", "schedules"], deliveries: ["reports", "deliveries"] } as const;

export const fetchReportTemplates = async () =>
  fetchJson("/reports/templates");

export const fetchReportSchedules = async () =>
  fetchJson("/reports/schedules");

export const fetchReportDeliveries = async (limit = 50) =>
  fetchJson(`/reports/deliveries?limit=${limit}`);

// ---------------------------------------------------------------------------
// External Signals (08-06)
// ---------------------------------------------------------------------------
export const externalSignalKeys = { signals: ["external-signals"], sources: ["external-signal-sources"], decomposition: (i: string, l: string) => ["decomposition", i, l] } as const;

export const fetchExternalSignals = async (days = 90, limit = 100) =>
  fetchJson(`/demand-signals/external?days=${days}&limit=${limit}`);

export const fetchSignalSources = async () =>
  fetchJson("/demand-signals/external/sources");

export const fetchDemandDecomposition = async (itemNo: string, loc: string) =>
  fetchJson(`/demand-signals/external/decomposition?item_id=${itemNo}&loc=${loc}`);

// Stale times
export const STALE_PLATFORM = 5 * 60 * 1000; // 5 min
