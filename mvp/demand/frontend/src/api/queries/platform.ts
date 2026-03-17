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
export interface DQDomainScore { domain: string; score: number; passed: number; failed: number; warnings: number; total: number; }
export interface DQCheck { check_id: number; check_name: string; check_type: string; domain: string; table_name: string; severity: string; enabled: boolean; last_status: string | null; last_value: number | null; last_run: string | null; }

export const dqKeys = { dashboard: ["dq", "dashboard"], checks: ["dq", "checks"], freshness: ["dq", "freshness"] } as const;

export const fetchDQDashboard = async (): Promise<{ domains: DQDomainScore[] }> =>
  fetchJson("/data-quality/dashboard");

export const fetchDQChecks = async (): Promise<{ checks: DQCheck[] }> =>
  fetchJson("/data-quality/checks");

export const fetchDQFreshness = async () =>
  fetchJson("/data-quality/freshness");

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
export const fvaKeys = { waterfall: (m: number) => ["fva", "waterfall", m], interventions: ["fva", "interventions"], roi: (m: number) => ["fva", "roi", m] } as const;

export const fetchFVAWaterfall = async (months = 12) =>
  fetchJson(`/fva/waterfall?months=${months}`);

export const fetchFVAInterventions = async (limit = 50, offset = 0) =>
  fetchJson(`/fva/interventions?limit=${limit}&offset=${offset}`);

export const fetchFVAROI = async (months = 12) =>
  fetchJson(`/fva/roi-summary?months=${months}`);

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
  fetchJson(`/demand-signals/external/decomposition?item_no=${itemNo}&loc=${loc}`);

// Stale times
export const STALE_PLATFORM = 5 * 60 * 1000; // 5 min
