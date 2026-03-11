/**
 * Platform query functions for Specs 08-01 through 08-10.
 * Covers: auth, data quality, notifications, collaboration, FVA, reports, webhooks.
 */

const API = "";

// ---------------------------------------------------------------------------
// Auth (08-02)
// ---------------------------------------------------------------------------
export interface LoginPayload { email: string; password: string; }
export interface TokenResponse { access_token: string; refresh_token: string; token_type: string; user: UserProfile; }
export interface UserProfile { user_id: string; email: string; display_name: string; role: string; is_active?: boolean; created_at?: string; last_login_at?: string; }
export interface AuditEntry { audit_id: number; user_id: string | null; email: string | null; display_name: string | null; action: string; resource_type: string; resource_id: string; old_value: Record<string, unknown> | null; new_value: Record<string, unknown> | null; created_at: string | null; }

export const fetchLogin = async (payload: LoginPayload): Promise<TokenResponse> => {
  const r = await fetch(`${API}/auth/login`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  if (!r.ok) throw new Error((await r.json()).detail || "Login failed");
  return r.json();
};

export const fetchMe = async (): Promise<UserProfile> => {
  const token = localStorage.getItem("ds_access_token");
  const r = await fetch(`${API}/auth/me`, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  if (!r.ok) throw new Error("Not authenticated");
  return r.json();
};

export const fetchUsers = async (limit = 50, offset = 0) => {
  const token = localStorage.getItem("ds_access_token");
  const r = await fetch(`${API}/users?limit=${limit}&offset=${offset}`, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  return r.json();
};

export const fetchAuditLog = async (limit = 50, offset = 0) => {
  const token = localStorage.getItem("ds_access_token");
  const r = await fetch(`${API}/users/audit-log?limit=${limit}&offset=${offset}`, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  return r.json();
};

// ---------------------------------------------------------------------------
// Data Quality (08-01)
// ---------------------------------------------------------------------------
export interface DQDomainScore { domain: string; score: number; passed: number; failed: number; warnings: number; total: number; }
export interface DQCheck { check_id: number; check_name: string; check_type: string; domain: string; table_name: string; severity: string; enabled: boolean; last_status: string | null; last_value: number | null; last_run: string | null; }

export const dqKeys = { dashboard: ["dq", "dashboard"], checks: ["dq", "checks"], freshness: ["dq", "freshness"] } as const;

export const fetchDQDashboard = async (): Promise<{ domains: DQDomainScore[] }> => {
  const r = await fetch(`${API}/data-quality/dashboard`);
  return r.json();
};

export const fetchDQChecks = async (): Promise<{ checks: DQCheck[] }> => {
  const r = await fetch(`${API}/data-quality/checks`);
  return r.json();
};

export const fetchDQFreshness = async () => {
  const r = await fetch(`${API}/data-quality/freshness`);
  return r.json();
};

// ---------------------------------------------------------------------------
// Notifications (08-04)
// ---------------------------------------------------------------------------
export const notifKeys = { history: ["notifications", "history"], channels: ["notifications", "channels"] } as const;

export const fetchNotificationHistory = async (limit = 50) => {
  const r = await fetch(`${API}/notifications/history?limit=${limit}`);
  return r.json();
};

export const fetchNotificationChannels = async () => {
  const r = await fetch(`${API}/notifications/channels`);
  return r.json();
};

// ---------------------------------------------------------------------------
// Collaboration (08-05)
// ---------------------------------------------------------------------------
export interface Annotation { annotation_id: number; user_id: string | null; email: string | null; display_name: string | null; parent_id: number | null; body: string; mentions: string[] | null; is_resolved: boolean; created_at: string | null; updated_at: string | null; }
export interface SharedView { view_id: number; user_id: string | null; title: string; tab: string; filters: Record<string, unknown>; layout: Record<string, unknown>; is_public: boolean; created_at: string | null; }

export const collabKeys = { annotations: (rt: string, rid: string) => ["annotations", rt, rid], mentions: ["mentions", "me"], sharedViews: ["shared-views"] } as const;

export const fetchAnnotations = async (resourceType: string, resourceId: string): Promise<{ annotations: Annotation[] }> => {
  const r = await fetch(`${API}/collaboration/annotations?resource_type=${resourceType}&resource_id=${resourceId}`);
  return r.json();
};

export const fetchMyMentions = async (limit = 20) => {
  const r = await fetch(`${API}/collaboration/mentions/me?limit=${limit}`);
  return r.json();
};

export const fetchSharedViews = async (): Promise<{ views: SharedView[] }> => {
  const r = await fetch(`${API}/collaboration/shared-views`);
  return r.json();
};

// ---------------------------------------------------------------------------
// FVA Tracking (08-07)
// ---------------------------------------------------------------------------
export const fvaKeys = { waterfall: (m: number) => ["fva", "waterfall", m], interventions: ["fva", "interventions"], roi: (m: number) => ["fva", "roi", m] } as const;

export const fetchFVAWaterfall = async (months = 12) => {
  const r = await fetch(`${API}/fva/waterfall?months=${months}`);
  return r.json();
};

export const fetchFVAInterventions = async (limit = 50, offset = 0) => {
  const r = await fetch(`${API}/fva/interventions?limit=${limit}&offset=${offset}`);
  return r.json();
};

export const fetchFVAROI = async (months = 12) => {
  const r = await fetch(`${API}/fva/roi-summary?months=${months}`);
  return r.json();
};

// ---------------------------------------------------------------------------
// Reports (08-08)
// ---------------------------------------------------------------------------
export const reportKeys = { templates: ["reports", "templates"], schedules: ["reports", "schedules"], deliveries: ["reports", "deliveries"] } as const;

export const fetchReportTemplates = async () => {
  const r = await fetch(`${API}/reports/templates`);
  return r.json();
};

export const fetchReportSchedules = async () => {
  const r = await fetch(`${API}/reports/schedules`);
  return r.json();
};

export const fetchReportDeliveries = async (limit = 50) => {
  const r = await fetch(`${API}/reports/deliveries?limit=${limit}`);
  return r.json();
};

// ---------------------------------------------------------------------------
// External Signals (08-06)
// ---------------------------------------------------------------------------
export const externalSignalKeys = { signals: ["external-signals"], sources: ["external-signal-sources"], decomposition: (i: string, l: string) => ["decomposition", i, l] } as const;

export const fetchExternalSignals = async (days = 90, limit = 100) => {
  const r = await fetch(`${API}/demand-signals/external?days=${days}&limit=${limit}`);
  return r.json();
};

export const fetchSignalSources = async () => {
  const r = await fetch(`${API}/demand-signals/external/sources`);
  return r.json();
};

export const fetchDemandDecomposition = async (itemNo: string, loc: string) => {
  const r = await fetch(`${API}/demand-signals/external/decomposition?item_no=${itemNo}&loc=${loc}`);
  return r.json();
};

// Stale times
export const STALE_PLATFORM = 5 * 60 * 1000; // 5 min
