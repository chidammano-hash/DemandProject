import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature1: Demand Variability
// ---------------------------------------------------------------------------

export interface VariabilitySummaryPayload {
  total_dfus: number;
  by_class: { low: number; medium: number; high: number; lumpy: number };
  cv_percentiles: { p25: number | null; p50: number | null; p75: number | null; p95: number | null };
  avg_cv: number | null;
  avg_intermittency_ratio: number | null;
  top_volatile: {
    item_no: string;
    loc: string;
    abc_vol: string | null;
    cluster_assignment: string | null;
    demand_mean: number | null;
    demand_std: number | null;
    demand_cv: number | null;
    demand_mad: number | null;
    intermittency_ratio: number | null;
    variability_class: string | null;
  }[];
}

export interface VariabilityDetailRow {
  item_no: string;
  loc: string;
  abc_vol: string | null;
  cluster_assignment: string | null;
  demand_mean: number | null;
  demand_std: number | null;
  demand_cv: number | null;
  demand_mad: number | null;
  demand_p50: number | null;
  demand_p90: number | null;
  demand_skewness: number | null;
  demand_kurtosis: number | null;
  zero_demand_months: number | null;
  total_demand_months: number | null;
  intermittency_ratio: number | null;
  variability_class: string | null;
  demand_profile_ts: string | null;
}

export interface VariabilityDetailPayload {
  total: number;
  rows: VariabilityDetailRow[];
}

export async function fetchVariabilitySummary(params: {
  abc_vol?: string;
  cluster_assignment?: string;
}): Promise<VariabilitySummaryPayload> {
  const qs = new URLSearchParams();
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  if (params.cluster_assignment?.trim()) qs.set("cluster_assignment", params.cluster_assignment.trim());
  return fetchJson(`/inv-planning/variability/summary?${qs}`);
}

export async function fetchVariabilityDetail(params: {
  item?: string;
  location?: string;
  abc_vol?: string;
  variability_class?: string;
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: string;
}): Promise<VariabilityDetailPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 50),
    offset: String(params.offset ?? 0),
    sort_by: params.sort_by ?? "demand_cv",
    sort_dir: params.sort_dir ?? "desc",
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  if (params.variability_class?.trim()) qs.set("variability_class", params.variability_class.trim());
  return fetchJson(`/inv-planning/variability/detail?${qs}`);
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature2: Lead Time Variability
// ---------------------------------------------------------------------------

export interface LtSummaryPayload {
  total_profiles: number;
  by_class: { stable: number; moderate: number; volatile: number };
  avg_lt_cv: number | null;
  avg_lt_mean_days: number | null;
  lt_cv_p50: number | null;
  lt_cv_p95: number | null;
  top_volatile: {
    item_no: string;
    loc: string;
    lt_mean_days: number | null;
    lt_std_days: number | null;
    lt_cv: number | null;
    lt_min_days: number | null;
    lt_max_days: number | null;
    observation_count: number | null;
    lt_variability_class: string | null;
  }[];
}

export interface LtProfileRow {
  item_no: string;
  loc: string;
  lt_mean_days: number | null;
  lt_std_days: number | null;
  lt_cv: number | null;
  lt_min_days: number | null;
  lt_max_days: number | null;
  lt_p25_days: number | null;
  lt_p50_days: number | null;
  lt_p75_days: number | null;
  lt_p95_days: number | null;
  observation_count: number | null;
  observation_months: number | null;
  lt_variability_class: string | null;
  computed_at: string | null;
}

export interface LtProfilePayload {
  total: number;
  rows: LtProfileRow[];
}

export async function fetchLtSummary(params: {
  abc_vol?: string;
}): Promise<LtSummaryPayload> {
  const qs = new URLSearchParams();
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  return fetchJson(`/inv-planning/lead-time/summary?${qs}`);
}

export async function fetchLtProfile(params: {
  item?: string;
  location?: string;
  lt_variability_class?: string;
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: string;
}): Promise<LtProfilePayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 50),
    offset: String(params.offset ?? 0),
    sort_by: params.sort_by ?? "lt_cv",
    sort_dir: params.sort_dir ?? "desc",
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  if (params.lt_variability_class?.trim()) qs.set("lt_variability_class", params.lt_variability_class.trim());
  return fetchJson(`/inv-planning/lead-time/profile?${qs}`);
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature3: Safety Stock
// ---------------------------------------------------------------------------

export const safetyStockKeys = {
  summary: (filters?: Record<string, string>) => ["safety-stock", "summary", filters] as const,
  detail: (params?: Record<string, unknown>) => ["safety-stock", "detail", params] as const,
  waterfall: (itemNo: string, loc: string) => ["safety-stock", "waterfall", itemNo, loc] as const,
  config: () => ["safety-stock", "config"] as const,
};

export interface SafetyStockSummary {
  total_dfus: number;
  below_ss_count: number;
  avg_ss_coverage: number;
  avg_ss_days: number;
  by_abc: Array<{ abc_vol: string; count: number; below_ss_count: number; avg_coverage: number }>;
}

export interface SafetyStockRow {
  item_no: string;
  loc: string;
  ss_combined: number;
  ss_coverage: number;
  is_below_ss: boolean;
  reorder_point: number;
  abc_vol: string;
  ss_demand_only: number;
  ss_lt_only: number;
}

export interface SafetyStockWaterfall {
  item_no: string;
  loc: string;
  ss_demand_only: number;
  ss_lt_only: number;
  ss_combined: number;
  reorder_point: number;
  avg_daily_demand: number;
  lt_mean_days: number;
}

export async function fetchSafetyStockSummary(filters?: Record<string, string>): Promise<SafetyStockSummary> {
  const params = new URLSearchParams(filters ?? {});
  const res = await fetch(`/inv-planning/safety-stock/summary?${params}`);
  if (!res.ok) throw new Error("Failed to fetch safety stock summary");
  return res.json();
}

export async function fetchSafetyStockDetail(params?: {
  is_below_ss?: boolean;
  abc_vol?: string;
  item?: string;
  loc?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; rows: SafetyStockRow[] }> {
  const p = new URLSearchParams();
  if (params?.is_below_ss !== undefined) p.set("is_below_ss", String(params.is_below_ss));
  if (params?.abc_vol) p.set("abc_vol", params.abc_vol);
  if (params?.item) p.set("item", params.item);
  if (params?.loc) p.set("loc", params.loc);
  if (params?.limit !== undefined) p.set("limit", String(params.limit));
  if (params?.offset !== undefined) p.set("offset", String(params.offset));
  const res = await fetch(`/inv-planning/safety-stock/detail?${p}`);
  if (!res.ok) throw new Error("Failed to fetch safety stock detail");
  return res.json();
}

export async function fetchSafetyStockWaterfall(itemNo: string, loc: string): Promise<SafetyStockWaterfall> {
  const res = await fetch(
    `/inv-planning/safety-stock/waterfall?item_no=${encodeURIComponent(itemNo)}&loc=${encodeURIComponent(loc)}`,
  );
  if (!res.ok) throw new Error("Failed to fetch safety stock waterfall");
  return res.json();
}

export async function fetchSafetyStockConfig(): Promise<Record<string, unknown>> {
  const res = await fetch("/inv-planning/safety-stock/config");
  if (!res.ok) throw new Error("Failed to fetch safety stock config");
  return res.json();
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature4: EOQ & Cycle Stock
// ---------------------------------------------------------------------------

export interface EoqAbcEntry {
  abc_vol: string;
  count: number;
  avg_eoq: number | null;
  total_cycle_stock: number | null;
  total_annual_cost: number | null;
  avg_order_frequency: number | null;
}

export interface EoqSummaryPayload {
  total_dfus: number;
  avg_effective_eoq: number | null;
  total_cycle_stock: number | null;
  avg_order_frequency: number | null;
  total_annual_cost: number | null;
  by_abc: EoqAbcEntry[];
}

export interface EoqDetailRow {
  item_no: string;
  loc: string;
  abc_vol: string | null;
  demand_mean_monthly: number | null;
  annual_demand: number | null;
  ordering_cost: number | null;
  holding_cost_pct: number | null;
  unit_cost: number | null;
  moq: number | null;
  eoq: number | null;
  effective_eoq: number | null;
  eoq_cycle_stock: number | null;
  order_frequency: number | null;
  annual_holding_cost: number | null;
  annual_order_cost: number | null;
  total_annual_cost: number | null;
  computed_at: string | null;
}

export interface EoqDetailPayload {
  total: number;
  limit: number;
  offset: number;
  rows: EoqDetailRow[];
}

export interface EoqSensitivityPoint {
  ordering_cost: number;
  eoq: number;
  effective_eoq: number;
  total_annual_cost: number;
}

export interface EoqSensitivityPayload {
  item_no: string | null;
  loc: string | null;
  avg_demand_monthly: number;
  curve: EoqSensitivityPoint[];
}

export async function fetchEoqSummary(params: { abc_vol?: string }): Promise<EoqSummaryPayload> {
  const qs = new URLSearchParams();
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  return fetchJson(`/inv-planning/eoq/summary?${qs}`);
}

export async function fetchEoqDetail(params: {
  item?: string;
  loc?: string;
  abc_vol?: string;
  sort_by?: string;
  sort_dir?: string;
  limit?: number;
  offset?: number;
}): Promise<EoqDetailPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 50),
    offset: String(params.offset ?? 0),
    sort_by: params.sort_by ?? "total_annual_cost",
    sort_dir: params.sort_dir ?? "desc",
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.loc?.trim()) qs.set("loc", params.loc.trim());
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  return fetchJson(`/inv-planning/eoq/detail?${qs}`);
}

export async function fetchEoqSensitivity(params: {
  item?: string;
  loc?: string;
}): Promise<EoqSensitivityPayload> {
  const qs = new URLSearchParams();
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.loc?.trim()) qs.set("loc", params.loc.trim());
  return fetchJson(`/inv-planning/eoq/sensitivity?${qs}`);
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature5: Replenishment Policy Management
// ---------------------------------------------------------------------------

export interface ReplenishmentPolicy {
  policy_id: string;
  policy_name: string;
  policy_type: "continuous_rop" | "periodic_review" | "min_max" | "manual";
  segment: string | null;
  review_cycle_days: number | null;
  service_level: number | null;
  use_eoq: boolean;
  use_safety_stock: boolean;
  active: boolean;
  dfu_count: number;
}

export interface PolicyListPayload {
  policies: ReplenishmentPolicy[];
}

export interface PolicyAssignmentRow {
  item_no: string;
  loc: string;
  policy_id: string;
  policy_name: string;
  policy_type: string;
  override_reason: string | null;
  assigned_by: string;
  effective_date: string | null;
}

export interface PolicyAssignmentsPayload {
  total: number;
  rows: PolicyAssignmentRow[];
}

export interface PolicyComplianceByPolicy {
  policy_name: string;
  policy_type: string;
  dfu_count: number;
  below_ss_pct: number | null;
  avg_ss_coverage: number | null;
  avg_dos: number | null;
}

export interface PolicyCompliancePayload {
  total_dfus: number;
  assigned_count: number;
  unassigned_count: number;
  assignment_pct: number;
  by_policy: Record<string, PolicyComplianceByPolicy>;
}

export interface PolicyAssignResult {
  assigned_count: number;
  failed_count: number;
  already_assigned_count: number;
}

export async function fetchPolicies(): Promise<PolicyListPayload> {
  return fetchJson("/inv-planning/policies");
}

export async function createPolicy(body: {
  policy_id: string;
  policy_name: string;
  policy_type: string;
  segment?: string;
  review_cycle_days?: number;
  service_level?: number;
  use_eoq?: boolean;
  use_safety_stock?: boolean;
  notes?: string;
}): Promise<ReplenishmentPolicy> {
  return fetchJson("/inv-planning/policies", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function updatePolicy(
  policyId: string,
  body: Partial<{
    policy_name: string;
    policy_type: string;
    segment: string;
    review_cycle_days: number;
    service_level: number;
    use_eoq: boolean;
    use_safety_stock: boolean;
    active: boolean;
    notes: string;
  }>,
): Promise<ReplenishmentPolicy> {
  return fetchJson(`/inv-planning/policies/${encodeURIComponent(policyId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function fetchPolicyAssignments(params: {
  item?: string;
  location?: string;
  policy_id?: string;
  assigned_by?: string;
  limit?: number;
  offset?: number;
}): Promise<PolicyAssignmentsPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 50),
    offset: String(params.offset ?? 0),
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  if (params.policy_id?.trim()) qs.set("policy_id", params.policy_id.trim());
  if (params.assigned_by?.trim()) qs.set("assigned_by", params.assigned_by.trim());
  return fetchJson(`/inv-planning/policy-assignments?${qs}`);
}

export async function assignPolicy(body: {
  item_no?: string;
  loc?: string;
  policy_id?: string;
  override_reason?: string;
  segment?: string;
}): Promise<PolicyAssignResult> {
  return fetchJson("/inv-planning/policy-assignments/assign", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function fetchPolicyCompliance(): Promise<PolicyCompliancePayload> {
  return fetchJson("/inv-planning/policy-assignments/compliance");
}

// ---------------------------------------------------------------------------
// IPfeature6 — Inventory Health Score
// ---------------------------------------------------------------------------

export const healthKeys = {
  summary: (filters?: HealthSummaryFilters) => ["health-summary", filters ?? {}] as const,
  detail:  (params?: HealthDetailParams)   => ["health-detail",  params ?? {}]   as const,
  heatmap: (groupX?: string, groupY?: string) => ["health-heatmap", groupX ?? "abc_vol", groupY ?? "variability_class"] as const,
};

export interface HealthSummaryFilters {
  abc_vol?: string;
  cluster_assignment?: string;
  region?: string;
  variability_class?: string;
}

export interface HealthTierBreakdown {
  healthy: number;
  monitor: number;
  at_risk: number;
  critical: number;
}

export interface HealthComponentAvgs {
  ss_coverage: number | null;
  dos_target: number | null;
  stockout_risk: number | null;
  forecast_accuracy: number | null;
}

export interface HealthHistogramBucket {
  bucket: string;
  count: number;
}

export interface HealthSummaryPayload {
  total_dfus: number;
  by_tier: HealthTierBreakdown;
  avg_health_score: number | null;
  component_avgs: HealthComponentAvgs;
  score_histogram: HealthHistogramBucket[];
}

export interface HealthDetailParams {
  item?: string;
  location?: string;
  health_tier?: string;
  abc_vol?: string;
  cluster_assignment?: string;
  variability_class?: string;
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: string;
}

export interface HealthDetailRow {
  item_no: string;
  loc: string;
  abc_vol: string | null;
  variability_class: string | null;
  cluster_assignment: string | null;
  health_score: number;
  health_tier: string;
  score_ss_coverage: number;
  score_dos_target: number;
  score_stockout_risk: number;
  score_forecast_accuracy: number;
  ss_coverage: number | null;
  current_dos: number | null;
  target_dos_min: number | null;
  target_dos_max: number | null;
  is_below_ss: boolean | null;
  recent_wape: number | null;
  stockout_count_3m: number | null;
}

export interface HealthDetailPayload {
  total: number;
  rows: HealthDetailRow[];
}

export interface HealthHeatmapCell {
  x: string;
  y: string;
  avg_health_score: number | null;
  count: number;
  critical_count: number;
}

export interface HealthHeatmapPayload {
  x_labels: string[];
  y_labels: string[];
  cells: HealthHeatmapCell[];
}

export async function fetchHealthSummary(
  filters: HealthSummaryFilters = {},
): Promise<HealthSummaryPayload> {
  const qs = new URLSearchParams();
  if (filters.abc_vol) qs.set("abc_vol", filters.abc_vol);
  if (filters.cluster_assignment) qs.set("cluster_assignment", filters.cluster_assignment);
  if (filters.region) qs.set("region", filters.region);
  if (filters.variability_class) qs.set("variability_class", filters.variability_class);
  const q = qs.toString();
  return fetchJson(`/inv-planning/health/summary${q ? "?" + q : ""}`);
}

export async function fetchHealthDetail(
  params: HealthDetailParams = {},
): Promise<HealthDetailPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 100),
    offset: String(params.offset ?? 0),
    sort_by: params.sort_by ?? "health_score",
    sort_dir: params.sort_dir ?? "asc",
  });
  if (params.item) qs.set("item", params.item);
  if (params.location) qs.set("location", params.location);
  if (params.health_tier) qs.set("health_tier", params.health_tier);
  if (params.abc_vol) qs.set("abc_vol", params.abc_vol);
  if (params.cluster_assignment) qs.set("cluster_assignment", params.cluster_assignment);
  if (params.variability_class) qs.set("variability_class", params.variability_class);
  return fetchJson(`/inv-planning/health/detail?${qs}`);
}

export async function fetchHealthHeatmap(
  group_x = "abc_vol",
  group_y = "variability_class",
): Promise<HealthHeatmapPayload> {
  return fetchJson(
    `/inv-planning/health/heatmap?group_x=${encodeURIComponent(group_x)}&group_y=${encodeURIComponent(group_y)}`,
  );
}

// ---------------------------------------------------------------------------
// IPfeature7 — Exception Queue & Replenishment Recommendations
// ---------------------------------------------------------------------------

export interface ExceptionSummaryFilters {
  status?: string;
}

export interface ExceptionListParams {
  exception_type?: string;
  severity?: string;
  status?: string;
  item?: string;
  location?: string;
  sort_by?: string;
  sort_dir?: string;
  limit?: number;
  offset?: number;
}

export interface ExceptionRow {
  exception_id: string;
  item_no: string;
  loc: string;
  exception_date: string;
  exception_type: string;
  severity: string;
  current_qty_on_hand: number | null;
  current_dos: number | null;
  ss_combined: number | null;
  reorder_point: number | null;
  recommended_order_qty: number | null;
  recommended_order_by: string | null;
  expected_receipt_date: string | null;
  estimated_order_value: number | null;
  policy_id: string | null;
  status: string;
  acknowledged_by: string | null;
  notes: string | null;
}

export interface ExceptionListPayload {
  total: number;
  limit: number;
  offset: number;
  rows: ExceptionRow[];
}

export interface ExceptionSummaryPayload {
  open_count: number;
  by_type: Record<string, number>;
  by_severity: { critical: number; high: number; medium: number; low: number };
  total_recommended_order_value: number;
  oldest_open_days: number;
}

export interface ExceptionGeneratePayload {
  generated_count: number;
  skipped_dedup: number;
  by_type: Record<string, number>;
}

export const exceptionKeys = {
  list:    (p?: ExceptionListParams)      => ["exception-list",    p ?? {}] as const,
  summary: (f?: ExceptionSummaryFilters)  => ["exception-summary", f ?? {}] as const,
};

export async function fetchExceptions(
  params: ExceptionListParams = {},
): Promise<ExceptionListPayload> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/inv-planning/exceptions${q ? `?${q}` : ""}`);
}

export async function fetchExceptionSummary(
  filters: ExceptionSummaryFilters = {},
): Promise<ExceptionSummaryPayload> {
  const qs = new URLSearchParams();
  if (filters.status) qs.set("status", filters.status);
  const q = qs.toString();
  return fetchJson(`/inv-planning/exceptions/summary${q ? `?${q}` : ""}`);
}

export async function acknowledgeException(
  exceptionId: string,
  acknowledgedBy: string,
  notes?: string,
): Promise<ExceptionRow> {
  return fetchJson(`/inv-planning/exceptions/${encodeURIComponent(exceptionId)}/acknowledge`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ acknowledged_by: acknowledgedBy, notes }),
  });
}

export async function updateExceptionStatus(
  exceptionId: string,
  status: "ordered" | "resolved",
  notes?: string,
): Promise<ExceptionRow> {
  return fetchJson(`/inv-planning/exceptions/${encodeURIComponent(exceptionId)}/status`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status, notes }),
  });
}

export async function generateExceptions(): Promise<ExceptionGeneratePayload> {
  return fetchJson("/inv-planning/exceptions/generate", { method: "POST" });
}

// ---------------------------------------------------------------------------
// IPfeature11: ABC-XYZ Classification
// ---------------------------------------------------------------------------

export interface AbcXyzCell {
  abc_vol: string;
  xyz_class: string;
  segment: string;
  dfu_count: number;
  avg_service_level: number | null;
  avg_dos_min: number | null;
  avg_dos_max: number | null;
}

export interface AbcXyzDetailRow {
  dmdunit: string;
  dmdgroup: string;
  loc: string;
  abc_vol: string | null;
  xyz_class: string | null;
  abc_xyz_segment: string | null;
  demand_cv: number | null;
  intermittency_ratio: number | null;
  abc_xyz_dos_min: number | null;
  abc_xyz_dos_max: number | null;
  abc_xyz_service_level: number | null;
}

export const abcXyzKeys = {
  matrix:  () => ["abc-xyz-matrix"] as const,
  summary: () => ["abc-xyz-summary"] as const,
  detail:  (f?: Record<string, unknown>) => ["abc-xyz-detail", f ?? {}] as const,
};

export const fetchAbcXyzMatrix = (): Promise<{ cells: AbcXyzCell[]; total_classified: number }> =>
  fetchJson("/inv-planning/abc-xyz/matrix");

export const fetchAbcXyzSummary = (): Promise<Record<string, number | null>> =>
  fetchJson("/inv-planning/abc-xyz/summary");

export async function fetchAbcXyzDetail(
  params: Record<string, unknown> = {},
): Promise<{ total: number; rows: AbcXyzDetailRow[] }> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/inv-planning/abc-xyz/detail${q ? `?${q}` : ""}`);
}

// ---------------------------------------------------------------------------
// IPfeature12: Supplier Performance
// ---------------------------------------------------------------------------

export interface SupplierRow {
  supplier_no: string;
  supplier_name: string | null;
  sku_loc_count: number;
  distinct_items: number;
  avg_lt_mean_days: number | null;
  avg_lt_cv: number | null;
  avg_lt_std_days: number | null;
  pct_stable_lt: number | null;
  pct_volatile_lt: number | null;
  total_safety_stock_units: number | null;
  total_ss_value: number | null;
  supplier_reliability_score: number | null;
}

export const supplierKeys = {
  summary: () => ["supplier-perf-summary"] as const,
  detail:  (f?: Record<string, unknown>) => ["supplier-perf-detail", f ?? {}] as const,
  items:   (supplierNo: string) => ["supplier-perf-items", supplierNo] as const,
};

export const fetchSupplierSummary = (): Promise<Record<string, number | null>> =>
  fetchJson("/inv-planning/supplier-performance/summary");

export async function fetchSupplierDetail(
  params: Record<string, unknown> = {},
): Promise<{ total: number; rows: SupplierRow[] }> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/inv-planning/supplier-performance/detail${q ? `?${q}` : ""}`);
}

// ---------------------------------------------------------------------------
// IPfeature14: Intra-Month Stockout Detection
// ---------------------------------------------------------------------------

export interface IntramonthStockoutRow {
  item_no: string;
  loc: string;
  month_start: string;
  snapshot_days: number;
  stockout_days: number;
  stockout_day_rate: number | null;
  min_qty_on_hand: number | null;
  max_qty_on_hand: number | null;
  avg_qty_on_hand: number | null;
  est_lost_sales: number | null;
  had_full_stockout: boolean;
  had_extended_stockout: boolean;
  abc_vol: string | null;
  abc_xyz_segment: string | null;
  cluster_assignment: string | null;
}

export const intramonthKeys = {
  summary: (f?: Record<string, unknown>) => ["intramonth-summary", f ?? {}] as const,
  detail:  (f?: Record<string, unknown>) => ["intramonth-detail", f ?? {}] as const,
};

export async function fetchIntramonthSummary(
  params: Record<string, unknown> = {},
): Promise<Record<string, number | null>> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/inv-planning/intramonth-stockouts/summary${q ? `?${q}` : ""}`);
}

export async function fetchIntramonthDetail(
  params: Record<string, unknown> = {},
): Promise<{ total: number; rows: IntramonthStockoutRow[] }> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/inv-planning/intramonth-stockouts/detail${q ? `?${q}` : ""}`);
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature9: Demand Signals
// ---------------------------------------------------------------------------

export const demandSignalsKeys = {
  summary: (date?: string) => ["demand-signals", "summary", date] as const,
  list: (params?: Record<string, unknown>) => ["demand-signals", "list", params] as const,
  item: (itemNo: string, loc: string) => ["demand-signals", "item", itemNo, loc] as const,
};

export interface DemandSignalSummary {
  signal_date: string;
  above_plan_count: number;
  below_plan_count: number;
  on_plan_count: number;
  urgent_count: number;
  watch_count: number;
  projected_stockouts: number;
}

export interface DemandSignalRow {
  item_no: string;
  loc: string;
  signal_date: string;
  signal_type: string;
  signal_strength: number;
  demand_vs_forecast_pct: number;
  projected_stockout: boolean;
  alert_priority: string;
  mtd_actual: number;
  projected_monthly: number;
  forecast_monthly: number;
  current_on_hand: number;
  is_below_ss: boolean;
}

export async function fetchDemandSignalsSummary(date?: string): Promise<DemandSignalSummary> {
  const p = date ? `?signal_date=${date}` : "";
  const res = await fetch(`/inv-planning/demand-signals/summary${p}`);
  if (!res.ok) throw new Error("Failed to fetch demand signals summary");
  return res.json();
}

export async function fetchDemandSignals(params?: {
  signal_type?: string;
  alert_priority?: string;
  item?: string;
  loc?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; rows: DemandSignalRow[] }> {
  const p = new URLSearchParams();
  if (params?.signal_type) p.set("signal_type", params.signal_type);
  if (params?.alert_priority) p.set("alert_priority", params.alert_priority);
  if (params?.item) p.set("item", params.item);
  if (params?.loc) p.set("loc", params.loc);
  if (params?.limit !== undefined) p.set("limit", String(params.limit));
  if (params?.offset !== undefined) p.set("offset", String(params.offset));
  const res = await fetch(`/inv-planning/demand-signals?${p}`);
  if (!res.ok) throw new Error("Failed to fetch demand signals");
  return res.json();
}

export async function fetchDemandSignalItem(itemNo: string, loc: string): Promise<DemandSignalRow> {
  const res = await fetch(
    `/inv-planning/demand-signals/item?item_no=${encodeURIComponent(itemNo)}&loc=${encodeURIComponent(loc)}`,
  );
  if (!res.ok) throw new Error("Failed to fetch demand signal item");
  return res.json();
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature10: Safety Stock Monte Carlo Simulation
// ---------------------------------------------------------------------------

export const simulationKeys = {
  results: (params?: Record<string, unknown>) => ["simulation", "results", params] as const,
  compare: (itemNo: string, loc: string) => ["simulation", "compare", itemNo, loc] as const,
  status: (simRunId: string) => ["simulation", "status", simRunId] as const,
};

export interface SimulationResult {
  sim_run_id: string;
  item_no: string;
  loc: string;
  simulation_date: string;
  n_simulations: number;
  target_csl: number;
  recommended_ss: number;
  recommended_ss_days: number;
  analytical_ss: number;
  sim_vs_analytical_pct: number;
  run_duration_secs: number;
  results_by_ss_level: Array<{ ss_qty: number; csl: number }>;
}

export async function fetchSimulationResults(params?: {
  item?: string;
  loc?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; rows: SimulationResult[] }> {
  const p = new URLSearchParams();
  if (params?.item) p.set("item", params.item);
  if (params?.loc) p.set("loc", params.loc);
  if (params?.limit !== undefined) p.set("limit", String(params.limit));
  const res = await fetch(`/inv-planning/simulation/results?${p}`);
  if (!res.ok) throw new Error("Failed to fetch simulation results");
  return res.json();
}

export async function fetchSimulationCompare(itemNo: string, loc: string): Promise<SimulationResult[]> {
  const res = await fetch(
    `/inv-planning/simulation/compare?item_no=${encodeURIComponent(itemNo)}&loc=${encodeURIComponent(loc)}`,
  );
  if (!res.ok) throw new Error("Failed to fetch simulation compare");
  return res.json();
}

export async function runSimulation(body: {
  item_no: string;
  loc: string;
  n_simulations?: number;
  target_csl?: number;
}): Promise<SimulationResult> {
  const res = await fetch("/inv-planning/simulation/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("Failed to run simulation");
  return res.json();
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature13: Investment Plan
// ---------------------------------------------------------------------------

export const investmentKeys = {
  summary: (planId?: string) => ["investment", "summary", planId] as const,
  detail: (params?: Record<string, unknown>) => ["investment", "detail", params] as const,
  frontier: (planId?: string) => ["investment", "frontier", planId] as const,
};

export interface InvestmentSummary {
  plan_id: string;
  computation_date: string;
  total_items: number;
  total_current_investment: number;
  total_recommended_investment: number;
  total_investment_gap: number;
  avg_current_csl: number;
  avg_recommended_csl: number;
}

export interface InvestmentRow {
  item_no: string;
  loc: string;
  abc_vol: string;
  abc_xyz_segment: string;
  current_ss_qty: number;
  current_ss_value: number;
  current_csl: number;
  recommended_ss_qty: number;
  recommended_ss_value: number;
  recommended_csl: number;
  ss_increment_qty: number;
  investment_increment: number;
  csl_increment: number;
  marginal_roi: number;
  investment_rank: number;
  cumulative_investment: number;
}

export interface FrontierPoint {
  plan_id: string;
  budget_point: number;
  items_funded: number;
  achievable_csl: number;
  marginal_item: string;
}

export async function fetchInvestmentSummary(planId?: string): Promise<InvestmentSummary> {
  const p = planId ? `?plan_id=${planId}` : "";
  const res = await fetch(`/inv-planning/investment/summary${p}`);
  if (!res.ok) throw new Error("Failed to fetch investment summary");
  return res.json();
}

export async function fetchInvestmentDetail(params?: {
  plan_id?: string;
  abc_vol?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; rows: InvestmentRow[] }> {
  const p = new URLSearchParams();
  if (params?.plan_id) p.set("plan_id", params.plan_id);
  if (params?.abc_vol) p.set("abc_vol", params.abc_vol);
  if (params?.limit !== undefined) p.set("limit", String(params.limit));
  if (params?.offset !== undefined) p.set("offset", String(params.offset));
  const res = await fetch(`/inv-planning/investment/detail?${p}`);
  if (!res.ok) throw new Error("Failed to fetch investment detail");
  return res.json();
}

export async function fetchInvestmentFrontier(planId?: string): Promise<FrontierPoint[]> {
  const p = planId ? `?plan_id=${planId}` : "";
  const res = await fetch(`/inv-planning/investment/efficient-frontier${p}`);
  if (!res.ok) throw new Error("Failed to fetch investment frontier");
  return res.json();
}

export async function runInvestmentPlan(params?: {
  budget?: number;
  target_csl?: number;
}): Promise<{ plan_id: string; total_items: number; total_investment_gap: number }> {
  const p = new URLSearchParams();
  if (params?.budget !== undefined) p.set("budget", String(params.budget));
  if (params?.target_csl !== undefined) p.set("target_csl", String(params.target_csl));
  const res = await fetch(`/inv-planning/investment/plan?${p}`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to run investment plan");
  return res.json();
}
