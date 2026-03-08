import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// F1.1 — Production Forecast (future-period ML predictions)
// ---------------------------------------------------------------------------

export interface ProductionForecastPoint {
  forecast_month: string;
  forecast_qty: number | null;
  forecast_qty_lower: number | null;
  forecast_qty_upper: number | null;
  model_id: string;
  cluster_id: number | null;
  horizon_months: number;
  is_recursive: boolean;
  lag_source: string | null;
}

export interface ProductionForecastPayload {
  item_no: string;
  loc: string;
  plan_version: string;
  model_id: string;
  generated_at: string | null;
  horizon_months: number;
  is_recursive: boolean;
  forecasts: ProductionForecastPoint[];
}

export interface ProductionForecastAbcRow {
  abc_class: string;
  dfu_count: number;
  forecast_qty: number;
}

export interface ProductionForecastSummaryPayload {
  plan_version: string | null;
  horizon_months: number;
  total_dfu_count: number;
  total_forecast_qty: number;
  generated_at: string | null;
  by_abc_class: ProductionForecastAbcRow[];
}

export interface ProductionForecastVersion {
  plan_version: string;
  dfu_count: number;
  total_rows: number;
  generated_at: string | null;
}

export interface ProductionForecastVersionsPayload {
  versions: ProductionForecastVersion[];
}

export async function fetchProductionForecast(params: {
  item_no: string;
  loc: string;
  horizon?: number;
  plan_version?: string;
}): Promise<ProductionForecastPayload> {
  const qs = new URLSearchParams({
    item_no: params.item_no,
    loc: params.loc,
    horizon: String(params.horizon ?? 12),
  });
  if (params.plan_version) qs.set("plan_version", params.plan_version);
  return fetchJson(`/forecast/production?${qs}`);
}

export async function fetchProductionForecastSummary(params: {
  plan_version?: string;
  horizon_months?: number;
  brand?: string;
  category?: string;
}): Promise<ProductionForecastSummaryPayload> {
  const qs = new URLSearchParams({
    horizon_months: String(params.horizon_months ?? 3),
  });
  if (params.plan_version) qs.set("plan_version", params.plan_version);
  if (params.brand?.trim()) qs.set("brand", params.brand.trim());
  if (params.category?.trim()) qs.set("category", params.category.trim());
  return fetchJson(`/forecast/production/summary?${qs}`);
}

export async function fetchProductionForecastVersions(): Promise<ProductionForecastVersionsPayload> {
  return fetchJson("/forecast/production/versions");
}

// ---------------------------------------------------------------------------
// F2.2 — Multi-Horizon Quantile Demand Plan
// ---------------------------------------------------------------------------

export interface DemandPlanRow {
  plan_month: string;
  horizon_months: number;
  p10?: number | null;
  p50?: number | null;
  p90?: number | null;
  sigma_forecast?: number | null;
  sigma_demand?: number | null;
  sigma_combined?: number | null;
}

export interface DemandPlanPayload {
  item_no: string;
  loc: string;
  plan_version: string;
  generated_at: string | null;
  horizon_months: number;
  rows: DemandPlanRow[];
}

export interface DemandPlanVersion {
  plan_version: string;
  plan_date: string | null;
  plan_label: string | null;
  model_id: string;
  horizon_months: number;
  dfu_count: number | null;
  status: string;
  generated_at: string | null;
}

export interface DemandPlanVersionsPayload {
  versions: DemandPlanVersion[];
}

export interface DemandPlanWeekRow {
  plan_week: string;
  iso_week: number;
  iso_year: number;
  parent_month: string | null;
  weekly_weight: number | null;
  p10_weekly?: number | null;
  p50_weekly?: number | null;
  p90_weekly?: number | null;
}

export interface DemandPlanWeeklyPayload {
  item_no: string;
  loc: string;
  plan_version: string;
  weeks: DemandPlanWeekRow[];
}

export interface DemandPlanComparisonMonth {
  plan_month: string;
  v1_p10?: number | null;
  v1_p50?: number | null;
  v1_p90?: number | null;
  v2_p10?: number | null;
  v2_p50?: number | null;
  v2_p90?: number | null;
  delta_p50?: number | null;
  delta_pct?: number | null;
}

export interface DemandPlanComparisonPayload {
  item_no: string;
  loc: string;
  v1: string;
  v2: string;
  months: DemandPlanComparisonMonth[];
}

export async function fetchDemandPlan(params: {
  item_no: string;
  loc: string;
  plan_version?: string;
  quantile?: number;
  horizon?: number;
}): Promise<DemandPlanPayload> {
  const qs = new URLSearchParams({
    item_no: params.item_no,
    loc: params.loc,
    horizon: String(params.horizon ?? 12),
  });
  if (params.plan_version) qs.set("plan_version", params.plan_version);
  if (params.quantile != null) qs.set("quantile", String(params.quantile));
  return fetchJson(`/forecast/demand-plan?${qs}`);
}

export async function fetchDemandPlanVersions(): Promise<DemandPlanVersionsPayload> {
  return fetchJson("/forecast/demand-plan/versions");
}

export async function fetchDemandPlanWeekly(params: {
  item_no: string;
  loc: string;
  plan_version?: string;
  weeks_ahead?: number;
}): Promise<DemandPlanWeeklyPayload> {
  const qs = new URLSearchParams({
    item_no: params.item_no,
    loc: params.loc,
    weeks_ahead: String(params.weeks_ahead ?? 8),
  });
  if (params.plan_version) qs.set("plan_version", params.plan_version);
  return fetchJson(`/forecast/demand-plan/weekly?${qs}`);
}

export async function fetchDemandPlanComparison(params: {
  v1: string;
  v2: string;
  item_no: string;
  loc: string;
}): Promise<DemandPlanComparisonPayload> {
  const qs = new URLSearchParams({
    v1: params.v1,
    v2: params.v2,
    item_no: params.item_no,
    loc: params.loc,
  });
  return fetchJson(`/forecast/demand-plan/comparison?${qs}`);
}

// ---------------------------------------------------------------------------
// F2.3 — Consensus Forecasting & Planner Overrides
// ---------------------------------------------------------------------------

export interface OverrideRow {
  override_id: number;
  item_no: string;
  loc: string;
  override_month: string;
  override_type: string;
  override_qty?: number | null;
  override_multiplier?: number | null;
  override_additive_qty?: number;
  is_hard_override: boolean;
  override_reason: string;
  override_note?: string | null;
  created_by: string;
  created_at: string;
  valid_from: string;
  valid_to: string;
  approved_by?: string | null;
  approved_at?: string | null;
  rejected_by?: string | null;
  rejected_at?: string | null;
  rejection_reason?: string | null;
  status: string;
  requires_approval: boolean;
  priority_rank: number;
  statistical_qty_at_creation?: number | null;
  estimated_impact_units?: number | null;
  estimated_impact_value?: number | null;
  currency?: string | null;
}

export interface OverrideListPayload {
  total: number;
  page: number;
  overrides: OverrideRow[];
}

export interface OverrideSummaryPayload {
  by_status: {
    pending_approval: number;
    approved: number;
    rejected: number;
    expired: number;
    superseded: number;
  };
  dfu_count_overridden: number;
  total_uplift_units: number;
  total_uplift_value: number;
  by_type: Record<string, number>;
}

export interface ConsensusPlanMonth {
  plan_month: string;
  statistical_qty: number | null;
  statistical_p10?: number | null;
  statistical_p90?: number | null;
  override_qty: number;
  consensus_qty: number | null;
  consensus_p10?: number | null;
  consensus_p90?: number | null;
  override_applied: boolean;
  override_type?: string | null;
  override_multiplier?: number | null;
  is_hard_override?: boolean | null;
  overrider?: string | null;
  approver?: string | null;
  uplift_pct: number;
}

export interface ConsensusPlanPayload {
  plan_version: string;
  item_no: string;
  loc: string;
  months: ConsensusPlanMonth[];
}

export async function fetchOverrides(params?: {
  item_no?: string;
  loc?: string;
  status?: string;
  override_type?: string;
  page?: number;
  page_size?: number;
}): Promise<OverrideListPayload> {
  const qs = new URLSearchParams();
  if (params?.item_no) qs.set("item_no", params.item_no);
  if (params?.loc) qs.set("loc", params.loc);
  if (params?.status) qs.set("status", params.status);
  if (params?.override_type) qs.set("override_type", params.override_type);
  if (params?.page) qs.set("page", String(params.page));
  if (params?.page_size) qs.set("page_size", String(params.page_size));
  return fetchJson(`/forecast/overrides?${qs}`);
}

export async function fetchOverrideSummary(): Promise<OverrideSummaryPayload> {
  return fetchJson("/forecast/overrides/summary");
}

export async function submitOverride(body: {
  item_no: string;
  loc: string;
  override_month: string;
  override_type: string;
  override_reason: string;
  created_by: string;
  valid_from: string;
  valid_to: string;
  override_multiplier?: number;
  override_additive_qty?: number;
  override_qty?: number;
  is_hard_override?: boolean;
  statistical_qty?: number;
  priority_rank?: number;
}): Promise<{ override_id: number; status: string; requires_approval: boolean; message: string }> {
  return fetchJson("/forecast/overrides", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function approveOverride(
  id: number,
  approvedBy: string,
): Promise<{ override_id: number; status: string; approved_by: string; approved_at: string }> {
  return fetchJson(`/forecast/overrides/${id}/approve`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved_by: approvedBy }),
  });
}

export async function rejectOverride(
  id: number,
  rejectionReason: string,
): Promise<{ override_id: number; status: string }> {
  return fetchJson(`/forecast/overrides/${id}/reject`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rejected_by: "manager", rejection_reason: rejectionReason }),
  });
}

export async function fetchConsensusPlan(params: {
  item_no: string;
  loc: string;
  plan_version?: string;
  month_from?: string;
  month_to?: string;
}): Promise<ConsensusPlanPayload> {
  const qs = new URLSearchParams({
    item_no: params.item_no,
    loc: params.loc,
  });
  if (params.plan_version) qs.set("plan_version", params.plan_version);
  if (params.month_from) qs.set("month_from", params.month_from);
  if (params.month_to) qs.set("month_to", params.month_to);
  return fetchJson(`/forecast/consensus-plan?${qs}`);
}
