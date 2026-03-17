/**
 * evolution.ts — API queries for 06-evolution_to_operations features
 * F3.1 Bias Corrections, F3.2 Service Level, F3.3 Lead Time Learning,
 * F3.4 Blended Demand, F3.5 Echelon Planning,
 * F4.1 Financial Plan, F4.2 S&OP, F4.3 Events, F4.4 Supply Scenarios
 */

import { fetchJson } from "./core";
import { buildSearchParams, buildQuerySuffix } from "./helpers";

export const STALE_EVO = { FIVE_MIN: 5 * 60 * 1000, ONE_MIN: 60 * 1000 };

// ---------------------------------------------------------------------------
// F3.1 — Bias Corrections
// ---------------------------------------------------------------------------

export interface BiasCorrectionSummary {
  total_corrections: number;
  dfu_count: number;
  flagged_count: number;
  clipped_count: number;
  avg_rolling_bias: number | null;
  avg_correction_factor: number | null;
  last_computed_at: string | null;
  plan_month: string | null;
}

export interface BiasCorrectionRow {
  item_no: string;
  loc: string;
  plan_month: string | null;
  segment_type: string;
  segment_value: string;
  rolling_bias_3m: number;
  correction_factor: number;
  correction_was_clipped: boolean;
  correction_pct: number;
  flagged_for_review: boolean;
  correction_applied: boolean;
  months_of_data: number;
  computed_at: string | null;
}

export interface FlaggedBiasRow {
  item_no: string;
  loc: string;
  plan_month: string | null;
  segment_type: string;
  rolling_bias_3m: number;
  correction_factor_raw: number;
  correction_factor: number;
  correction_was_clipped: boolean;
  months_of_data: number;
}

export const biasKeys = {
  summary: (plan_month?: string) => ["bias-corrections", "summary", plan_month] as const,
  list: (params: object) => ["bias-corrections", "list", params] as const,
  flagged: (plan_month?: string) => ["bias-corrections", "flagged", plan_month] as const,
  history: (segment_type: string, segment_value?: string) => ["bias-corrections", "history", segment_type, segment_value] as const,
};

export async function fetchBiasCorrectionSummary(plan_month?: string): Promise<BiasCorrectionSummary> {
  return fetchJson(`/forecast/bias-corrections/summary${buildQuerySuffix({ plan_month })}`);
}

export async function fetchBiasCorrections(params: {
  plan_month?: string; item_no?: string; loc?: string; page?: number; page_size?: number;
}): Promise<{ total: number; page: number; corrections: BiasCorrectionRow[] }> {
  const qs = buildSearchParams(params as Record<string, string | number | undefined>);
  return fetchJson(`/forecast/bias-corrections?${qs}`);
}

export async function fetchFlaggedBiasCorrections(plan_month?: string): Promise<{ total: number; page: number; flagged: FlaggedBiasRow[] }> {
  return fetchJson(`/forecast/bias-corrections/flagged${buildQuerySuffix({ plan_month })}`);
}

// ---------------------------------------------------------------------------
// F3.2 — Service Level Achievement
// ---------------------------------------------------------------------------

export interface ServiceLevelSummary {
  total_dfus: number;
  meeting_target: number;
  below_target: number;
  chronic_misses: number;
  avg_achieved: number | null;
  avg_target: number | null;
}

export interface ServiceLevelRow {
  item_no: string;
  loc: string;
  abc_class: string;
  target_service_level: number;
  achieved_service_level: number | null;
  gap_pct: number | null;
  is_chronic_miss: boolean;
  flagged_for_review: boolean;
  months_evaluated: number;
}

export interface ChronicMissRow {
  item_no: string;
  loc: string;
  abc_class: string;
  miss_count: number;
  target_service_level: number;
  avg_achieved: number | null;
  last_miss_month: string | null;
}

export const serviceLevelKeys = {
  summary: (params?: object) => ["service-level", "summary", params] as const,
  detail: (params: object) => ["service-level", "detail", params] as const,
  chronicMisses: (params?: object) => ["service-level", "chronic-misses", params] as const,
};

export async function fetchServiceLevelSummary(params?: { item_no?: string; loc?: string }): Promise<ServiceLevelSummary> {
  return fetchJson(`/analytics/service-level/summary${buildQuerySuffix(params ?? {})}`);
}

export async function fetchServiceLevelDetail(params: { item_no?: string; loc?: string; abc_class?: string; flagged_only?: boolean; page?: number; page_size?: number }): Promise<{ total: number; page: number; items: ServiceLevelRow[] }> {
  const qs = buildSearchParams(params as Record<string, string | number | boolean | undefined>);
  return fetchJson(`/analytics/service-level/detail?${qs}`);
}

export async function fetchChronicMisses(params?: { abc_class?: string; min_misses?: number }): Promise<{ total: number; items: ChronicMissRow[] }> {
  return fetchJson(`/analytics/service-level/chronic-misses${buildQuerySuffix(params ?? {})}`);
}

// ---------------------------------------------------------------------------
// F3.3 — Lead Time Learning
// ---------------------------------------------------------------------------

export interface LeadTimeRow {
  item_no: string;
  supplier_id: string | null;
  loc: string;
  quoted_lt_days: number;
  actual_lt_days_avg: number | null;
  lt_cv: number | null;
  reliability_band: string | null;
  data_points: number;
  last_updated: string | null;
}

export interface LeadTimeSummary {
  total_items: number;
  avg_quoted_lt: number | null;
  avg_actual_lt: number | null;
  reliable_count: number;
  unreliable_count: number;
  avg_lt_cv: number | null;
}

export interface LeadTimeAlert {
  trigger_id: string;
  item_no: string;
  loc: string;
  alert_type: string;
  severity: string;
  message: string;
  created_at: string | null;
  acknowledged: boolean;
}

export const leadTimeKeys = {
  list: (params: object) => ["lead-time", "list", params] as const,
  summary: (params?: object) => ["lead-time", "summary", params] as const,
  alerts: (params?: object) => ["lead-time", "alerts", params] as const,
};

export async function fetchLeadTimeLearning(params: { item_no?: string; loc?: string; supplier_id?: string; page?: number; page_size?: number }): Promise<{ total: number; page: number; items: LeadTimeRow[] }> {
  const qs = buildSearchParams(params as Record<string, string | number | undefined>);
  return fetchJson(`/supply/supplier-lead-times?${qs}`);
}

export async function fetchLeadTimeSummary(params?: { supplier_id?: string }): Promise<LeadTimeSummary> {
  return fetchJson(`/supply/supplier-lead-times/summary${buildQuerySuffix(params ?? {})}`);
}

export async function fetchLeadTimeAlerts(params?: { severity?: string; page?: number }): Promise<{ total: number; items: LeadTimeAlert[] }> {
  return fetchJson(`/supply/lead-time-alerts${buildQuerySuffix(params ?? {})}`);
}

// ---------------------------------------------------------------------------
// F3.4 — Blended Demand Forecast
// ---------------------------------------------------------------------------

export interface BlendedForecastRow {
  item_no: string;
  loc: string;
  week_start: string;
  plan_version: string;
  alpha_weight: number;
  sensing_signal_qty: number;
  statistical_forecast_qty: number;
  blended_qty: number;
  velocity_spike_ratio: number;
  is_outlier_capped: boolean;
}

export interface BlendedSummary {
  total_dfus: number;
  total_weeks: number;
  avg_alpha: number | null;
  capped_count: number;
  plan_version: string | null;
  latest_week: string | null;
}

export interface SensingActive {
  item_no: string;
  loc: string;
  is_active: boolean;
  alpha_current_week: number | null;
  last_signal_date: string | null;
}

export const blendedKeys = {
  list: (params: object) => ["blended-forecast", "list", params] as const,
  summary: (params?: object) => ["blended-forecast", "summary", params] as const,
  sensingActive: (params: object) => ["sensing-active", params] as const,
};

export async function fetchBlendedForecast(params: { item_no?: string; loc?: string; weeks_ahead?: number; plan_version?: string; page?: number; page_size?: number }): Promise<{ total: number; page: number; rows: BlendedForecastRow[] }> {
  const qs = buildSearchParams(params as Record<string, string | number | undefined>);
  return fetchJson(`/forecast/blended?${qs}`);
}

export async function fetchBlendedSummary(params?: { plan_version?: string }): Promise<BlendedSummary> {
  return fetchJson(`/forecast/blended/summary${buildQuerySuffix(params ?? {})}`);
}

// ---------------------------------------------------------------------------
// F3.5 — Multi-Echelon Safety Stock
// ---------------------------------------------------------------------------

export interface EchelonNetworkNode {
  location_id: string;
  node_type: string;
  downstream_count: number;
}

export interface EchelonTargetRow {
  item_no: string;
  loc: string;
  node_type: string;
  pooled_sigma: number | null;
  echelon_ss: number | null;
  echelon_rop: number | null;
  cascade_risk_score: number | null;
  cascade_risk_severity: string | null;
  downstream_coverage_days: number | null;
  computed_at: string | null;
}

export interface EchelonSummary {
  total_nodes: number;
  critical_count: number;
  high_count: number;
  avg_risk_score: number | null;
  avg_coverage_days: number | null;
}

export const echelonKeys = {
  network: () => ["echelon", "network"] as const,
  targets: (params: object) => ["echelon", "targets", params] as const,
  summary: (params?: object) => ["echelon", "summary", params] as const,
  ropList: (params: object) => ["echelon", "rop", params] as const,
};

export async function fetchEchelonNetwork(): Promise<{ nodes: EchelonNetworkNode[]; total: number }> {
  return fetchJson("/supply/echelon/network");
}

export async function fetchEchelonTargets(params: { item_no?: string; node_type?: string; severity?: string; page?: number; page_size?: number }): Promise<{ total: number; page: number; rows: EchelonTargetRow[] }> {
  const qs = buildSearchParams(params as Record<string, string | number | undefined>);
  return fetchJson(`/supply/echelon/targets?${qs}`);
}

export async function fetchEchelonSummary(params?: { item_no?: string }): Promise<EchelonSummary> {
  return fetchJson(`/supply/echelon/summary${buildQuerySuffix(params ?? {})}`);
}

// ---------------------------------------------------------------------------
// F4.1 — Financial Inventory Plan
// ---------------------------------------------------------------------------

export interface InventoryPlanRow {
  item_no: string;
  loc: string;
  abc_class: string | null;
  qty_on_hand: number | null;
  unit_cost: number | null;
  inventory_value: number | null;
  carrying_cost_monthly: number | null;
  excess_value: number | null;
  excess_qty: number | null;
  plan_month: string | null;
}

export interface BudgetStatus {
  budget_id: string;
  category: string;
  budget_cap: number;
  committed_spend: number;
  utilization_pct: number;
  is_breached: boolean;
  effective_from: string | null;
}

export interface WorkingCapitalPoint {
  month: string;
  inventory_value: number;
  carrying_cost: number;
  excess_value: number;
}

export const financialPlanKeys = {
  plan: (params: object) => ["financial-plan", "plan", params] as const,
  budget: (params?: object) => ["financial-plan", "budget", params] as const,
  workingCapital: (params?: object) => ["financial-plan", "working-capital", params] as const,
  excess: (params: object) => ["financial-plan", "excess", params] as const,
};

export async function fetchInventoryPlan(params: { plan_month?: string; abc_class?: string; page?: number; page_size?: number }): Promise<{ total: number; page: number; rows: InventoryPlanRow[] }> {
  const qs = buildSearchParams(params as Record<string, string | number | undefined>);
  return fetchJson(`/finance/inventory-plan?${qs}`);
}

export async function fetchBudgetStatus(params?: { category?: string }): Promise<{ budgets: BudgetStatus[]; total: number }> {
  return fetchJson(`/finance/budget-status${buildQuerySuffix(params ?? {})}`);
}

export async function fetchWorkingCapitalTrend(params?: { months_ahead?: number }): Promise<{ months: WorkingCapitalPoint[] }> {
  return fetchJson(`/finance/working-capital-trend${buildQuerySuffix(params ?? {})}`);
}

// ---------------------------------------------------------------------------
// F4.2 — S&OP Cycle
// ---------------------------------------------------------------------------

export interface SopCycle {
  cycle_id: string;
  cycle_month: string;
  current_stage: string;
  created_at: string | null;
  updated_at: string | null;
  demand_review_date: string | null;
  supply_review_date: string | null;
  pre_sop_date: string | null;
  executive_sop_date: string | null;
}

export interface SopGap {
  gap_id: string;
  category: string;
  gap_type: string;
  gap_qty: number | null;
  gap_value: number | null;
  severity: string;
  resolution_options: string | null;
  mitigation_status: string;
}

export interface ApprovedPlanRow {
  item_no: string;
  loc: string;
  plan_month: string;
  approved_qty: number;
  approved_by: string | null;
  approved_at: string | null;
}

export const sopKeys = {
  cycles: (params?: object) => ["sop", "cycles", params] as const,
  cycle: (cycle_id: string) => ["sop", "cycle", cycle_id] as const,
  gaps: (cycle_id: string) => ["sop", "gaps", cycle_id] as const,
  approvedPlan: (params?: object) => ["sop", "approved-plan", params] as const,
};

export async function fetchSopCycles(params?: { stage?: string; limit?: number }): Promise<{ cycles: SopCycle[]; total: number }> {
  return fetchJson(`/sop/cycles${buildQuerySuffix(params ?? {})}`);
}

export async function fetchSopCycle(cycle_id: string): Promise<SopCycle & { supply_constraints: unknown[]; demand_review_data: unknown[] }> {
  return fetchJson(`/sop/cycles/${cycle_id}`);
}

export async function fetchSopGaps(cycle_id: string): Promise<{ gaps: SopGap[]; total: number }> {
  return fetchJson(`/sop/cycles/${cycle_id}/gaps`);
}

export async function fetchApprovedPlan(params?: { plan_month?: string; item_no?: string; loc?: string; page?: number }): Promise<{ total: number; page: number; rows: ApprovedPlanRow[] }> {
  return fetchJson(`/sop/approved-plan${buildQuerySuffix(params ?? {})}`);
}

// ---------------------------------------------------------------------------
// F4.3 — Event & Promotion Planning
// ---------------------------------------------------------------------------

export interface CalendarEvent {
  event_id: string;
  event_name: string;
  event_type: string;
  start_date: string;
  end_date: string;
  item_no: string | null;
  loc: string | null;
  uplift_multiplier: number;
  additive_qty: number;
  is_hard_override: boolean;
  override_qty: number | null;
  status: string;
  created_by: string | null;
  created_at: string | null;
}

export interface EventImpactPreview {
  event_id: string | null;
  item_no: string;
  loc: string;
  week_start: string;
  base_qty: number;
  adjusted_qty: number;
  uplift_delta: number;
  impact_value: number;
  is_outlier_capped: boolean;
}

export const eventKeys = {
  calendar: (params: object) => ["events", "calendar", params] as const,
  event: (event_id: string) => ["events", "event", event_id] as const,
  impactPreview: (params: object) => ["events", "impact-preview", params] as const,
  performance: (params?: object) => ["events", "performance", params] as const,
};

export async function fetchEventCalendar(params: { year?: number; month?: number; event_type?: string; status?: string }): Promise<{ events: CalendarEvent[]; total: number }> {
  const qs = buildSearchParams(params as Record<string, string | number | undefined>);
  return fetchJson(`/events/calendar?${qs}`);
}

export async function fetchEventImpactPreview(params: { item_no: string; loc: string; uplift_multiplier?: number; additive_qty?: number }): Promise<{ rows: EventImpactPreview[]; total_impact_value: number }> {
  const qs = buildSearchParams(params as Record<string, string | number | undefined>);
  return fetchJson(`/events/impact-preview?${qs}`);
}

// ---------------------------------------------------------------------------
// F4.4 — Supply Chain Scenario Planning
// ---------------------------------------------------------------------------

export interface SupplyScenario {
  scenario_id: string;
  scenario_name: string;
  disruption_type: string;
  item_no: string | null;
  loc: string | null;
  impact_pct: number;
  duration_weeks: number;
  status: string;
  created_at: string | null;
}

export interface ScenarioResult {
  scenario_id: string;
  item_no: string;
  loc: string;
  adjusted_lt_days: number | null;
  lt_increase_days: number | null;
  available_supply: number | null;
  supply_reduction: number | null;
  stockout_days: number | null;
  stockout_cost: number | null;
  holding_cost: number | null;
  total_impact: number | null;
}

export const scenarioKeys = {
  list: (params?: object) => ["supply-scenarios", "list", params] as const,
  scenario: (scenario_id: string) => ["supply-scenarios", "scenario", scenario_id] as const,
  results: (scenario_id: string) => ["supply-scenarios", "results", scenario_id] as const,
};

export async function fetchSupplyScenarios(params?: { disruption_type?: string; status?: string; page?: number }): Promise<{ scenarios: SupplyScenario[]; total: number }> {
  return fetchJson(`/scenarios/supply${buildQuerySuffix(params ?? {})}`);
}

export async function fetchScenarioResults(scenario_id: string): Promise<{ scenario_id: string; items: ScenarioResult[]; total_impact: number; total_stockout_days: number }> {
  return fetchJson(`/scenarios/supply/${scenario_id}/results`);
}
