import { buildSearchParams } from "./helpers";
import { fetchJson } from "./request";

import type { CustomerForecastRouteModelId } from "@/lib/customerForecast";

export type CustomerForecastRunStatus =
  | "queued"
  | "generating"
  | "completed"
  | "failed"
  | "cancelled";

export interface CustomerForecastReadiness {
  ready: boolean;
  planning_month: string;
  history_start: string;
  history_end: string;
  forecast_start: string;
  forecast_end: string;
  history_months: number;
  horizon_months: number;
  source_latest_month: string | null;
  total_series: number;
  eligible_series: number;
  moving_average_series: number;
  trailing_average_series: number;
  seasonal_repeat_series: number;
  tsb_series: number;
  adida_series: number;
  croston_series: number;
  ses_series: number;
  holt_damped_series: number;
  model_route_counts: Record<CustomerForecastRouteModelId, number>;
  dormant_series: number;
  forecastable_series: number;
  skipped_series: number;
  invalid_key_rows: number;
  duplicate_grains: number;
  negative_rows: number;
  blockers: string[];
}

export interface CustomerForecastRun {
  run_id: string;
  job_id: string | null;
  status: CustomerForecastRunStatus;
  planning_month: string;
  history_start: string;
  history_end: string;
  forecast_start: string;
  forecast_end: string;
  eligible_series: number;
  row_count: number;
  skipped_series: number;
  model_id: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error_summary: string | null;
  skip_reason_counts: Record<string, number>;
  model_route_counts: Record<string, number>;
  total_series: number;
  completed_series: number;
  total_batches: number;
  completed_batches: number;
  progress_pct: number;
  eta_seconds: number | null;
}

export interface CustomerForecastFilters {
  item_id: string;
  location_id: string;
  customer_no: string;
  run_id?: string;
}

export interface CustomerForecastSeries {
  run: CustomerForecastRun | null;
  item_id: string;
  location_id: string;
  customer_no: string;
  history: { month: string; actual_qty: number }[];
  forecast: {
    month: string;
    forecast_qty: number;
    lower_bound: number | null;
    upper_bound: number | null;
    model_id: CustomerForecastRouteModelId | "chronos2_enriched";
  }[];
}

export type CustomerForecastBacktestStatus =
  | "queued"
  | "generating"
  | "completed"
  | "failed"
  | "cancelled";

export type CustomerForecastComparisonModelId =
  | "champion"
  | "customer_bottom_up"
  | "customer_bottom_up_blend";

export interface CustomerForecastBacktestMetric {
  model_id: CustomerForecastComparisonModelId;
  observations: number;
  actual_qty: number | null;
  absolute_error: number | null;
  mae: number | null;
  wape_pct: number | null;
  bias_pct: number | null;
  accuracy_pct: number | null;
}

export interface CustomerForecastBacktest {
  run_id: string;
  job_id: string | null;
  status: CustomerForecastBacktestStatus;
  customer_run_id: string;
  planning_month: string | null;
  common_months: number;
  common_dfus: number;
  common_rows: number;
  actual_qty: number | null;
  component_checksum: string | null;
  completed_at: string | null;
  gate_passed: boolean | null;
  gate_reason: string | null;
  blend_wape_degradation_pct: number | null;
  min_common_months: number | null;
  min_common_dfus: number | null;
  max_wape_degradation_pct: number | null;
  error_summary: string | null;
  metrics: CustomerForecastBacktestMetric[];
}

export interface CustomerForecastJobSubmitted {
  run_id: string;
  job_id: string;
  status: "queued";
}

export interface CustomerBlendRun {
  run_id: string;
  status: "generating" | "ready" | "invalid" | "promoted" | "archived";
  planning_month: string | null;
  horizon_months: number;
  row_count: number;
  dfu_count: number;
  completed_at: string | null;
  job_id: string | null;
  invalid_reason: string | null;
  model_id: string | null;
  customer_run_id: string | null;
  source_run_id: string | null;
  source_production_run_id: string | null;
  source_promotion_id: number | null;
  backtest_run_id: string | null;
  blended_row_count: number;
  champion_fallback_row_count: number;
  customer_only_excluded_count: number;
  bottom_up_staging_run_id: string | null;
  bottom_up_staging_status: "ready" | "invalid" | null;
  bottom_up_staging_row_count: number;
  bottom_up_staging_dfu_count: number;
  promotion_enabled: boolean;
  backtest_gate: Record<string, unknown> | null;
}

export interface CustomerBlendReadiness {
  ready: boolean;
  blockers: string[];
  customer_run_id: string | null;
  source_promotion_id: number | null;
  source_run_id: string | null;
  source_production_run_id: string | null;
  backtest_run_id: string | null;
  backtest_gate_passed: boolean;
  promotion_enabled: boolean;
  promotion_reason: string | null;
}

export interface CustomerBlendSeriesFilters {
  item_id: string;
  location_id: string;
  run_id?: string;
}

export interface CustomerBlendSeriesMonth {
  forecast_month: string;
  raw_customer_demand_qty: number | null;
  normalized_customer_qty: number | null;
  champion_qty: number;
  blended_qty: number;
  lower_bound: number | null;
  upper_bound: number | null;
  fulfillment_ratio: number | null;
  effective_customer_weight: number;
  coverage_status: "blended" | "champion_fallback";
  interval_method: "champion_width_shift" | "champion_passthrough" | "none";
}

export interface CustomerBlendSeries {
  run_id: string;
  customer_run_id: string;
  source_run_id: string;
  source_production_run_id: string;
  item_id: string;
  location_id: string;
  months: CustomerBlendSeriesMonth[];
}

export interface CustomerBlendTrendFilters {
  run_id: string;
  window?: number;
  item_id?: string | string[];
  location_id?: string | string[];
  brand?: string[];
  category?: string[];
  market?: string[];
  channel?: string[];
  cluster?: string[];
}

export interface CustomerBlendTrendMonth {
  month: string;
  phase: "backtest" | "staged";
  actual_qty: number | null;
  customer_bottom_up_qty: number | null;
  source_champion_qty: number | null;
  customer_blend_qty: number | null;
  lower_bound: number | null;
  upper_bound: number | null;
  blended_dfu_count: number;
  fallback_dfu_count: number;
}

export interface CustomerBlendTrend {
  run_id: string;
  status: "ready" | "promoted";
  planning_month: string;
  completed_at: string | null;
  backtest_run_id: string;
  bottom_up_staging_run_id: string;
  backtest_gate: Record<string, unknown> | null;
  filters_applied: Record<string, string[]>;
  filter_notes: string[];
  accuracy: {
    common_actual_qty: number;
    customer_bottom_up_wape_pct: number | null;
    source_champion_wape_pct: number | null;
    customer_blend_wape_pct: number | null;
  };
  coverage: {
    blended_rows: number;
    champion_fallback_rows: number;
    global_customer_only_excluded_count: number;
  };
  months: CustomerBlendTrendMonth[];
}

export const customerForecastKeys = {
  all: ["customer-forecast"] as const,
  readiness: ["customer-forecast", "readiness"] as const,
  latestRun: ["customer-forecast", "latest-run"] as const,
  latestCompletedRun: ["customer-forecast", "latest-completed-run"] as const,
  series: (filters: CustomerForecastFilters) => ["customer-forecast", "series", filters] as const,
  latestBacktest: ["customer-forecast", "backtest", "latest"] as const,
  blendReadiness: (customerRunId?: string) =>
    ["customer-forecast", "blend", "readiness", customerRunId ?? null] as const,
  latestBlend: ["customer-forecast", "blend", "latest"] as const,
  blendSeriesAll: ["customer-forecast", "blend", "series"] as const,
  blendSeries: (filters: CustomerBlendSeriesFilters) =>
    ["customer-forecast", "blend", "series", filters] as const,
  blendTrend: (filters: CustomerBlendTrendFilters) =>
    ["customer-forecast", "blend", "trend", filters] as const,
};

export function fetchCustomerForecastReadiness(): Promise<CustomerForecastReadiness> {
  return fetchJson("/customer-forecast/readiness");
}

export async function fetchLatestCustomerForecastRun(
  completedOnly = false
): Promise<CustomerForecastRun | null> {
  try {
    const suffix = completedOnly ? "?completed_only=true" : "";
    return await fetchJson(`/customer-forecast/runs/latest${suffix}`);
  } catch (error) {
    if (error !== null && typeof error === "object" && "status" in error && error.status === 404) {
      return null;
    }
    throw error;
  }
}

export function generateCustomerForecast(): Promise<{
  run_id: string;
  job_id: string;
  status: "queued";
}> {
  return fetchJson("/customer-forecast/generate", { method: "POST" });
}

export function cancelCustomerForecastRun(
  runId: string
): Promise<{ run_id: string; status: "cancelled" }> {
  return fetchJson(`/customer-forecast/runs/${encodeURIComponent(runId)}/cancel`, {
    method: "POST",
  });
}

export function retryCustomerForecastRun(
  runId: string
): Promise<{ run_id: string; job_id: string; status: "queued" }> {
  return fetchJson(`/customer-forecast/runs/${encodeURIComponent(runId)}/retry`, {
    method: "POST",
  });
}

export function fetchCustomerForecastSeries(
  filters: CustomerForecastFilters
): Promise<CustomerForecastSeries> {
  const params = buildSearchParams({
    item_id: filters.item_id,
    location_id: filters.location_id,
    customer_no: filters.customer_no,
    run_id: filters.run_id,
  });
  return fetchJson(`/customer-forecast/series?${params}`);
}

export function customerForecastExportUrl(filters: CustomerForecastFilters): string {
  const params = buildSearchParams({
    item_id: filters.item_id,
    location_id: filters.location_id,
    customer_no: filters.customer_no,
    run_id: filters.run_id,
  });
  return `/customer-forecast/export?${params}`;
}

function isNotFound(error: unknown): boolean {
  return error !== null && typeof error === "object" && "status" in error && error.status === 404;
}

export async function fetchLatestCustomerForecastBacktest(): Promise<CustomerForecastBacktest | null> {
  try {
    return await fetchJson("/customer-forecast/backtest/latest");
  } catch (error) {
    if (isNotFound(error)) return null;
    throw error;
  }
}

export function generateCustomerForecastBacktest(): Promise<CustomerForecastJobSubmitted> {
  return fetchJson("/customer-forecast/backtest/generate", { method: "POST" });
}

export function fetchCustomerBlendReadiness(
  customerRunId?: string
): Promise<CustomerBlendReadiness> {
  const suffix = customerRunId ? `?${buildSearchParams({ customer_run_id: customerRunId })}` : "";
  return fetchJson(`/customer-forecast/blend/readiness${suffix}`);
}

export async function fetchLatestCustomerBlend(): Promise<CustomerBlendRun | null> {
  try {
    return await fetchJson("/customer-forecast/blend/latest");
  } catch (error) {
    if (isNotFound(error)) return null;
    throw error;
  }
}

export function generateCustomerBlend(
  customerRunId?: string
): Promise<CustomerForecastJobSubmitted> {
  const suffix = customerRunId ? `?${buildSearchParams({ customer_run_id: customerRunId })}` : "";
  return fetchJson(`/customer-forecast/blend/generate${suffix}`, { method: "POST" });
}

export function fetchCustomerBlendSeries(
  filters: CustomerBlendSeriesFilters
): Promise<CustomerBlendSeries> {
  const params = buildSearchParams({
    item_id: filters.item_id,
    location_id: filters.location_id,
    run_id: filters.run_id,
  });
  return fetchJson(`/customer-forecast/blend/series?${params}`);
}

function commaSeparated(value?: string | string[]): string | undefined {
  if (Array.isArray(value)) return value.length > 0 ? value.join(",") : undefined;
  return value || undefined;
}

export function fetchCustomerBlendTrend(
  filters: CustomerBlendTrendFilters
): Promise<CustomerBlendTrend> {
  const params = buildSearchParams({
    run_id: filters.run_id,
    window: filters.window,
    item_id: commaSeparated(filters.item_id),
    location_id: commaSeparated(filters.location_id),
    brand: commaSeparated(filters.brand),
    category: commaSeparated(filters.category),
    market: commaSeparated(filters.market),
    channel: commaSeparated(filters.channel),
    cluster_assignment: commaSeparated(filters.cluster),
  });
  return fetchJson(`/customer-forecast/blend/trend?${params}`);
}
