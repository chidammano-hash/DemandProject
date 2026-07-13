import { buildSearchParams } from "./helpers";
import { fetchJson } from "./request";

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
  }[];
}

export const customerForecastKeys = {
  all: ["customer-forecast"] as const,
  readiness: ["customer-forecast", "readiness"] as const,
  latestRun: ["customer-forecast", "latest-run"] as const,
  latestCompletedRun: ["customer-forecast", "latest-completed-run"] as const,
  series: (filters: CustomerForecastFilters) => ["customer-forecast", "series", filters] as const,
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
