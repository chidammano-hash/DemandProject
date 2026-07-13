import { fetchJson } from "./request";
import type {
  DashboardKpis,
  Alert,
  Mover,
  HeatmapRow,
  DistinctValuesPayload,
} from "@/types/theme";

// ---------------------------------------------------------------------------
// Distinct values (Feature 36 -- global filter dropdowns)
// ---------------------------------------------------------------------------
export interface CascadeFilterParams {
  brand?: string;
  category?: string;
  item?: string;
  location?: string;
  market?: string;
  channel?: string;
  cluster?: string;
}

export async function fetchDistinctValues(
  domain: string,
  column: string,
  search?: string,
  limit = 100,
  cascade?: CascadeFilterParams,
): Promise<DistinctValuesPayload> {
  const qs = new URLSearchParams({ column, limit: String(limit) });
  if (search) qs.set("search", search);
  if (cascade) {
    if (cascade.brand) qs.set("brand", cascade.brand);
    if (cascade.category) qs.set("category", cascade.category);
    if (cascade.item) qs.set("item", cascade.item);
    if (cascade.location) qs.set("location", cascade.location);
    if (cascade.market) qs.set("market", cascade.market);
    if (cascade.channel) qs.set("channel", cascade.channel);
    if (cascade.cluster) qs.set("cluster", cascade.cluster);
  }
  return fetchJson(`/domains/${encodeURIComponent(domain)}/distinct?${qs}`);
}

// ---------------------------------------------------------------------------
// Dashboard queries (Feature 36)
// ---------------------------------------------------------------------------
export interface DashboardFilterParams {
  brand?: string[];
  category?: string[];
  market?: string[];
  channel?: string[];
  item?: string[];
  location?: string[];
  cluster?: string[];
  time_grain?: "month" | "quarter";
}

function appendFilterParams(qs: URLSearchParams, params?: DashboardFilterParams) {
  if (!params) return;
  if (params.brand?.length) qs.set("brand", params.brand.join(","));
  if (params.category?.length) qs.set("category", params.category.join(","));
  if (params.market?.length) qs.set("market", params.market.join(","));
  if (params.channel?.length) qs.set("channel", params.channel.join(","));
  if (params.item?.length) qs.set("item", params.item.join(","));
  if (params.location?.length) qs.set("location", params.location.join(","));
  if (params.cluster?.length) qs.set("cluster_assignment", params.cluster.join(","));
  if (params.time_grain) qs.set("time_grain", params.time_grain);
}

// ---------------------------------------------------------------------------
// Planning date
// ---------------------------------------------------------------------------

export interface PlanningDateInfo {
  planning_date: string;   // ISO date string e.g. "2026-02-24"
  system_date: string;     // ISO date string e.g. "2026-03-09"
  is_frozen: boolean;      // true when planning_date !== system_date
  days_behind: number;     // system_date - planning_date in days
}

export async function fetchPlanningDate(): Promise<PlanningDateInfo> {
  return fetchJson("/dashboard/planning-date");
}

// ---------------------------------------------------------------------------
// Pipeline readiness — derivable staleness of downstream ML stages
// ---------------------------------------------------------------------------

/** A one-click remediation the UI can offer for a stale stage — currently a
 *  deep-link to the tab where the user fixes it (e.g. Clustering). */
export interface PipelineReadinessAction {
  kind: "navigate";
  target: string;   // tab id, e.g. "clusters"
  label: string;
}

export interface PipelineReadinessCheck {
  stage: string;                       // e.g. "clustering"
  status: "stale" | "ok";
  severity: "high" | "medium" | "low";
  title: string;
  detail: string;
  action: PipelineReadinessAction | null;
}

export interface PipelineReadiness {
  ready: boolean;                      // true when nothing is stale
  checks: PipelineReadinessCheck[];
}

/** Report whether downstream ML stages (clustering, …) are in sync with dim_sku.
 *  Staleness is derived live — it auto-clears once the dependent stage re-runs. */
export async function fetchPipelineReadiness(): Promise<PipelineReadiness> {
  return fetchJson("/dashboard/pipeline-readiness");
}

export const pipelineReadinessKeys = {
  readiness: ["dashboard", "pipeline-readiness"] as const,
};

export async function fetchDashboardKpis(
  window = 3,
  filters?: DashboardFilterParams,
  model = "external",
): Promise<DashboardKpis> {
  const qs = new URLSearchParams({ window: String(window), model });
  appendFilterParams(qs, filters);
  return fetchJson(`/dashboard/kpis?${qs}`);
}

export async function fetchDashboardAlerts(
  limit = 10,
  filters?: DashboardFilterParams,
): Promise<{ alerts: Alert[] }> {
  const qs = new URLSearchParams({ limit: String(limit) });
  appendFilterParams(qs, filters);
  return fetchJson(`/dashboard/alerts?${qs}`);
}

export async function fetchDashboardTopMovers(
  limit = 5,
  direction: "up" | "down" | "both" = "both",
  filters?: DashboardFilterParams,
): Promise<{ movers: Mover[] }> {
  const qs = new URLSearchParams({ limit: String(limit), direction });
  appendFilterParams(qs, filters);
  return fetchJson(`/dashboard/top-movers?${qs}`);
}

export type HeatmapGrain = "category" | "brand" | "location" | "class" | "sub_class" | "date";

export async function fetchDashboardHeatmap(
  grain: HeatmapGrain = "category",
  periods = 4,
  filters?: DashboardFilterParams,
  colGrain: HeatmapGrain = "date",
  model = "external",
): Promise<{ rows: HeatmapRow[]; period_labels: string[]; metric: string }> {
  const qs = new URLSearchParams({ grain, col_grain: colGrain, periods: String(periods), model });
  appendFilterParams(qs, filters);
  return fetchJson(`/dashboard/heatmap?${qs}`);
}

export interface TrendPoint {
  month: string;
  forecast: number;
  actual: number;
}

export async function fetchDashboardTrend(
  window = 12,
  filters?: DashboardFilterParams,
  model = "external",
): Promise<{ months: TrendPoint[] }> {
  const qs = new URLSearchParams({ window: String(window), model });
  appendFilterParams(qs, filters);
  return fetchJson(`/dashboard/trend?${qs}`);
}

// ---------------------------------------------------------------------------
// Customer Map queries
// ---------------------------------------------------------------------------

export interface CustomerMapLocation {
  label: string;
  customer_count: number;
  state?: string;
  lat?: number;
  lon?: number;
}

export async function fetchCustomerMap(
  groupBy: "state" | "zip" | "city" = "state",
): Promise<{ locations: CustomerMapLocation[]; group_by: string; total: number }> {
  return fetchJson(`/dashboard/customer-map?group_by=${groupBy}`);
}
