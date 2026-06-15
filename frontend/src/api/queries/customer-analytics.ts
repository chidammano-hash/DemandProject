import { buildQuerySuffix } from "./helpers";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CustomerAnalyticsFilters {
  item_id?: string;
  date_from?: string;
  date_to?: string;
  channel?: string;
  store_type?: string;
  state?: string;
}

// ---------------------------------------------------------------------------
// Channel Mix extended payload
// ---------------------------------------------------------------------------

export interface ChannelMixPayloadExtended extends ChannelMixPayload {
  grand_total?: number;
  total_customers?: number;
  top_channel?: string;
}

// ---------------------------------------------------------------------------
// KPI types
// ---------------------------------------------------------------------------

export interface KpiMetric {
  value: number;
  // null when the backend has no prior-period anchor to compute MoM (U3.4).
  delta: number | null;
}

export interface KpiPayload {
  total_demand: KpiMetric;
  fill_rate: KpiMetric;
  lost_sales_oos: KpiMetric;
  active_customers: KpiMetric;
  demand_concentration: KpiMetric;
  order_to_demand_ratio: KpiMetric;
}

// ---------------------------------------------------------------------------
// Filter options
// ---------------------------------------------------------------------------

export interface FilterOptionsPayload {
  channels: string[];
  store_types: string[];
  states: string[];
}

// ---------------------------------------------------------------------------
// Lifecycle types
// ---------------------------------------------------------------------------

export interface CohortCell {
  cohort_month: string;
  months_since: number;
  retention_pct: number;
}

export interface WaterfallBar {
  label: string;
  value: number;
  type: "new" | "churned" | "net";
}

export interface LifecyclePayload {
  cohort_heatmap: CohortCell[];
  cohort_months: string[];
  max_months_since: number;
  waterfall: WaterfallBar[];
}

// ---------------------------------------------------------------------------
// Demand at risk
// ---------------------------------------------------------------------------

export interface RiskBar {
  label: string;
  value: number;
  type: "total" | "risk" | "secure";
}

export interface DemandAtRiskPayload {
  bars: RiskBar[];
}

// ---------------------------------------------------------------------------
// Customer-item affinity
// ---------------------------------------------------------------------------

export interface AffinityCell {
  customer: string;
  item: string;
  demand_qty: number;
}

export interface AffinityPayload {
  customers: string[];
  items: string[];
  cells: AffinityCell[];
}

// ---------------------------------------------------------------------------
// Order patterns
// ---------------------------------------------------------------------------

export interface FrequencyBin {
  bin: string;
  count: number;
}

export interface RegularityPoint {
  customer: string;
  avg_interval: number;
  cv: number;
  total_orders: number;
}

export interface OrderPatternsPayload {
  frequency: FrequencyBin[];
  regularity: RegularityPoint[];
}

// ---------------------------------------------------------------------------
// Demand flow sankey
// ---------------------------------------------------------------------------

export interface SankeyNode {
  name: string;
}

export interface SankeyLink {
  source: string;
  target: string;
  value: number;
}

export interface DemandFlowPayload {
  nodes: SankeyNode[];
  links: SankeyLink[];
}

// ---------------------------------------------------------------------------
// Alerts
// ---------------------------------------------------------------------------

export interface AlertEntry {
  alert_type: string;
  severity: "red" | "amber";
  message: string;
  item_id: string | null;
  loc: string | null;
  value: number;
  threshold: number;
}

export interface AlertsPayload {
  alerts: AlertEntry[];
}

export interface MapLocation {
  label: string;
  state: string;
  customer_count: number;
  demand_qty: number;
  sales_qty: number;
  oos_qty: number;
  fill_rate: number;
  lat?: number;
  lon?: number;
}

export interface MapPayload {
  locations: MapLocation[];
  group_by: string;
  metric: string;
  total_demand: number;
  total_customers: number;
}

export interface TreemapNode {
  name: string;
  value: number;
  fill_rate?: number;
  children?: TreemapNode[];
}

export interface TreemapPayload {
  tree: TreemapNode[];
}

export interface HeatmapItem {
  item_id: string;
  item_desc: string;
}

export interface HeatmapCell {
  item_id: string;
  state: string;
  demand_qty: number;
  customer_count: number;
  fill_rate: number;
}

export interface HeatmapPayload {
  items: HeatmapItem[];
  states: string[];
  cells: HeatmapCell[];
  metric: string;
}

export interface SunburstNode {
  name: string;
  value: number;
  customer_count?: number;
  children?: SunburstNode[];
}

export interface ChannelMixPayload {
  tree: SunburstNode[];
}

export interface TrendPoint {
  month: string;
  demand_qty: number;
  sales_qty: number;
  fill_rate: number;
}

export interface SegmentRow {
  segment: string;
  total_demand: number;
  total_customers: number;
  fill_rate: number;
  mom_change: number;
  trend: TrendPoint[];
}

export interface SegmentTrendsPayload {
  segments: SegmentRow[];
  segment_by: string;
}

export interface RankedCustomer {
  customer_no: string;
  customer_name: string;
  state: string;
  channel: string;
  demand_qty: number;
  sales_qty: number;
  oos_qty: number;
  fill_rate: number;
}

export interface RankingPayload {
  customers: RankedCustomer[];
  sort: string;
  top_n: number;
}

export interface OosBubble {
  label: string;
  state: string;
  channel: string;
  demand_qty: number;
  sales_qty: number;
  oos_qty: number;
  fill_rate: number;
  customer_no?: string;
}

export interface OosImpactPayload {
  bubbles: OosBubble[];
  grain: string;
}

export interface ItemOption {
  item_id: string;
  item_desc: string;
}

export interface ItemsPayload {
  items: ItemOption[];
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

function filterParams(f?: CustomerAnalyticsFilters): Record<string, string | undefined> {
  if (!f) return {};
  return {
    item_id: f.item_id || undefined,
    date_from: f.date_from || undefined,
    date_to: f.date_to || undefined,
    channel: f.channel || undefined,
    store_type: f.store_type || undefined,
    state: f.state || undefined,
  };
}

// ---------------------------------------------------------------------------
// Query keys
// ---------------------------------------------------------------------------

export const customerAnalyticsKeys = {
  map: (metric: string, groupBy: string, f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-map", metric, groupBy, f] as const,
  treemap: (f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-treemap", f] as const,
  heatmap: (metric: string, topN: number, f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-heatmap", metric, topN, f] as const,
  channelMix: (f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-channel-mix", f] as const,
  segmentTrends: (segmentBy: string, f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-segment-trends", segmentBy, f] as const,
  ranking: (sort: string, topN: number, f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-ranking", sort, topN, f] as const,
  oosImpact: (grain: string, f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-oos-impact", grain, f] as const,
  items: (search: string) => ["customer-analytics-items", search] as const,
  kpis: (f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-kpis", f] as const,
  filterOptions: () => ["customer-analytics-filter-options"] as const,
  lifecycle: (f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-lifecycle", f] as const,
  demandAtRisk: (f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-demand-at-risk", f] as const,
  affinity: (f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-affinity", f] as const,
  orderPatterns: (f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-order-patterns", f] as const,
  demandFlow: (f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-demand-flow", f] as const,
  alerts: (f?: CustomerAnalyticsFilters) =>
    ["customer-analytics-alerts", f] as const,
};

// ---------------------------------------------------------------------------
// Fetch functions
// ---------------------------------------------------------------------------

export function fetchCustomerAnalyticsMap(
  metric: string,
  groupBy: string,
  filters?: CustomerAnalyticsFilters,
): Promise<MapPayload> {
  const qs = buildQuerySuffix({ metric, group_by: groupBy, ...filterParams(filters) });
  return fetchJson(`/customer-analytics/map${qs}`);
}

export function fetchCustomerAnalyticsTreemap(
  filters?: CustomerAnalyticsFilters,
): Promise<TreemapPayload> {
  const qs = buildQuerySuffix(filterParams(filters));
  return fetchJson(`/customer-analytics/treemap${qs}`);
}

export function fetchCustomerAnalyticsHeatmap(
  metric: string,
  topN: number,
  filters?: CustomerAnalyticsFilters,
): Promise<HeatmapPayload> {
  const qs = buildQuerySuffix({ metric, top_n: String(topN), ...filterParams(filters) });
  return fetchJson(`/customer-analytics/heatmap${qs}`);
}

export function fetchCustomerAnalyticsChannelMix(
  filters?: CustomerAnalyticsFilters,
): Promise<ChannelMixPayload> {
  const qs = buildQuerySuffix(filterParams(filters));
  return fetchJson(`/customer-analytics/channel-mix${qs}`);
}

export function fetchCustomerAnalyticsSegmentTrends(
  segmentBy: string,
  filters?: CustomerAnalyticsFilters,
): Promise<SegmentTrendsPayload> {
  const qs = buildQuerySuffix({ segment_by: segmentBy, ...filterParams(filters) });
  return fetchJson(`/customer-analytics/segment-trends${qs}`);
}

export function fetchCustomerAnalyticsRanking(
  sort: string,
  topN: number,
  filters?: CustomerAnalyticsFilters,
): Promise<RankingPayload> {
  const qs = buildQuerySuffix({ sort, top_n: String(topN), ...filterParams(filters) });
  return fetchJson(`/customer-analytics/ranking${qs}`);
}

export function fetchCustomerAnalyticsOosImpact(
  grain: string,
  filters?: CustomerAnalyticsFilters,
): Promise<OosImpactPayload> {
  const qs = buildQuerySuffix({ grain, ...filterParams(filters) });
  return fetchJson(`/customer-analytics/oos-impact${qs}`);
}

export function fetchCustomerAnalyticsItems(
  search: string,
): Promise<ItemsPayload> {
  const qs = buildQuerySuffix({ search });
  return fetchJson(`/customer-analytics/items${qs}`);
}

export function fetchCustomerAnalyticsKpis(
  filters?: CustomerAnalyticsFilters,
): Promise<KpiPayload> {
  const qs = buildQuerySuffix(filterParams(filters));
  return fetchJson(`/customer-analytics/kpis${qs}`);
}

/**
 * U3.3 — the source MV serves dirty state codes (`.`, `00`, `0D`, `XX`, `null`,
 * numeric/junk) intermixed with real 2-letter codes, making the State filter
 * unscannable. Whitelist only canonical US state/territory + Canadian province
 * codes (both are 2 alpha chars) so placeholders like `XX` are also dropped;
 * uppercased, de-duped, sorted. Minimum-safe client-side normalization until
 * the MV (sql/173) is cleaned upstream.
 */
const _VALID_STATE_CODES = new Set<string>([
  // US states + DC
  "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID",
  "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
  "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
  "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
  // US territories
  "PR", "VI", "GU", "AS", "MP",
  // Canadian provinces/territories
  "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT",
]);

export function normalizeStateOptions(states: readonly string[] | null | undefined): string[] {
  if (!states) return [];
  const seen = new Set<string>();
  for (const raw of states) {
    if (typeof raw !== "string") continue;
    const code = raw.trim().toUpperCase();
    if (_VALID_STATE_CODES.has(code)) seen.add(code);
  }
  return Array.from(seen).sort();
}

const _NULLISH_LABELS = new Set<string>(["", "null", "undefined", "n/a", "na"]);

/**
 * F4.5 / U4.2 — free-text facets (Channel, Store Type) arrive from the MV with
 * trailing-whitespace duplicates, case-variant duplicates, and literal
 * `null`/`undefined` entries. This trims each value, drops nullish/empty ones,
 * de-dupes case-insensitively (keeping the FIRST canonical casing seen so the
 * emitted value still matches a real DB value for the WHERE clause), and sorts
 * case-insensitively. Mirrors the State whitelist treatment without forcing an
 * uppercase that could break case-sensitive predicates.
 */
export function normalizeLabelOptions(
  values: readonly string[] | null | undefined,
): string[] {
  if (!values) return [];
  const byKey = new Map<string, string>();
  for (const raw of values) {
    if (typeof raw !== "string") continue;
    const trimmed = raw.trim();
    const key = trimmed.toLowerCase();
    if (_NULLISH_LABELS.has(key)) continue;
    if (!byKey.has(key)) byKey.set(key, trimmed);
  }
  return Array.from(byKey.values()).sort((a, b) =>
    a.toLowerCase().localeCompare(b.toLowerCase()),
  );
}

export async function fetchCustomerAnalyticsFilterOptions(): Promise<FilterOptionsPayload> {
  const payload = await fetchJson<FilterOptionsPayload>("/customer-analytics/filter-options");
  return {
    ...payload,
    states: normalizeStateOptions(payload.states),
    channels: normalizeLabelOptions(payload.channels),
    store_types: normalizeLabelOptions(payload.store_types),
  };
}

export function fetchCustomerAnalyticsLifecycle(
  filters?: CustomerAnalyticsFilters,
): Promise<LifecyclePayload> {
  const qs = buildQuerySuffix(filterParams(filters));
  return fetchJson(`/customer-analytics/lifecycle${qs}`);
}

export function fetchCustomerAnalyticsDemandAtRisk(
  filters?: CustomerAnalyticsFilters,
): Promise<DemandAtRiskPayload> {
  const qs = buildQuerySuffix(filterParams(filters));
  return fetchJson(`/customer-analytics/demand-at-risk${qs}`);
}

export function fetchCustomerAnalyticsAffinity(
  filters?: CustomerAnalyticsFilters,
): Promise<AffinityPayload> {
  const qs = buildQuerySuffix(filterParams(filters));
  return fetchJson(`/customer-analytics/affinity${qs}`);
}

export function fetchCustomerAnalyticsOrderPatterns(
  filters?: CustomerAnalyticsFilters,
): Promise<OrderPatternsPayload> {
  const qs = buildQuerySuffix(filterParams(filters));
  return fetchJson(`/customer-analytics/order-patterns${qs}`);
}

export function fetchCustomerAnalyticsDemandFlow(
  filters?: CustomerAnalyticsFilters,
): Promise<DemandFlowPayload> {
  const qs = buildQuerySuffix(filterParams(filters));
  return fetchJson(`/customer-analytics/demand-flow${qs}`);
}

export function fetchCustomerAnalyticsAlerts(
  filters?: CustomerAnalyticsFilters,
): Promise<AlertsPayload> {
  const qs = buildQuerySuffix(filterParams(filters));
  return fetchJson(`/customer-analytics/alerts${qs}`);
}
