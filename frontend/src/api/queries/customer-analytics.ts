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
