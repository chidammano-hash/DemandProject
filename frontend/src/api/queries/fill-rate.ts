import { fetchJson } from "./request";
import { buildQuerySuffix } from "./helpers";

// ---------------------------------------------------------------------------
// IPfeature8: Fill Rate Analytics
// ---------------------------------------------------------------------------

export interface FillRateSummaryPayload {
  portfolio_fill_rate: number | null;
  total_ordered: number;
  total_shipped: number;
  total_shortage_qty: number;
  partial_fulfillment_events: number;
  by_abc: Record<string, { avg_fill_rate: number | null; total_shortage_qty: number; events: number }>;
  worst_items: Array<{ item_id: string; loc: string; fill_rate: number | null; shortage_qty: number | null; abc_vol: string | null }>;
  trend: Array<{ month_start: string; portfolio_fill_rate: number | null; total_shortage_qty: number }>;
}

export interface FillRateTrendRow {
  month_start: string;
  fill_rate: number | null;
  total_ordered: number;
  total_shipped: number;
  shortage_qty: number;
}

export interface FillRateDetailRow {
  item_id: string;
  loc: string;
  month_start: string;
  total_ordered: number | null;
  total_shipped: number | null;
  fill_rate: number | null;
  shortage_qty: number | null;
  had_partial_fulfillment: boolean;
  abc_vol: string | null;
  cluster_assignment: string | null;
  region: string | null;
}

export interface GapDecompositionItem {
  cause: string;
  impact_pct: number;
  sku_count: number;
  shortage_qty: number;
}

export interface FillRateGapAnalysisPayload {
  target_fill_rate: number;
  actual_fill_rate: number | null;
  gap_pct: number | null;
  decomposition: GapDecompositionItem[];
  month: string | null;
}

export const fillRateKeys = {
  summary:     (f?: Record<string, unknown>) => ["fill-rate-summary", f ?? {}] as const,
  trend:       (f?: Record<string, unknown>) => ["fill-rate-trend", f ?? {}] as const,
  detail:      (f?: Record<string, unknown>) => ["fill-rate-detail", f ?? {}] as const,
  gapAnalysis: (f?: Record<string, unknown>) => ["fill-rate-gap-analysis", f ?? {}] as const,
};

export async function fetchFillRateSummary(
  params: Record<string, unknown> = {},
): Promise<FillRateSummaryPayload> {
  return fetchJson(`/fill-rate/summary${buildQuerySuffix(params as Record<string, string>)}`);
}

export async function fetchFillRateTrend(
  params: Record<string, unknown> = {},
): Promise<{ months: FillRateTrendRow[] }> {
  return fetchJson(`/fill-rate/trend${buildQuerySuffix(params as Record<string, string>)}`);
}

export async function fetchFillRateDetail(
  params: Record<string, unknown> = {},
): Promise<{ total: number; rows: FillRateDetailRow[] }> {
  return fetchJson(`/fill-rate/detail${buildQuerySuffix(params as Record<string, string>)}`);
}

export async function fetchFillRateGapAnalysis(
  params?: { month?: string; abc_vol?: string },
): Promise<FillRateGapAnalysisPayload> {
  const qs = new URLSearchParams();
  if (params?.month) qs.set("month", params.month);
  if (params?.abc_vol) qs.set("abc_vol", params.abc_vol);
  const suffix = qs.toString();
  return fetchJson(`/fill-rate/gap-analysis${suffix ? `?${suffix}` : ""}`);
}
