import { fetchJson } from "./core";

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
  worst_items: Array<{ item_no: string; loc: string; fill_rate: number | null; shortage_qty: number | null; abc_vol: string | null }>;
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
  item_no: string;
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

export const fillRateKeys = {
  summary: (f?: Record<string, unknown>) => ["fill-rate-summary", f ?? {}] as const,
  trend:   (f?: Record<string, unknown>) => ["fill-rate-trend", f ?? {}] as const,
  detail:  (f?: Record<string, unknown>) => ["fill-rate-detail", f ?? {}] as const,
};

export async function fetchFillRateSummary(
  params: Record<string, unknown> = {},
): Promise<FillRateSummaryPayload> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/fill-rate/summary${q ? `?${q}` : ""}`);
}

export async function fetchFillRateTrend(
  params: Record<string, unknown> = {},
): Promise<{ months: FillRateTrendRow[] }> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/fill-rate/trend${q ? `?${q}` : ""}`);
}

export async function fetchFillRateDetail(
  params: Record<string, unknown> = {},
): Promise<{ total: number; rows: FillRateDetailRow[] }> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const q = qs.toString();
  return fetchJson(`/fill-rate/detail${q ? `?${q}` : ""}`);
}
