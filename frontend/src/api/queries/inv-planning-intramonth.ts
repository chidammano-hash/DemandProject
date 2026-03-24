import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature14: Intra-Month Stockout Detection
// ---------------------------------------------------------------------------

export interface IntramonthStockoutRow {
  item_id: string;
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
