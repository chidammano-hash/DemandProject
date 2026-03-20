import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature4: EOQ & Cycle Stock
// ---------------------------------------------------------------------------

export interface EoqAbcEntry {
  abc_vol: string;
  count: number;
  avg_eoq: number | null;
  total_cycle_stock: number | null;
  total_annual_cost: number | null;
  avg_order_frequency: number | null;
}

export interface EoqSummaryPayload {
  total_dfus: number;
  avg_effective_eoq: number | null;
  total_cycle_stock: number | null;
  avg_order_frequency: number | null;
  total_annual_cost: number | null;
  by_abc: EoqAbcEntry[];
}

export interface EoqDetailRow {
  item_no: string;
  loc: string;
  abc_vol: string | null;
  demand_mean_monthly: number | null;
  annual_demand: number | null;
  ordering_cost: number | null;
  holding_cost_pct: number | null;
  unit_cost: number | null;
  moq: number | null;
  eoq: number | null;
  effective_eoq: number | null;
  eoq_cycle_stock: number | null;
  order_frequency: number | null;
  annual_holding_cost: number | null;
  annual_order_cost: number | null;
  total_annual_cost: number | null;
  computed_at: string | null;
}

export interface EoqDetailPayload {
  total: number;
  limit: number;
  offset: number;
  rows: EoqDetailRow[];
}

export interface EoqSensitivityPoint {
  ordering_cost: number;
  eoq: number;
  effective_eoq: number;
  total_annual_cost: number;
}

export interface EoqSensitivityPayload {
  item_no: string | null;
  loc: string | null;
  avg_demand_monthly: number;
  curve: EoqSensitivityPoint[];
}

export async function fetchEoqSummary(params: { abc_vol?: string }): Promise<EoqSummaryPayload> {
  const qs = new URLSearchParams();
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  return fetchJson(`/inv-planning/eoq/summary?${qs}`);
}

export async function fetchEoqDetail(params: {
  item?: string;
  loc?: string;
  abc_vol?: string;
  sort_by?: string;
  sort_dir?: string;
  limit?: number;
  offset?: number;
}): Promise<EoqDetailPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 50),
    offset: String(params.offset ?? 0),
    sort_by: params.sort_by ?? "total_annual_cost",
    sort_dir: params.sort_dir ?? "desc",
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.loc?.trim()) qs.set("loc", params.loc.trim());
  if (params.abc_vol?.trim()) qs.set("abc_vol", params.abc_vol.trim());
  return fetchJson(`/inv-planning/eoq/detail?${qs}`);
}

export async function fetchEoqSensitivity(params: {
  item?: string;
  loc?: string;
}): Promise<EoqSensitivityPayload> {
  const qs = new URLSearchParams();
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.loc?.trim()) qs.set("loc", params.loc.trim());
  return fetchJson(`/inv-planning/eoq/sensitivity?${qs}`);
}
