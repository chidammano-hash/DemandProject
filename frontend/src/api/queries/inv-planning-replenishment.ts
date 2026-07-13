import { fetchJson } from "./request";

// ---------------------------------------------------------------------------
// Inventory Planning — Replenishment Plan (forward-looking SS + EOQ plan)
// ---------------------------------------------------------------------------

export interface ReplenishmentSummary {
  plan_version: string;
  total_skus: number;
  below_ss_count: number;
  below_ss_pct: number;
  avg_ss: number | null;
  avg_eoq: number | null;
  avg_ss_delta_pct: number | null;
  computed_at?: string | null;
  by_policy_type: ReplenishmentPolicyBreakdown[];
}

export interface ReplenishmentPolicyBreakdown {
  policy_type: string;
  sku_count: number;
  avg_ss: number | null;
  avg_eoq: number | null;
  total_order_qty: number | null;
}

export interface ReplenishmentDetailRow {
  item_id: string;
  loc: string;
  plan_month: string;
  abc_vol: string | null;
  policy_type: string | null;
  forecast_qty: number | null;
  ss_combined: number | null;
  historical_ss: number | null;
  ss_delta: number | null;
  ss_delta_pct: number | null;
  eoq: number | null;
  cycle_stock: number | null;
  reorder_point: number | null;
  order_qty: number | null;
  order_up_to_level: number | null;
  is_below_ss: boolean | null;
}

export interface ReplenishmentDetailPayload {
  total: number;
  limit: number;
  offset: number;
  rows: ReplenishmentDetailRow[];
}

export interface ReplenishmentAbcBreakdown {
  abc_vol: string;
  sku_count: number;
  avg_forecast_ss: number | null;
  avg_historical_ss: number | null;
  avg_ss_delta: number | null;
  avg_ss_delta_pct: number | null;
  count_increased: number;
  count_decreased: number;
  count_unchanged: number;
}

export interface ReplenishmentComparison {
  plan_version: string;
  by_abc: ReplenishmentAbcBreakdown[];
  total_increased: number;
  total_decreased: number;
}

export interface ReplenishmentDfuPoint {
  plan_month: string;
  horizon_months: number;
  forecast_qty: number | null;
  forecast_qty_lower: number | null;
  forecast_qty_upper: number | null;
  ss_combined: number | null;
  historical_ss: number | null;
  ss_delta: number | null;
  eoq: number | null;
  cycle_stock: number | null;
  reorder_point: number | null;
  order_qty: number | null;
  order_up_to_level: number | null;
  avg_daily_demand: number | null;
  is_below_ss: boolean | null;
  sigma_method: string | null;
}

export interface ReplenishmentDfuPayload {
  item_id: string;
  loc: string;
  plan_version: string;
  series: ReplenishmentDfuPoint[];
}

// ---- Query keys ----

export const replenishmentKeys = {
  summary: (planVersion?: string, policyType?: string, abcVol?: string) =>
    ["replenishment", "summary", planVersion, policyType, abcVol] as const,
  detail: (params: object) => ["replenishment", "detail", params] as const,
  comparison: (planVersion?: string, abcVol?: string, policyType?: string) =>
    ["replenishment", "comparison", planVersion, abcVol, policyType] as const,
  sku: (itemNo: string, loc: string, planVersion?: string) =>
    ["replenishment", "sku", itemNo, loc, planVersion] as const,
};

// ---- Fetch functions ----

export async function fetchReplenishmentSummary(params?: {
  plan_version?: string;
  policy_type?: string;
  abc_vol?: string;
}): Promise<ReplenishmentSummary> {
  const qs = new URLSearchParams();
  if (params?.plan_version) qs.set("plan_version", params.plan_version);
  if (params?.policy_type) qs.set("policy_type", params.policy_type);
  if (params?.abc_vol) qs.set("abc_vol", params.abc_vol);
  return fetchJson(`/inv-planning/replenishment/summary?${qs}`);
}

export async function fetchReplenishmentDetail(params: {
  item?: string;
  location?: string;
  policy_type?: string;
  abc_vol?: string;
  is_below_ss?: boolean;
  plan_version?: string;
  plan_month?: string;
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: string;
}): Promise<ReplenishmentDetailPayload> {
  const qs = new URLSearchParams();
  if (params.item) qs.set("item", params.item);
  if (params.location) qs.set("location", params.location);
  if (params.policy_type) qs.set("policy_type", params.policy_type);
  if (params.abc_vol) qs.set("abc_vol", params.abc_vol);
  if (params.is_below_ss != null) qs.set("is_below_ss", String(params.is_below_ss));
  if (params.plan_version) qs.set("plan_version", params.plan_version);
  if (params.plan_month) qs.set("plan_month", params.plan_month);
  qs.set("limit", String(params.limit ?? 50));
  qs.set("offset", String(params.offset ?? 0));
  if (params.sort_by) qs.set("sort_by", params.sort_by);
  if (params.sort_dir) qs.set("sort_dir", params.sort_dir);
  return fetchJson(`/inv-planning/replenishment/detail?${qs}`);
}

export async function fetchReplenishmentComparison(params?: {
  plan_version?: string;
  abc_vol?: string;
  policy_type?: string;
}): Promise<ReplenishmentComparison> {
  const qs = new URLSearchParams();
  if (params?.plan_version) qs.set("plan_version", params.plan_version);
  if (params?.abc_vol) qs.set("abc_vol", params.abc_vol);
  if (params?.policy_type) qs.set("policy_type", params.policy_type);
  return fetchJson(`/inv-planning/replenishment/comparison?${qs}`);
}

export async function fetchReplenishmentSku(params: {
  item_id: string;
  loc: string;
  plan_version?: string;
}): Promise<ReplenishmentDfuPayload> {
  const qs = new URLSearchParams({ item_id: params.item_id, loc: params.loc });
  if (params.plan_version) qs.set("plan_version", params.plan_version);
  return fetchJson(`/inv-planning/replenishment/sku?${qs}`);
}
