import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Inventory Rebalancing — Cross-location transfer optimization
// ---------------------------------------------------------------------------

export interface RebalancingKpis {
  total_multi_loc_items: number;
  avg_dos_cv: number | null;
  network_balance_score: number | null;
  imbalanced_items: number;
  total_excess_locs: number;
  total_shortage_locs: number;
  latest_plan: {
    plan_id: string;
    total_transfer_qty: number | null;
    total_transfer_cost: number | null;
    total_avoided_stockout_value: number | null;
    net_roi: number | null;
    items_rebalanced: number;
    status: string;
    computation_date: string | null;
  } | null;
}

export interface TransferLane {
  lane_id: string;
  source_loc: string;
  dest_loc: string;
  transfer_mode: string;
  cost_per_unit: number | null;
  handling_cost: number | null;
  freight_cost: number | null;
  receiving_cost: number | null;
  fixed_cost_per_shipment: number | null;
  transfer_lt_days: number;
  min_transfer_qty: number;
  max_transfer_qty: number | null;
  batch_size: number;
}

export interface Imbalance {
  item_id: string;
  location_count: number;
  avg_on_hand: number | null;
  avg_dos: number | null;
  dos_cv: number | null;
  excess_loc_count: number;
  shortage_loc_count: number;
}

export interface RebalancingPlan {
  plan_id: string;
  computation_date: string | null;
  solver_method: string;
  objective: string;
  total_transfer_qty: number | null;
  total_transfer_cost: number | null;
  total_avoided_stockout_value: number | null;
  net_roi: number | null;
  items_rebalanced: number;
  lanes_used: number;
  status: string;
  solver_runtime_ms: number;
  created_ts: string | null;
}

export interface RebalancingTransfer {
  transfer_id: string;
  item_id: string;
  source_loc: string;
  dest_loc: string;
  transfer_mode: string;
  recommended_qty: number | null;
  approved_qty: number | null;
  source_on_hand: number | null;
  source_dos: number | null;
  source_ss_target: number | null;
  source_excess_qty: number | null;
  dest_on_hand: number | null;
  dest_dos: number | null;
  dest_ss_target: number | null;
  dest_shortage_qty: number | null;
  transfer_cost: number | null;
  carrying_cost_saved: number | null;
  stockout_cost_avoided: number | null;
  net_benefit: number | null;
  roi: number | null;
  planned_ship_date: string | null;
  expected_arrival_date: string | null;
  transfer_lt_days: number;
  priority_score: number | null;
  abc_class: string | null;
  urgency: string;
  status: string;
  approved_by: string | null;
  rejection_reason: string | null;
  notes: string | null;
}

// Query key factory
export const rebalancingKeys = {
  kpis: () => ["rebalancing-kpis"] as const,
  network: (params?: Record<string, unknown>) => ["rebalancing-network", params ?? {}] as const,
  imbalances: (params?: Record<string, unknown>) => ["rebalancing-imbalances", params ?? {}] as const,
  plans: (params?: Record<string, unknown>) => ["rebalancing-plans", params ?? {}] as const,
  planDetail: (planId: string) => ["rebalancing-plan", planId] as const,
  transfers: (planId: string, params?: Record<string, unknown>) => ["rebalancing-transfers", planId, params ?? {}] as const,
};

// Fetch functions

export async function fetchRebalancingKpis(): Promise<RebalancingKpis> {
  return fetchJson("/inv-planning/rebalancing/kpis");
}

export async function fetchTransferLanes(
  params: { source_loc?: string; dest_loc?: string; limit?: number; offset?: number } = {},
): Promise<{ total: number; rows: TransferLane[] }> {
  const qs = new URLSearchParams();
  if (params.source_loc) qs.set("source_loc", params.source_loc);
  if (params.dest_loc) qs.set("dest_loc", params.dest_loc);
  if (params.limit) qs.set("limit", String(params.limit));
  if (params.offset) qs.set("offset", String(params.offset));
  const q = qs.toString();
  return fetchJson(`/inv-planning/rebalancing/network${q ? "?" + q : ""}`);
}

export async function fetchImbalances(
  params: { item?: string; limit?: number; offset?: number } = {},
): Promise<{ total: number; rows: Imbalance[] }> {
  const qs = new URLSearchParams();
  if (params.item) qs.set("item", params.item);
  if (params.limit) qs.set("limit", String(params.limit));
  if (params.offset) qs.set("offset", String(params.offset));
  const q = qs.toString();
  return fetchJson(`/inv-planning/rebalancing/imbalances${q ? "?" + q : ""}`);
}

export async function fetchRebalancingPlans(
  params: { status?: string; limit?: number; offset?: number } = {},
): Promise<{ total: number; rows: RebalancingPlan[] }> {
  const qs = new URLSearchParams();
  if (params.status) qs.set("status", params.status);
  if (params.limit) qs.set("limit", String(params.limit));
  if (params.offset) qs.set("offset", String(params.offset));
  const q = qs.toString();
  return fetchJson(`/inv-planning/rebalancing/plans${q ? "?" + q : ""}`);
}

export async function fetchPlanDetail(planId: string): Promise<RebalancingPlan & {
  horizon_weeks: number;
  network_balance_before: number | null;
  network_balance_after: number | null;
  approved_by: string | null;
  approved_ts: string | null;
}> {
  return fetchJson(`/inv-planning/rebalancing/plans/${encodeURIComponent(planId)}`);
}

export async function fetchPlanTransfers(
  planId: string,
  params: { urgency?: string; status?: string; item?: string; sort_by?: string; sort_dir?: string; limit?: number; offset?: number } = {},
): Promise<{ total: number; rows: RebalancingTransfer[] }> {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 50),
    offset: String(params.offset ?? 0),
    sort_by: params.sort_by ?? "priority_score",
    sort_dir: params.sort_dir ?? "desc",
  });
  if (params.urgency) qs.set("urgency", params.urgency);
  if (params.status) qs.set("status", params.status);
  if (params.item) qs.set("item", params.item);
  return fetchJson(`/inv-planning/rebalancing/plans/${encodeURIComponent(planId)}/transfers?${qs}`);
}

export async function computeRebalancingPlan(
  body: { solver?: string; horizon_weeks?: number; budget_cap?: number | null } = {},
): Promise<{ status: string; message: string }> {
  return fetchJson("/inv-planning/rebalancing/compute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function approveTransfer(
  transferId: string,
  body: { approved_by: string; approved_qty?: number; notes?: string },
): Promise<{ transfer_id: string; status: string }> {
  return fetchJson(`/inv-planning/rebalancing/transfers/${encodeURIComponent(transferId)}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function rejectTransfer(
  transferId: string,
  body: { rejection_reason: string; notes?: string },
): Promise<{ transfer_id: string; status: string }> {
  return fetchJson(`/inv-planning/rebalancing/transfers/${encodeURIComponent(transferId)}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function approveAllTransfers(
  planId: string,
  body: { approved_by: string },
): Promise<{ plan_id: string; approved_count: number; status: string }> {
  return fetchJson(`/inv-planning/rebalancing/plans/${encodeURIComponent(planId)}/approve-all`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
