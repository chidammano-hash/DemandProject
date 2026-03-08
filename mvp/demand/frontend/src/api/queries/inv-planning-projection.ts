/**
 * F1.2 — Forward Inventory Projection query functions
 */
import { fetchJson } from "./core";

export interface ProjectionKeyDate {
  reorder_trigger_date: string | null;
  stockout_date: string | null;
  days_until_stockout: number | null;
  excess_date: string | null;
}

export interface ProjectionRow {
  projection_date: string;
  daily_demand_rate: number;
  receipts_expected: number;
  no_order_qty?: number;
  no_order_stockout_risk?: boolean;
  no_order_reorder_triggered?: boolean;
  with_open_po_qty?: number;
  with_open_po_stockout_risk?: boolean;
  with_open_po_reorder_triggered?: boolean;
  with_planned_orders_qty?: number;
  with_planned_orders_stockout_risk?: boolean;
  with_planned_orders_reorder_triggered?: boolean;
}

export interface ProjectionPayload {
  item_no: string;
  loc: string;
  current_qty_on_hand: number;
  safety_stock: number;
  reorder_point: number;
  forecast_source: "production_forecast" | "fallback_avg";
  plan_version: string | null;
  open_po_data_available: boolean;
  computed_at: string | null;
  key_dates: Record<string, ProjectionKeyDate>;
  projection: ProjectionRow[];
}

export interface AtRiskItem {
  item_no: string;
  loc: string;
  stockout_date: string | null;
  days_until_stockout: number | null;
  reorder_trigger_date: string | null;
  current_qty: number;
  safety_stock: number;
  severity: "critical" | "high" | "medium";
}

export interface AtRiskPayload {
  total: number;
  horizon_days: number;
  page: number;
  page_size: number;
  items: AtRiskItem[];
}

export const projectionKeys = {
  dfu: (params: Record<string, unknown>) => ["projection", "dfu", params] as const,
  atRisk: (horizon_days: number) => ["projection", "at-risk", horizon_days] as const,
};

export async function fetchProjection(params: {
  item_no: string;
  loc: string;
  horizon_days?: number;
  scenario?: string;
}): Promise<ProjectionPayload> {
  const qs = new URLSearchParams({
    item_no: params.item_no,
    loc: params.loc,
    horizon_days: String(params.horizon_days ?? 90),
  });
  if (params.scenario) qs.set("scenario", params.scenario);
  return fetchJson(`/inv-planning/projection?${qs}`);
}

export async function fetchProjectionAtRisk(params: {
  horizon_days?: number;
  page?: number;
  page_size?: number;
}): Promise<AtRiskPayload> {
  const qs = new URLSearchParams({
    horizon_days: String(params.horizon_days ?? 30),
    page: String(params.page ?? 1),
    page_size: String(params.page_size ?? 50),
  });
  return fetchJson(`/inv-planning/projection/at-risk?${qs}`);
}

export async function refreshProjection(params: {
  item_no: string;
  loc: string;
  horizon_days?: number;
}): Promise<{ status: string; rows_written: number; run_id: string }> {
  return fetchJson("/inv-planning/projection/refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      item_no: params.item_no,
      loc: params.loc,
      horizon_days: params.horizon_days ?? 90,
    }),
  });
}
