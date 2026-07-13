/**
 * F1.3 — Open Purchase Order API queries
 */

import { fetchJson } from "./request";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface OpenPOLine {
  po_number: string;
  po_line_number: number;
  item_id: string;
  loc: string;
  supplier_id: string | null;
  supplier_name: string | null;
  po_date: string | null;
  ordered_qty: number | null;
  confirmed_qty: number | null;
  received_qty: number | null;
  open_qty: number | null;
  unit_cost: number | null;
  line_value: number | null;
  promised_delivery_date: string | null;
  confirmed_delivery_date: string | null;
  revised_delivery_date: string | null;
  effective_delivery_date: string | null;
  days_past_due: number;
  line_status: string;
}

export interface OpenPOPayload {
  total: number;
  open_po_data_available: boolean;
  last_loaded_at: string | null;
  page: number;
  page_size: number;
  items: OpenPOLine[];
}

export interface OpenPOSummaryPayload {
  total_open_lines: number;
  total_open_value_usd: number;
  total_open_qty_by_status: { open: number; partially_received: number };
  past_due_lines: number;
  past_due_value_usd: number;
  avg_days_past_due: number | null;
  suppliers_with_open_pos: number;
  last_loaded_at: string | null;
}

export interface PastDuePOLine {
  po_number: string;
  item_id: string;
  loc: string;
  supplier_name: string | null;
  open_qty: number | null;
  confirmed_delivery_date: string | null;
  days_past_due: number;
  line_value: number | null;
  severity: string;
}

export interface PastDuePOPayload {
  total: number;
  items: PastDuePOLine[];
}

// ---------------------------------------------------------------------------
// Fetch functions
// ---------------------------------------------------------------------------

export async function fetchOpenPOs(params: {
  item_id?: string;
  loc?: string;
  supplier_id?: string;
  status?: string;
  past_due_only?: boolean;
  page?: number;
  page_size?: number;
}): Promise<OpenPOPayload> {
  const qs = new URLSearchParams({
    page: String(params.page ?? 1),
    page_size: String(params.page_size ?? 50),
  });
  if (params.item_id?.trim()) qs.set("item_id", params.item_id.trim());
  if (params.loc?.trim()) qs.set("loc", params.loc.trim());
  if (params.supplier_id?.trim()) qs.set("supplier_id", params.supplier_id.trim());
  if (params.status) qs.set("status", params.status);
  if (params.past_due_only) qs.set("past_due_only", "true");
  return fetchJson(`/supply/open-pos?${qs}`);
}

export async function fetchOpenPOSummary(): Promise<OpenPOSummaryPayload> {
  return fetchJson("/supply/open-pos/summary");
}

export async function fetchPastDuePOs(params: {
  min_days_past_due?: number;
  supplier_id?: string;
  page?: number;
  page_size?: number;
}): Promise<PastDuePOPayload> {
  const qs = new URLSearchParams({
    min_days_past_due: String(params.min_days_past_due ?? 7),
    page: String(params.page ?? 1),
    page_size: String(params.page_size ?? 50),
  });
  if (params.supplier_id?.trim()) qs.set("supplier_id", params.supplier_id.trim());
  return fetchJson(`/supply/past-due-pos?${qs}`);
}

// ---------------------------------------------------------------------------
// F2.1 — Planned Orders
// ---------------------------------------------------------------------------

export interface PlannedOrder {
  id: number;
  item_id: string;
  loc: string;
  supplier_id: string | null;
  supplier_name: string | null;
  net_requirement_qty: number | null;
  recommended_qty: number | null;
  moq: number | null;
  unit_cost: number | null;
  order_value: number | null;
  currency: string;
  trigger_date: string | null;
  trigger_reason: string;
  order_by_date: string | null;
  expected_receipt_date: string | null;
  lead_time_days: number;
  current_qty_on_hand: number | null;
  safety_stock: number | null;
  reorder_point: number | null;
  confirmed_inbound_qty: number | null;
  lt_forecast_demand: number | null;
  plan_version: string | null;
  confidence_score: number | null;
  confidence_reason: string | null;
  is_past_due: boolean;
  status: string;
  created_at: string | null;
  approved_by: string | null;
  approved_at: string | null;
}

export interface PlannedOrdersPayload {
  total: number;
  total_order_value_usd: number;
  past_due_count: number;
  page: number;
  page_size: number;
  items: PlannedOrder[];
}

export interface PlannedOrdersSummaryPayload {
  status_counts: { proposed: number; approved: number; released: number; rejected: number };
  total_proposed_value_usd: number;
  total_approved_value_usd: number;
  past_due_proposed_count: number;
  past_due_proposed_value_usd: number;
  avg_confidence_score: number | null;
  low_confidence_count: number;
  generated_at: string | null;
}

export async function fetchPlannedOrders(params: {
  item_id?: string;
  loc?: string;
  status?: string;
  past_due_only?: boolean;
  supplier_id?: string;
  page?: number;
  page_size?: number;
}): Promise<PlannedOrdersPayload> {
  const qs = new URLSearchParams({
    page: String(params.page ?? 1),
    page_size: String(params.page_size ?? 50),
  });
  if (params.item_id?.trim()) qs.set("item_id", params.item_id.trim());
  if (params.loc?.trim()) qs.set("loc", params.loc.trim());
  if (params.status) qs.set("status", params.status);
  if (params.supplier_id?.trim()) qs.set("supplier_id", params.supplier_id.trim());
  if (params.past_due_only) qs.set("past_due_only", "true");
  return fetchJson(`/supply/planned-orders?${qs}`);
}

export async function fetchPlannedOrdersSummary(): Promise<PlannedOrdersSummaryPayload> {
  return fetchJson("/supply/planned-orders/summary");
}

export async function approvePlannedOrder(
  id: number,
  approvedBy: string
): Promise<{ id: number; status: string; approved_by: string; approved_at: string }> {
  return fetchJson(`/supply/planned-orders/${id}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved_by: approvedBy }),
  });
}

export async function rejectPlannedOrder(
  id: number,
  rejectionReason: string
): Promise<{ id: number; status: string }> {
  return fetchJson(`/supply/planned-orders/${id}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rejection_reason: rejectionReason }),
  });
}

export async function generatePlannedOrders(body?: {
  item_id?: string;
  loc?: string;
}): Promise<{ status: string; job_id: string }> {
  return fetchJson("/supply/planned-orders/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
}

// ---------------------------------------------------------------------------
// F2.4 — Procurement Workflow & Order Release (fact_purchase_orders)
// ---------------------------------------------------------------------------

export interface PurchaseOrderLine {
  po_number: string;
  line_number: number;
  item_id: string;
  item_description: string | null;
  loc: string;
  supplier_id: string | null;
  supplier_name: string | null;
  ordered_qty: number | null;
  unit_cost: number | null;
  total_value: number | null;
  currency: string;
  po_date: string | null;
  requested_delivery_date: string | null;
  confirmed_delivery_date: string | null;
  status: string;
  source_exception_id: number | null;
  created_by: string;
  planner_approved_by: string | null;
  buyer_released_by: string | null;
  erp_po_number: string | null;
}

export interface PurchaseOrdersPayload {
  total: number;
  total_value: number;
  page: number;
  orders: PurchaseOrderLine[];
}

export interface POTimelineEvent {
  action: string;
  performed_by: string;
  performed_at: string | null;
  old_status: string | null;
  new_status: string | null;
  old_qty: number | null;
  new_qty: number | null;
  reason: string | null;
  note: string | null;
}

export interface POTimelinePayload {
  po_number: string;
  current_status: string;
  timeline: POTimelineEvent[];
}

export interface POExportPayload {
  filename: string;
  line_count: number;
  total_value: number;
  csv_content: string;
}

export async function fetchPurchaseOrders(params?: {
  status?: string;
  supplier_id?: string;
  item_id?: string;
  loc?: string;
  po_date_from?: string;
  po_date_to?: string;
  page?: number;
  page_size?: number;
}): Promise<PurchaseOrdersPayload> {
  const qs = new URLSearchParams();
  if (params?.status) qs.set("status", params.status);
  if (params?.supplier_id) qs.set("supplier_id", params.supplier_id);
  if (params?.item_id) qs.set("item_id", params.item_id);
  if (params?.loc) qs.set("loc", params.loc);
  if (params?.po_date_from) qs.set("po_date_from", params.po_date_from);
  if (params?.po_date_to) qs.set("po_date_to", params.po_date_to);
  if (params?.page) qs.set("page", String(params.page));
  if (params?.page_size) qs.set("page_size", String(params.page_size));
  return fetchJson(`/supply/purchase-orders?${qs}`);
}

export async function createPOFromException(
  exceptionId: number,
  body: {
    performed_by: string;
    ordered_qty?: number;
    requested_delivery_date?: string;
    notes?: string;
  }
): Promise<{ po_number: string; status: string; total_value: number | null; requested_delivery_date: string | null }> {
  return fetchJson(`/supply/purchase-orders/from-exception/${exceptionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function approvePurchaseOrder(
  poNumber: string,
  body: { approved_by: string; new_qty?: number }
): Promise<{ po_number: string; status: string; approved_by: string }> {
  return fetchJson(`/supply/purchase-orders/${poNumber}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function releasePurchaseOrder(
  poNumber: string,
  body: { released_by: string; confirmed_delivery_date?: string; notes?: string }
): Promise<{ po_number: string; status: string; released_by: string }> {
  return fetchJson(`/supply/purchase-orders/${poNumber}/release`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function exportPOsCSV(body: {
  po_numbers: string[];
  exported_by: string;
}): Promise<POExportPayload> {
  return fetchJson("/supply/purchase-orders/export-csv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function fetchPOTimeline(poNumber: string): Promise<POTimelinePayload> {
  return fetchJson(`/supply/purchase-orders/${poNumber}/timeline`);
}
