/**
 * Purchase Orders API queries — comprehensive PO history
 */

import { fetchJson } from "./request";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PORow {
  po_ck: number;
  po_number: string;
  site_id: string | null;
  loc: string;
  source: string | null;
  item_id: string;
  ordered_qty: number | null;
  net_price: number | null;
  gross_value: number | null;
  closure_code: string | null;
  po_hdr_status: string | null;
  po_line_status: string | null;
  receipt_status: string | null;
  supplier_id: string | null;
  supplier_name: string | null;
  carrier_name: string | null;
  delivery_date: string | null;
  original_delivery_date: string | null;
  current_ship_date: string | null;
  original_ship_date: string | null;
  po_type: string | null;
  is_closed: boolean;
  lead_time_planned: number | null;
  lead_time_actual: number | null;
}

export interface POPayload {
  total: number;
  rows: PORow[];
}

export interface POSummary {
  total_lines: number;
  closed_lines: number;
  open_lines: number;
  distinct_pos: number;
  distinct_suppliers: number;
  distinct_items: number;
  total_value: number;
  open_value: number;
  closed_value: number;
}

export interface POAgingBucket {
  age_bucket: string;
  line_count: number;
  total_value: number;
}

export interface SupplierOTD {
  supplier_id: string;
  supplier_name: string | null;
  total_closed: number;
  on_time: number;
  otd_pct: number | null;
  avg_lead_time_days: number | null;
}

// ---------------------------------------------------------------------------
// Fetchers
// ---------------------------------------------------------------------------

export async function fetchPORows(params: {
  po_number?: string;
  item?: string;
  loc?: string;
  supplier?: string;
  status?: string;
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: string;
}): Promise<POPayload> {
  const sp = new URLSearchParams();
  if (params.po_number) sp.set("po_number", params.po_number);
  if (params.item) sp.set("item", params.item);
  if (params.loc) sp.set("loc", params.loc);
  if (params.supplier) sp.set("supplier", params.supplier);
  if (params.status) sp.set("status", params.status);
  sp.set("limit", String(params.limit ?? 50));
  sp.set("offset", String(params.offset ?? 0));
  if (params.sort_by) sp.set("sort_by", params.sort_by);
  if (params.sort_dir) sp.set("sort_dir", params.sort_dir);
  return fetchJson(`/purchase-orders/rows?${sp}`);
}

export async function fetchPOSearch(q: string): Promise<{ rows: PORow[] }> {
  return fetchJson(`/purchase-orders/search?q=${encodeURIComponent(q)}`);
}

export async function fetchPOByNumber(poNum: string): Promise<POPayload> {
  return fetchJson(`/purchase-orders/by-po/${encodeURIComponent(poNum)}`);
}

export async function fetchPOSummary(): Promise<POSummary> {
  return fetchJson("/purchase-orders/summary");
}

export async function fetchPOAging(): Promise<{ buckets: POAgingBucket[] }> {
  return fetchJson("/purchase-orders/aging");
}

export async function fetchPOOnTimeDelivery(supplier?: string): Promise<{ suppliers: SupplierOTD[] }> {
  const sp = new URLSearchParams();
  if (supplier) sp.set("supplier", supplier);
  return fetchJson(`/purchase-orders/otd?${sp}`);
}
