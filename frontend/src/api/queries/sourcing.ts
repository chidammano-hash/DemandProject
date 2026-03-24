/**
 * Sourcing API queries — item-location supply source mapping
 */

import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SourcingRow {
  sourcing_ck: number;
  site_id: string;
  item_id: string;
  loc: string;
  source_cd: string;
  transit_mode: string | null;
  supplier_id: string | null;
  plant_id: string | null;
}

export interface SourcingPayload {
  total: number;
  rows: SourcingRow[];
}

export interface SourcingNetworkPayload {
  total_rows: number;
  supplier_count: number;
  item_location_count: number;
  single_source_count: number;
  multi_source_count: number;
  transit_modes: { transit_mode: string | null; count: number }[];
}

// ---------------------------------------------------------------------------
// Fetchers
// ---------------------------------------------------------------------------

export async function fetchSourcingRows(params: {
  item?: string;
  loc?: string;
  supplier?: string;
  transit_mode?: string;
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: string;
}): Promise<SourcingPayload> {
  const sp = new URLSearchParams();
  if (params.item) sp.set("item", params.item);
  if (params.loc) sp.set("loc", params.loc);
  if (params.supplier) sp.set("supplier", params.supplier);
  if (params.transit_mode) sp.set("transit_mode", params.transit_mode);
  sp.set("limit", String(params.limit ?? 50));
  sp.set("offset", String(params.offset ?? 0));
  if (params.sort_by) sp.set("sort_by", params.sort_by);
  if (params.sort_dir) sp.set("sort_dir", params.sort_dir);
  return fetchJson(`/sourcing/rows?${sp}`);
}

export async function fetchSourcingSearch(q: string): Promise<{ rows: SourcingRow[] }> {
  return fetchJson(`/sourcing/search?q=${encodeURIComponent(q)}`);
}

export async function fetchSourcingByItem(itemId: string): Promise<SourcingPayload> {
  return fetchJson(`/sourcing/by-item/${encodeURIComponent(itemId)}`);
}

export async function fetchSourcingBySupplier(supplierId: string): Promise<SourcingPayload> {
  return fetchJson(`/sourcing/by-supplier/${encodeURIComponent(supplierId)}`);
}

export async function fetchSourcingNetwork(): Promise<SourcingNetworkPayload> {
  return fetchJson("/sourcing/network");
}
