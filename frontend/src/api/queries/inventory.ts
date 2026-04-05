import { buildSearchParams } from "./helpers";
import { fetchJson } from "./core";
import type {
  InventoryPositionPayload,
  InventoryKpis,
  InventoryTrendPayload,
  InventoryItemDetailPayload,
} from "@/types";

// ---------------------------------------------------------------------------
// Inventory queries
// ---------------------------------------------------------------------------
export interface InventoryPositionParams {
  item?: string;
  location?: string;
  limit: number;
  offset: number;
  sort_by: string;
  sort_dir: string;
}

export async function fetchInventoryPosition(params: InventoryPositionParams): Promise<InventoryPositionPayload> {
  const qs = new URLSearchParams({
    limit: String(params.limit),
    offset: String(params.offset),
    sort_by: params.sort_by,
    sort_dir: params.sort_dir,
  });
  if (params.item?.trim()) qs.set("item", params.item.trim());
  if (params.location?.trim()) qs.set("location", params.location.trim());
  return fetchJson(`/inventory/position?${qs}`);
}

export async function fetchInventoryKpis(params: { item?: string; location?: string; months?: number }): Promise<InventoryKpis> {
  const qs = buildSearchParams({
    item: params.item?.trim() || undefined,
    location: params.location?.trim() || undefined,
    months: params.months,
  });
  return fetchJson(`/inventory/kpis?${qs}`);
}

export async function fetchInventoryTrend(params: { item?: string; location?: string; months?: number }): Promise<InventoryTrendPayload> {
  const qs = buildSearchParams({
    item: params.item?.trim() || undefined,
    location: params.location?.trim() || undefined,
    months: params.months,
  });
  return fetchJson(`/inventory/trend?${qs}`);
}

export async function fetchInventoryItemDetail(params: { item: string; location: string; months?: number }): Promise<InventoryItemDetailPayload> {
  const qs = new URLSearchParams({ item: params.item.trim(), location: params.location.trim() });
  if (params.months) qs.set("months", String(params.months));
  return fetchJson(`/inventory/item-detail?${qs}`);
}
