import { fetchJson } from "./core";
import type { GlobalFilters } from "@/types/theme";

export interface DfuCountPayload {
  count: number;
}

export const filterMetaKeys = {
  dfuCount: (filters: Partial<GlobalFilters>) => ["dfu-count", filters] as const,
};

export async function fetchDfuCount(filters: Partial<GlobalFilters>): Promise<DfuCountPayload> {
  const qs = new URLSearchParams();
  if (filters.brand?.length) qs.set("brand", filters.brand.join(","));
  if (filters.category?.length) qs.set("category", filters.category.join(","));
  if (filters.item?.length) qs.set("item", filters.item.join(","));
  if (filters.location?.length) qs.set("location", filters.location.join(","));
  if (filters.market?.length) qs.set("market", filters.market.join(","));
  if (filters.channel?.length) qs.set("channel", filters.channel.join(","));
  if (filters.cluster?.length) qs.set("cluster", filters.cluster.join(","));
  const q = qs.toString();
  return fetchJson(`/domains/dfu/count${q ? `?${q}` : ""}`);
}
