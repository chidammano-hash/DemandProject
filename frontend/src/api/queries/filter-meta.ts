import { fetchJson } from "./core";
import type { GlobalFilters } from "@/types/theme";

export interface SkuCountPayload {
  count: number;
}

export const filterMetaKeys = {
  skuCount: (filters: Partial<GlobalFilters>) => ["sku-count", filters] as const,
};

export async function fetchSkuCount(filters: Partial<GlobalFilters>): Promise<SkuCountPayload> {
  const qs = new URLSearchParams();
  if (filters.brand?.length) qs.set("brand", filters.brand.join(","));
  if (filters.category?.length) qs.set("category", filters.category.join(","));
  if (filters.item?.length) qs.set("item", filters.item.join(","));
  if (filters.location?.length) qs.set("location", filters.location.join(","));
  if (filters.market?.length) qs.set("market", filters.market.join(","));
  if (filters.channel?.length) qs.set("channel", filters.channel.join(","));
  if (filters.cluster?.length) qs.set("cluster", filters.cluster.join(","));
  const q = qs.toString();
  return fetchJson(`/domains/sku/count${q ? `?${q}` : ""}`);
}
