import { fetchJson } from "./core";
import type {
  SkuAnalysisPayload,
  SkuAnalysisMode,
  MarketIntelPayload,
} from "@/types";

// ---------------------------------------------------------------------------
// SKU Analysis queries
// ---------------------------------------------------------------------------
export interface SkuAnalysisParams {
  mode: SkuAnalysisMode;
  item: string;
  location: string;
  points: number;
}

export async function fetchSkuAnalysis(params: SkuAnalysisParams): Promise<SkuAnalysisPayload> {
  const qs = new URLSearchParams({
    mode: params.mode,
    item: params.item.trim(),
    location: params.location.trim(),
    points: String(params.points),
  });
  return fetchJson(`/sku/analysis?${qs}`);
}

// ---------------------------------------------------------------------------
// Market Intelligence
// ---------------------------------------------------------------------------
export async function fetchMarketIntel(item: string, location: string): Promise<MarketIntelPayload> {
  return fetchJson("/market-intelligence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ item_id: item.trim(), location_id: location.trim() }),
  });
}
