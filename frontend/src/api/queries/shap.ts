import { fetchJson } from "./core";
import type {
  ShapModelsPayload,
  ShapSummaryPayload,
  ShapTimeframesPayload,
  ShapTimeframeDetailPayload,
  ShapFilterParams,
  SkuShapPayload,
} from "@/types/shap";

// ---------------------------------------------------------------------------
// SHAP feature importance queries (Feature 42)
// ---------------------------------------------------------------------------
export async function fetchShapModels(): Promise<ShapModelsPayload> {
  return fetchJson("/forecast/shap/models");
}

export async function fetchShapSummary(modelId: string, topN = 15, filters?: ShapFilterParams): Promise<ShapSummaryPayload> {
  const qs = new URLSearchParams({ top_n: String(topN) });
  if (filters?.item) qs.set("item", filters.item);
  if (filters?.location) qs.set("location", filters.location);
  if (filters?.brand) qs.set("brand", filters.brand);
  if (filters?.category) qs.set("category", filters.category);
  if (filters?.market) qs.set("market", filters.market);
  return fetchJson(`/forecast/shap/${encodeURIComponent(modelId)}/summary?${qs}`);
}

export async function fetchShapTimeframes(modelId: string): Promise<ShapTimeframesPayload> {
  return fetchJson(`/forecast/shap/${encodeURIComponent(modelId)}/timeframes`);
}

export async function fetchShapTimeframeDetail(
  modelId: string,
  idx: number,
  topN = 15,
  cluster = "all",
  filters?: ShapFilterParams,
): Promise<ShapTimeframeDetailPayload> {
  const qs = new URLSearchParams({ top_n: String(topN), cluster });
  if (filters?.item) qs.set("item", filters.item);
  if (filters?.location) qs.set("location", filters.location);
  if (filters?.brand) qs.set("brand", filters.brand);
  if (filters?.category) qs.set("category", filters.category);
  if (filters?.market) qs.set("market", filters.market);
  return fetchJson(`/forecast/shap/${encodeURIComponent(modelId)}/timeframe/${idx}?${qs}`);
}

export interface ShapClustersPayload {
  model_id: string;
  clusters: string[];
}

export async function fetchShapClusters(modelId: string): Promise<ShapClustersPayload> {
  return fetchJson(`/forecast/shap/${encodeURIComponent(modelId)}/clusters`);
}

export async function fetchSkuShap(
  modelId: string,
  itemNo: string,
  loc: string,
  topN = 10,
): Promise<SkuShapPayload> {
  const qs = new URLSearchParams({
    item_id: itemNo,
    loc,
    top_n: String(topN),
  });
  return fetchJson(`/forecast/shap/${encodeURIComponent(modelId)}/sku?${qs}`);
}
