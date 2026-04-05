import { buildSearchParams } from "./helpers";
import { fetchJson } from "./core";
import type {
  AccuracySlicePayload,
  LagCurvePayload,
} from "@/types";

// ---------------------------------------------------------------------------
// Accuracy queries
// ---------------------------------------------------------------------------
export interface SliceParams {
  group_by: string;
  lag: number;
  models: string;
  month_from: string;
  common_skus: boolean;
  include_sku_count: boolean;
  item?: string;
  location?: string;
  seasonality_profile?: string;
  time_grain?: string;
  brand?: string;
  category?: string;
  market?: string;
  cluster_assignment?: string;
}

export async function fetchAccuracySlice(params: SliceParams): Promise<AccuracySlicePayload> {
  const qs = buildSearchParams({
    group_by: params.group_by,
    lag: params.lag,
    models: params.models.trim() || undefined,
    month_from: params.month_from,
    common_dfus: params.common_skus ? "true" : undefined,
    include_dfu_count: params.include_sku_count ? "true" : undefined,
    item: params.item,
    location: params.location,
    seasonality_profile: params.seasonality_profile,
    time_grain: params.time_grain,
    brand: params.brand,
    category: params.category,
    market: params.market,
    cluster_assignment: params.cluster_assignment,
  });
  return fetchJson(`/forecast/accuracy/slice?${qs}`);
}

export interface LagCurveParams {
  models: string;
  month_from: string;
  common_skus: boolean;
  include_sku_count: boolean;
  item?: string;
  location?: string;
  seasonality_profile?: string;
  time_grain?: string;
  brand?: string;
  category?: string;
  market?: string;
  cluster_assignment?: string;
}

export async function fetchLagCurve(params: LagCurveParams): Promise<LagCurvePayload> {
  const qs = buildSearchParams({
    models: params.models.trim() || undefined,
    month_from: params.month_from,
    common_dfus: params.common_skus ? "true" : undefined,
    include_dfu_count: params.include_sku_count ? "true" : undefined,
    item: params.item,
    location: params.location,
    seasonality_profile: params.seasonality_profile,
    time_grain: params.time_grain,
    brand: params.brand,
    category: params.category,
    market: params.market,
    cluster_assignment: params.cluster_assignment,
  });
  return fetchJson(`/forecast/accuracy/lag-curve?${qs}`);
}
