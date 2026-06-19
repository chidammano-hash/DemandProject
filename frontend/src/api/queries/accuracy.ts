import { buildSearchParams } from "./helpers";
import { fetchJson } from "./core";
import type {
  AccuracySlicePayload,
  LagCurvePayload,
  AccuracyDecompositionPayload,
  ErrorContributorsPayload,
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

export interface LagLeaderboardEntry {
  rank: number;
  model_id: string;
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  n_rows: number;
}

export interface LagLeaderboardLag {
  lag: number;
  rankings: LagLeaderboardEntry[];
}

export interface LagLeaderboardPayload {
  lags: LagLeaderboardLag[];
  limit: number;
  source: string;
}

export interface LagLeaderboardParams {
  month_from?: string;
  month_to?: string;
  limit?: number;
}

export const lagLeaderboardKeys = {
  list: (params?: LagLeaderboardParams) => ["lag-leaderboard", params ?? {}] as const,
};

export async function fetchLagLeaderboard(
  params?: LagLeaderboardParams,
): Promise<LagLeaderboardPayload> {
  const qs = buildSearchParams({
    month_from: params?.month_from,
    month_to: params?.month_to,
    limit: params?.limit,
  });
  return fetchJson(`/forecast/accuracy/lag-leaderboard?${qs}`);
}

// ---------------------------------------------------------------------------
// Per-DFU accuracy decomposition (diagnostic layer)
// ---------------------------------------------------------------------------
export interface DecompositionParams {
  group_by: string;
  lag: number;
  models?: string;
  month_from?: string;
  cluster_assignment?: string;
  seasonality_profile?: string;
}

export const accuracyDecompositionKeys = {
  list: (params: DecompositionParams) => ["accuracy-decomposition", params] as const,
};

export async function fetchAccuracyDecomposition(
  params: DecompositionParams,
): Promise<AccuracyDecompositionPayload> {
  const qs = buildSearchParams({
    group_by: params.group_by,
    lag: params.lag,
    models: params.models?.trim() || undefined,
    month_from: params.month_from,
    cluster_assignment: params.cluster_assignment,
    seasonality_profile: params.seasonality_profile,
  });
  return fetchJson(`/forecast/accuracy/decomposition?${qs}`);
}

export interface ErrorContributorsParams {
  lag: number;
  limit?: number;
  models?: string;
  month_from?: string;
  cluster_assignment?: string;
  seasonality_profile?: string;
}

export const errorContributorsKeys = {
  list: (params: ErrorContributorsParams) => ["error-contributors", params] as const,
};

export async function fetchErrorContributors(
  params: ErrorContributorsParams,
): Promise<ErrorContributorsPayload> {
  const qs = buildSearchParams({
    lag: params.lag,
    limit: params.limit,
    models: params.models?.trim() || undefined,
    month_from: params.month_from,
    cluster_assignment: params.cluster_assignment,
    seasonality_profile: params.seasonality_profile,
  });
  return fetchJson(`/forecast/accuracy/error-contributors?${qs}`);
}
