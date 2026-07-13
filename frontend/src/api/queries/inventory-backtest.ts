import { buildSearchParams } from "./helpers";
import { fetchJson } from "./request";
import type {
  InvBacktestSummaryPayload,
  InvBacktestTrendPayload,
  InvBacktestRootCausePayload,
  InvBacktestDetailPayload,
} from "@/types";

// ---------------------------------------------------------------------------
// Inventory Backtest queries (Feature 37)
// ---------------------------------------------------------------------------
export interface InvBacktestFilterParams {
  models?: string;
  month_from?: string;
  month_to?: string;
  item?: string;
  location?: string;
  cluster_assignment?: string;
  abc_vol?: string;
  region?: string;
  excess_dos_threshold?: number;
}

function invBacktestParams(params: InvBacktestFilterParams): Record<string, string | number | undefined> {
  return {
    models: params.models?.trim() || undefined,
    month_from: params.month_from?.trim() || undefined,
    month_to: params.month_to?.trim() || undefined,
    item: params.item?.trim() || undefined,
    location: params.location?.trim() || undefined,
    cluster_assignment: params.cluster_assignment?.trim() || undefined,
    abc_vol: params.abc_vol?.trim() || undefined,
    region: params.region?.trim() || undefined,
    excess_dos_threshold: params.excess_dos_threshold ?? undefined,
  };
}

export async function fetchInvBacktestSummary(params: InvBacktestFilterParams): Promise<InvBacktestSummaryPayload> {
  const qs = buildSearchParams(invBacktestParams(params));
  return fetchJson(`/inventory-backtest/summary?${qs}`);
}

export async function fetchInvBacktestTrend(params: InvBacktestFilterParams): Promise<InvBacktestTrendPayload> {
  const qs = buildSearchParams(invBacktestParams(params));
  return fetchJson(`/inventory-backtest/trend?${qs}`);
}

export async function fetchInvBacktestRootCause(
  params: InvBacktestFilterParams & { model_id: string },
): Promise<InvBacktestRootCausePayload> {
  const qs = buildSearchParams({ model_id: params.model_id, ...invBacktestParams(params) });
  return fetchJson(`/inventory-backtest/root-cause?${qs}`);
}

export interface InvBacktestDetailParams extends InvBacktestFilterParams {
  event_type?: string;
  limit: number;
  offset: number;
  sort_by: string;
  sort_dir: string;
}

export async function fetchInvBacktestDetail(params: InvBacktestDetailParams): Promise<InvBacktestDetailPayload> {
  const qs = buildSearchParams({
    limit: params.limit,
    offset: params.offset,
    sort_by: params.sort_by,
    sort_dir: params.sort_dir,
    event_type: params.event_type && params.event_type !== "all" ? params.event_type : undefined,
    ...invBacktestParams(params),
  });
  return fetchJson(`/inventory-backtest/detail?${qs}`);
}
