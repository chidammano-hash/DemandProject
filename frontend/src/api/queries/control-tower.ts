import { fetchJson } from "./core";
import { buildSearchParams } from "./helpers";

// ---------------------------------------------------------------------------
// IPfeature15: Control Tower
// ---------------------------------------------------------------------------

export interface ControlTowerKpis {
  computed_at: string | null;
  health: {
    total_skus: number; healthy_count: number; monitor_count: number;
    at_risk_count: number; critical_count: number;
    avg_health_score: number | null; avg_ss_coverage: number | null;
    below_ss_count: number; below_ss_pct: number | null; avg_portfolio_dos: number | null;
  };
  exceptions: {
    open_exceptions_total: number; critical_exceptions: number;
    high_exceptions: number; recommended_order_value: number | null;
  };
  fill_rate: { portfolio_fill_rate_3m: number | null; total_shortage_qty_3m: number | null };
  demand_signals: { urgent_demand_signals: number; projected_stockouts_today: number };
  intramonth: { items_with_stockout_this_month: number; extended_stockouts_this_month: number };
}

export interface ControlTowerAlert {
  alert_id: string;
  source: string;
  severity: string;
  item_id: string;
  loc: string;
  alert_type: string;
  description: string;
  action: string;
  alert_ts: string | null;
  abc_vol: string | null;
}

export interface ControlTowerCriticalItem {
  item_id: string; loc: string; abc_vol: string | null; abc_xyz_segment: string | null;
  health_score: number | null; health_tier: string | null;
  ss_coverage: number | null; is_below_ss: boolean;
  current_dos: number | null; target_dos_min: number | null; target_dos_max: number | null;
  open_exception_count: number; recommended_order_qty: number | null;
  fill_rate_last_3m: number | null; stockout_days_this_month: number;
}

export interface ControlTowerFilterParams {
  location?: string[];
  brand?: string[];
  category?: string[];
  market?: string[];
  item?: string[];
}

export const controlTowerKeys = {
  kpis:       () => ["ct-kpis"] as const,
  alerts:     (f?: Record<string, unknown>) => ["ct-alerts", f ?? {}] as const,
  topCritical:(f?: Record<string, unknown>) => ["ct-top-critical", f ?? {}] as const,
  trend:      (months?: number) => ["ct-trend", months ?? 6] as const,
};

export const fetchControlTowerKpis = (): Promise<ControlTowerKpis> =>
  fetchJson("/control-tower/kpis");

export async function fetchControlTowerAlerts(
  params: { limit?: number; severity?: string } & ControlTowerFilterParams = {},
): Promise<{ total: number; alerts: ControlTowerAlert[] }> {
  const qs = buildSearchParams({
    limit: params.limit,
    severity: params.severity,
    location: params.location?.length === 1 ? params.location[0] : undefined,
    brand: params.brand?.length === 1 ? params.brand[0] : undefined,
    category: params.category?.length === 1 ? params.category[0] : undefined,
    market: params.market?.length === 1 ? params.market[0] : undefined,
    item: params.item?.length === 1 ? params.item[0] : undefined,
  });
  const q = qs.toString();
  return fetchJson(`/control-tower/alerts${q ? `?${q}` : ""}`);
}

export async function fetchControlTowerTopCritical(
  params: { limit?: number } & ControlTowerFilterParams = {},
): Promise<{ items: ControlTowerCriticalItem[] }> {
  const qs = buildSearchParams({
    limit: params.limit ?? 10,
    location: params.location?.length === 1 ? params.location[0] : undefined,
    brand: params.brand?.length === 1 ? params.brand[0] : undefined,
    category: params.category?.length === 1 ? params.category[0] : undefined,
    market: params.market?.length === 1 ? params.market[0] : undefined,
    item: params.item?.length === 1 ? params.item[0] : undefined,
  });
  return fetchJson(`/control-tower/top-critical?${qs.toString()}`);
}

export const fetchControlTowerTrend = (
  months = 6,
): Promise<{ trend: Array<Record<string, number | string | null>> }> =>
  fetchJson(`/control-tower/trend?months=${months}`);
