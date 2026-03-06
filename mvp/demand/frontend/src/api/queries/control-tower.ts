import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// IPfeature15: Control Tower
// ---------------------------------------------------------------------------

export interface ControlTowerKpis {
  computed_at: string | null;
  health: {
    total_dfus: number; healthy_count: number; monitor_count: number;
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
  item_no: string;
  loc: string;
  alert_type: string;
  description: string;
  action: string;
  alert_ts: string | null;
  abc_vol: string | null;
}

export interface ControlTowerCriticalItem {
  item_no: string; loc: string; abc_vol: string | null; abc_xyz_segment: string | null;
  health_score: number | null; health_tier: string | null;
  ss_coverage: number | null; is_below_ss: boolean;
  current_dos: number | null; target_dos_min: number | null; target_dos_max: number | null;
  open_exception_count: number; recommended_order_qty: number | null;
  fill_rate_last_3m: number | null; stockout_days_this_month: number;
}

export const controlTowerKeys = {
  kpis:       () => ["ct-kpis"] as const,
  alerts:     (f?: Record<string, unknown>) => ["ct-alerts", f ?? {}] as const,
  topCritical:(limit?: number) => ["ct-top-critical", limit ?? 10] as const,
  trend:      (months?: number) => ["ct-trend", months ?? 6] as const,
};

export const fetchControlTowerKpis = (): Promise<ControlTowerKpis> =>
  fetchJson("/control-tower/kpis");

export async function fetchControlTowerAlerts(
  params: { limit?: number; severity?: string } = {},
): Promise<{ total: number; alerts: ControlTowerAlert[] }> {
  const qs = new URLSearchParams();
  if (params.limit) qs.set("limit", String(params.limit));
  if (params.severity) qs.set("severity", params.severity);
  const q = qs.toString();
  return fetchJson(`/control-tower/alerts${q ? `?${q}` : ""}`);
}

export const fetchControlTowerTopCritical = (
  limit = 10,
): Promise<{ items: ControlTowerCriticalItem[] }> =>
  fetchJson(`/control-tower/top-critical?limit=${limit}`);

export const fetchControlTowerTrend = (
  months = 6,
): Promise<{ trend: Array<Record<string, number | string | null>> }> =>
  fetchJson(`/control-tower/trend?months=${months}`);
