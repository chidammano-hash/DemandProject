import { fetchJson } from "./core";

// ---------------------------------------------------------------------------
// Stale times
// ---------------------------------------------------------------------------
export const STALE_INSIGHTS = { ONE_MIN: 60_000, FIVE_MIN: 300_000 } as const;

// ---------------------------------------------------------------------------
// Query key factories
// ---------------------------------------------------------------------------
export const insightKeys = {
  actionFeed: () => ["inv-planning", "action-feed"] as const,
  rootCause: (item: string, loc: string) =>
    ["inv-planning", "root-cause", item, loc] as const,
  segmentDashboard: (segment: string) =>
    ["inv-planning", "segment-dashboard", segment] as const,
  ssCostBenefit: (params: Record<string, unknown>) =>
    ["inv-planning", "ss-cost-benefit", params] as const,
  serviceLevelWaterfall: () =>
    ["inv-planning", "service-level-waterfall"] as const,
  networkHeatmap: () => ["inv-planning", "network-heatmap"] as const,
  planningScorecard: () => ["inv-planning", "planning-scorecard"] as const,
  cashFlow: () => ["inv-planning", "cash-flow-timeline"] as const,
  constrainedOpt: (budget: number) =>
    ["inv-planning", "constrained-opt", budget] as const,
  proactiveRebalancing: () =>
    ["inv-planning", "proactive-rebalancing"] as const,
};

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

export interface ActionFeedItem {
  id: string;
  source: string;
  severity: string;
  item_no: string;
  loc: string;
  title: string;
  detail: string;
  financial_impact: number | null;
  action_url: string | null;
  created_at: string;
}

export interface ActionFeedPayload {
  total: number;
  actions: ActionFeedItem[];
  summary: {
    total: number;
    critical: number;
    high: number;
    financial_at_risk: number | null;
  };
}

export interface RootCausePayload {
  item_no: string;
  loc: string;
  causes: {
    factor: string;
    contribution_pct: number;
    description: string;
  }[];
}

export interface SegmentDashboardPayload {
  segment: string;
  dfu_count: number;
  open_exceptions: number;
  below_ss_count: number;
  avg_fill_rate: number | null;
  avg_health_score: number | null;
  policy_distribution: { policy_id: string; count: number }[];
  top_exceptions: {
    item_no: string;
    loc: string;
    exception_type: string;
    severity: string;
    detail: string;
  }[];
  recommended_actions: string[];
}

export interface ServiceLevelWaterfallPayload {
  base_forecast_csl: number;
  ss_buffer_contribution: number;
  lt_buffer_contribution: number;
  sensing_contribution: number;
  achieved_csl: number;
  segments: {
    lever: string;
    contribution: number;
    cumulative: number;
    color: string;
  }[];
}

export interface NetworkHeatmapCell {
  location: string;
  category: string;
  avg_dos: number | null;
  item_count: number;
}

export interface NetworkHeatmapPayload {
  cells: NetworkHeatmapCell[];
  locations: string[];
  categories: string[];
  summary: {
    total_locations: number;
    critical_cells: number;
    excess_cells: number;
    avg_network_dos: number | null;
  };
}

export interface PlanningMetric {
  name: string;
  current: number | null;
  prior: number | null;
  trend: "up" | "down" | "flat";
  unit: string;
  sparkline: number[];
}

export interface PlanningScorecardPayload {
  health_score: number | null;
  metrics: PlanningMetric[];
  period: string;
}

export interface CashFlowMonth {
  month: string;
  po_committed: number;
  planned_orders: number;
  ss_investment: number;
  total: number;
}

export interface CashFlowPayload {
  months: CashFlowMonth[];
  summary: {
    total_6m_outflow: number | null;
    largest_month: string | null;
    avg_monthly: number | null;
  };
}

export interface ConstrainedOptItem {
  item_no: string;
  loc: string;
  current_ss: number;
  recommended_ss: number;
  investment: number;
  csl_before: number;
  csl_after: number;
}

export interface ConstrainedOptPayload {
  budget: number;
  allocated: number;
  items_improved: number;
  csl_before: number | null;
  csl_after: number | null;
  items: ConstrainedOptItem[];
}

// ---------------------------------------------------------------------------
// Fetch functions
// ---------------------------------------------------------------------------

export async function fetchActionFeed(): Promise<ActionFeedPayload> {
  return fetchJson("/inv-planning/action-feed");
}

export async function fetchRootCause(
  item: string,
  loc: string,
): Promise<RootCausePayload> {
  return fetchJson(
    `/inv-planning/exceptions/${encodeURIComponent(item)}/${encodeURIComponent(loc)}/root-cause`,
  );
}

export async function fetchSegmentDashboard(
  segment: string,
): Promise<SegmentDashboardPayload> {
  return fetchJson(
    `/inv-planning/segment-dashboard?segment=${encodeURIComponent(segment)}`,
  );
}

export async function fetchSsCostBenefit(
  params: Record<string, string>,
): Promise<unknown> {
  const qs = new URLSearchParams(params);
  return fetchJson(`/inv-planning/ss-cost-benefit?${qs}`);
}

export async function fetchServiceLevelWaterfall(): Promise<ServiceLevelWaterfallPayload> {
  return fetchJson("/inv-planning/service-level-waterfall");
}

export async function fetchNetworkHeatmap(): Promise<NetworkHeatmapPayload> {
  return fetchJson("/inv-planning/network-heatmap");
}

export async function fetchPlanningScorecard(): Promise<PlanningScorecardPayload> {
  return fetchJson("/inv-planning/planning-scorecard");
}

export async function fetchCashFlowTimeline(): Promise<CashFlowPayload> {
  return fetchJson("/inv-planning/cash-flow-timeline");
}

export async function fetchConstrainedOpt(
  budget: number,
): Promise<ConstrainedOptPayload> {
  return fetchJson(
    `/inv-planning/constrained-optimization?budget=${budget}`,
  );
}

export async function fetchProactiveRebalancing(): Promise<unknown> {
  return fetchJson("/inv-planning/proactive-rebalancing");
}
