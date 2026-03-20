// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature9: Demand Signals
// ---------------------------------------------------------------------------

export const demandSignalsKeys = {
  summary: (date?: string) => ["demand-signals", "summary", date] as const,
  list: (params?: Record<string, unknown>) => ["demand-signals", "list", params] as const,
  item: (itemNo: string, loc: string) => ["demand-signals", "item", itemNo, loc] as const,
};

export interface DemandSignalSummary {
  signal_date: string;
  above_plan_count: number;
  below_plan_count: number;
  on_plan_count: number;
  urgent_count: number;
  watch_count: number;
  projected_stockouts: number;
}

export interface DemandSignalRow {
  item_no: string;
  loc: string;
  signal_date: string;
  signal_type: string;
  signal_strength: number;
  demand_vs_forecast_pct: number;
  projected_stockout: boolean;
  alert_priority: string;
  mtd_actual: number;
  projected_monthly: number;
  forecast_monthly: number;
  current_on_hand: number;
  is_below_ss: boolean;
}

export async function fetchDemandSignalsSummary(date?: string): Promise<DemandSignalSummary> {
  const p = date ? `?signal_date=${date}` : "";
  const res = await fetch(`/inv-planning/demand-signals/summary${p}`);
  if (!res.ok) throw new Error("Failed to fetch demand signals summary");
  return res.json();
}

export async function fetchDemandSignals(params?: {
  signal_type?: string;
  alert_priority?: string;
  item?: string;
  loc?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; rows: DemandSignalRow[] }> {
  const p = new URLSearchParams();
  if (params?.signal_type) p.set("signal_type", params.signal_type);
  if (params?.alert_priority) p.set("alert_priority", params.alert_priority);
  if (params?.item) p.set("item", params.item);
  if (params?.loc) p.set("loc", params.loc);
  if (params?.limit !== undefined) p.set("limit", String(params.limit));
  if (params?.offset !== undefined) p.set("offset", String(params.offset));
  const res = await fetch(`/inv-planning/demand-signals?${p}`);
  if (!res.ok) throw new Error("Failed to fetch demand signals");
  return res.json();
}

export async function fetchDemandSignalItem(itemNo: string, loc: string): Promise<DemandSignalRow> {
  const res = await fetch(
    `/inv-planning/demand-signals/item?item_no=${encodeURIComponent(itemNo)}&loc=${encodeURIComponent(loc)}`,
  );
  if (!res.ok) throw new Error("Failed to fetch demand signal item");
  return res.json();
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature10: Safety Stock Monte Carlo Simulation
// ---------------------------------------------------------------------------

export const simulationKeys = {
  results: (params?: Record<string, unknown>) => ["simulation", "results", params] as const,
  compare: (itemNo: string, loc: string) => ["simulation", "compare", itemNo, loc] as const,
  status: (simRunId: string) => ["simulation", "status", simRunId] as const,
};

export interface SimulationResult {
  sim_run_id: string;
  item_no: string;
  loc: string;
  simulation_date: string;
  n_simulations: number;
  target_csl: number;
  recommended_ss: number;
  recommended_ss_days: number;
  analytical_ss: number;
  sim_vs_analytical_pct: number;
  run_duration_secs: number;
  results_by_ss_level: Array<{ ss_qty: number; csl: number }>;
}

export async function fetchSimulationResults(params?: {
  item?: string;
  loc?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; rows: SimulationResult[] }> {
  const p = new URLSearchParams();
  if (params?.item) p.set("item", params.item);
  if (params?.loc) p.set("loc", params.loc);
  if (params?.limit !== undefined) p.set("limit", String(params.limit));
  const res = await fetch(`/inv-planning/simulation/results?${p}`);
  if (!res.ok) throw new Error("Failed to fetch simulation results");
  return res.json();
}

export async function fetchSimulationCompare(itemNo: string, loc: string): Promise<SimulationResult[]> {
  const res = await fetch(
    `/inv-planning/simulation/compare?item_no=${encodeURIComponent(itemNo)}&loc=${encodeURIComponent(loc)}`,
  );
  if (!res.ok) throw new Error("Failed to fetch simulation compare");
  return res.json();
}

export async function runSimulation(body: {
  item_no: string;
  loc: string;
  n_simulations?: number;
  target_csl?: number;
}): Promise<SimulationResult> {
  const res = await fetch("/inv-planning/simulation/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("Failed to run simulation");
  return res.json();
}

// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature13: Investment Plan
// ---------------------------------------------------------------------------

export const investmentKeys = {
  summary: (planId?: string, filters?: Record<string, unknown>) => ["investment", "summary", planId, filters ?? {}] as const,
  detail: (params?: Record<string, unknown>) => ["investment", "detail", params] as const,
  frontier: (planId?: string, filters?: Record<string, unknown>) => ["investment", "frontier", planId, filters ?? {}] as const,
};

export interface InvestmentSummary {
  plan_id: string;
  computation_date: string;
  total_items: number;
  total_current_investment: number;
  total_recommended_investment: number;
  total_investment_gap: number;
  avg_current_csl: number;
  avg_recommended_csl: number;
}

export interface InvestmentRow {
  item_no: string;
  loc: string;
  abc_vol: string;
  abc_xyz_segment: string;
  current_ss_qty: number;
  current_ss_value: number;
  current_csl: number;
  recommended_ss_qty: number;
  recommended_ss_value: number;
  recommended_csl: number;
  ss_increment_qty: number;
  investment_increment: number;
  csl_increment: number;
  marginal_roi: number;
  investment_rank: number;
  cumulative_investment: number;
}

export interface FrontierPoint {
  plan_id: string;
  budget_point: number;
  items_funded: number;
  achievable_csl: number;
  marginal_item: string;
}

export async function fetchInvestmentSummary(
  paramsOrPlanId?: string | Record<string, unknown>,
): Promise<InvestmentSummary> {
  const qs = new URLSearchParams();
  if (typeof paramsOrPlanId === "string") {
    if (paramsOrPlanId) qs.set("plan_id", paramsOrPlanId);
  } else if (paramsOrPlanId) {
    Object.entries(paramsOrPlanId).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
    });
  }
  const q = qs.toString();
  const res = await fetch(`/inv-planning/investment/summary${q ? `?${q}` : ""}`);
  if (!res.ok) throw new Error("Failed to fetch investment summary");
  return res.json();
}

export async function fetchInvestmentDetail(params?: Record<string, unknown>): Promise<{ total: number; rows: InvestmentRow[] }> {
  const p = new URLSearchParams();
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") p.set(k, String(v));
    });
  }
  const res = await fetch(`/inv-planning/investment/detail?${p}`);
  if (!res.ok) throw new Error("Failed to fetch investment detail");
  return res.json();
}

export async function fetchInvestmentFrontier(
  paramsOrPlanId?: string | Record<string, unknown>,
): Promise<FrontierPoint[]> {
  const qs = new URLSearchParams();
  if (typeof paramsOrPlanId === "string") {
    if (paramsOrPlanId) qs.set("plan_id", paramsOrPlanId);
  } else if (paramsOrPlanId) {
    Object.entries(paramsOrPlanId).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
    });
  }
  const q = qs.toString();
  const res = await fetch(`/inv-planning/investment/efficient-frontier${q ? `?${q}` : ""}`);
  if (!res.ok) throw new Error("Failed to fetch investment frontier");
  return res.json();
}

export async function runInvestmentPlan(params?: {
  budget?: number;
  target_csl?: number;
}): Promise<{ plan_id: string; total_items: number; total_investment_gap: number }> {
  const p = new URLSearchParams();
  if (params?.budget !== undefined) p.set("budget", String(params.budget));
  if (params?.target_csl !== undefined) p.set("target_csl", String(params.target_csl));
  const res = await fetch(`/inv-planning/investment/plan?${p}`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to run investment plan");
  return res.json();
}
