// ---------------------------------------------------------------------------
// Inventory Planning — IPfeature3: Safety Stock
// ---------------------------------------------------------------------------

export const safetyStockKeys = {
  summary: (filters?: Record<string, string>) => ["safety-stock", "summary", filters] as const,
  detail: (params?: Record<string, unknown>) => ["safety-stock", "detail", params] as const,
  waterfall: (itemNo: string, loc: string) => ["safety-stock", "waterfall", itemNo, loc] as const,
  explain: (itemId: string, loc: string) => ["safety-stock", "explain", itemId, loc] as const,
  config: () => ["safety-stock", "config"] as const,
};

export interface SafetyStockSummary {
  total_skus: number;
  below_ss_count: number;
  avg_ss_coverage: number;
  avg_ss_days: number;
  computed_at?: string | null;
  by_abc: Array<{ abc_vol: string; count: number; below_ss_count: number; avg_coverage: number }>;
}

export interface SafetyStockRow {
  item_id: string;
  loc: string;
  ss_combined: number;
  ss_coverage: number;
  is_below_ss: boolean;
  reorder_point: number;
  abc_vol: string;
  ss_demand_only: number;
  ss_lt_only: number;
  z_score: number | null;
  ss_gap: number | null;
  current_qty_on_hand: number | null;
  current_dos: number | null;
  service_level_target: number | null;
}

export interface SafetyStockWaterfall {
  item_id: string;
  loc: string;
  ss_demand_only: number;
  ss_lt_only: number;
  ss_combined: number;
  reorder_point: number;
  avg_daily_demand: number;
  lt_mean_days: number;
}

export async function fetchSafetyStockSummary(filters?: Record<string, string>): Promise<SafetyStockSummary> {
  const params = new URLSearchParams(filters ?? {});
  const res = await fetch(`/inv-planning/safety-stock/summary?${params}`);
  if (!res.ok) throw new Error("Failed to fetch safety stock summary");
  return res.json();
}

export async function fetchSafetyStockDetail(params?: {
  is_below_ss?: boolean;
  abc_vol?: string;
  item?: string;
  loc?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; rows: SafetyStockRow[] }> {
  const p = new URLSearchParams();
  if (params?.is_below_ss !== undefined) p.set("is_below_ss", String(params.is_below_ss));
  if (params?.abc_vol) p.set("abc_vol", params.abc_vol);
  if (params?.item) p.set("item", params.item);
  if (params?.loc) p.set("loc", params.loc);
  if (params?.limit !== undefined) p.set("limit", String(params.limit));
  if (params?.offset !== undefined) p.set("offset", String(params.offset));
  const res = await fetch(`/inv-planning/safety-stock/detail?${p}`);
  if (!res.ok) throw new Error("Failed to fetch safety stock detail");
  return res.json();
}

export async function fetchSafetyStockWaterfall(itemNo: string, loc: string): Promise<SafetyStockWaterfall> {
  const res = await fetch(
    `/inv-planning/safety-stock/waterfall?item_id=${encodeURIComponent(itemNo)}&loc=${encodeURIComponent(loc)}`,
  );
  if (!res.ok) throw new Error("Failed to fetch safety stock waterfall");
  return res.json();
}

export async function fetchSafetyStockConfig(): Promise<Record<string, unknown>> {
  const res = await fetch("/inv-planning/safety-stock/config");
  if (!res.ok) throw new Error("Failed to fetch safety stock config");
  return res.json();
}

// ── Safety Stock Explainability ──────────────────────────────────────────

export interface SsComponentDetail {
  label: string;
  value: number;
  pct_of_total?: number;
  formula: string;
  inputs?: Record<string, number | null>;
}

export interface SsSensitivityScenario {
  scenario: string;
  ss_result: number;
  delta: string;
}

export interface SsExplanation {
  item_id: string;
  loc: string;
  abc_vol: string | null;
  abc_xyz_segment: string | null;
  service_level: number | null;
  z_score: number | null;
  formula: string;
  formula_substituted: string;
  components: {
    demand_component: SsComponentDetail;
    leadtime_component: SsComponentDetail;
    combined: SsComponentDetail;
  };
  sensitivity: SsSensitivityScenario[];
  context: {
    current_on_hand: number | null;
    current_dos: number | null;
    reorder_point: number | null;
    is_below_ss: boolean | null;
    gap_qty: number | null;
    forecast_source: string;
    avg_daily_demand: number | null;
  };
}

export async function fetchSafetyStockExplain(itemId: string, loc: string): Promise<SsExplanation> {
  const res = await fetch(
    `/inv-planning/safety-stock/explain?item_id=${encodeURIComponent(itemId)}&loc=${encodeURIComponent(loc)}`,
  );
  if (!res.ok) throw new Error("Failed to fetch safety stock explanation");
  return res.json();
}

// ── What-If Scenario Builder ────────────────────────────────────────────

export interface WhatIfCurrentSimulated {
  ss_combined: number;
  reorder_point: number;
  monthly_holding_cost: number;
}

export interface WhatIfDelta {
  ss_change: number;
  ss_change_pct: number;
  rop_change: number;
  holding_cost_change_monthly: number;
}

export interface WhatIfInputsUsed {
  demand_mean: number;
  demand_std: number;
  lt_mean_days: number;
  lt_std_days: number;
  service_level: number;
  z_score: number;
  unit_cost: number;
}

export interface WhatIfResult {
  current: WhatIfCurrentSimulated;
  simulated: WhatIfCurrentSimulated;
  delta: WhatIfDelta;
  inputs_used: WhatIfInputsUsed;
}

export async function fetchSsWhatIf(params: {
  item_id: string;
  loc: string;
  demand_change_pct: number;
  lt_change_days: number;
  service_level_override?: string;
}): Promise<WhatIfResult> {
  const p = new URLSearchParams({
    item_id: params.item_id,
    loc: params.loc,
    demand_change_pct: String(params.demand_change_pct),
    lt_change_days: String(params.lt_change_days),
  });
  if (params.service_level_override) p.set("service_level_override", params.service_level_override);
  const res = await fetch(`/inv-planning/safety-stock/what-if?${p}`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to fetch what-if simulation");
  return res.json();
}
