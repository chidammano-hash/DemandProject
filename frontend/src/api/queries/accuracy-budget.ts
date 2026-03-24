/**
 * Accuracy Budget — API types, query keys, and fetchers.
 * Maps to: /accuracy-budget/decomposition, /abc-breakdown, /model-comparison,
 *          /monthly-trend, /forecast-value
 */

// ---------------------------------------------------------------------------
// Types — match actual API response shapes
// ---------------------------------------------------------------------------

export interface AbcBreakdownRow {
  abc_class: string;
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  n_dfus: number;
  volume_share: number | null;
  error_share: number | null;
}

export interface DecompositionComponent {
  name: string;
  estimated_gain_pp: number;
  rationale: string;
  cluster_ids?: number[];
  months?: number[];
  abc_class?: string;
}

export interface DecompositionResponse {
  current_accuracy: number | null;
  current_wape: number | null;
  current_bias: number | null;
  n_dfus: number | null;
  model_id: string;
  oracle_ceiling: number | null;
  oracle_wape: number | null;
  naive_baseline: number | null;
  naive_wape: number | null;
  forecast_value_added: number | null;
  addressable_gap: number | null;
  abc_breakdown: AbcBreakdownRow[];
  cluster_breakdown: { ml_cluster: number; accuracy_pct: number | null; wape: number | null; n_dfus: number }[];
  components: DecompositionComponent[];
  irreducible_noise: number | null;
}

export interface ModelComparisonRow {
  model_id: string;
  accuracy_pct: number | null;
  wape: number | null;
  bias: number | null;
  n_dfus: number | null;
}

export interface MonthlyTrendPoint {
  month: number;
  accuracy: number | null;
  wape: number | null;
  bias: number | null;
  n_dfus: number;
  flag: string | null;
}

export interface ForecastValueBaseline {
  name: string;
  description: string;
  accuracy: number | null;
  wape: number | null;
}

// ---------------------------------------------------------------------------
// Query keys
// ---------------------------------------------------------------------------

export const accuracyBudgetKeys = {
  all: ["accuracy-budget"] as const,
  decomposition: (modelId?: string) =>
    [...accuracyBudgetKeys.all, "decomposition", modelId ?? "lgbm_cluster"] as const,
  abc: () => [...accuracyBudgetKeys.all, "abc-breakdown"] as const,
  models: () => [...accuracyBudgetKeys.all, "model-comparison"] as const,
  monthly: () => [...accuracyBudgetKeys.all, "monthly-trend"] as const,
  forecastValue: () => [...accuracyBudgetKeys.all, "forecast-value"] as const,
};

// ---------------------------------------------------------------------------
// Fetchers — match actual backend routes
// ---------------------------------------------------------------------------

export async function fetchAccuracyDecomposition(
  modelId = "lgbm_cluster",
): Promise<DecompositionResponse> {
  const res = await fetch(
    `/accuracy-budget/decomposition?model_id=${encodeURIComponent(modelId)}`,
  );
  if (!res.ok) throw new Error(`fetchAccuracyDecomposition: ${res.status}`);
  return res.json();
}

export async function fetchAbcBreakdown(): Promise<{
  classes: AbcBreakdownRow[];
}> {
  const res = await fetch("/accuracy-budget/abc-breakdown");
  if (!res.ok) throw new Error(`fetchAbcBreakdown: ${res.status}`);
  return res.json();
}

export async function fetchModelComparison(): Promise<{
  models: ModelComparisonRow[];
  oracle_ceiling: { accuracy: number; wape: number } | null;
}> {
  const res = await fetch("/accuracy-budget/model-comparison");
  if (!res.ok) throw new Error(`fetchModelComparison: ${res.status}`);
  return res.json();
}

export async function fetchMonthlyTrend(): Promise<{
  months: MonthlyTrendPoint[];
  worst_month: MonthlyTrendPoint | null;
  best_month: MonthlyTrendPoint | null;
}> {
  const res = await fetch("/accuracy-budget/monthly-trend");
  if (!res.ok) throw new Error(`fetchMonthlyTrend: ${res.status}`);
  return res.json();
}

export async function fetchForecastValue(): Promise<{
  baselines: ForecastValueBaseline[];
  ml_model: { name: string; accuracy: number | null; wape: number | null } | null;
  value_added: Record<string, number | null> | null;
}> {
  const res = await fetch("/accuracy-budget/forecast-value");
  if (!res.ok) throw new Error(`fetchForecastValue: ${res.status}`);
  return res.json();
}
