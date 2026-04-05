/** Shared display constants for models and strategies. */

export const MODEL_LABELS: Record<string, string> = {
  lgbm_cluster: "LightGBM",
  catboost_cluster: "CatBoost",
  xgboost_cluster: "XGBoost",
  lgbm_cust_enriched: "LightGBM (Cust)",
  catboost_cust_enriched: "CatBoost (Cust)",
  xgboost_cust_enriched: "XGBoost (Cust)",
  chronos: "Chronos T5",
  chronos_bolt: "Chronos Bolt",
  chronos2: "Chronos 2",
  chronos2_enriched: "Chronos 2E",
  bolt_hierarchical: "Bolt Hierarchical",
  mstl: "MSTL",
  nbeats: "N-BEATS",
  nhits: "N-HiTS",
  seasonal_naive: "Seasonal Naive",
  rolling_mean: "Rolling Mean",
};

export const MODEL_TYPE_COLORS: Record<string, string> = {
  tree: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200",
  foundation: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  statistical: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200",
  deep_learning: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
};

export function modelLabel(id: string): string {
  return MODEL_LABELS[id] || id.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}
