// ---------------------------------------------------------------------------
// Chart color maps + layout constants for the Item Analysis unified chart.
// Hex values relocated verbatim from UnifiedChartPanel.tsx (no theme change).
// ---------------------------------------------------------------------------
export const PROD_FORECAST_COLOR = "#7c3aed";
export const CHART_MARGIN = { top: 8, right: 40, left: 18, bottom: 8 };
export const DESELECT_OPACITY = 0.25;

/** Per-model colors for staging forecast lines */
export const STAGING_COLORS: Record<string, string> = {
  lgbm_cluster: "#2563eb",
  catboost_cluster: "#dc2626",
  xgboost_cluster: "#16a34a",
  lgbm_cust_enriched: "#1d4ed8",
  catboost_cust_enriched: "#b91c1c",
  xgboost_cust_enriched: "#15803d",
  nbeats: "#ea580c",
  nhits: "#9333ea",
  chronos_bolt: "#0891b2",
  chronos: "#0e7490",
  chronos2: "#155e75",
  chronos2_enriched: "#164e63",
  bolt_hierarchical: "#0d9488",
  mstl: "#ca8a04",
  seasonal_naive: "#64748b",
  rolling_mean: "#78716c",
};
export const STAGING_FALLBACK_COLOR = "#6b7280";

export const SUPPLY_COLORS: Record<string, string> = {
  total_on_hand: "#2563EB",
  total_on_order: "#0D9488",
  total_position: "#a855f7",
  inv_monthly_sales: "#0891B2",
  dos: "#DC2626",
  avg_lead_time: "#D97706",
  safety_stock: "#8b5cf6",
  cycle_stock: "#06b6d4",
};
