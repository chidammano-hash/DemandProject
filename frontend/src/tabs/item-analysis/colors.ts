// ---------------------------------------------------------------------------
// Chart color maps + layout constants for the Item Analysis unified chart.
// Hex values relocated verbatim from UnifiedChartPanel.tsx (no theme change).
// ---------------------------------------------------------------------------
import type { ChartRoles } from "@/constants/palette";
export const PROD_FORECAST_COLOR = "#7c3aed";
/** Amber line for the saved AI Champion forward forecast (matches the panel's Sparkles accent). */
export const AI_CHAMPION_COLOR = "#f59e0b";
export const CHART_MARGIN = { top: 8, right: 40, left: 18, bottom: 8 };
export const DESELECT_OPACITY = 0.25;

/** Per-model colors for staging forecast lines */
export const STAGING_COLORS: Record<string, string> = {
  lgbm_cluster: "#2563eb",
  nbeats: "#ea580c",
  nhits: "#9333ea",
  chronos2_enriched: "#164e63",
  mstl: "#ca8a04",
};
export const STAGING_FALLBACK_COLOR = "#6b7280";

export function stagingModelColor(
  modelId: string,
  roles: ChartRoles,
  fallback = STAGING_FALLBACK_COLOR
): string {
  if (modelId === "customer_bottom_up") return roles.good;
  if (modelId === "customer_source_champion") return roles.champion;
  if (modelId === "customer_bottom_up_blend") return roles.ai;
  return STAGING_COLORS[modelId] ?? fallback;
}

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

/** Red used for DQ original (pre-correction) overlay series. */
export const DQ_ORIG_COLOR = "#DC2626";

/** dataKey → human label for chart tooltips and DQ-overlay pills. */
export const TOOLTIP_LABELS: Record<string, string> = {
  tothist_dmd: "Sale Qty (external)",
  sales_qty: "Sale Qty",
  sales_qty_orig: "Sale Qty (original)",
  qty_shipped: "Qty Shipped",
  qty_shipped_orig: "Shipped (original)",
  qty_ordered: "Qty Ordered",
  production_forecast: "Production Forecast",
  ai_champion: "AI Champion",
  total_on_hand: "On Hand",
  total_on_hand_orig: "On Hand (original)",
  total_on_order: "On Order",
  total_position: "Total Position",
  inv_monthly_sales: "Inv Monthly Sales",
  dos: "Days of Supply",
  avg_lead_time: "Avg Lead Time",
  safety_stock: "Safety Stock",
  cycle_stock: "Cycle Stock",
};
