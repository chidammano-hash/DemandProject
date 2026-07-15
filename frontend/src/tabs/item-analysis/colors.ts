// ---------------------------------------------------------------------------
// Chart color maps + layout constants for the Item Analysis unified chart.
// All colors derive from the semantic palette (constants/palette.ts) per
// color mode — no hex literals. Components resolve the mode via
// `useChartColors().theme` and call the builders below.
// ---------------------------------------------------------------------------
import { PALETTE, type ColorMode, type ChartRoles } from "@/constants/palette";

export const CHART_MARGIN = { top: 8, right: 40, left: 18, bottom: 8 };
export const DESELECT_OPACITY = 0.25;

export interface ItemAnalysisColors {
  /** Forward staged production forecast — same petrol as the champion, so the
   * past champion overlay and its staged future read as one storyline. */
  prodForecast: string;
  /** Saved AI Champion adjustment — the violet AI-semantic role. */
  aiChampion: string;
  /** DQ original (pre-correction) overlay — error red. */
  dqOrig: string;
  /** Per-model colors for staging forecast lines (muted fallback ramp — the
   * candidates are deliberately secondary to the champion/actual lines). */
  staging: Record<string, string>;
  stagingFallback: string;
  supply: Record<string, string>;
}

export function getItemAnalysisColors(mode: ColorMode): ItemAnalysisColors {
  const { roles, fallback, series } = PALETTE[mode].charts;
  return {
    prodForecast: roles.forecast,
    aiChampion: roles.ai,
    dqOrig: roles.error,
    staging: {
      lgbm_cluster: fallback[0],
      nbeats: fallback[3],
      nhits: fallback[4],
      chronos2_enriched: fallback[2],
      mstl: fallback[5],
    },
    stagingFallback: fallback[1],
    supply: {
      total_on_hand: series[0],
      total_on_order: roles.ceiling,
      total_position: roles.ai,
      inv_monthly_sales: roles.reference,
      dos: roles.error,
      avg_lead_time: roles.warning,
      safety_stock: fallback[4],
      cycle_stock: fallback[5],
    },
  };
}

/**
 * Color for one staging model line. Customer-derived lines carry their
 * semantic role; classic models use the muted candidate ramp.
 */
export function stagingModelColor(
  modelId: string,
  roles: ChartRoles,
  colors: ItemAnalysisColors,
): string {
  if (modelId === "customer_bottom_up") return roles.good;
  if (modelId === "customer_source_champion") return roles.champion;
  if (modelId === "customer_bottom_up_blend") return roles.ai;
  return colors.staging[modelId] ?? colors.stagingFallback;
}

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
