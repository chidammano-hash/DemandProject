/**
 * Shared constants and types for InvBacktest tab sub-components.
 */

export const CHART_MARGIN = { top: 8, right: 16, left: 8, bottom: 8 };
export const PAGE_SIZE = 50;

export const TREND_METRICS = [
  { key: "stockout_rate", label: "Stockout Rate %" },
  { key: "excess_rate", label: "Excess Rate %" },
  { key: "avg_dos", label: "Avg DOS (days)" },
  { key: "wape", label: "WAPE %" },
] as const;

export type DetailSortCol =
  | "item_id"
  | "loc"
  | "month_start"
  | "model_id"
  | "forecast"
  | "actual_demand"
  | "eom_qty_on_hand"
  | "dos"
  | "forecast_error";
