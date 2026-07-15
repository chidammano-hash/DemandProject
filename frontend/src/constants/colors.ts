import type { Theme } from "@/types";
import { PALETTE } from "@/constants/palette";

/**
 * Legacy chart-color exports, now DERIVED from the single palette source
 * (`constants/palette.ts`). Export names and shapes are unchanged so existing
 * importers keep working during the redesign migration.
 *
 * @deprecated for new code — read `useChartColors().series` / `.roles`
 * instead of these positional arrays. These compat exports are removed once
 * the per-tab sweep completes.
 */

export const TREND_COLORS_BY_THEME: Record<Theme, string[]> = {
  light: [...PALETTE.light.charts.series],
  dark: [...PALETTE.dark.charts.series],
  soft: [...PALETTE.soft.charts.series],
};

/**
 * Colorblind-aware categorical palette.
 *
 * Historically the fixed Okabe-Ito 8; now an alias of the light-mode series,
 * which keeps the colorblind-safe intent but matches the product palette.
 * Prefer `useChartColors().okabeIto` (mode-aware) or `.series`.
 */
export const OKABE_ITO: string[] = [...PALETTE.light.charts.series];

export const CHART_COLORS: Record<
  Theme,
  { grid: string; axis: string; tooltip_bg: string; tooltip_border: string }
> = {
  light: {
    grid: PALETTE.light.charts.grid,
    axis: PALETTE.light.charts.axis,
    tooltip_bg: PALETTE.light.charts.tooltipBg,
    tooltip_border: PALETTE.light.charts.tooltipBorder,
  },
  dark: {
    grid: PALETTE.dark.charts.grid,
    axis: PALETTE.dark.charts.axis,
    tooltip_bg: PALETTE.dark.charts.tooltipBg,
    tooltip_border: PALETTE.dark.charts.tooltipBorder,
  },
  soft: {
    grid: PALETTE.soft.charts.grid,
    axis: PALETTE.soft.charts.axis,
    tooltip_bg: PALETTE.soft.charts.tooltipBg,
    tooltip_border: PALETTE.soft.charts.tooltipBorder,
  },
};

/**
 * Demand-history series colors (light values; per-mode migration happens with
 * the tab sweep via `useChartColors().roles`).
 */
export const SKU_SALES_COLORS: Record<string, string> = {
  tothist_dmd: PALETTE.light.charts.roles.error,
  sales_qty: PALETTE.light.charts.roles.ai,
  qty_shipped: PALETTE.light.charts.roles.forecast,
  qty_ordered: PALETTE.light.charts.roles.good,
};

const DFU_MODEL_COLORS: Record<string, string> = {
  champion: PALETTE.light.charts.roles.champion,
  ceiling: PALETTE.light.charts.roles.ceiling,
  external: PALETTE.light.charts.roles.reference,
  lgbm_cluster: PALETTE.light.charts.roles.good,
};

const DFU_MODEL_FALLBACK_COLORS = [...PALETTE.light.charts.fallback];

export function skuModelColor(model: string, idx: number): string {
  return (
    DFU_MODEL_COLORS[model] ?? DFU_MODEL_FALLBACK_COLORS[idx % DFU_MODEL_FALLBACK_COLORS.length]
  );
}
