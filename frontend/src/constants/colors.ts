import type { Theme } from "@/types";

export const TREND_COLORS_BY_THEME: Record<Theme, string[]> = {
  light: ["#2563EB", "#0D9488", "#D97706", "#0891B2", "#DC2626", "#0284C7"],
  dark: ["#60A5FA", "#2DD4BF", "#FBBF24", "#22D3EE", "#FCA5A5", "#7DD3FC"],
  soft: ["#2667C7", "#0E9E72", "#D4890A", "#0891B2", "#D44040", "#0598B0"],
};

/**
 * Okabe-Ito color-blind-safe categorical palette (UX-3).
 *
 * Published by Okabe & Ito (2008) — 8 colors chosen so every pair is
 * distinguishable for deuteranopia, protanopia, and tritanopia. Prefer this
 * over `TREND_COLORS_BY_THEME` for new charts that encode category/series
 * (not magnitude).
 *
 * Legacy theme-keyed palettes stay in place for existing charts with
 * signed-off visuals; new code can opt in via `useChartColors().okabeIto`.
 */
export const OKABE_ITO: string[] = [
  "#E69F00", // orange
  "#56B4E9", // sky blue
  "#009E73", // bluish green
  "#F0E442", // yellow
  "#0072B2", // blue
  "#D55E00", // vermillion
  "#CC79A7", // reddish purple
  "#000000", // black
];

export const CHART_COLORS: Record<
  Theme,
  { grid: string; axis: string; tooltip_bg: string; tooltip_border: string }
> = {
  light: { grid: "#eaeef4", axis: "#7c8798", tooltip_bg: "#ffffff", tooltip_border: "#eaeef4" },
  dark: { grid: "#283449", axis: "#9aa6b8", tooltip_bg: "#1e2433", tooltip_border: "#283449" },
  soft: { grid: "#E5E0D8", axis: "#9A9088", tooltip_bg: "#FDFCFA", tooltip_border: "#E5E0D8" },
};

export const SKU_SALES_COLORS: Record<string, string> = {
  tothist_dmd: "#e11d48",
  sales_qty: "#9333ea",
  qty_shipped: "#2563eb",
  qty_ordered: "#059669",
};

const DFU_MODEL_COLORS: Record<string, string> = {
  champion: "#D97706",
  ceiling: "#0891B2",
  external: "#06B6D4",
  lgbm_cluster: "#0D9488",
};

const DFU_MODEL_FALLBACK_COLORS = [
  "#64748B",
  "#78716C",
  "#0F766E",
  "#B45309",
  "#0891B2",
  "#EA580C",
];

export function skuModelColor(model: string, idx: number): string {
  return (
    DFU_MODEL_COLORS[model] ?? DFU_MODEL_FALLBACK_COLORS[idx % DFU_MODEL_FALLBACK_COLORS.length]
  );
}
