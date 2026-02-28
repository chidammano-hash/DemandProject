import type { Theme } from "@/types";

export const TREND_COLORS_BY_THEME: Record<Theme, string[]> = {
  light: ["#4f46e5", "#0d9488", "#d97706", "#7c3aed", "#dc2626", "#0284c7"],
  dark: ["#818cf8", "#2dd4bf", "#fbbf24", "#a78bfa", "#fca5a5", "#7dd3fc"],
  soft: ["#3574C4", "#0E9E72", "#D4890A", "#7A4ED4", "#D44040", "#0598B0"],
};

export const CHART_COLORS: Record<Theme, { grid: string; axis: string; tooltip_bg: string; tooltip_border: string }> = {
  light: { grid: "#e2e8f0", axis: "#64748b", tooltip_bg: "#ffffff", tooltip_border: "#e2e8f0" },
  dark: { grid: "#2d3548", axis: "#94a3b8", tooltip_bg: "#1e2433", tooltip_border: "#2d3548" },
  soft: { grid: "#DDD8D0", axis: "#8A8078", tooltip_bg: "#FDFCFA", tooltip_border: "#DDD8D0" },
};

export const DFU_SALES_COLORS: Record<string, string> = {
  tothist_dmd: "#e11d48",
  qty_shipped: "#2563eb",
  qty_ordered: "#059669",
};

export const DFU_MODEL_COLORS: Record<string, string> = {
  champion: "#f59e0b",
  ceiling: "#8b5cf6",
  external: "#06b6d4",
  lgbm_global: "#84cc16",
  lgbm_cluster: "#14b8a6",
  lgbm_transfer: "#f97316",
  catboost_global: "#ec4899",
  catboost_cluster: "#6366f1",
  catboost_transfer: "#a3e635",
  xgboost_global: "#a855f7",
  xgboost_cluster: "#0ea5e9",
  xgboost_transfer: "#fb923c",
};

export const DFU_MODEL_FALLBACK_COLORS = ["#64748b", "#78716c", "#0f766e", "#b45309", "#9333ea", "#e879f9"];

export function dfuModelColor(model: string, idx: number): string {
  return DFU_MODEL_COLORS[model] ?? DFU_MODEL_FALLBACK_COLORS[idx % DFU_MODEL_FALLBACK_COLORS.length];
}
