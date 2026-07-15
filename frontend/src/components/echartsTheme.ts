/**
 * Palette -> ECharts option-defaults bridge.
 *
 * `ModularReactECharts` (echarts-modular.tsx) stays a thin, theme-agnostic
 * wrapper — it never reads the palette itself. Panels that want their ECharts
 * `option` chrome (axis lines, split lines, tooltip, categorical series
 * color) to follow the product palette merge these defaults in, e.g.:
 *
 *   const themeDefaults = useEChartsThemeDefaults();
 *   const option = { ...themeDefaults, ...panelSpecificOption };
 *
 * Not wired into the 8 Customer-Analytics panels here — that merge happens
 * where each panel builds its `option` (tracked separately).
 */
import { useMemo } from "react";
import { useChartColors } from "@/hooks/useChartColors";

export interface EChartsThemeDefaults {
  /** Categorical series ramp — feeds ECharts' top-level `color` array. */
  color: string[];
  textStyle: { color: string };
  axisLine: { lineStyle: { color: string } };
  axisLabel: { color: string };
  splitLine: { lineStyle: { color: string } };
  tooltip: {
    backgroundColor: string;
    borderColor: string;
    textStyle: { color: string };
  };
}

/** Pure builder — takes the `useChartColors()` result so it's testable without rendering a hook. */
export function buildEChartsThemeDefaults(
  chartColors: ReturnType<typeof useChartColors>,
): EChartsThemeDefaults {
  const { series, chartColors: chrome } = chartColors;
  return {
    color: series,
    textStyle: { color: chrome.axis },
    axisLine: { lineStyle: { color: chrome.axis } },
    axisLabel: { color: chrome.axis },
    splitLine: { lineStyle: { color: chrome.grid } },
    tooltip: {
      backgroundColor: chrome.tooltip_bg,
      borderColor: chrome.tooltip_border,
      textStyle: { color: chrome.axis },
    },
  };
}

/** Mode-aware ECharts option defaults derived from the current palette. */
export function useEChartsThemeDefaults(): EChartsThemeDefaults {
  const chartColors = useChartColors();
  // useChartColors returns a fresh object per render; the palette only
  // changes with the mode, so memoize on that.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => buildEChartsThemeDefaults(chartColors), [chartColors.theme]);
}
