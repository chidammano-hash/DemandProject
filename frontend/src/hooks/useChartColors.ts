import { useThemeContext } from "@/context/ThemeContext";
import { CHART_COLORS, TREND_COLORS_BY_THEME } from "@/constants/colors";
import { PALETTE } from "@/constants/palette";

/**
 * Returns chart styling colors for the current theme mode.
 *
 * New code reads:
 *   - `roles`   — fixed semantic colors (forecast, actual, error, ...). The
 *                 same concept always wears the same color on every screen.
 *   - `series`  — the 8-color categorical ramp for unnamed series.
 *   - `heatmap` — good -> bad 5-stop scale.
 *   - `chartColors` — grid / axis / tooltip chrome.
 *
 * `trendColors` and `okabeIto` are compat aliases of `series` kept while the
 * per-tab migration runs; see `frontend/CONTRIBUTING.md §5`.
 */
export function useChartColors() {
  const { theme } = useThemeContext();
  const charts = PALETTE[theme].charts;
  return {
    theme,
    chartColors: CHART_COLORS[theme],
    trendColors: TREND_COLORS_BY_THEME[theme],
    okabeIto: [...charts.series],
    series: [...charts.series],
    roles: charts.roles,
    fallback: [...charts.fallback],
    heatmap: [...charts.heatmapScale],
    sequential: [...charts.sequential],
  };
}
