import { useThemeContext } from "@/context/ThemeContext";
import { CHART_COLORS, OKABE_ITO, TREND_COLORS_BY_THEME } from "@/constants/colors";

/**
 * Returns chart styling colors based on the current theme context.
 * Replaces direct `CHART_COLORS[theme]` / `TREND_COLORS_BY_THEME[theme]` usage.
 *
 * `okabeIto` is the color-blind-safe 8-color categorical palette (UX-3).
 * Prefer it over `trendColors` for new categorical charts; see
 * `frontend/CONTRIBUTING.md §5`.
 */
export function useChartColors() {
  const { theme } = useThemeContext();
  return {
    theme,
    chartColors: CHART_COLORS[theme],
    trendColors: TREND_COLORS_BY_THEME[theme],
    okabeIto: OKABE_ITO,
  };
}
