import { useThemeContext } from "@/context/ThemeContext";
import { CHART_COLORS, TREND_COLORS_BY_THEME } from "@/constants/colors";

/**
 * Returns chart styling colors based on the current theme context.
 * Replaces direct `CHART_COLORS[theme]` / `TREND_COLORS_BY_THEME[theme]` usage.
 */
export function useChartColors() {
  const { theme } = useThemeContext();
  return {
    theme,
    chartColors: CHART_COLORS[theme],
    trendColors: TREND_COLORS_BY_THEME[theme],
  };
}
