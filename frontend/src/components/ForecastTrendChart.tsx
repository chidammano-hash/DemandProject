import { memo } from "react";
import { EChartContainer } from "@/components/EChartContainer";
import { useChartColors } from "@/hooks/useChartColors";
import { formatInt } from "@/lib/formatters";

interface ForecastTrendPoint {
  month: string;
  forecast: number;
  actual: number;
  /** Lower 80% quantile. When paired with `upper_80` and `includeCI` is true, renders as a shaded band. */
  lower_80?: number;
  /** Upper 80% quantile. */
  upper_80?: number;
}

interface ForecastTrendChartProps {
  data: ForecastTrendPoint[];
  /** Optional override; defaults to ThemeContext. */
  theme?: "light" | "dark";
  /** Optional override; defaults to useChartColors() values. */
  chartColors?: { grid: string; axis: string; tooltip: string };
  /** Optional override; defaults to useChartColors().trendColors. */
  seriesColors?: string[];
  /** UX-3: render the 80% confidence-interval band when data has lower_80/upper_80. */
  includeCI?: boolean;
}

export const ForecastTrendChart = memo(function ForecastTrendChart({
  data,
  theme: themeProp,
  chartColors: chartColorsProp,
  seriesColors: seriesColorsProp,
  includeCI = false,
}: ForecastTrendChartProps) {
  const { theme: ctxTheme, chartColors: ctxChartColors, trendColors } = useChartColors();
  const theme: "light" | "dark" = themeProp ?? (ctxTheme === "dark" ? "dark" : "light");
  const chartColors = chartColorsProp ?? {
    grid: ctxChartColors.grid,
    axis: ctxChartColors.axis,
    tooltip: ctxChartColors.tooltip_bg,
  };
  const seriesColors = seriesColorsProp ?? trendColors;
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
        No forecast data available
      </div>
    );
  }

  // The CI band renders as two stacked "line" series: the lower bound drawn
  // invisibly, then the delta (upper - lower) stacked on top with areaStyle.
  // This is the canonical ECharts pattern for a filled band between two lines.
  const hasCI =
    includeCI &&
    data.every((d) => d.lower_80 != null && d.upper_80 != null);
  const lowers = hasCI ? data.map((d) => d.lower_80 as number) : [];
  const bandHeights = hasCI
    ? data.map((d) => (d.upper_80 as number) - (d.lower_80 as number))
    : [];

  const option = buildForecastTrendOption({
    data,
    theme,
    chartColors,
    seriesColors,
    hasCI,
    lowers,
    bandHeights,
  });

  return <EChartContainer option={option} theme={theme} height={260} />;
});

interface BuildOptionArgs {
  data: ForecastTrendPoint[];
  theme: "light" | "dark";
  chartColors: { grid: string; axis: string; tooltip: string };
  seriesColors: string[];
  hasCI: boolean;
  lowers: number[];
  bandHeights: number[];
}

/**
 * Pure ECharts option builder for the Forecast-vs-Actual trend chart. Extracted
 * so the tooltip `valueFormatter` (U4.2) can be unit-tested without mounting
 * ECharts. The tooltip thousands-separates hover values (`formatInt`) so a
 * 7-digit point reads "2,157,763" consistent with the compact K/M axis and the
 * KPI tiles above the chart.
 */
export function buildForecastTrendOption({
  data,
  theme,
  chartColors,
  seriesColors,
  hasCI,
  lowers,
  bandHeights,
}: BuildOptionArgs) {
  const months = data.map((d) => d.month);
  const forecasts = data.map((d) => d.forecast);
  const actuals = data.map((d) => d.actual);

  return {
    tooltip: {
      trigger: "axis" as const,
      backgroundColor: chartColors.tooltip,
      borderColor: chartColors.grid,
      textStyle: { color: theme === "dark" ? "#e5e5e5" : "#171717", fontSize: 12 },
      valueFormatter: (v: unknown) => formatInt(typeof v === "number" ? v : null),
    },
    legend: {
      data: hasCI ? ["Forecast", "Actual", "80% CI"] : ["Forecast", "Actual"],
      bottom: 0,
      textStyle: { color: chartColors.axis, fontSize: 11 },
    },
    grid: { top: 16, right: 16, bottom: 36, left: 60, containLabel: false },
    xAxis: {
      type: "category" as const,
      data: months,
      axisLine: { lineStyle: { color: chartColors.grid } },
      axisLabel: { color: chartColors.axis, fontSize: 10 },
    },
    yAxis: {
      type: "value" as const,
      axisLine: { show: false },
      splitLine: { lineStyle: { color: chartColors.grid, type: "dashed" as const } },
      axisLabel: {
        color: chartColors.axis,
        fontSize: 10,
        formatter: (v: number) => {
          if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
          if (v >= 1_000) return `${(v / 1_000).toFixed(0)}K`;
          return String(v);
        },
      },
    },
    series: [
      // CI band (drawn first so it sits behind the forecast/actual lines).
      ...(hasCI
        ? [
            {
              name: "_ci_lower",
              type: "line" as const,
              data: lowers,
              stack: "ci",
              symbol: "none",
              lineStyle: { opacity: 0 },
              itemStyle: { opacity: 0 },
              silent: true,
              showInLegend: false,
              tooltip: { show: false },
            },
            {
              name: "80% CI",
              type: "line" as const,
              data: bandHeights,
              stack: "ci",
              symbol: "none",
              lineStyle: { opacity: 0 },
              areaStyle: { color: seriesColors[0], opacity: 0.18 },
              tooltip: { show: false },
            },
          ]
        : []),
      {
        name: "Forecast",
        type: "line" as const,
        data: forecasts,
        smooth: true,
        areaStyle: { opacity: 0.15 },
        lineStyle: { width: 2, color: seriesColors[0] },
        itemStyle: { color: seriesColors[0] },
        symbol: "none",
      },
      {
        name: "Actual",
        type: "line" as const,
        data: actuals,
        smooth: true,
        lineStyle: { width: 2, color: seriesColors[1], type: "dashed" as const },
        itemStyle: { color: seriesColors[1] },
        symbol: "circle",
        symbolSize: 4,
      },
    ],
  };
}
