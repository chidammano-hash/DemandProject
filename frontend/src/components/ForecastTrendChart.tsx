import { memo, useMemo } from "react";
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { useChartColors } from "@/hooks/useChartColors";

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

function formatY(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(0)}K`;
  return String(v);
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

  const hasCI = useMemo(
    () => includeCI && data.every((d) => d.lower_80 != null && d.upper_80 != null),
    [data, includeCI],
  );

  // Recharts renders a CI band as an Area driven by [lower, upper] tuples.
  // Reshape the data once so the chart can pull the band directly off
  // each row instead of computing it per render.
  const chartData = useMemo(
    () =>
      data.map((d) => ({
        ...d,
        ci: hasCI && d.lower_80 != null && d.upper_80 != null
          ? [d.lower_80, d.upper_80]
          : undefined,
      })),
    [data, hasCI],
  );

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
        No forecast data available
      </div>
    );
  }

  const tooltipBg = chartColors.tooltip;
  const tooltipFg = theme === "dark" ? "#e5e5e5" : "#171717";

  return (
    <div role="img" aria-label="Forecast vs actual trend chart">
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={chartData} margin={{ top: 16, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke={chartColors.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="month"
            tick={{ fontSize: 10, fill: chartColors.axis }}
            stroke={chartColors.grid}
          />
          <YAxis
            tick={{ fontSize: 10, fill: chartColors.axis }}
            tickFormatter={formatY}
            axisLine={false}
            stroke={chartColors.grid}
          />
          <Tooltip
            contentStyle={{ backgroundColor: tooltipBg, border: `1px solid ${chartColors.grid}`, color: tooltipFg, fontSize: 12 }}
            labelStyle={{ color: tooltipFg }}
            formatter={(value: number | [number, number], name: string) => {
              if (Array.isArray(value)) {
                return [`${formatY(value[0])} – ${formatY(value[1])}`, name];
              }
              return [formatY(value), name];
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: chartColors.axis }} />
          {hasCI && (
            <Area
              type="monotone"
              dataKey="ci"
              name="80% CI"
              fill={seriesColors[0]}
              fillOpacity={0.18}
              stroke="transparent"
              isAnimationActive={false}
            />
          )}
          <Line
            type="monotone"
            dataKey="forecast"
            name="Forecast"
            stroke={seriesColors[0]}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="actual"
            name="Actual"
            stroke={seriesColors[1]}
            strokeWidth={2}
            strokeDasharray="4 4"
            dot={{ r: 2, fill: seriesColors[1] }}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
});
