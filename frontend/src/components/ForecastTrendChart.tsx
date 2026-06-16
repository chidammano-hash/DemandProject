import { memo, useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useChartColors } from "@/hooks/useChartColors";
import { formatInt, formatCompactNumber } from "@/lib/formatters";

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
  /** Optional override; defaults to useChartColors().trendColors. */
  seriesColors?: string[];
  /** UX-3: render the 80% confidence-interval band when data has lower_80/upper_80. */
  includeCI?: boolean;
}

/** Forecast-vs-Actual trend chart (recharts). Replaces the retired ECharts
 *  rendering; renders an optional shaded 80% CI band behind the two lines. */
export const ForecastTrendChart = memo(function ForecastTrendChart({
  data,
  seriesColors: seriesColorsProp,
  includeCI = false,
}: ForecastTrendChartProps) {
  const { chartColors, trendColors } = useChartColors();
  const seriesColors = seriesColorsProp ?? trendColors;

  const hasCI =
    includeCI && data.length > 0 && data.every((d) => d.lower_80 != null && d.upper_80 != null);

  // recharts renders a band from a single dataKey whose value is a [low, high]
  // tuple, so precompute it per point.
  const chartData = useMemo(
    () =>
      data.map((d) => ({
        ...d,
        ...(hasCI ? { ci: [d.lower_80 as number, d.upper_80 as number] } : {}),
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

  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={chartData} margin={{ top: 16, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} vertical={false} />
        <XAxis
          dataKey="month"
          tick={{ fill: chartColors.axis, fontSize: 10 }}
          axisLine={{ stroke: chartColors.grid }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: chartColors.axis, fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={56}
          tickFormatter={(v: number) => formatCompactNumber(v)}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: chartColors.tooltip_bg,
            borderColor: chartColors.tooltip_border,
            fontSize: 12,
          }}
          formatter={(value: number | string) =>
            formatInt(typeof value === "number" ? value : null)
          }
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {hasCI && (
          <Area
            type="monotone"
            dataKey="ci"
            name="80% CI"
            stroke="none"
            fill={seriesColors[0]}
            fillOpacity={0.18}
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
          strokeDasharray="5 3"
          dot={{ r: 3 }}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
});
