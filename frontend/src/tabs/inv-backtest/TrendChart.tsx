/**
 * TrendChart — Monthly trend line chart for inventory backtest metrics.
 */
import { memo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { LoadingElement } from "@/components/LoadingElement";
import { skuModelColor } from "@/constants/colors";
import { CHART_MARGIN, TREND_METRICS } from "./invBacktestShared";

export const TrendChart = memo(function TrendChart({
  trendChartData,
  loadingTrend,
  trendMetric,
  models,
  onTrendMetricChange,
  chartColors,
}: {
  trendChartData: Record<string, string | number | null>[];
  loadingTrend: boolean;
  trendMetric: string;
  models: string[];
  onTrendMetricChange: (metric: string) => void;
  chartColors: { grid: string; axis: string; tooltip_bg: string; tooltip_border: string };
}) {
  if (loadingTrend) {
    return <LoadingElement tabKey="invBacktest" message="Loading trend..." />;
  }

  if (trendChartData.length === 0) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">
          Monthly Trend
        </p>
        <select
          className="h-7 rounded border border-input bg-background px-2 text-xs"
          value={trendMetric}
          onChange={(e) => onTrendMetricChange(e.target.value)}
        >
          {TREND_METRICS.map((tm) => (
            <option key={tm.key} value={tm.key}>{tm.label}</option>
          ))}
        </select>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={trendChartData} margin={CHART_MARGIN}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
          <XAxis dataKey="month" tick={{ fontSize: 11, fill: chartColors.axis }} />
          <YAxis
            tick={{ fontSize: 11, fill: chartColors.axis }}
            tickFormatter={(v: number) =>
              trendMetric === "avg_dos" ? `${v}d` : `${v}%`
            }
          />
          <Tooltip
            contentStyle={{
              backgroundColor: chartColors.tooltip_bg,
              borderColor: chartColors.tooltip_border,
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {models.map((mid, idx) => (
            <Line
              key={mid}
              type="monotone"
              dataKey={mid}
              name={mid}
              stroke={skuModelColor(mid, idx)}
              strokeWidth={2}
              dot={{ r: 2 }}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});
