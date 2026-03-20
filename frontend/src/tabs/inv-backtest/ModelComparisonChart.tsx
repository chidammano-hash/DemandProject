/**
 * ModelComparisonChart — Stockout vs Excess Rate bar chart with WAPE overlay line.
 */
import {
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatNumber } from "@/lib/formatters";
import { CHART_MARGIN } from "./invBacktestShared";

export function ModelComparisonChart({
  comparisonData,
  chartColors,
  trendColors,
}: {
  comparisonData: { model: string; stockout_rate: number; excess_rate: number; wape: number }[];
  chartColors: { grid: string; axis: string; tooltip_bg: string; tooltip_border: string };
  trendColors: string[];
}) {
  if (comparisonData.length === 0) return null;

  return (
    <div className="space-y-2">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">
        Model Comparison — Stockout vs Excess Rate
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={comparisonData} margin={CHART_MARGIN}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
          <XAxis
            dataKey="model"
            tick={{ fontSize: 10, fill: chartColors.axis }}
            angle={-20}
            textAnchor="end"
            height={50}
          />
          <YAxis
            yAxisId="left"
            tick={{ fontSize: 11, fill: chartColors.axis }}
            tickFormatter={(v: number) => `${v}%`}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={{ fontSize: 11, fill: chartColors.axis }}
            tickFormatter={(v: number) => `${v}%`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: chartColors.tooltip_bg,
              borderColor: chartColors.tooltip_border,
            }}
            formatter={(value: number, name: string) => [
              `${formatNumber(value)}%`,
              name === "stockout_rate"
                ? "Stockout Rate"
                : name === "excess_rate"
                  ? "Excess Rate"
                  : "WAPE",
            ]}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar yAxisId="left" dataKey="stockout_rate" name="Stockout Rate" fill="#ef4444" barSize={20}>
            {comparisonData.map((_, idx) => (
              <Cell key={idx} fill="#ef4444" />
            ))}
          </Bar>
          <Bar yAxisId="left" dataKey="excess_rate" name="Excess Rate" fill="#f59e0b" barSize={20}>
            {comparisonData.map((_, idx) => (
              <Cell key={idx} fill="#f59e0b" />
            ))}
          </Bar>
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="wape"
            name="WAPE"
            stroke={trendColors[0]}
            strokeWidth={2}
            dot={{ r: 4 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
