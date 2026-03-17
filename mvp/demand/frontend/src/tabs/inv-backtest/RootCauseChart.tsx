/**
 * RootCauseChart — Forecast bias correlation stacked bar chart.
 */
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { LoadingElement } from "@/components/LoadingElement";
import { CHART_MARGIN } from "./invBacktestShared";

export function RootCauseChart({
  rootCauseChartData,
  loadingRootCause,
  rootCauseModel,
  models,
  onRootCauseModelChange,
  chartColors,
}: {
  rootCauseChartData: { event: string; under_forecast: number; over_forecast: number; exact: number }[];
  loadingRootCause: boolean;
  rootCauseModel: string;
  models: string[] | undefined;
  onRootCauseModelChange: (model: string) => void;
  chartColors: { grid: string; axis: string; tooltip_bg: string; tooltip_border: string };
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">
          Forecast Bias Correlation
        </p>
        {models && (
          <select
            className="h-7 rounded border border-input bg-background px-2 text-xs"
            value={rootCauseModel}
            onChange={(e) => onRootCauseModelChange(e.target.value)}
          >
            {models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        )}
      </div>
      <p className="text-xs text-muted-foreground italic">
        Correlation between forecast bias direction and inventory events — not causal attribution
      </p>
      {loadingRootCause ? (
        <LoadingElement tabKey="invBacktest" message="Loading root cause..." />
      ) : rootCauseChartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={rootCauseChartData} layout="vertical" margin={CHART_MARGIN}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
            <XAxis type="number" tick={{ fontSize: 11, fill: chartColors.axis }} />
            <YAxis
              dataKey="event"
              type="category"
              tick={{ fontSize: 11, fill: chartColors.axis }}
              width={60}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: chartColors.tooltip_bg,
                borderColor: chartColors.tooltip_border,
              }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="under_forecast" name="Under-Forecast" stackId="a" fill="#ef4444" />
            <Bar dataKey="over_forecast" name="Over-Forecast" stackId="a" fill="#f59e0b" />
            <Bar dataKey="exact" name="Exact" stackId="a" fill="#94a3b8" />
          </BarChart>
        </ResponsiveContainer>
      ) : null}
    </div>
  );
}
