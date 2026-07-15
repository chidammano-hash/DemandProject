import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { CustomerBlendTrend, CustomerBlendTrendMonth } from "@/api/queries/customerForecast";
import { useChartColors } from "@/hooks/useChartColors";
import { formatCompactNumber, formatNumber } from "@/lib/formatters";

function formatMonth(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(value));
}

function formatWape(value: number | null): string {
  return value == null ? "—" : `${value.toFixed(1)}%`;
}

function toChartPoint(month: CustomerBlendTrendMonth) {
  const historical = month.phase === "backtest";
  return {
    month: month.month.slice(0, 7),
    actual_qty: historical ? month.actual_qty : null,
    customer_bottom_up_backtest: historical ? month.customer_bottom_up_qty : null,
    source_champion_backtest: historical ? month.source_champion_qty : null,
    customer_blend_backtest: historical ? month.customer_blend_qty : null,
    customer_bottom_up_staged: historical ? null : month.customer_bottom_up_qty,
    source_champion_staged: historical ? null : month.source_champion_qty,
    customer_blend_staged: historical ? null : month.customer_blend_qty,
    blend_interval:
      !historical && month.lower_bound != null && month.upper_bound != null
        ? [month.lower_bound, month.upper_bound]
        : null,
  };
}

export function CustomerForecastTrendChart({ trend }: { trend: CustomerBlendTrend }) {
  const { chartColors, roles } = useChartColors();
  const chartData = useMemo(() => trend.months.map(toChartPoint), [trend.months]);
  const planningMonth = trend.planning_month.slice(0, 7);

  if (trend.months.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
        No customer blend trend is available for these filters.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded-full bg-muted px-2 py-1 font-medium text-foreground">
          Customer Bottom-Up WAPE {formatWape(trend.accuracy.customer_bottom_up_wape_pct)}
        </span>
        <span className="rounded-full bg-muted px-2 py-1 font-medium text-foreground">
          Source Champion WAPE {formatWape(trend.accuracy.source_champion_wape_pct)}
        </span>
        <span className="rounded-full bg-primary/10 px-2 py-1 font-medium text-primary">
          Customer Blend WAPE {formatWape(trend.accuracy.customer_blend_wape_pct)}
        </span>
        <span className="rounded-full bg-muted px-2 py-1 text-muted-foreground">
          {trend.coverage.blended_rows.toLocaleString()} blended ·{" "}
          {trend.coverage.champion_fallback_rows.toLocaleString()} fallback
        </span>
        <span className="font-mono text-muted-foreground" title={trend.run_id}>
          {formatMonth(trend.planning_month)} · run {trend.run_id.slice(0, 7)}
        </span>
      </div>

      {trend.filter_notes.map((note) => (
        <p key={note} className="text-xs text-muted-foreground" role="note">
          {note}
        </p>
      ))}

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={chartData} margin={{ top: 16, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} vertical={false} />
          <XAxis
            dataKey="month"
            tick={{ fill: chartColors.axis, fontSize: 10 }}
            axisLine={{ stroke: chartColors.grid }}
            tickLine={false}
          />
          <YAxis
            width={62}
            tick={{ fill: chartColors.axis, fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(value: number) => formatCompactNumber(value)}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: chartColors.tooltip_bg,
              borderColor: chartColors.tooltip_border,
              fontSize: 12,
            }}
            formatter={(value: number | string, name: string) => [
              formatNumber(typeof value === "number" ? value : null),
              name,
            ]}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <ReferenceLine
            x={planningMonth}
            stroke={chartColors.axis}
            strokeDasharray="3 3"
            label={{ value: "Plan", fill: chartColors.axis, fontSize: 10 }}
          />
          <Area
            type="monotone"
            dataKey="blend_interval"
            name="Customer Blend interval"
            stroke="none"
            fill={roles.ai}
            fillOpacity={0.14}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="actual_qty"
            name="Actual sales"
            stroke={roles.actual}
            strokeWidth={2.5}
            dot={{ r: 3 }}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="customer_bottom_up_backtest"
            name="Customer Bottom-Up (backtest)"
            stroke={roles.good}
            strokeWidth={1.75}
            strokeDasharray="2 3"
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="customer_bottom_up_staged"
            name="Customer Bottom-Up (staged)"
            stroke={roles.good}
            strokeWidth={2.25}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="source_champion_backtest"
            name="Source Champion (backtest)"
            stroke={roles.champion}
            strokeWidth={1.5}
            strokeDasharray="2 3"
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="source_champion_staged"
            name="Source Champion (future)"
            stroke={roles.champion}
            strokeWidth={1.75}
            strokeDasharray="6 3"
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="customer_blend_backtest"
            name="Customer Blend (backtest)"
            stroke={roles.ai}
            strokeWidth={1.75}
            strokeDasharray="2 3"
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="customer_blend_staged"
            name="Customer Blend (staged)"
            stroke={roles.ai}
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <table className="sr-only" aria-label="Customer forecast monthly comparison">
        <thead>
          <tr>
            <th scope="col">Month</th>
            <th scope="col">Phase</th>
            <th scope="col">Actual sales</th>
            <th scope="col">Customer Bottom-Up</th>
            <th scope="col">Source Champion</th>
            <th scope="col">Customer Blend</th>
          </tr>
        </thead>
        <tbody>
          {trend.months.map((month) => (
            <tr key={`${month.phase}-${month.month}`}>
              <th scope="row">{formatMonth(month.month)}</th>
              <td>{month.phase}</td>
              <td>{formatNumber(month.actual_qty)}</td>
              <td>{formatNumber(month.customer_bottom_up_qty)}</td>
              <td>{formatNumber(month.source_champion_qty)}</td>
              <td>{formatNumber(month.customer_blend_qty)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
