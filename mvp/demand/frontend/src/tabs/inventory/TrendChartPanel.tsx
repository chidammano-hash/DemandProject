import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { LoadingElement } from "@/components/LoadingElement";
import { useChartColors } from "@/hooks/useChartColors";
import { formatCompactNumber } from "@/lib/formatters";
import type { InventoryTrendPoint } from "@/types";

const CHART_MARGIN = { top: 8, right: 16, left: 8, bottom: 8 };
const CHART_DOT = { r: 3 };

interface TrendChartPanelProps {
  trendData: InventoryTrendPoint[];
  isLoading: boolean;
}

export function TrendChartPanel({ trendData, isLoading }: TrendChartPanelProps) {
  const { chartColors, trendColors } = useChartColors();

  if (isLoading) {
    return (
      <LoadingElement tabKey="inventory" message="Loading trend data..." />
    );
  }

  if (trendData.length === 0) return null;

  const chartData = trendData.map((pt) => ({
    month: pt.month,
    total_on_hand: pt.total_on_hand,
    total_on_order: pt.total_on_order,
    monthly_sales: pt.monthly_sales,
    avg_lead_time: pt.avg_lead_time,
    dos: pt.dos,
  }));

  // Compute reorder-point threshold: avg lead time × 1.5 (safety buffer)
  const validLTs = trendData.map((pt) => pt.avg_lead_time).filter((v): v is number => v != null && v > 0);
  const ropThreshold = validLTs.length > 0
    ? (validLTs.reduce((a, b) => a + b, 0) / validLTs.length) * 1.5
    : null;

  const TOOLTIP_LABELS: Record<string, string> = {
    total_on_hand: "On Hand",
    total_on_order: "On Order",
    monthly_sales: "Monthly Sales",
    avg_lead_time: "Avg Lead Time",
    dos: "Days of Supply",
  };

  return (
    <div className="space-y-2">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">
        Monthly Inventory Trend
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData} margin={CHART_MARGIN}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
          <XAxis
            dataKey="month"
            tick={{ fontSize: 11, fill: chartColors.axis }}
          />
          <YAxis
            yAxisId="left"
            tick={{ fontSize: 11, fill: chartColors.axis }}
            tickFormatter={(v: number) => formatCompactNumber(v)}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={{ fontSize: 11, fill: chartColors.axis }}
            tickFormatter={(v: number) => `${Number(v).toFixed(0)}d`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: chartColors.tooltip_bg,
              borderColor: chartColors.tooltip_border,
            }}
            formatter={(value: number, name: string) => {
              const formatted =
                name === "avg_lead_time" || name === "dos"
                  ? `${Number(value).toFixed(1)} days`
                  : formatCompactNumber(value);
              return [formatted, TOOLTIP_LABELS[name] ?? name];
            }}
          />
          <Legend
            wrapperStyle={{ fontSize: 11 }}
            formatter={(value: string) => TOOLTIP_LABELS[value] ?? value}
          />
          <Line
            type="monotone"
            dataKey="total_on_hand"
            yAxisId="left"
            stroke={trendColors[0]}
            strokeWidth={2}
            dot={CHART_DOT}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="total_on_order"
            yAxisId="left"
            stroke={trendColors[1]}
            strokeWidth={2}
            dot={CHART_DOT}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="monthly_sales"
            yAxisId="left"
            stroke={trendColors[3]}
            strokeWidth={2}
            dot={CHART_DOT}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="dos"
            yAxisId="right"
            stroke={trendColors[4]}
            strokeWidth={2.5}
            dot={CHART_DOT}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="avg_lead_time"
            yAxisId="right"
            stroke={trendColors[2]}
            strokeWidth={2}
            strokeDasharray="5 3"
            dot={CHART_DOT}
            connectNulls
          />
          {/* Reorder Point threshold: LT × 1.5 — DOS below this line needs attention */}
          {ropThreshold != null && (
            <ReferenceLine
              yAxisId="right"
              y={ropThreshold}
              stroke="#ef4444"
              strokeDasharray="4 3"
              strokeWidth={1.5}
              label={{ value: `ROP ${ropThreshold.toFixed(0)}d`, position: "insideTopRight", fontSize: 10, fill: "#ef4444" }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
