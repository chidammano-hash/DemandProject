import { memo, useState, useCallback } from "react";
import {
  CartesianGrid,
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
import type { InventoryTrendPoint, InventoryTrendParams } from "@/types";

const CHART_MARGIN = { top: 8, right: 40, left: 8, bottom: 8 };
const CHART_DOT = { r: 3 };

type SeriesKey =
  | "total_on_hand"
  | "total_on_order"
  | "total_position"
  | "monthly_sales"
  | "dos"
  | "avg_lead_time"
  | "safety_stock"
  | "cycle_stock";

const ALL_BASE_SERIES: SeriesKey[] = [
  "total_on_hand",
  "total_on_order",
  "total_position",
  "monthly_sales",
  "dos",
  "avg_lead_time",
];

interface TrendChartPanelProps {
  trendData: InventoryTrendPoint[];
  isLoading: boolean;
  params?: InventoryTrendParams;
}

export const TrendChartPanel = memo(function TrendChartPanel({ trendData, isLoading, params }: TrendChartPanelProps) {
  const { chartColors, trendColors } = useChartColors();
  const [hiddenSeries, setHiddenSeries] = useState<Set<SeriesKey>>(new Set());

  const toggleSeries = useCallback((key: SeriesKey) => {
    setHiddenSeries((prev) => {
      const next = new Set(prev);
      if (next.has(key)) { next.delete(key); } else { next.add(key); }
      return next;
    });
  }, []);

  const show = useCallback((key: SeriesKey) => !hiddenSeries.has(key), [hiddenSeries]);

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
    total_position: pt.total_on_hand + pt.total_on_order,
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
    total_position: "Total Position (On Hand + On Order)",
    monthly_sales: "Monthly Sales",
    avg_lead_time: "Avg Lead Time",
    dos: "Days of Supply",
    safety_stock: "Safety Stock",
    cycle_stock: "Cycle Stock",
  };

  const ss = params?.safety_stock ?? null;
  const ropUnits = params?.reorder_point_units ?? null;
  const hasSs = ss != null && trendData.some((d) => d.safety_stock != null);

  const ALL_SERIES: { key: SeriesKey; label: string; color: string; axis: "left" | "right" }[] = [
    { key: "total_on_hand",   label: "On Hand",         color: trendColors[0], axis: "left" },
    { key: "total_on_order",  label: "On Order",        color: trendColors[1], axis: "left" },
    { key: "total_position",  label: "Total Position",  color: "#a855f7",      axis: "left" },
    { key: "monthly_sales",   label: "Monthly Sales",   color: trendColors[3], axis: "left" },
    { key: "dos",             label: "Days of Supply",  color: trendColors[4], axis: "right" },
    { key: "avg_lead_time",   label: "Avg Lead Time",   color: trendColors[2], axis: "right" },
    ...(hasSs ? [
      { key: "safety_stock" as SeriesKey, label: "Safety Stock", color: "#8b5cf6", axis: "left" as const },
      { key: "cycle_stock"  as SeriesKey, label: "Cycle Stock",  color: "#06b6d4", axis: "left" as const },
    ] : []),
  ];

  const allVisible = ALL_SERIES.every((s) => !hiddenSeries.has(s.key));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">
          Monthly Inventory Trend
        </p>
        <button
          className="text-xs text-primary underline-offset-2 hover:underline"
          onClick={() => setHiddenSeries(allVisible ? new Set(ALL_SERIES.map((s) => s.key)) : new Set())}
        >
          {allVisible ? "Deselect All" : "Select All"}
        </button>
      </div>
      {/* Series toggle pills */}
      <div className="flex flex-wrap gap-2 mb-1">
        {ALL_SERIES.map((s) => {
          const active = !hiddenSeries.has(s.key);
          return (
            <button
              key={s.key}
              onClick={() => toggleSeries(s.key)}
              className={`flex items-center gap-1 rounded-full px-2 py-0.5 text-xs border transition-opacity ${
                active ? "opacity-100" : "opacity-40"
              }`}
              style={{ borderColor: s.color, color: active ? s.color : undefined }}
            >
              <span className="inline-block w-3 h-0.5" style={{ backgroundColor: s.color }} />
              {s.label}
              <span className="text-muted-foreground text-[10px]">({s.axis})</span>
            </button>
          );
        })}
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={CHART_MARGIN}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
          <XAxis
            dataKey="month"
            tick={{ fontSize: 11, fill: chartColors.axis }}
            angle={trendData.length > 8 ? -30 : 0}
            textAnchor={trendData.length > 8 ? "end" : "middle"}
            height={trendData.length > 8 ? 40 : 30}
          />
          <YAxis
            yAxisId="left"
            tick={{ fontSize: 11, fill: chartColors.axis }}
            tickFormatter={(v: number) => formatCompactNumber(v)}
            label={{ value: "Units", angle: -90, position: "insideLeft", fontSize: 10, offset: 10 }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={{ fontSize: 11, fill: chartColors.axis }}
            tickFormatter={(v: number) => `${Number(v).toFixed(0)}d`}
            label={{ value: "Days", angle: 90, position: "insideRight", fontSize: 10, offset: 10 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: chartColors.tooltip_bg,
              borderColor: chartColors.tooltip_border,
            }}
            labelFormatter={(label: string) => `Month: ${label}`}
            formatter={(value: number, name: string) => {
              const formatted =
                name === "avg_lead_time" || name === "dos"
                  ? `${Number(value).toFixed(1)} days`
                  : formatCompactNumber(value);
              return [formatted, TOOLTIP_LABELS[name] ?? name];
            }}
          />
          <Line
            type="monotone"
            dataKey="total_on_hand"
            yAxisId="left"
            stroke={trendColors[0]}
            strokeWidth={2}
            dot={CHART_DOT}
            connectNulls
            hide={!show("total_on_hand")}
          />
          <Line
            type="monotone"
            dataKey="total_on_order"
            yAxisId="left"
            stroke={trendColors[1]}
            strokeWidth={2}
            dot={CHART_DOT}
            connectNulls
            hide={!show("total_on_order")}
          />
          <Line
            type="monotone"
            dataKey="total_position"
            yAxisId="left"
            stroke="#a855f7"
            strokeWidth={2.5}
            strokeDasharray="8 3"
            dot={CHART_DOT}
            connectNulls
            hide={!show("total_position")}
          />
          <Line
            type="monotone"
            dataKey="monthly_sales"
            yAxisId="left"
            stroke={trendColors[3]}
            strokeWidth={2}
            dot={CHART_DOT}
            connectNulls
            hide={!show("monthly_sales")}
          />
          <Line
            type="monotone"
            dataKey="dos"
            yAxisId="right"
            stroke={trendColors[4]}
            strokeWidth={3}
            dot={CHART_DOT}
            connectNulls
            hide={!show("dos")}
          />
          <Line
            type="monotone"
            dataKey="avg_lead_time"
            yAxisId="right"
            stroke={trendColors[2]}
            strokeWidth={1.5}
            strokeDasharray="5 3"
            dot={CHART_DOT}
            connectNulls
            hide={!show("avg_lead_time")}
          />
          {/* Safety Stock: horizontal reference line in units (left axis) */}
          {hasSs && show("safety_stock") && (
            <ReferenceLine
              yAxisId="left"
              y={ss!}
              stroke="#8b5cf6"
              strokeDasharray="6 3"
              strokeWidth={1.5}
              label={{ value: `SS ${ss!.toFixed(0)}u`, position: "insideTopLeft", fontSize: 10, fill: "#8b5cf6" }}
            />
          )}
          {/* Reorder Point in units (left axis): SS + lead_time_demand */}
          {ropUnits != null && show("safety_stock") && (
            <ReferenceLine
              yAxisId="left"
              y={ropUnits}
              stroke="#f97316"
              strokeDasharray="4 2"
              strokeWidth={1.5}
              label={{ value: `ROP ${ropUnits.toFixed(0)}u`, position: "insideBottomLeft", fontSize: 10, fill: "#f97316" }}
            />
          )}
          {/* Safety Stock line (time-series, shows constant level) */}
          {hasSs && (
            <Line
              type="monotone"
              dataKey="safety_stock"
              yAxisId="left"
              stroke="#8b5cf6"
              strokeWidth={1.5}
              hide={!show("safety_stock")}
              strokeDasharray="6 3"
              dot={false}
              connectNulls
            />
          )}
          {/* Cycle Stock line: inventory above safety stock */}
          {hasSs && (
            <Line
              type="monotone"
              dataKey="cycle_stock"
              yAxisId="left"
              stroke="#06b6d4"
              strokeWidth={2}
              dot={CHART_DOT}
              connectNulls
              hide={!show("cycle_stock")}
            />
          )}
          {/* Reorder Point threshold in days (right axis): LT × 1.5 */}
          {ropThreshold != null && show("avg_lead_time") && (
            <ReferenceLine
              yAxisId="right"
              y={ropThreshold}
              stroke="#ef4444"
              strokeDasharray="4 3"
              strokeWidth={1.5}
              label={{ value: `ROP ${ropThreshold.toFixed(0)}d`, position: "insideBottomRight", offset: -4, fontSize: 10, fill: "#ef4444" }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
      <p className="text-xs text-muted-foreground text-center mt-1">
        Right axis: Days of Supply (red) · Avg Lead Time (dashed).{" "}
        {hasSs
          ? "Left axis reference lines: Safety Stock (purple dashed) · Reorder Point in units (orange). Cycle Stock = On Hand − Safety Stock."
          : "ROP line (right axis) = Lead Time × 1.5 safety buffer. Apply item+location filter to see Safety Stock and Cycle Stock lines."}
      </p>
      {/* Inventory parameters summary — shown when item+location filter returns SS/EOQ data */}
      {params && (params.safety_stock != null || params.eoq != null || params.order_policy != null) && (
        <div className="mt-3 rounded border border-muted bg-muted/30 px-3 py-2 text-xs">
          <p className="font-semibold text-foreground mb-1">Inventory Parameters</p>
          <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-muted-foreground">
            {params.order_policy != null && (
              <span><strong>Order Policy:</strong> {params.order_policy} ({params.policy_type?.replace(/_/g, " ")})</span>
            )}
            {params.service_level_target != null && (
              <span title="Target fill probability used to size safety stock">
                <strong>Service Level:</strong> {(params.service_level_target * 100).toFixed(0)}% (Z={params.z_score?.toFixed(2)})
              </span>
            )}
            {params.safety_stock != null && (
              <span title="Formula: Z × √(LT × σ²_demand + demand² × σ²_LT)">
                <strong>Safety Stock:</strong> {params.safety_stock.toLocaleString()} units
              </span>
            )}
            {params.reorder_point_units != null && (
              <span title="Formula: avg_daily_demand × lead_time_days + safety_stock">
                <strong>Reorder Point:</strong> {params.reorder_point_units.toLocaleString()} units
              </span>
            )}
            {params.eoq != null && (
              <span title="Formula: √(2 × annual_demand × ordering_cost / holding_cost_per_unit)">
                <strong>EOQ:</strong> {params.eoq.toLocaleString()} units
              </span>
            )}
            {params.eoq_cycle_stock != null && (
              <span title="EOQ ÷ 2 = average cycle stock between replenishments">
                <strong>Cycle Stock (EOQ/2):</strong> {params.eoq_cycle_stock.toLocaleString()} units
              </span>
            )}
            {params.order_frequency != null && (
              <span title="Annual demand ÷ EOQ = orders per year">
                <strong>Order Frequency:</strong> {params.order_frequency.toFixed(1)}×/year
              </span>
            )}
            {params.demand_cv != null && (
              <span title="Coefficient of Variation = std_dev / mean. Higher = more variable demand.">
                <strong>Demand CV:</strong> {params.demand_cv.toFixed(3)} ({params.demand_cv < 0.3 ? "stable" : params.demand_cv < 0.8 ? "moderate" : "volatile"})
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
});
