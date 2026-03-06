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
import { ChartColumn } from "lucide-react";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";

import { useChartColors } from "@/hooks/useChartColors";
import { DFU_SALES_COLORS, dfuModelColor } from "@/constants/colors";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
import type { DfuAnalysisPayload } from "@/types";

// ---------------------------------------------------------------------------
// Hoisted constants for Recharts inline props
// ---------------------------------------------------------------------------
const DFU_CHART_MARGIN = { top: 8, right: 16, left: 18, bottom: 8 };
const ACTIVE_DOT_SM = { r: 4 };
const ACTIVE_DOT_MD = { r: 5 };

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
export interface OverlayChartPanelProps {
  dfuData: DfuAnalysisPayload;
  dfuFilteredSeries: Record<string, unknown>[];
  dfuMonths: string[];
  dfuTimeStart: string;
  setDfuTimeStart: (v: string) => void;
  dfuTimeEnd: string;
  setDfuTimeEnd: (v: string) => void;
  dfuDefaultStart: string;
  dfuVisibleSeries: Set<string>;
  setDfuVisibleSeries: (updater: (prev: Set<string>) => Set<string>) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function OverlayChartPanel({
  dfuData,
  dfuFilteredSeries,
  dfuMonths,
  dfuTimeStart,
  setDfuTimeStart,
  dfuTimeEnd,
  setDfuTimeEnd,
  dfuDefaultStart,
  dfuVisibleSeries,
  setDfuVisibleSeries,
}: OverlayChartPanelProps) {
  const { chartColors } = useChartColors();

  const allKeys = new Set([
    "tothist_dmd",
    "qty_shipped",
    "qty_ordered",
    ...dfuData.models.map((m) => `forecast_${m}`),
  ]);
  const allSelected = [...allKeys].every((k) => dfuVisibleSeries.has(k));

  const salesMeasures: { key: string; label: string; color: string }[] = [
    {
      key: "tothist_dmd",
      label: "Sale Qty (external)",
      color: DFU_SALES_COLORS.tothist_dmd,
    },
    {
      key: "qty_shipped",
      label: "Qty Shipped",
      color: DFU_SALES_COLORS.qty_shipped,
    },
    {
      key: "qty_ordered",
      label: "Qty Ordered",
      color: DFU_SALES_COLORS.qty_ordered,
    },
  ];

  function toggleSeries(key: string, checked: boolean | string) {
    setDfuVisibleSeries((prev) => {
      const next = new Set(prev);
      if (checked) next.add(key);
      else next.delete(key);
      return next;
    });
  }

  return (
    <>
      {/* Measure toggles */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Visible Measures
          </span>
          <button
            className="text-xs font-medium text-primary hover:underline"
            onClick={() =>
              setDfuVisibleSeries(() => (allSelected ? new Set() : allKeys))
            }
          >
            {allSelected ? "Deselect All" : "Select All"}
          </button>
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 rounded-md border border-input bg-background p-2">
          {salesMeasures.map(({ key, label, color }) => (
            <label
              key={key}
              className="flex items-center gap-2 text-xs font-medium"
            >
              <Checkbox
                checked={dfuVisibleSeries.has(key)}
                onCheckedChange={(v) => toggleSeries(key, v)}
              />
              <span className="flex items-center gap-1">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: color }}
                />
                {label}
              </span>
            </label>
          ))}
          {dfuData.models.map((model, idx) => {
            const seriesKey = `forecast_${model}`;
            return (
              <label
                key={model}
                className="flex items-center gap-2 text-xs font-medium"
              >
                <Checkbox
                  checked={dfuVisibleSeries.has(seriesKey)}
                  onCheckedChange={(v) => toggleSeries(seriesKey, v)}
                />
                <span className="flex items-center gap-1">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: dfuModelColor(model, idx) }}
                  />
                  {model}
                </span>
              </label>
            );
          })}
        </div>
      </div>

      {/* Time range controls */}
      <div className="flex flex-wrap items-end gap-3">
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          From
          <select
            className="h-8 w-36 rounded-md border border-input bg-background px-2 text-sm"
            value={dfuTimeStart || dfuMonths[0] || ""}
            onChange={(e) => setDfuTimeStart(e.target.value)}
          >
            {dfuMonths.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          To
          <select
            className="h-8 w-36 rounded-md border border-input bg-background px-2 text-sm"
            value={dfuTimeEnd || dfuMonths[dfuMonths.length - 1] || ""}
            onChange={(e) => setDfuTimeEnd(e.target.value)}
          >
            {dfuMonths.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <button
          className="h-8 rounded-md border border-input bg-background px-3 text-xs font-medium text-muted-foreground hover:text-foreground"
          onClick={() => {
            setDfuTimeStart("");
            setDfuTimeEnd("");
          }}
        >
          Show All
        </button>
        <button
          className="h-8 rounded-md border border-input bg-background px-3 text-xs font-medium text-muted-foreground hover:text-foreground"
          onClick={() => {
            setDfuTimeStart(dfuDefaultStart);
            setDfuTimeEnd("");
          }}
        >
          Default
        </button>
      </div>

      {/* Chart */}
      <Card className="min-w-0 border-muted shadow-none">
        <CardHeader className="pb-0">
          <CardTitle className="flex items-center gap-2 text-sm">
            <ChartColumn className="h-4 w-4" /> Sales vs Forecast Overlay
            {dfuData.scope_count != null && (
              <span className="ml-2 rounded bg-muted px-2 py-0.5 text-xs font-normal text-muted-foreground">
                {dfuData.mode === "item_at_all_locations"
                  ? `${dfuData.scope_count} locations aggregated`
                  : `${dfuData.scope_count} items aggregated`}
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="h-[380px] pt-2">
          <div className="h-full overflow-x-scroll overflow-y-hidden pb-2 [scrollbar-gutter:stable]">
            <div
              className="h-full"
              style={{
                minWidth: `${Math.max(1200, dfuFilteredSeries.length * 100)}px`,
              }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dfuFilteredSeries} margin={DFU_CHART_MARGIN}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={chartColors.grid}
                  />
                  <XAxis
                    dataKey="month"
                    tick={{ fill: chartColors.axis }}
                  />
                  <YAxis
                    yAxisId="left"
                    width={84}
                    tickFormatter={formatCompactNumber}
                    tickMargin={10}
                    tick={{ fill: chartColors.axis }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartColors.tooltip_bg,
                      borderColor: chartColors.tooltip_border,
                    }}
                    formatter={(value: number, name: string) => [
                      formatNumber(
                        Number.isFinite(Number(value)) ? Number(value) : null,
                      ),
                      String(name),
                    ]}
                  />
                  <Legend />
                  {dfuVisibleSeries.has("tothist_dmd") ? (
                    <Line
                      type="monotone"
                      dataKey="tothist_dmd"
                      yAxisId="left"
                      name="Sale Qty (external)"
                      stroke={DFU_SALES_COLORS.tothist_dmd}
                      strokeWidth={2.5}
                      dot={false}
                      activeDot={ACTIVE_DOT_MD}
                    />
                  ) : null}
                  {dfuVisibleSeries.has("qty_shipped") ? (
                    <Line
                      type="monotone"
                      dataKey="qty_shipped"
                      yAxisId="left"
                      name="Qty Shipped"
                      stroke={DFU_SALES_COLORS.qty_shipped}
                      strokeWidth={2}
                      dot={false}
                      activeDot={ACTIVE_DOT_SM}
                    />
                  ) : null}
                  {dfuVisibleSeries.has("qty_ordered") ? (
                    <Line
                      type="monotone"
                      dataKey="qty_ordered"
                      yAxisId="left"
                      name="Qty Ordered"
                      stroke={DFU_SALES_COLORS.qty_ordered}
                      strokeWidth={2}
                      dot={false}
                      activeDot={ACTIVE_DOT_SM}
                    />
                  ) : null}
                  {dfuData.models
                    .filter((m) => dfuVisibleSeries.has(`forecast_${m}`))
                    .map((model, idx) => (
                      <Line
                        key={model}
                        type="monotone"
                        dataKey={`forecast_${model}`}
                        yAxisId="left"
                        name={model}
                        stroke={dfuModelColor(model, idx)}
                        strokeWidth={model === "champion" ? 2.5 : 1.5}
                        strokeDasharray={
                          model === "champion" ? undefined : "5 3"
                        }
                        dot={false}
                        activeDot={ACTIVE_DOT_SM}
                      />
                    ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </CardContent>
      </Card>
    </>
  );
}
