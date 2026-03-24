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
import { SKU_SALES_COLORS, skuModelColor } from "@/constants/colors";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
import type { SkuAnalysisPayload } from "@/types";
import type { ProductionForecastPayload } from "@/api/queries/production-forecast";

const PROD_FORECAST_COLOR = "#7c3aed"; // violet-700

// ---------------------------------------------------------------------------
// Hoisted constants for Recharts inline props
// ---------------------------------------------------------------------------
const DFU_CHART_MARGIN = { top: 8, right: 16, left: 18, bottom: 8 };
const ACTIVE_DOT_SM = { r: 4 };
const ACTIVE_DOT_MD = { r: 5 };

// ---------------------------------------------------------------------------
// Custom legend renderer — forecast model items are clickable for SHAP
// ---------------------------------------------------------------------------
interface LegendPayloadItem {
  value: string;
  color: string;
  dataKey?: string;
}

function renderCustomLegend(
  props: { payload?: LegendPayloadItem[] },
  modelNames: string[],
  selectedModel: string | null,
  onModelSelect?: (model: string | null) => void,
) {
  const { payload = [] } = props;
  return (
    <ul className="flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs">
      {payload.map((entry) => {
        const isModelEntry = modelNames.includes(entry.value);
        const isSelected = selectedModel === entry.value;
        return (
          <li
            key={entry.value}
            className={[
              "flex items-center gap-1.5",
              isModelEntry && onModelSelect
                ? "cursor-pointer select-none"
                : "",
              isModelEntry && !isSelected && selectedModel !== null
                ? "opacity-40"
                : "",
            ]
              .filter(Boolean)
              .join(" ")}
            onClick={() => {
              if (isModelEntry && onModelSelect) {
                onModelSelect(isSelected ? null : entry.value);
              }
            }}
          >
            <span
              className={[
                "inline-block h-2.5 w-2.5 shrink-0 rounded-full",
                isSelected ? "ring-2 ring-offset-1 ring-primary" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              style={{ backgroundColor: entry.color }}
            />
            <span className={isSelected ? "font-semibold text-primary" : ""}>
              {entry.value}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

// Opacity for de-emphasised lines when another is selected
const DESELECT_OPACITY = 0.25;

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
export interface OverlayChartPanelProps {
  skuData: SkuAnalysisPayload;
  skuFilteredSeries: Record<string, unknown>[];
  skuMonths: string[];
  skuTimeStart: string;
  setSkuTimeStart: (v: string) => void;
  skuTimeEnd: string;
  setSkuTimeEnd: (v: string) => void;
  skuDefaultStart: string;
  skuVisibleSeries: Set<string>;
  setSkuVisibleSeries: (updater: (prev: Set<string>) => Set<string>) => void;
  prodForecastData?: ProductionForecastPayload | null;
  /** Currently selected model for SHAP exploration (raw model_id, e.g. "lgbm_cluster") */
  selectedModel?: string | null;
  /** Called when user clicks a forecast line to select/deselect it */
  onModelSelect?: (model: string | null) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function OverlayChartPanel({
  skuData,
  skuFilteredSeries,
  skuMonths,
  skuTimeStart,
  setSkuTimeStart,
  skuTimeEnd,
  setSkuTimeEnd,
  skuDefaultStart,
  skuVisibleSeries,
  setSkuVisibleSeries,
  prodForecastData,
  selectedModel = null,
  onModelSelect,
}: OverlayChartPanelProps) {
  const { chartColors } = useChartColors();

  const hasProdForecast = (prodForecastData?.forecasts.length ?? 0) > 0;
  const promotedRun = prodForecastData?.promoted_run ?? null;
  const prodForecastLabel = hasProdForecast
    ? promotedRun
      ? `Production Forecast (${prodForecastData!.model_id} · run #${promotedRun.run_id}${promotedRun.accuracy_pct != null ? ` · ${promotedRun.accuracy_pct.toFixed(1)}%` : ""})`
      : `Production Forecast (${prodForecastData!.model_id})`
    : "Production Forecast";

  const allKeys = new Set([
    "tothist_dmd",
    "qty_shipped",
    "qty_ordered",
    ...(hasProdForecast ? ["production_forecast"] : []),
    ...skuData.models.map((m) => `forecast_${m}`),
  ]);
  const allSelected = [...allKeys].every((k) => skuVisibleSeries.has(k));

  const salesMeasures: { key: string; label: string; color: string }[] = [
    {
      key: "tothist_dmd",
      label: "Sale Qty (external)",
      color: SKU_SALES_COLORS.tothist_dmd,
    },
    {
      key: "qty_shipped",
      label: "Qty Shipped",
      color: SKU_SALES_COLORS.qty_shipped,
    },
    {
      key: "qty_ordered",
      label: "Qty Ordered",
      color: SKU_SALES_COLORS.qty_ordered,
    },
  ];

  function toggleSeries(key: string, checked: boolean | string) {
    setSkuVisibleSeries((prev) => {
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
          <span className="text-xs italic text-muted-foreground">
            Click a forecast model in the legend to explore SHAP
          </span>
          <button
            className="text-xs font-medium text-primary hover:underline"
            onClick={() =>
              setSkuVisibleSeries(() => (allSelected ? new Set() : allKeys))
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
                checked={skuVisibleSeries.has(key)}
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
          {skuData.models.map((model, idx) => {
            const seriesKey = `forecast_${model}`;
            return (
              <label
                key={model}
                className="flex items-center gap-2 text-xs font-medium"
              >
                <Checkbox
                  checked={skuVisibleSeries.has(seriesKey)}
                  onCheckedChange={(v) => toggleSeries(seriesKey, v)}
                />
                <span className="flex items-center gap-1">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: skuModelColor(model, idx) }}
                  />
                  {model}
                </span>
              </label>
            );
          })}
          {hasProdForecast && (
            <label className="flex items-center gap-2 text-xs font-medium">
              <Checkbox
                checked={skuVisibleSeries.has("production_forecast")}
                onCheckedChange={(v) => toggleSeries("production_forecast", v)}
              />
              <span className="flex items-center gap-1">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: PROD_FORECAST_COLOR }}
                />
                {prodForecastLabel}
              </span>
            </label>
          )}
        </div>
      </div>

      {/* Time range controls */}
      <div className="flex flex-wrap items-end gap-3">
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          From
          <select
            className="h-8 w-36 rounded-md border border-input bg-background px-2 text-sm"
            value={skuTimeStart || skuMonths[0] || ""}
            onChange={(e) => setSkuTimeStart(e.target.value)}
          >
            {skuMonths.map((m) => (
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
            value={skuTimeEnd || skuMonths[skuMonths.length - 1] || ""}
            onChange={(e) => setSkuTimeEnd(e.target.value)}
          >
            {skuMonths.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <button
          className="h-8 rounded-md border border-input bg-background px-3 text-xs font-medium text-muted-foreground hover:text-foreground"
          onClick={() => {
            setSkuTimeStart("");
            setSkuTimeEnd("");
          }}
        >
          Show All
        </button>
        <button
          className="h-8 rounded-md border border-input bg-background px-3 text-xs font-medium text-muted-foreground hover:text-foreground"
          onClick={() => {
            setSkuTimeStart(skuDefaultStart);
            setSkuTimeEnd("");
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
            {skuData.scope_count != null && (
              <span className="ml-2 rounded bg-muted px-2 py-0.5 text-xs font-normal text-muted-foreground">
                {skuData.mode === "item_at_all_locations"
                  ? `${skuData.scope_count} locations aggregated`
                  : `${skuData.scope_count} items aggregated`}
              </span>
            )}
            {selectedModel && (
              <span className="ml-auto rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                SHAP: {selectedModel}
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="h-[380px] pt-2">
          <div className="h-full overflow-x-scroll overflow-y-hidden pb-2 [scrollbar-gutter:stable]">
            <div
              className="h-full"
              style={{
                minWidth: `${Math.max(1200, skuFilteredSeries.length * 100)}px`,
              }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={skuFilteredSeries} margin={DFU_CHART_MARGIN}>
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
                  <Legend
                    content={(props) =>
                      renderCustomLegend(
                        props as { payload?: LegendPayloadItem[] },
                        skuData.models,
                        selectedModel,
                        onModelSelect,
                      )
                    }
                  />
                  {skuVisibleSeries.has("tothist_dmd") ? (
                    <Line
                      type="monotone"
                      dataKey="tothist_dmd"
                      yAxisId="left"
                      name="Sale Qty (external)"
                      stroke={SKU_SALES_COLORS.tothist_dmd}
                      strokeWidth={2.5}
                      dot={false}
                      activeDot={ACTIVE_DOT_MD}
                    />
                  ) : null}
                  {skuVisibleSeries.has("qty_shipped") ? (
                    <Line
                      type="monotone"
                      dataKey="qty_shipped"
                      yAxisId="left"
                      name="Qty Shipped"
                      stroke={SKU_SALES_COLORS.qty_shipped}
                      strokeWidth={2}
                      dot={false}
                      activeDot={ACTIVE_DOT_SM}
                    />
                  ) : null}
                  {skuVisibleSeries.has("qty_ordered") ? (
                    <Line
                      type="monotone"
                      dataKey="qty_ordered"
                      yAxisId="left"
                      name="Qty Ordered"
                      stroke={SKU_SALES_COLORS.qty_ordered}
                      strokeWidth={2}
                      dot={false}
                      activeDot={ACTIVE_DOT_SM}
                    />
                  ) : null}
                  {skuData.models
                    .filter((m) => skuVisibleSeries.has(`forecast_${m}`))
                    .map((model, idx) => {
                      const isSelected = selectedModel === model;
                      const isOtherSelected = selectedModel !== null && selectedModel !== model;
                      return (
                        <Line
                          key={model}
                          type="monotone"
                          dataKey={`forecast_${model}`}
                          yAxisId="left"
                          name={model}
                          stroke={skuModelColor(model, idx)}
                          strokeWidth={isSelected ? 3 : model === "champion" ? 2.5 : 1.5}
                          strokeDasharray={
                            model === "champion" ? undefined : "5 3"
                          }
                          dot={false}
                          style={{ opacity: isOtherSelected ? DESELECT_OPACITY : 1 }}
                          activeDot={{ r: isSelected ? 7 : 4 }}
                        />
                      );
                    })}
                  {hasProdForecast && skuVisibleSeries.has("production_forecast") ? (
                    <Line
                      type="monotone"
                      dataKey="production_forecast"
                      yAxisId="left"
                      name={prodForecastLabel}
                      stroke={PROD_FORECAST_COLOR}
                      strokeWidth={2.5}
                      strokeDasharray="6 3"
                      dot={false}
                      activeDot={ACTIVE_DOT_MD}
                    />
                  ) : null}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </CardContent>
      </Card>
    </>
  );
}
