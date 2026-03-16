import { useMemo, useState, useCallback } from "react";
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

import { useChartColors } from "@/hooks/useChartColors";
import { DFU_SALES_COLORS, dfuModelColor } from "@/constants/colors";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
import type {
  DfuAnalysisPayload,
  InventoryTrendPoint,
  InventoryTrendParams,
} from "@/types";
import type { ProductionForecastPayload } from "@/api/queries/production-forecast";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const PROD_FORECAST_COLOR = "#7c3aed";
const CHART_MARGIN = { top: 8, right: 40, left: 18, bottom: 8 };
const DESELECT_OPACITY = 0.25;

const SUPPLY_COLORS: Record<string, string> = {
  total_on_hand: "#2563EB",
  total_on_order: "#0D9488",
  total_position: "#a855f7",
  inv_monthly_sales: "#0891B2",
  dos: "#DC2626",
  avg_lead_time: "#D97706",
  safety_stock: "#8b5cf6",
  cycle_stock: "#06b6d4",
};

interface SupplySeriesDef {
  key: string;
  label: string;
  color: string;
  axis: "left" | "right";
  defaultVisible: boolean;
  dashArray?: string;
  strokeWidth?: number;
}

const SUPPLY_SERIES_DEFS: SupplySeriesDef[] = [
  { key: "total_on_hand", label: "On Hand", color: SUPPLY_COLORS.total_on_hand, axis: "left", defaultVisible: true },
  { key: "total_on_order", label: "On Order", color: SUPPLY_COLORS.total_on_order, axis: "left", defaultVisible: false },
  { key: "total_position", label: "Position", color: SUPPLY_COLORS.total_position, axis: "left", defaultVisible: false, dashArray: "8 3" },
  { key: "inv_monthly_sales", label: "Inv Sales", color: SUPPLY_COLORS.inv_monthly_sales, axis: "left", defaultVisible: false },
  { key: "dos", label: "DOS", color: SUPPLY_COLORS.dos, axis: "right", defaultVisible: true, strokeWidth: 2.5 },
  { key: "avg_lead_time", label: "Lead Time", color: SUPPLY_COLORS.avg_lead_time, axis: "right", defaultVisible: false, dashArray: "5 3" },
  { key: "safety_stock", label: "Safety Stock", color: SUPPLY_COLORS.safety_stock, axis: "left", defaultVisible: false, dashArray: "6 3" },
  { key: "cycle_stock", label: "Cycle Stock", color: SUPPLY_COLORS.cycle_stock, axis: "left", defaultVisible: false },
];

const DEFAULT_HIDDEN_SUPPLY = new Set(
  SUPPLY_SERIES_DEFS.filter((s) => !s.defaultVisible).map((s) => s.key),
);

const TOOLTIP_LABELS: Record<string, string> = {
  tothist_dmd: "Sale Qty (external)",
  qty_shipped: "Qty Shipped",
  qty_ordered: "Qty Ordered",
  production_forecast: "Production Forecast",
  total_on_hand: "On Hand",
  total_on_order: "On Order",
  total_position: "Total Position",
  inv_monthly_sales: "Inv Monthly Sales",
  dos: "Days of Supply",
  avg_lead_time: "Avg Lead Time",
  safety_stock: "Safety Stock",
  cycle_stock: "Cycle Stock",
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
export interface UnifiedChartPanelProps {
  // Demand data
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
  prodForecastData?: ProductionForecastPayload | null;
  selectedModel?: string | null;
  onModelSelect?: (model: string | null) => void;
  // Supply data (optional)
  trendData?: InventoryTrendPoint[];
  trendParams?: InventoryTrendParams;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function UnifiedChartPanel({
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
  prodForecastData,
  selectedModel = null,
  onModelSelect,
  trendData = [],
  trendParams,
}: UnifiedChartPanelProps) {
  const { chartColors } = useChartColors();
  const [hiddenSupply, setHiddenSupply] = useState<Set<string>>(DEFAULT_HIDDEN_SUPPLY);

  const hasProdForecast = (prodForecastData?.forecasts.length ?? 0) > 0;
  const prodForecastLabel = hasProdForecast
    ? `Prod (${prodForecastData!.model_id})`
    : "Prod Forecast";
  const hasSupplyData = trendData.length > 0;
  const ss = trendParams?.safety_stock ?? null;
  const ropUnits = trendParams?.reorder_point_units ?? null;
  const hasSs = ss != null;

  // Merge demand + supply data on month key
  const mergedData = useMemo(() => {
    if (!hasSupplyData) return dfuFilteredSeries;
    const supplyMap = new Map<string, Record<string, number | null>>();
    for (const pt of trendData) {
      const key = String(pt.month).slice(0, 7);
      supplyMap.set(key, {
        total_on_hand: pt.total_on_hand,
        total_on_order: pt.total_on_order,
        total_position: pt.total_on_hand + pt.total_on_order,
        inv_monthly_sales: pt.monthly_sales,
        dos: pt.dos,
        avg_lead_time: pt.avg_lead_time,
        ...(pt.safety_stock != null
          ? {
              safety_stock: pt.safety_stock,
              cycle_stock: Math.max(0, pt.total_on_hand - pt.safety_stock),
            }
          : {}),
      });
    }
    return dfuFilteredSeries.map((pt) => {
      const key = String(pt.month).slice(0, 7);
      const supply = supplyMap.get(key);
      return supply ? { ...pt, ...supply } : pt;
    });
  }, [dfuFilteredSeries, trendData, hasSupplyData]);

  // Demand toggle helper
  function toggleDemandSeries(key: string, checked: boolean) {
    setDfuVisibleSeries((prev) => {
      const next = new Set(prev);
      if (checked) next.add(key);
      else next.delete(key);
      return next;
    });
  }

  // Supply toggle
  const toggleSupply = useCallback((key: string) => {
    setHiddenSupply((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const showSupply = useCallback((key: string) => !hiddenSupply.has(key), [hiddenSupply]);

  // Right axis needed?
  const hasRightAxis =
    hasSupplyData &&
    SUPPLY_SERIES_DEFS.some((s) => s.axis === "right" && !hiddenSupply.has(s.key));

  // Available supply series (conditionally include SS-dependent ones)
  const availableSupply = useMemo(() => {
    return SUPPLY_SERIES_DEFS.filter((s) => {
      if (s.key === "safety_stock" || s.key === "cycle_stock") return hasSs;
      return true;
    });
  }, [hasSs]);

  // Sales measures
  const salesMeasures = [
    { key: "tothist_dmd", label: "Sale Qty", color: DFU_SALES_COLORS.tothist_dmd },
    { key: "qty_shipped", label: "Shipped", color: DFU_SALES_COLORS.qty_shipped },
    { key: "qty_ordered", label: "Ordered", color: DFU_SALES_COLORS.qty_ordered },
  ];

  return (
    <div className="space-y-3">
      {/* ---- Toggle pills ---- */}
      <div className="space-y-1.5">
        {/* Demand pills */}
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="w-16 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
            Demand
          </span>
          {salesMeasures.map(({ key, label, color }) => (
            <TogglePill
              key={key}
              label={label}
              color={color}
              active={dfuVisibleSeries.has(key)}
              onClick={() => toggleDemandSeries(key, !dfuVisibleSeries.has(key))}
            />
          ))}
          {dfuData.models.map((model, idx) => {
            const key = `forecast_${model}`;
            const color = dfuModelColor(model, idx);
            const isVisible = dfuVisibleSeries.has(key);
            const isShapSelected = selectedModel === model;
            return (
              <span key={key} className="inline-flex items-center gap-0.5">
                <TogglePill
                  label={model}
                  color={color}
                  active={isVisible}
                  onClick={() => toggleDemandSeries(key, !isVisible)}
                  dashed={model !== "champion"}
                  ring={isShapSelected}
                />
                {isVisible && onModelSelect && (
                  <button
                    onClick={() => onModelSelect(isShapSelected ? null : model)}
                    className={`rounded px-1 py-0.5 text-[9px] font-semibold leading-none transition-colors ${
                      isShapSelected
                        ? "bg-primary text-primary-foreground"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    }`}
                    title="Select for SHAP analysis"
                  >
                    S
                  </button>
                )}
              </span>
            );
          })}
          {hasProdForecast && (
            <TogglePill
              label={prodForecastLabel}
              color={PROD_FORECAST_COLOR}
              active={dfuVisibleSeries.has("production_forecast")}
              onClick={() =>
                toggleDemandSeries("production_forecast", !dfuVisibleSeries.has("production_forecast"))
              }
              dashed
            />
          )}
        </div>

        {/* Supply pills */}
        {hasSupplyData && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="w-16 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
              Supply
            </span>
            {availableSupply.map((s) => (
              <TogglePill
                key={s.key}
                label={s.label}
                color={s.color}
                active={showSupply(s.key)}
                onClick={() => toggleSupply(s.key)}
                suffix={s.axis === "right" ? "d" : "u"}
              />
            ))}
          </div>
        )}
      </div>

      {/* ---- Time range ---- */}
      <div className="flex flex-wrap items-center gap-2.5 text-xs">
        <label className="flex items-center gap-1.5">
          <span className="font-semibold uppercase tracking-wider text-muted-foreground">From</span>
          <select
            className="h-7 w-28 rounded border border-input bg-background px-2 text-xs"
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
        <label className="flex items-center gap-1.5">
          <span className="font-semibold uppercase tracking-wider text-muted-foreground">To</span>
          <select
            className="h-7 w-28 rounded border border-input bg-background px-2 text-xs"
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
          className="h-7 rounded border border-input bg-background px-2.5 text-xs text-muted-foreground hover:text-foreground"
          onClick={() => {
            setDfuTimeStart("");
            setDfuTimeEnd("");
          }}
        >
          All
        </button>
        <button
          className="h-7 rounded border border-input bg-background px-2.5 text-xs text-muted-foreground hover:text-foreground"
          onClick={() => {
            setDfuTimeStart(dfuDefaultStart);
            setDfuTimeEnd("");
          }}
        >
          Default
        </button>
        {selectedModel && (
          <span className="ml-auto rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
            SHAP: {selectedModel}
          </span>
        )}
        {dfuData.scope_count != null && (
          <span className="ml-auto rounded bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
            {dfuData.mode === "item_at_all_locations"
              ? `${dfuData.scope_count} locations`
              : `${dfuData.scope_count} items`}
          </span>
        )}
      </div>

      {/* ---- Chart ---- */}
      <div className="h-[400px] overflow-x-auto overflow-y-hidden pb-2 [scrollbar-gutter:stable]">
        <div
          className="h-full"
          style={{ minWidth: `${Math.max(1200, mergedData.length * 100)}px` }}
        >
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mergedData} margin={CHART_MARGIN}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
              <XAxis dataKey="month" tick={{ fill: chartColors.axis, fontSize: 11 }} />
              <YAxis
                yAxisId="left"
                width={78}
                tickFormatter={formatCompactNumber}
                tick={{ fill: chartColors.axis, fontSize: 11 }}
                label={{ value: "Units", angle: -90, position: "insideLeft", fontSize: 10, offset: 10 }}
              />
              {hasRightAxis && (
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  tick={{ fill: chartColors.axis, fontSize: 11 }}
                  tickFormatter={(v: number) => `${Number(v).toFixed(0)}d`}
                  label={{ value: "Days", angle: 90, position: "insideRight", fontSize: 10, offset: 10 }}
                />
              )}
              <Tooltip
                contentStyle={{
                  backgroundColor: chartColors.tooltip_bg,
                  borderColor: chartColors.tooltip_border,
                }}
                formatter={(value: number, name: string) => {
                  const label = TOOLTIP_LABELS[name] ?? name;
                  if (name === "dos" || name === "avg_lead_time")
                    return [`${Number(value).toFixed(1)} days`, label];
                  return [
                    formatNumber(Number.isFinite(Number(value)) ? Number(value) : null),
                    label,
                  ];
                }}
              />

              {/* ---- Demand lines ---- */}
              {dfuVisibleSeries.has("tothist_dmd") && (
                <Line
                  type="monotone"
                  dataKey="tothist_dmd"
                  yAxisId="left"
                  name="tothist_dmd"
                  stroke={DFU_SALES_COLORS.tothist_dmd}
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              )}
              {dfuVisibleSeries.has("qty_shipped") && (
                <Line
                  type="monotone"
                  dataKey="qty_shipped"
                  yAxisId="left"
                  name="qty_shipped"
                  stroke={DFU_SALES_COLORS.qty_shipped}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              )}
              {dfuVisibleSeries.has("qty_ordered") && (
                <Line
                  type="monotone"
                  dataKey="qty_ordered"
                  yAxisId="left"
                  name="qty_ordered"
                  stroke={DFU_SALES_COLORS.qty_ordered}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              )}
              {dfuData.models
                .filter((m) => dfuVisibleSeries.has(`forecast_${m}`))
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
                      stroke={dfuModelColor(model, idx)}
                      strokeWidth={isSelected ? 3 : model === "champion" ? 2.5 : 1.5}
                      strokeDasharray={model === "champion" ? undefined : "5 3"}
                      dot={false}
                      style={{ opacity: isOtherSelected ? DESELECT_OPACITY : 1 }}
                      activeDot={{ r: isSelected ? 7 : 4 }}
                    />
                  );
                })}
              {hasProdForecast && dfuVisibleSeries.has("production_forecast") && (
                <Line
                  type="monotone"
                  dataKey="production_forecast"
                  yAxisId="left"
                  name="production_forecast"
                  stroke={PROD_FORECAST_COLOR}
                  strokeWidth={2.5}
                  strokeDasharray="6 3"
                  dot={false}
                  activeDot={{ r: 5 }}
                />
              )}

              {/* ---- Supply lines ---- */}
              {hasSupplyData &&
                availableSupply
                  .filter((s) => !hiddenSupply.has(s.key))
                  .map((s) => (
                    <Line
                      key={s.key}
                      type="monotone"
                      dataKey={s.key}
                      yAxisId={s.axis}
                      name={s.key}
                      stroke={s.color}
                      strokeWidth={s.strokeWidth ?? 2}
                      strokeDasharray={s.dashArray}
                      dot={false}
                      connectNulls
                      activeDot={{ r: 3 }}
                    />
                  ))}

              {/* ---- Reference lines ---- */}
              {hasSs && showSupply("safety_stock") && (
                <ReferenceLine
                  yAxisId="left"
                  y={ss!}
                  stroke="#8b5cf6"
                  strokeDasharray="6 3"
                  strokeWidth={1.5}
                  label={{
                    value: `SS ${ss!.toFixed(0)}u`,
                    position: "insideTopLeft",
                    fontSize: 10,
                    fill: "#8b5cf6",
                  }}
                />
              )}
              {ropUnits != null && showSupply("safety_stock") && (
                <ReferenceLine
                  yAxisId="left"
                  y={ropUnits}
                  stroke="#f97316"
                  strokeDasharray="4 2"
                  strokeWidth={1.5}
                  label={{
                    value: `ROP ${ropUnits.toFixed(0)}u`,
                    position: "insideBottomLeft",
                    fontSize: 10,
                    fill: "#f97316",
                  }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ---- Inventory parameters summary ---- */}
      {trendParams &&
        (trendParams.safety_stock != null ||
          trendParams.eoq != null ||
          trendParams.order_policy != null) && (
          <div className="rounded border border-muted bg-muted/30 px-3 py-2 text-xs">
            <p className="mb-1 font-semibold text-foreground">Inventory Parameters</p>
            <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-muted-foreground sm:grid-cols-4">
              {trendParams.order_policy != null && (
                <span>
                  <strong>Policy:</strong> {trendParams.order_policy} (
                  {trendParams.policy_type?.replace(/_/g, " ")})
                </span>
              )}
              {trendParams.service_level_target != null && (
                <span>
                  <strong>SL:</strong> {(trendParams.service_level_target * 100).toFixed(0)}% (Z=
                  {trendParams.z_score?.toFixed(2)})
                </span>
              )}
              {trendParams.safety_stock != null && (
                <span>
                  <strong>SS:</strong> {trendParams.safety_stock.toLocaleString()}u
                </span>
              )}
              {trendParams.reorder_point_units != null && (
                <span>
                  <strong>ROP:</strong> {trendParams.reorder_point_units.toLocaleString()}u
                </span>
              )}
              {trendParams.eoq != null && (
                <span>
                  <strong>EOQ:</strong> {trendParams.eoq.toLocaleString()}u
                </span>
              )}
              {trendParams.demand_cv != null && (
                <span>
                  <strong>CV:</strong> {trendParams.demand_cv.toFixed(3)} (
                  {trendParams.demand_cv < 0.3
                    ? "stable"
                    : trendParams.demand_cv < 0.8
                      ? "moderate"
                      : "volatile"}
                  )
                </span>
              )}
            </div>
          </div>
        )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// TogglePill — reusable pill button for series visibility
// ---------------------------------------------------------------------------
function TogglePill({
  label,
  color,
  active,
  onClick,
  dashed,
  ring,
  suffix,
}: {
  label: string;
  color: string;
  active: boolean;
  onClick: () => void;
  dashed?: boolean;
  ring?: boolean;
  suffix?: string;
}) {
  return (
    <button
      onClick={onClick}
      className={[
        "flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] transition-opacity",
        active ? "opacity-100" : "opacity-30",
        ring ? "ring-2 ring-primary ring-offset-1" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      style={{ borderColor: color, color: active ? color : undefined }}
    >
      <span
        className="inline-block h-0.5 w-3"
        style={
          dashed
            ? { borderTop: `2px dashed ${color}`, backgroundColor: "transparent" }
            : { backgroundColor: color }
        }
      />
      {label}
      {suffix && (
        <span className="text-[9px] text-muted-foreground">({suffix})</span>
      )}
    </button>
  );
}
