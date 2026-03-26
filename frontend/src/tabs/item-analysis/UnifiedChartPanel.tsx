import { memo, useMemo, useState, useCallback } from "react";
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
import { SKU_SALES_COLORS, skuModelColor } from "@/constants/colors";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
import type {
  SkuAnalysisPayload,
  InventoryTrendPoint,
  InventoryTrendParams,
} from "@/types";
import type { ProductionForecastPayload } from "@/api/queries/production-forecast";
import type { DQCorrection } from "@/api/queries/platform";

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

const DQ_ORIG_COLOR = "#DC2626"; // red for original (pre-DQ) values

// Map DB column_name + table → chart dataKey for the original series
const DQ_COLUMN_MAP: Record<string, { dataKey: string; origKey: string; label: string }> = {
  "fact_sales_monthly:qty": { dataKey: "sales_qty", origKey: "sales_qty_orig", label: "Sale Qty (original)" },
  "fact_sales_monthly:qty_shipped": { dataKey: "qty_shipped", origKey: "qty_shipped_orig", label: "Shipped (original)" },
  "fact_inventory_snapshot:qty_on_hand": { dataKey: "total_on_hand", origKey: "total_on_hand_orig", label: "On Hand (original)" },
};

const TOOLTIP_LABELS: Record<string, string> = {
  tothist_dmd: "Sale Qty (external)",
  sales_qty: "Sale Qty",
  sales_qty_orig: "Sale Qty (original)",
  qty_shipped: "Qty Shipped",
  qty_shipped_orig: "Shipped (original)",
  qty_ordered: "Qty Ordered",
  production_forecast: "Production Forecast",
  total_on_hand: "On Hand",
  total_on_hand_orig: "On Hand (original)",
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
  selectedModel?: string | null;
  onModelSelect?: (model: string | null) => void;
  // Supply data (optional)
  trendData?: InventoryTrendPoint[];
  trendParams?: InventoryTrendParams;
  // DQ corrections overlay (optional)
  corrections?: DQCorrection[];
  showCorrections?: boolean;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export const UnifiedChartPanel = memo(function UnifiedChartPanel({
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
  trendData = [],
  trendParams,
  corrections = [],
  showCorrections = false,
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

  // Build corrections overlay map: month → { origKey: old_value }
  const correctionOverlay = useMemo(() => {
    if (!showCorrections || corrections.length === 0) return null;
    const map = new Map<string, Record<string, number>>();
    for (const c of corrections) {
      if (c.old_value == null || c.period == null) continue;
      // Skip corrections where old == new (no actual change)
      if (c.new_value != null && Math.abs(c.old_value - c.new_value) < 0.01) continue;
      const mapKey = `${c.table_name}:${c.column_name}`;
      const def = DQ_COLUMN_MAP[mapKey];
      if (!def) continue;
      const month = c.period.slice(0, 7);
      const existing = map.get(month) ?? {};
      existing[def.origKey] = c.old_value;
      map.set(month, existing);
    }
    return map;
  }, [corrections, showCorrections]);

  // Which correction series are active?
  const activeCorrectionSeries = useMemo(() => {
    if (!correctionOverlay) return [];
    const keys = new Set<string>();
    for (const vals of correctionOverlay.values()) {
      for (const k of Object.keys(vals)) keys.add(k);
    }
    return Array.from(keys);
  }, [correctionOverlay]);

  // Merge demand + supply + corrections data on month key
  const mergedData = useMemo(() => {
    let data = skuFilteredSeries;
    if (hasSupplyData) {
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
      data = data.map((pt) => {
        const key = String(pt.month).slice(0, 7);
        const supply = supplyMap.get(key);
        return supply ? { ...pt, ...supply } : pt;
      });
    }
    // Overlay DQ corrections original values
    if (correctionOverlay && correctionOverlay.size > 0) {
      data = data.map((pt) => {
        const key = String(pt.month).slice(0, 7);
        const orig = correctionOverlay.get(key);
        return orig ? { ...pt, ...orig } : pt;
      });
    }
    return data;
  }, [skuFilteredSeries, trendData, hasSupplyData, correctionOverlay]);

  // Demand toggle helper
  function toggleDemandSeries(key: string, checked: boolean) {
    setSkuVisibleSeries((prev) => {
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
    { key: "tothist_dmd", label: "Sale Qty (ext)", color: SKU_SALES_COLORS.tothist_dmd },
    { key: "sales_qty", label: "Sale Qty", color: SKU_SALES_COLORS.sales_qty },
    { key: "qty_shipped", label: "Shipped", color: SKU_SALES_COLORS.qty_shipped },
    { key: "qty_ordered", label: "Ordered", color: SKU_SALES_COLORS.qty_ordered },
  ];

  // All demand keys for select/deselect all
  const allDemandKeys = useMemo(() => {
    const keys = salesMeasures.map((s) => s.key);
    keys.push(...skuData.models.map((m) => `forecast_${m}`));
    if (hasProdForecast) keys.push("production_forecast");
    return keys;
  }, [skuData.models, hasProdForecast]); // eslint-disable-line react-hooks/exhaustive-deps

  const allDemandOn = allDemandKeys.every((k) => skuVisibleSeries.has(k));

  const toggleAllDemand = useCallback(() => {
    setSkuVisibleSeries((prev) => {
      const next = new Set(prev);
      if (allDemandOn) {
        for (const k of allDemandKeys) next.delete(k);
      } else {
        for (const k of allDemandKeys) next.add(k);
      }
      return next;
    });
  }, [allDemandKeys, allDemandOn, setSkuVisibleSeries]);

  // Supply select/deselect all
  const allSupplyOn = availableSupply.every((s) => !hiddenSupply.has(s.key));

  const toggleAllSupply = useCallback(() => {
    setHiddenSupply(() => {
      if (allSupplyOn) {
        return new Set(availableSupply.map((s) => s.key));
      }
      return new Set();
    });
  }, [allSupplyOn, availableSupply]);

  return (
    <div className="space-y-3">
      {/* ---- Toggle pills ---- */}
      <div className="space-y-1.5">
        {/* Demand pills */}
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            onClick={toggleAllDemand}
            className="w-16 text-[10px] font-bold uppercase tracking-widest text-muted-foreground hover:text-foreground transition-colors text-left"
            title={allDemandOn ? "Deselect all demand series" : "Select all demand series"}
          >
            {allDemandOn ? "Demand \u2212" : "Demand +"}
          </button>
          {salesMeasures.map(({ key, label, color }) => (
            <TogglePill
              key={key}
              label={label}
              color={color}
              active={skuVisibleSeries.has(key)}
              onClick={() => toggleDemandSeries(key, !skuVisibleSeries.has(key))}
            />
          ))}
          {skuData.models.map((model, idx) => {
            const key = `forecast_${model}`;
            const color = skuModelColor(model, idx);
            const isVisible = skuVisibleSeries.has(key);
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
              active={skuVisibleSeries.has("production_forecast")}
              onClick={() =>
                toggleDemandSeries("production_forecast", !skuVisibleSeries.has("production_forecast"))
              }
              dashed
            />
          )}
        </div>

        {/* DQ corrections indicator */}
        {showCorrections && activeCorrectionSeries.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="w-16 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
              DQ Orig
            </span>
            {activeCorrectionSeries.map((origKey) => {
              const label = TOOLTIP_LABELS[origKey] ?? origKey;
              return (
                <TogglePill
                  key={origKey}
                  label={label}
                  color={DQ_ORIG_COLOR}
                  active
                  onClick={() => {}}
                  dashed
                />
              );
            })}
            <span className="text-[10px] text-muted-foreground">
              ({corrections.length} corrections)
            </span>
          </div>
        )}

        {/* Supply pills */}
        {hasSupplyData && (
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              onClick={toggleAllSupply}
              className="w-16 text-[10px] font-bold uppercase tracking-widest text-muted-foreground hover:text-foreground transition-colors text-left"
              title={allSupplyOn ? "Deselect all supply series" : "Select all supply series"}
            >
              {allSupplyOn ? "Supply \u2212" : "Supply +"}
            </button>
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
        <label className="flex items-center gap-1.5">
          <span className="font-semibold uppercase tracking-wider text-muted-foreground">To</span>
          <select
            className="h-7 w-28 rounded border border-input bg-background px-2 text-xs"
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
          className="h-7 rounded border border-input bg-background px-2.5 text-xs text-muted-foreground hover:text-foreground"
          onClick={() => {
            setSkuTimeStart("");
            setSkuTimeEnd("");
          }}
        >
          All
        </button>
        <button
          className="h-7 rounded border border-input bg-background px-2.5 text-xs text-muted-foreground hover:text-foreground"
          onClick={() => {
            setSkuTimeStart(skuDefaultStart);
            setSkuTimeEnd("");
          }}
        >
          Default
        </button>
        {selectedModel && (
          <span className="ml-auto rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
            SHAP: {selectedModel}
          </span>
        )}
        {skuData.scope_count != null && (
          <span className="ml-auto rounded bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
            {skuData.mode === "item_at_all_locations"
              ? `${skuData.scope_count} locations`
              : `${skuData.scope_count} items`}
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
              {skuVisibleSeries.has("tothist_dmd") && (
                <Line
                  type="monotone"
                  dataKey="tothist_dmd"
                  yAxisId="left"
                  name="tothist_dmd"
                  stroke={SKU_SALES_COLORS.tothist_dmd}
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              )}
              {skuVisibleSeries.has("sales_qty") && (
                <Line
                  type="monotone"
                  dataKey="sales_qty"
                  yAxisId="left"
                  name="sales_qty"
                  stroke={SKU_SALES_COLORS.sales_qty}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              )}
              {skuVisibleSeries.has("qty_shipped") && (
                <Line
                  type="monotone"
                  dataKey="qty_shipped"
                  yAxisId="left"
                  name="qty_shipped"
                  stroke={SKU_SALES_COLORS.qty_shipped}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              )}
              {skuVisibleSeries.has("qty_ordered") && (
                <Line
                  type="monotone"
                  dataKey="qty_ordered"
                  yAxisId="left"
                  name="qty_ordered"
                  stroke={SKU_SALES_COLORS.qty_ordered}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              )}
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
                      strokeDasharray={model === "champion" ? undefined : "5 3"}
                      dot={false}
                      style={{ opacity: isOtherSelected ? DESELECT_OPACITY : 1 }}
                      activeDot={{ r: isSelected ? 7 : 4 }}
                    />
                  );
                })}
              {hasProdForecast && skuVisibleSeries.has("production_forecast") && (
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

              {/* ---- DQ correction original-value lines ---- */}
              {showCorrections &&
                activeCorrectionSeries.map((origKey) => (
                  <Line
                    key={origKey}
                    type="monotone"
                    dataKey={origKey}
                    yAxisId="left"
                    name={origKey}
                    stroke={DQ_ORIG_COLOR}
                    strokeWidth={2}
                    strokeDasharray="4 3"
                    dot={{ r: 3, fill: DQ_ORIG_COLOR }}
                    connectNulls={false}
                    activeDot={{ r: 5 }}
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

    </div>
  );
});

// ---------------------------------------------------------------------------
// TogglePill — reusable pill button for series visibility
// ---------------------------------------------------------------------------
const TogglePill = memo(function TogglePill({
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
});
