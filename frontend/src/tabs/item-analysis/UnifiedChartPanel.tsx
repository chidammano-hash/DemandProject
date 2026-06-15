import { memo, useMemo, useState, useCallback, useRef, useEffect } from "react";
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
import type { ProductionForecastPayload, StagingForecastsPayload } from "@/api/queries/production-forecast";
import { modelLabel } from "@/lib/model-labels";
import type { DQCorrection } from "@/api/queries/platform";
import { formatMonthLabel, isFromDisabled, isToDisabled } from "./monthRange";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const PROD_FORECAST_COLOR = "#7c3aed";
const CHART_MARGIN = { top: 8, right: 40, left: 18, bottom: 8 };
const DESELECT_OPACITY = 0.25;

/** Per-model colors for staging forecast lines */
const STAGING_COLORS: Record<string, string> = {
  lgbm_cluster: "#2563eb",
  catboost_cluster: "#dc2626",
  xgboost_cluster: "#16a34a",
  lgbm_cust_enriched: "#1d4ed8",
  catboost_cust_enriched: "#b91c1c",
  xgboost_cust_enriched: "#15803d",
  nbeats: "#ea580c",
  nhits: "#9333ea",
  chronos_bolt: "#0891b2",
  chronos: "#0e7490",
  chronos2: "#155e75",
  chronos2_enriched: "#164e63",
  bolt_hierarchical: "#0d9488",
  mstl: "#ca8a04",
  seasonal_naive: "#64748b",
  rolling_mean: "#78716c",
};
const STAGING_FALLBACK_COLOR = "#6b7280";

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

// ---------------------------------------------------------------------------
// Persistent default-measure preferences (localStorage)
// ---------------------------------------------------------------------------
const LS_KEY_DEMAND = "ds:itemAnalysis:defaultMeasures";
const LS_KEY_SUPPLY = "ds:itemAnalysis:defaultSupply";

/** All static demand measure keys (sales-side). */
const SALES_MEASURE_KEYS = ["tothist_dmd", "sales_qty", "qty_shipped", "qty_ordered"] as const;

/** Load saved demand defaults from localStorage. */
export function loadDefaultMeasures(): Set<string> {
  try {
    const raw = localStorage.getItem(LS_KEY_DEMAND);
    if (raw) return new Set(JSON.parse(raw) as string[]);
  } catch { /* ignore */ }
  return new Set<string>(SALES_MEASURE_KEYS);
}

/** Load saved supply hidden-set from localStorage. */
export function loadDefaultHiddenSupply(): Set<string> {
  try {
    const raw = localStorage.getItem(LS_KEY_SUPPLY);
    if (raw) return new Set(JSON.parse(raw) as string[]);
  } catch { /* ignore */ }
  return new Set(DEFAULT_HIDDEN_SUPPLY);
}

function saveDemandDefaults(keys: Set<string>) {
  localStorage.setItem(LS_KEY_DEMAND, JSON.stringify([...keys]));
}

function saveSupplyDefaults(hiddenKeys: Set<string>) {
  localStorage.setItem(LS_KEY_SUPPLY, JSON.stringify([...hiddenKeys]));
}

/** Return a new Set with `key` toggled (added if absent, removed if present). */
function toggleInSet<T>(prev: Set<T>, key: T): Set<T> {
  const next = new Set(prev);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  return next;
}

/**
 * Build the visible-series set for a new SKU load, using saved defaults.
 * Forecast model keys are always included; sales measures follow user prefs.
 */
export function buildInitialVisibleSeries(
  models: string[],
): Set<string> {
  const defaults = loadDefaultMeasures();
  const keys = new Set<string>();
  for (const k of SALES_MEASURE_KEYS) {
    if (defaults.has(k)) keys.add(k);
  }
  for (const m of models) keys.add(`forecast_${m}`);
  if (defaults.has("production_forecast")) keys.add("production_forecast");
  return keys;
}

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
  stagingForecastData?: StagingForecastsPayload | null;
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
  stagingForecastData,
  selectedModel = null,
  onModelSelect,
  trendData = [],
  trendParams,
  corrections = [],
  showCorrections = false,
}: UnifiedChartPanelProps) {
  const { chartColors } = useChartColors();
  const [hiddenSupply, setHiddenSupply] = useState<Set<string>>(() => loadDefaultHiddenSupply());
  // Tracks demand series whose pill is shown but chart line is hidden (dimmed pill)
  const [hiddenDemand, setHiddenDemand] = useState<Set<string>>(new Set());

  // Staging forecast model visibility (hidden set — hidden by default until toggled on)
  const [hiddenStaging, setHiddenStaging] = useState<Set<string>>(new Set());
  // Staging pills hidden from the toolbar entirely (Defaults menu unchecks them).
  // Pill click only dims via hiddenStaging; Defaults menu removes the pill.
  const [hiddenStagingPills, setHiddenStagingPills] = useState<Set<string>>(new Set());

  const hasProdForecast = (prodForecastData?.forecasts.length ?? 0) > 0;
  const prodForecastLabel = hasProdForecast
    ? `Prod (${prodForecastData!.model_id})`
    : "Prod Forecast";
  const hasSupplyData = trendData.length > 0;

  // Derive staging model IDs from the chart data (staging_* keys),
  // excluding the promoted production model to avoid duplicate lines.
  const promotedModelId = prodForecastData?.model_id ?? null;
  const stagingModelIds = useMemo(() => {
    if (!stagingForecastData?.models) return [];
    return Object.keys(stagingForecastData.models).filter(
      (id) => id !== promotedModelId,
    );
  }, [stagingForecastData, promotedModelId]);

  const hasStagingModels = stagingModelIds.length > 0;
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

  // Toggle chart line visibility — pill stays but dims (clicking pill)
  function toggleDemandLineVisibility(key: string) {
    setHiddenDemand((prev) => toggleInSet(prev, key));
  }

  // Staging model toggle
  const toggleStagingModel = useCallback((modelId: string) => {
    setHiddenStaging((prev) => toggleInSet(prev, modelId));
  }, []);

  // Toggle all staging models
  const allStagingOn = hasStagingModels && stagingModelIds.every((m) => !hiddenStaging.has(m));
  const toggleAllStaging = useCallback(() => {
    setHiddenStaging((prev) => {
      if (allStagingOn) {
        return new Set(stagingModelIds);
      }
      return new Set();
    });
  }, [allStagingOn, stagingModelIds]);

  // Supply toggle
  const toggleSupply = useCallback((key: string) => {
    setHiddenSupply((prev) => toggleInSet(prev, key));
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

  // "All on" means all enabled pills have their lines visible (not hidden)
  const enabledDemandKeys = allDemandKeys.filter((k) => skuVisibleSeries.has(k));
  const allDemandOn = enabledDemandKeys.length > 0 && enabledDemandKeys.every((k) => !hiddenDemand.has(k));

  const toggleAllDemand = useCallback(() => {
    setHiddenDemand((prev) => {
      if (allDemandOn) {
        // Hide all lines (dim all pills)
        const next = new Set(prev);
        for (const k of enabledDemandKeys) next.add(k);
        return next;
      } else {
        // Show all lines (un-dim all pills)
        const next = new Set(prev);
        for (const k of enabledDemandKeys) next.delete(k);
        return next;
      }
    });
  }, [enabledDemandKeys, allDemandOn]);

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
      {/* ---- Toggle pills + defaults gear ---- */}
      <div className="flex items-start gap-2">
        <div className="flex-1 space-y-1.5">
        {/* Demand pills */}
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            onClick={toggleAllDemand}
            className="w-16 text-[10px] font-bold uppercase tracking-widest text-muted-foreground hover:text-foreground transition-colors text-left"
            title={allDemandOn ? "Deselect all demand series" : "Select all demand series"}
          >
            {allDemandOn ? "Demand \u2212" : "Demand +"}
          </button>
          {salesMeasures.map(({ key, label, color }) =>
            skuVisibleSeries.has(key) ? (
              <TogglePill
                key={key}
                label={label}
                color={color}
                active={!hiddenDemand.has(key)}
                onClick={() => toggleDemandLineVisibility(key)}
              />
            ) : null,
          )}
          {skuData.models.map((model, idx) => {
            const key = `forecast_${model}`;
            const color = skuModelColor(model, idx);
            const isEnabled = skuVisibleSeries.has(key);
            if (!isEnabled) return null;
            const isShapSelected = selectedModel === model;
            return (
              <span key={key} className="inline-flex items-center gap-0.5">
                <TogglePill
                  label={model}
                  color={color}
                  active={!hiddenDemand.has(key)}
                  onClick={() => toggleDemandLineVisibility(key)}
                  dashed={model !== "champion"}
                  ring={isShapSelected}
                />
                {onModelSelect && (
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
          {hasProdForecast && skuVisibleSeries.has("production_forecast") && (
            <TogglePill
              label={prodForecastLabel}
              color={PROD_FORECAST_COLOR}
              active={!hiddenDemand.has("production_forecast")}
              onClick={() => toggleDemandLineVisibility("production_forecast")}
              dashed
            />
          )}
        </div>

        {/* Staging forecast model pills */}
        {hasStagingModels && (
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              onClick={toggleAllStaging}
              className="w-16 text-[10px] font-bold uppercase tracking-widest text-muted-foreground hover:text-foreground transition-colors text-left"
              title={allStagingOn ? "Hide all staging models" : "Show all staging models"}
            >
              {allStagingOn ? "Staging \u2212" : "Staging +"}
            </button>
            {stagingModelIds
              .filter((mid) => !hiddenStagingPills.has(mid))
              .map((mid) => {
                const color = STAGING_COLORS[mid] ?? STAGING_FALLBACK_COLOR;
                return (
                  <TogglePill
                    key={`staging_${mid}`}
                    label={modelLabel(mid)}
                    color={color}
                    active={!hiddenStaging.has(mid)}
                    onClick={() => toggleStagingModel(mid)}
                    dashed
                  />
                );
              })}
          </div>
        )}

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
            {availableSupply.map((s) =>
              showSupply(s.key) ? (
                <TogglePill
                  key={s.key}
                  label={s.label}
                  color={s.color}
                  active
                  onClick={() => toggleSupply(s.key)}
                  suffix={s.axis === "right" ? "d" : "u"}
                />
              ) : null,
            )}
          </div>
        )}
        </div>

        {/* Defaults gear — right side */}
        <MeasureDefaultsMenu
          models={skuData.models}
          hasProdForecast={hasProdForecast}
          prodForecastLabel={prodForecastLabel}
          hasSupplyData={hasSupplyData}
          availableSupply={availableSupply}
          stagingModelIds={stagingModelIds}
          hiddenStagingPills={hiddenStagingPills}
          setSkuVisibleSeries={setSkuVisibleSeries}
          setHiddenSupply={setHiddenSupply}
          setHiddenStagingPills={setHiddenStagingPills}
        />
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
              <option key={m} value={m} disabled={isFromDisabled(m, skuTimeEnd)}>
                {formatMonthLabel(m)}
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
              <option key={m} value={m} disabled={isToDisabled(m, skuTimeStart)}>
                {formatMonthLabel(m)}
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
          style={{ minWidth: `${Math.max(800, mergedData.length * 40)}px` }}
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
                  let label = TOOLTIP_LABELS[name] ?? name;
                  // Resolve staging model names to readable labels
                  if (name.startsWith("staging_")) {
                    const mid = name.slice("staging_".length);
                    label = `${modelLabel(mid)} (staging)`;
                  }
                  if (name === "dos" || name === "avg_lead_time")
                    return [`${Number(value).toFixed(1)} days`, label];
                  return [
                    formatNumber(Number.isFinite(Number(value)) ? Number(value) : null),
                    label,
                  ];
                }}
              />

              {/* ---- Demand lines ---- */}
              {skuVisibleSeries.has("tothist_dmd") && !hiddenDemand.has("tothist_dmd") && (
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
              {skuVisibleSeries.has("sales_qty") && !hiddenDemand.has("sales_qty") && (
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
              {skuVisibleSeries.has("qty_shipped") && !hiddenDemand.has("qty_shipped") && (
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
              {skuVisibleSeries.has("qty_ordered") && !hiddenDemand.has("qty_ordered") && (
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
                .filter((m) => skuVisibleSeries.has(`forecast_${m}`) && !hiddenDemand.has(`forecast_${m}`))
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
              {hasProdForecast && skuVisibleSeries.has("production_forecast") && !hiddenDemand.has("production_forecast") && (
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

              {/* ---- Staging forecast lines ---- */}
              {stagingModelIds
                .filter((mid) => !hiddenStaging.has(mid) && !hiddenStagingPills.has(mid))
                .map((mid) => {
                  const key = `staging_${mid}`;
                  const color = STAGING_COLORS[mid] ?? STAGING_FALLBACK_COLOR;
                  return (
                    <Line
                      key={key}
                      type="monotone"
                      dataKey={key}
                      yAxisId="left"
                      name={key}
                      stroke={color}
                      strokeWidth={1.5}
                      strokeDasharray="4 3"
                      dot={false}
                      connectNulls
                      activeDot={{ r: 3 }}
                    />
                  );
                })}

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
// MeasureDefaultsMenu — gear icon dropdown to set default visible measures
// ---------------------------------------------------------------------------
const MeasureDefaultsMenu = memo(function MeasureDefaultsMenu({
  models,
  hasProdForecast,
  prodForecastLabel,
  hasSupplyData,
  availableSupply,
  stagingModelIds,
  hiddenStagingPills,
  setSkuVisibleSeries,
  setHiddenSupply,
  setHiddenStagingPills,
}: {
  models: string[];
  hasProdForecast: boolean;
  prodForecastLabel: string;
  hasSupplyData: boolean;
  availableSupply: SupplySeriesDef[];
  stagingModelIds: string[];
  hiddenStagingPills: Set<string>;
  setSkuVisibleSeries: (updater: (prev: Set<string>) => Set<string>) => void;
  setHiddenSupply: React.Dispatch<React.SetStateAction<Set<string>>>;
  setHiddenStagingPills: React.Dispatch<React.SetStateAction<Set<string>>>;
}) {
  const [open, setOpen] = useState(false);
  const [demandDefaults, setDemandDefaults] = useState<Set<string>>(loadDefaultMeasures);
  const [supplyHidden, setSupplyHidden] = useState<Set<string>>(loadDefaultHiddenSupply);
  const menuRef = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const [pos, setPos] = useState<{ top: number; right: number }>({ top: 0, right: 0 });

  // Compute fixed position from button rect when opening
  useEffect(() => {
    if (open && btnRef.current) {
      const r = btnRef.current.getBoundingClientRect();
      setPos({ top: r.bottom + 4, right: window.innerWidth - r.right });
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (
        menuRef.current && !menuRef.current.contains(e.target as Node) &&
        btnRef.current && !btnRef.current.contains(e.target as Node)
      ) setOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  const toggleDemand = useCallback((key: string) => {
    setDemandDefaults((prev) => {
      const next = toggleInSet(prev, key);
      saveDemandDefaults(next);
      return next;
    });
    // Update live chart state immediately
    setSkuVisibleSeries((prev) => toggleInSet(prev, key));
  }, [setSkuVisibleSeries]);

  const toggleSupply = useCallback((key: string) => {
    setSupplyHidden((prev) => {
      const next = toggleInSet(prev, key);
      saveSupplyDefaults(next);
      return next;
    });
    // Update live chart state immediately
    setHiddenSupply((prev) => toggleInSet(prev, key));
  }, [setHiddenSupply]);

  const toggleStaging = useCallback((mid: string) => {
    setHiddenStagingPills((prev) => toggleInSet(prev, mid));
  }, [setHiddenStagingPills]);

  const demandItems: { key: string; label: string; color: string }[] = [
    { key: "tothist_dmd", label: "Sale Qty (ext)", color: SKU_SALES_COLORS.tothist_dmd },
    { key: "sales_qty", label: "Sale Qty", color: SKU_SALES_COLORS.sales_qty },
    { key: "qty_shipped", label: "Shipped", color: SKU_SALES_COLORS.qty_shipped },
    { key: "qty_ordered", label: "Ordered", color: SKU_SALES_COLORS.qty_ordered },
    ...models.map((m, i) => ({ key: `forecast_${m}`, label: m, color: skuModelColor(m, i) })),
    ...(hasProdForecast ? [{ key: "production_forecast", label: prodForecastLabel, color: PROD_FORECAST_COLOR }] : []),
  ];

  return (
    <>
      <button
        ref={btnRef}
        onClick={() => setOpen((v) => !v)}
        className={`flex h-6 items-center gap-1 rounded border px-2 py-0.5 text-[10px] font-medium transition-colors ${
          open
            ? "border-primary text-primary"
            : "border-input text-muted-foreground hover:text-foreground hover:border-foreground"
        }`}
        title="Configure which measures are visible by default when loading a new DFU"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
        Defaults
      </button>
      {open && (
        <div
          ref={menuRef}
          className="fixed z-[9999] min-w-[200px] overflow-y-auto rounded-md border bg-white p-2.5 shadow-lg dark:bg-zinc-900"
          style={{ top: pos.top, right: pos.right, maxHeight: `calc(100vh - ${pos.top}px - 16px)` }}
        >
          {/* Demand section */}
          <DefaultsSectionHeader label="Demand" withDivider={false} />
          {demandItems.map(({ key, label, color }) => (
            <DefaultCheckboxRow
              key={key}
              label={label}
              color={color}
              checked={demandDefaults.has(key)}
              onChange={() => toggleDemand(key)}
            />
          ))}

          {/* Staging section */}
          {stagingModelIds.length > 0 && (
            <>
              <DefaultsSectionHeader label="Staging" />
              {stagingModelIds.map((mid) => {
                const color = STAGING_COLORS[mid] ?? STAGING_FALLBACK_COLOR;
                return (
                  <DefaultCheckboxRow
                    key={`staging_${mid}`}
                    label={modelLabel(mid)}
                    color={color}
                    checked={!hiddenStagingPills.has(mid)}
                    onChange={() => toggleStaging(mid)}
                  />
                );
              })}
            </>
          )}

          {/* Supply section */}
          {hasSupplyData && availableSupply.length > 0 && (
            <>
              <DefaultsSectionHeader label="Supply" />
              {availableSupply.map((s) => (
                <DefaultCheckboxRow
                  key={s.key}
                  label={s.label}
                  color={s.color}
                  checked={!supplyHidden.has(s.key)}
                  onChange={() => toggleSupply(s.key)}
                />
              ))}
            </>
          )}
        </div>
      )}
    </>
  );
});

// ---------------------------------------------------------------------------
// Defaults menu row — checkbox + color dot + label (shared by Demand/Staging/Supply)
// ---------------------------------------------------------------------------
function DefaultsSectionHeader({ label, withDivider = true }: { label: string; withDivider?: boolean }) {
  return (
    <>
      {withDivider && <div className="my-1.5 border-t border-border" />}
      <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
    </>
  );
}

function DefaultCheckboxRow({
  label,
  color,
  checked,
  onChange,
}: {
  label: string;
  color: string;
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-[11px] hover:bg-muted">
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="h-3 w-3 rounded border-muted-foreground accent-current"
        style={{ accentColor: color }}
      />
      <span className="inline-block h-0.5 w-3 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </label>
  );
}

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
