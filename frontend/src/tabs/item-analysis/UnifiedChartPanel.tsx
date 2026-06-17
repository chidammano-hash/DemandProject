import { memo, useMemo, useState, useCallback } from "react";

import { SKU_SALES_COLORS, skuModelColor } from "@/constants/colors";
import type {
  SkuAnalysisPayload,
  InventoryTrendPoint,
  InventoryTrendParams,
} from "@/types";
import type { ProductionForecastPayload, StagingForecastsPayload } from "@/api/queries/production-forecast";
import { modelLabel } from "@/lib/model-labels";
import type { DQCorrection } from "@/api/queries/platform";
import { formatMonthLabel, isFromDisabled, isToDisabled } from "./monthRange";
import {
  PROD_FORECAST_COLOR,
  AI_CHAMPION_COLOR,
  STAGING_COLORS,
  STAGING_FALLBACK_COLOR,
  DQ_ORIG_COLOR,
  TOOLTIP_LABELS,
} from "./colors";
import {
  SUPPLY_SERIES_DEFS,
  loadDefaultMeasures,
  loadDefaultHiddenSupply,
  toggleInSet,
} from "./measures";
import { TogglePill } from "./TogglePill";
import { MeasureDefaultsMenu } from "./MeasureDefaultsMenu";
import { UnifiedChart } from "./UnifiedChart";

// Map DB column_name + table → chart dataKey for the original series
const DQ_COLUMN_MAP: Record<string, { dataKey: string; origKey: string; label: string }> = {
  "fact_sales_monthly:qty": { dataKey: "sales_qty", origKey: "sales_qty_orig", label: "Sale Qty (original)" },
  "fact_sales_monthly:qty_shipped": { dataKey: "qty_shipped", origKey: "qty_shipped_orig", label: "Shipped (original)" },
  "fact_inventory_snapshot:qty_on_hand": { dataKey: "total_on_hand", origKey: "total_on_hand_orig", label: "On Hand (original)" },
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
  // Saved AI Champion forward forecast overlay (optional)
  hasAiChampion?: boolean;
  aiChampionRecCode?: string | null;
  aiChampionRationale?: string | null;
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
  hasAiChampion = false,
  aiChampionRecCode = null,
  aiChampionRationale = null,
}: UnifiedChartPanelProps) {
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
    if (hasAiChampion) keys.push("ai_champion");
    return keys;
  }, [skuData.models, hasProdForecast, hasAiChampion]); // eslint-disable-line react-hooks/exhaustive-deps

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
          {hasAiChampion && skuVisibleSeries.has("ai_champion") && (
            <TogglePill
              label="AI Champion"
              color={AI_CHAMPION_COLOR}
              active={!hiddenDemand.has("ai_champion")}
              onClick={() => toggleDemandLineVisibility("ai_champion")}
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

      {/* ---- AI Champion rationale (the reasons behind the amber line) ---- */}
      {hasAiChampion && aiChampionRationale && !hiddenDemand.has("ai_champion") && (
        <p className="rounded-md border border-amber-300/60 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-700/60 dark:bg-amber-950/30 dark:text-amber-200">
          <span className="font-semibold" style={{ color: AI_CHAMPION_COLOR }}>
            AI Champion{aiChampionRecCode ? ` (${aiChampionRecCode})` : ""}:
          </span>{" "}
          {aiChampionRationale}
        </p>
      )}

      {/* ---- Chart ---- */}
      <UnifiedChart
        mergedData={mergedData}
        hasRightAxis={hasRightAxis}
        models={skuData.models}
        skuVisibleSeries={skuVisibleSeries}
        hiddenDemand={hiddenDemand}
        selectedModel={selectedModel}
        hasProdForecast={hasProdForecast}
        hasAiChampion={hasAiChampion}
        aiChampionLineHidden={hiddenDemand.has("ai_champion")}
        stagingModelIds={stagingModelIds}
        hiddenStaging={hiddenStaging}
        hiddenStagingPills={hiddenStagingPills}
        hasSupplyData={hasSupplyData}
        availableSupply={availableSupply}
        hiddenSupply={hiddenSupply}
        showSupply={showSupply}
        showCorrections={showCorrections}
        activeCorrectionSeries={activeCorrectionSeries}
        hasSs={hasSs}
        ss={ss}
        ropUnits={ropUnits}
      />

    </div>
  );
});
