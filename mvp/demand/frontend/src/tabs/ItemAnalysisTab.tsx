import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { LoadingElement } from "@/components/LoadingElement";

import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { useDebounce } from "@/hooks/useDebounce";
import { usePanelToggles } from "@/hooks/usePanelToggles";
import {
  queryKeys,
  STALE,
  fetchInventoryPosition,
  fetchInventoryKpis,
  fetchInventoryTrend,
  fetchLtProfile,
} from "@/api/queries";
import { fetchProductionForecast } from "@/api/queries/production-forecast";
import type { ProductionForecastPayload } from "@/api/queries/production-forecast";
import type {
  DfuAnalysisKpis,
  DfuAnalysisPayload,
  SamplePairPayload,
  SuggestPayload,
  InventoryKpis,
  InventoryTrendPoint,
} from "@/types";

// DFU Analysis panels
import { SelectorPanel } from "./dfu-analysis/SelectorPanel";
import { ModelKpiSection } from "./dfu-analysis/ModelKpiSection";
import { DfuShapPanel } from "./dfu-analysis/DfuShapPanel";

// Unified chart (demand + supply in one view)
import { UnifiedChartPanel } from "./item-analysis/UnifiedChartPanel";

// ---------------------------------------------------------------------------
// Panel toggle defaults
// ---------------------------------------------------------------------------
const PANEL_DEFAULTS: Record<string, boolean> = {
  overlay: true,
  shap: true,
  forecastKpis: true,
};

const DEMAND_PANELS = [
  { key: "overlay", label: "Chart" },
  { key: "shap", label: "SHAP" },
  { key: "forecastKpis", label: "Forecast KPIs" },
] as const;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function ItemAnalysisTab() {
  const { panels, toggle, setAll, allOn } = usePanelToggles("ds:itemAnalysis:panels", PANEL_DEFAULTS);

  // ========================================================================
  // DFU Analysis state — single DFU mode only
  // ========================================================================
  const [dfuItem, setDfuItem] = useState("");
  const [dfuLocation, setDfuLocation] = useState("");
  const [dfuPoints, setDfuPoints] = useState(36);
  const [dfuKpiMonths, setDfuKpiMonths] = useState(12);
  const [dfuData, setDfuData] = useState<DfuAnalysisPayload | null>(null);
  const [dfuVisibleSeries, setDfuVisibleSeries] = useState<Set<string>>(
    new Set(["tothist_dmd", "qty_shipped", "qty_ordered"]),
  );
  const [dfuTimeStart, setDfuTimeStart] = useState("");
  const [dfuTimeEnd, setDfuTimeEnd] = useState("");
  const [dfuDefaultStart, setDfuDefaultStart] = useState("");
  const [dfuLoading, setDfuLoading] = useState(false);
  const [dfuAutoSampled, setDfuAutoSampled] = useState(false);
  const [dfuItemSuggestions, setDfuItemSuggestions] = useState<string[]>([]);
  const [dfuLocationSuggestions, setDfuLocationSuggestions] = useState<string[]>([]);
  const [prodForecastData, setProdForecastData] = useState<ProductionForecastPayload | null>(null);
  // SHAP model selection via dropdown (null = none)
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const debouncedDfuItem = useDebounce(dfuItem, 500);
  const debouncedDfuLocation = useDebounce(dfuLocation, 500);

  // Deep-link: consume ?item=…&loc=… URL params on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const urlItem = params.get("item");
    const urlLoc = params.get("loc");
    if (urlItem) {
      setDfuItem(urlItem);
      if (urlLoc) setDfuLocation(urlLoc);
      setDfuAutoSampled(true);
      // Clean up URL params so they don't persist on refresh
      params.delete("item");
      params.delete("loc");
      const newUrl = params.toString()
        ? `${window.location.pathname}?${params}`
        : window.location.pathname;
      window.history.replaceState({}, "", newUrl);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ========================================================================
  // Global filter sync (shared)
  // ========================================================================
  const { filters: globalFilters } = useGlobalFilterContext();
  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setDfuItem(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setDfuLocation(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  // ========================================================================
  // DFU effects
  // ========================================================================

  // Auto-sample on first visit
  useEffect(() => {
    if (dfuAutoSampled) return;
    if (dfuItem.trim() || dfuLocation.trim()) { setDfuAutoSampled(true); return; }
    let cancelled = false;
    async function loadSample() {
      try {
        const res = await fetch("/domains/sales/sample-pair");
        if (!res.ok) throw new Error("HTTP error");
        const payload = (await res.json()) as SamplePairPayload;
        if (!cancelled) {
          if (payload.item) setDfuItem(String(payload.item));
          if (payload.location) setDfuLocation(String(payload.location));
        }
      } catch { /* non-blocking */ } finally {
        if (!cancelled) setDfuAutoSampled(true);
      }
    }
    loadSample();
    return () => { cancelled = true; };
  }, [dfuAutoSampled, dfuItem, dfuLocation]);

  // Fetch DFU analysis data (always item_location mode)
  const anyDemandPanelOn = panels.overlay || panels.shap || panels.forecastKpis;
  useEffect(() => {
    if (!anyDemandPanelOn) return;
    if (!debouncedDfuItem.trim() || !debouncedDfuLocation.trim()) return;
    setDfuData(null);
    let cancelled = false;
    async function loadAnalysis() {
      setDfuLoading(true);
      try {
        const params = new URLSearchParams({ mode: "item_location", item: debouncedDfuItem.trim(), location: debouncedDfuLocation.trim(), points: String(dfuPoints) });
        const res = await fetch(`/dfu/analysis?${params}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = (await res.json()) as DfuAnalysisPayload;
        if (!cancelled) {
          setDfuData(payload);
          setSelectedModel(null);
          const allKeys = new Set(["tothist_dmd", "qty_shipped", "qty_ordered", "production_forecast", ...payload.models.map((m) => `forecast_${m}`)]);
          setDfuVisibleSeries(allKeys);
          const measureKeys = new Set<string>();
          for (const pt of payload.series) for (const k of Object.keys(pt)) if (k !== "month") measureKeys.add(k);
          let smartStart = "";
          if (measureKeys.size > 0 && payload.series.length > 0) {
            const keys = Array.from(measureKeys);
            for (const pt of payload.series) { if (keys.every((k) => k in pt)) { smartStart = String(pt.month); break; } }
          }
          setDfuDefaultStart(smartStart);
          setDfuTimeStart(smartStart);
          setDfuTimeEnd("");
        }
      } catch { if (!cancelled) setDfuData(null); }
      finally { if (!cancelled) setDfuLoading(false); }
    }
    loadAnalysis();
    return () => { cancelled = true; };
  }, [debouncedDfuItem, debouncedDfuLocation, dfuPoints, anyDemandPanelOn]);

  // Fetch production forecast (future months)
  useEffect(() => {
    if (!debouncedDfuItem.trim() || !debouncedDfuLocation.trim()) { setProdForecastData(null); return; }
    let cancelled = false;
    fetchProductionForecast({ item_no: debouncedDfuItem.trim(), loc: debouncedDfuLocation.trim() })
      .then((payload) => { if (!cancelled) setProdForecastData(payload); })
      .catch(() => { if (!cancelled) setProdForecastData(null); });
    return () => { cancelled = true; };
  }, [debouncedDfuItem, debouncedDfuLocation]);

  // Item typeahead
  useEffect(() => {
    if (!dfuItem.trim()) { setDfuItemSuggestions([]); return; }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({ field: "dmdunit", q: dfuItem.trim(), limit: "12" });
        if (debouncedDfuLocation.trim()) params.set("filters", JSON.stringify({ loc: `=${debouncedDfuLocation.trim()}` }));
        const res = await fetch(`/domains/sales/suggest?${params}`);
        if (!res.ok) throw new Error("HTTP error");
        const payload = (await res.json()) as SuggestPayload;
        if (!cancelled) setDfuItemSuggestions(payload.values || []);
      } catch { if (!cancelled) setDfuItemSuggestions([]); }
    }, 180);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [dfuItem, debouncedDfuLocation]);

  // Location typeahead
  useEffect(() => {
    if (!dfuLocation.trim()) { setDfuLocationSuggestions([]); return; }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({ field: "loc", q: dfuLocation.trim(), limit: "12" });
        if (debouncedDfuItem.trim()) params.set("filters", JSON.stringify({ dmdunit: `=${debouncedDfuItem.trim()}` }));
        const res = await fetch(`/domains/sales/suggest?${params}`);
        if (!res.ok) throw new Error("HTTP error");
        const payload = (await res.json()) as SuggestPayload;
        if (!cancelled) setDfuLocationSuggestions(payload.values || []);
      } catch { if (!cancelled) setDfuLocationSuggestions([]); }
    }, 180);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [dfuLocation, debouncedDfuItem]);

  // ========================================================================
  // DFU computed values
  // ========================================================================
  const dfuKpis = useMemo<Record<string, DfuAnalysisKpis>>(() => {
    if (!dfuData?.model_monthly) return {};
    const result: Record<string, DfuAnalysisKpis> = {};
    for (const [modelId, rows] of Object.entries(dfuData.model_monthly)) {
      const window = rows.slice(0, dfuKpiMonths);
      if (window.length === 0) continue;
      let sumForecast = 0, sumActual = 0, sumAbsErr = 0;
      for (const r of window) { sumForecast += r.forecast; sumActual += r.actual; sumAbsErr += Math.abs(r.forecast - r.actual); }
      const absActual = Math.abs(sumActual);
      const wape = absActual > 0 ? (100 * sumAbsErr) / absActual : null;
      const accuracy = wape !== null ? 100 - wape : null;
      const bias = absActual > 0 ? sumForecast / sumActual - 1 : null;
      result[modelId] = { accuracy_pct: accuracy !== null ? Math.round(accuracy * 10000) / 10000 : null, wape: wape !== null ? Math.round(wape * 10000) / 10000 : null, bias: bias !== null ? Math.round(bias * 10000) / 10000 : null, sum_forecast: sumForecast, sum_actual: sumActual, months_covered: window.length };
    }
    return result;
  }, [dfuData, dfuKpiMonths]);

  const dfuMonths = useMemo(() => (!dfuData?.series.length ? [] : dfuData.series.map((p) => String(p.month))), [dfuData]);

  const dfuFilteredSeries = useMemo(() => {
    if (!dfuData?.series.length) return [];
    const start = dfuTimeStart || dfuMonths[0];
    const end = dfuTimeEnd || dfuMonths[dfuMonths.length - 1];
    return dfuData.series.filter((p) => { const m = String(p.month); return m >= start && m <= end; });
  }, [dfuData, dfuMonths, dfuTimeStart, dfuTimeEnd]);

  const mergedFilteredSeries = useMemo(() => {
    if (!prodForecastData?.forecasts.length) return dfuFilteredSeries as Record<string, unknown>[];
    const lastHistMonth = dfuMonths[dfuMonths.length - 1] ?? "";
    const prodMap = new Map<string, number | null>(
      prodForecastData.forecasts.map((pt) => [pt.forecast_month, pt.forecast_qty]),
    );
    const enhanced = dfuFilteredSeries.map((pt) => {
      const m = String(pt.month);
      return prodMap.has(m) ? { ...pt, production_forecast: prodMap.get(m) } : pt;
    });
    const futurePts = prodForecastData.forecasts
      .filter((pt) => pt.forecast_month > lastHistMonth)
      .sort((a, b) => a.forecast_month.localeCompare(b.forecast_month))
      .map((pt) => ({ month: pt.forecast_month, production_forecast: pt.forecast_qty }));
    return [...enhanced, ...futurePts] as Record<string, unknown>[];
  }, [dfuFilteredSeries, dfuMonths, prodForecastData]);

  // Available models for the SHAP dropdown
  const shapModelOptions = useMemo(() => {
    if (!dfuData?.models.length) return [];
    return dfuData.models;
  }, [dfuData]);

  function handleReset() {
    setDfuData(null);
    setDfuAutoSampled(false);
    setDfuItem("");
    setDfuLocation("");
    setSelectedModel(null);
  }

  const emptyMessage =
    !dfuItem.trim() || !dfuLocation.trim()
      ? "Enter both item and location to view analysis."
      : "No data available for the selected filters.";

  // ========================================================================
  // Inventory queries — DFU-level data for attributes
  // ========================================================================
  const invMonths = 12;
  const invKpiParams = useMemo(
    () => ({ item: debouncedDfuItem, location: debouncedDfuLocation, months: invMonths }),
    [debouncedDfuItem, debouncedDfuLocation],
  );
  const invTrendParams = useMemo(
    () => ({ item: debouncedDfuItem, location: debouncedDfuLocation, months: invMonths }),
    [debouncedDfuItem, debouncedDfuLocation],
  );
  const invPositionParams = useMemo(
    () => ({ item: debouncedDfuItem, location: debouncedDfuLocation, limit: 1, offset: 0, sort_by: "snapshot_date" as const, sort_dir: "desc" as const }),
    [debouncedDfuItem, debouncedDfuLocation],
  );

  const hasDfu = !!(debouncedDfuItem.trim() && debouncedDfuLocation.trim());

  // Always fetch KPIs for the DFU attributes section
  const { data: kpiData } = useQuery({
    queryKey: queryKeys.inventoryKpis(invKpiParams),
    queryFn: () => fetchInventoryKpis(invKpiParams),
    staleTime: STALE.FIVE_MIN,
    enabled: hasDfu,
  });

  const { data: trendPayload } = useQuery({
    queryKey: queryKeys.inventoryTrend(invTrendParams),
    queryFn: () => fetchInventoryTrend(invTrendParams),
    staleTime: STALE.FIVE_MIN,
    enabled: hasDfu,
  });

  const trendData: InventoryTrendPoint[] = trendPayload?.trend ?? [];
  const trendParams2 = trendPayload?.params;

  // Latest position snapshot for inline attributes
  const { data: positionPayload } = useQuery({
    queryKey: queryKeys.inventoryPosition(invPositionParams),
    queryFn: () => fetchInventoryPosition(invPositionParams),
    staleTime: STALE.TWO_MIN,
    enabled: hasDfu,
  });

  const positionRow = positionPayload?.positions?.[0] ?? null;

  // Single-DFU lead time profile for inline attributes
  const ltProfileParams = useMemo(
    () => ({ item: debouncedDfuItem, location: debouncedDfuLocation, limit: 1 }),
    [debouncedDfuItem, debouncedDfuLocation],
  );
  const { data: ltProfilePayload } = useQuery({
    queryKey: queryKeys.ltProfile(ltProfileParams),
    queryFn: () => fetchLtProfile(ltProfileParams),
    staleTime: STALE.FIVE_MIN,
    enabled: hasDfu,
  });
  const ltProfileRow = ltProfilePayload?.rows?.[0] ?? null;

  // ========================================================================
  // Render
  // ========================================================================
  return (
    <section className="mt-4 space-y-4">
      {/* ---- Selector (always visible) ---- */}
      <Card className="animate-fade-in">
        <SelectorPanel
          dfuItem={dfuItem} setDfuItem={setDfuItem}
          dfuLocation={dfuLocation} setDfuLocation={setDfuLocation}
          dfuPoints={dfuPoints} setDfuPoints={setDfuPoints}
          dfuItemSuggestions={dfuItemSuggestions}
          dfuLocationSuggestions={dfuLocationSuggestions}
          dfuData={dfuData}
          onReset={handleReset}
          kpiData={kpiData as InventoryKpis | undefined}
          trendParams={trendParams2}
          positionRow={positionRow}
          ltProfile={ltProfileRow}
        />

        {/* ---- Toggle toolbar ---- */}
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t px-6 py-2 text-xs">
          {/* Select All / Deselect All toggle */}
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-[10px] uppercase tracking-wider font-semibold"
            onClick={() => setAll(!allOn)}
          >
            {allOn ? "Deselect All" : "Select All"}
          </Button>
          <span className="hidden h-4 w-px bg-border sm:block" />

          {DEMAND_PANELS.map((p) => (
            <label key={p.key} className="flex cursor-pointer items-center gap-1.5 select-none">
              <Checkbox
                checked={panels[p.key]}
                onCheckedChange={() => toggle(p.key)}
                aria-label={`Toggle ${p.label}`}
              />
              <span className={panels[p.key] ? "text-foreground" : "text-muted-foreground"}>{p.label}</span>
            </label>
          ))}

          {/* SHAP model dropdown */}
          {panels.shap && shapModelOptions.length > 0 && (
            <>
              <span className="mx-1 hidden h-4 w-px bg-border sm:block" />
              <label className="flex items-center gap-1.5 select-none">
                <span className="font-semibold uppercase tracking-wider text-muted-foreground">SHAP</span>
                <select
                  className="h-7 rounded border border-input bg-background px-2 text-xs"
                  value={selectedModel ?? ""}
                  onChange={(e) => setSelectedModel(e.target.value || null)}
                >
                  <option value="">None</option>
                  {shapModelOptions.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </label>
            </>
          )}
        </div>
      </Card>

      {/* ---- Demand panels ---- */}
      {anyDemandPanelOn && (
        <Card>
          <CardContent className="space-y-4 pt-4">
            {dfuData && dfuData.series.length > 0 ? (
              <>
                {panels.overlay && (
                  <UnifiedChartPanel
                    dfuData={dfuData}
                    dfuFilteredSeries={mergedFilteredSeries}
                    dfuMonths={dfuMonths}
                    dfuTimeStart={dfuTimeStart} setDfuTimeStart={setDfuTimeStart}
                    dfuTimeEnd={dfuTimeEnd} setDfuTimeEnd={setDfuTimeEnd}
                    dfuDefaultStart={dfuDefaultStart}
                    dfuVisibleSeries={dfuVisibleSeries}
                    setDfuVisibleSeries={setDfuVisibleSeries}
                    prodForecastData={prodForecastData}
                    selectedModel={selectedModel}
                    onModelSelect={setSelectedModel}
                    trendData={trendData}
                    trendParams={trendParams2}
                  />
                )}
                {panels.shap && selectedModel && (
                  <DfuShapPanel
                    selectedModel={selectedModel}
                    itemNo={debouncedDfuItem}
                    loc={debouncedDfuLocation}
                    dfuMode="item_location"
                    visibleMonths={mergedFilteredSeries.map((p) => String(p.month))}
                  />
                )}
                {panels.forecastKpis && (
                  <ModelKpiSection
                    dfuData={dfuData}
                    dfuKpis={dfuKpis}
                    dfuKpiMonths={dfuKpiMonths}
                    setDfuKpiMonths={setDfuKpiMonths}
                    dfuVisibleSeries={dfuVisibleSeries}
                  />
                )}
              </>
            ) : dfuLoading ? (
              <div className="flex h-[320px] items-center justify-center">
                <LoadingElement message="Fetching analysis..." />
              </div>
            ) : (
              <div className="flex h-[320px] items-center justify-center text-sm text-muted-foreground">
                {emptyMessage}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </section>
  );
}
