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
  correctionKeys,
  fetchCorrectionsByItem,
} from "@/api/queries";
import { fetchProductionForecast, fetchStagingForecasts } from "@/api/queries/production-forecast";
import type { ProductionForecastPayload, StagingForecastsPayload } from "@/api/queries/production-forecast";
import type {
  SkuAnalysisKpis,
  SkuAnalysisPayload,
  SamplePairPayload,
  SuggestPayload,
  InventoryKpis,
  InventoryTrendPoint,
} from "@/types";

// DFU Analysis panels
import { SelectorPanel } from "./dfu-analysis/SelectorPanel";
import { ModelKpiSection } from "./dfu-analysis/ModelKpiSection";
import { SkuShapPanel } from "./dfu-analysis/DfuShapPanel";

// Unified chart (demand + supply in one view)
import { UnifiedChartPanel, loadDefaultMeasures, buildInitialVisibleSeries } from "./item-analysis/UnifiedChartPanel";

// UX-1: deep-state breadcrumbs.
import { Breadcrumbs } from "@/components/Breadcrumbs";

// ---------------------------------------------------------------------------
// Panel toggle defaults
// ---------------------------------------------------------------------------
const PANEL_DEFAULTS: Record<string, boolean> = {
  overlay: true,
  shap: true,
  forecastKpis: true,
  dqCorrections: false,
};

const DEMAND_PANELS = [
  { key: "overlay", label: "Chart" },
  { key: "shap", label: "SHAP" },
  { key: "forecastKpis", label: "Forecast KPIs" },
  { key: "dqCorrections", label: "DQ Corrections" },
] as const;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function ItemAnalysisTab() {
  const { panels, toggle, set: setPanel, setAll, allOn } = usePanelToggles("ds:itemAnalysis:panels", PANEL_DEFAULTS);

  // ========================================================================
  // DFU Analysis state — single DFU mode only
  // ========================================================================
  const [skuItem, setSkuItem] = useState("");
  const [skuLocation, setSkuLocation] = useState("");
  const [skuPoints, setSkuPoints] = useState(36);
  const [skuKpiMonths, setSkuKpiMonths] = useState(12);
  const [skuData, setSkuData] = useState<SkuAnalysisPayload | null>(null);
  const [skuVisibleSeries, setSkuVisibleSeries] = useState<Set<string>>(
    () => loadDefaultMeasures(),
  );
  const [skuTimeStart, setSkuTimeStart] = useState("");
  const [skuTimeEnd, setSkuTimeEnd] = useState("");
  const [skuDefaultStart, setSkuDefaultStart] = useState("");
  const [skuLoading, setSkuLoading] = useState(false);
  const [skuAutoSampled, setSkuAutoSampled] = useState(false);
  const [skuItemSuggestions, setSkuItemSuggestions] = useState<string[]>([]);
  const [skuLocationSuggestions, setSkuLocationSuggestions] = useState<string[]>([]);
  const [prodForecastData, setProdForecastData] = useState<ProductionForecastPayload | null>(null);
  // SHAP model selection via dropdown (null = none)
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const debouncedSkuItem = useDebounce(skuItem, 500);
  const debouncedSkuLocation = useDebounce(skuLocation, 500);

  // Deep-link: consume ?item=…&loc=… URL params on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const urlItem = params.get("item");
    const urlLoc = params.get("loc");
    if (urlItem) {
      setSkuItem(urlItem);
      if (urlLoc) setSkuLocation(urlLoc);
      setSkuAutoSampled(true);
      // Auto-enable DQ Corrections panel when navigating from DQ tab
      if (params.get("dqCorrections") === "1") {
        setPanel("dqCorrections", true);
      }
      // Clean up URL params so they don't persist on refresh
      params.delete("item");
      params.delete("loc");
      params.delete("dqCorrections");
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
    if (globalFilters.item.length === 1) setSkuItem(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setSkuLocation(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  // ========================================================================
  // DFU effects
  // ========================================================================

  // Auto-sample on first visit
  useEffect(() => {
    if (skuAutoSampled) return;
    if (skuItem.trim() || skuLocation.trim()) { setSkuAutoSampled(true); return; }
    let cancelled = false;
    async function loadSample() {
      try {
        const res = await fetch("/domains/sales/sample-pair");
        if (!res.ok) throw new Error("HTTP error");
        const payload = (await res.json()) as SamplePairPayload;
        if (!cancelled) {
          if (payload.item) setSkuItem(String(payload.item));
          if (payload.location) setSkuLocation(String(payload.location));
        }
      } catch { /* non-blocking */ } finally {
        if (!cancelled) setSkuAutoSampled(true);
      }
    }
    loadSample();
    return () => { cancelled = true; };
  }, [skuAutoSampled, skuItem, skuLocation]);

  // Fetch DFU analysis data (always item_location mode)
  const anyDemandPanelOn = panels.overlay || panels.shap || panels.forecastKpis;
  useEffect(() => {
    if (!anyDemandPanelOn) return;
    if (!debouncedSkuItem.trim() || !debouncedSkuLocation.trim()) return;
    setSkuData(null);
    let cancelled = false;
    async function loadAnalysis() {
      setSkuLoading(true);
      try {
        const params = new URLSearchParams({ mode: "item_location", item: debouncedSkuItem.trim(), location: debouncedSkuLocation.trim(), points: String(skuPoints) });
        const res = await fetch(`/sku/analysis?${params}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = (await res.json()) as SkuAnalysisPayload;
        if (!cancelled) {
          setSkuData(payload);
          setSelectedModel(null);
          setSkuVisibleSeries(buildInitialVisibleSeries(payload.models));
          const fcKeys = payload.models.map((m) => `forecast_${m}`);
          let smartStart = "";
          for (const pt of payload.series) { if (fcKeys.some((k) => k in pt)) { smartStart = String(pt.month); break; } }
          setSkuDefaultStart(smartStart);
          setSkuTimeStart(smartStart);
          setSkuTimeEnd("");
        }
      } catch { if (!cancelled) setSkuData(null); }
      finally { if (!cancelled) setSkuLoading(false); }
    }
    loadAnalysis();
    return () => { cancelled = true; };
  }, [debouncedSkuItem, debouncedSkuLocation, skuPoints, anyDemandPanelOn]);

  // Fetch production forecast (future months)
  useEffect(() => {
    if (!debouncedSkuItem.trim() || !debouncedSkuLocation.trim()) { setProdForecastData(null); return; }
    let cancelled = false;
    fetchProductionForecast({ item_id: debouncedSkuItem.trim(), loc: debouncedSkuLocation.trim() })
      .then((payload) => { if (!cancelled) setProdForecastData(payload); })
      .catch(() => { if (!cancelled) setProdForecastData(null); });
    return () => { cancelled = true; };
  }, [debouncedSkuItem, debouncedSkuLocation]);

  // Fetch staging forecasts (all algorithm lines, pre-promotion)
  const [stagingForecastData, setStagingForecastData] = useState<StagingForecastsPayload | null>(null);
  useEffect(() => {
    if (!debouncedSkuItem.trim() || !debouncedSkuLocation.trim()) { setStagingForecastData(null); return; }
    let cancelled = false;
    fetchStagingForecasts({ item_id: debouncedSkuItem.trim(), loc: debouncedSkuLocation.trim() })
      .then((payload) => { if (!cancelled) setStagingForecastData(payload); })
      .catch(() => { if (!cancelled) setStagingForecastData(null); });
    return () => { cancelled = true; };
  }, [debouncedSkuItem, debouncedSkuLocation]);

  // Item typeahead
  useEffect(() => {
    if (!skuItem.trim()) { setSkuItemSuggestions([]); return; }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({ field: "item_id", q: skuItem.trim(), limit: "12" });
        if (debouncedSkuLocation.trim()) params.set("filters", JSON.stringify({ loc: `=${debouncedSkuLocation.trim()}` }));
        const res = await fetch(`/domains/sales/suggest?${params}`);
        if (!res.ok) throw new Error("HTTP error");
        const payload = (await res.json()) as SuggestPayload;
        if (!cancelled) setSkuItemSuggestions(payload.values || []);
      } catch { if (!cancelled) setSkuItemSuggestions([]); }
    }, 180);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [skuItem, debouncedSkuLocation]);

  // Location typeahead
  useEffect(() => {
    if (!skuLocation.trim()) { setSkuLocationSuggestions([]); return; }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({ field: "loc", q: skuLocation.trim(), limit: "12" });
        if (debouncedSkuItem.trim()) params.set("filters", JSON.stringify({ item_id: `=${debouncedSkuItem.trim()}` }));
        const res = await fetch(`/domains/sales/suggest?${params}`);
        if (!res.ok) throw new Error("HTTP error");
        const payload = (await res.json()) as SuggestPayload;
        if (!cancelled) setSkuLocationSuggestions(payload.values || []);
      } catch { if (!cancelled) setSkuLocationSuggestions([]); }
    }, 180);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [skuLocation, debouncedSkuItem]);

  // ========================================================================
  // DFU computed values
  // ========================================================================
  const skuKpis = useMemo<Record<string, SkuAnalysisKpis>>(() => {
    if (!skuData?.model_monthly) return {};
    const result: Record<string, SkuAnalysisKpis> = {};
    for (const [modelId, rows] of Object.entries(skuData.model_monthly)) {
      const window = rows.slice(0, skuKpiMonths);
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
  }, [skuData, skuKpiMonths]);

  const skuMonths = useMemo(() => (!skuData?.series.length ? [] : skuData.series.map((p) => String(p.month))), [skuData]);

  const skuFilteredSeries = useMemo(() => {
    if (!skuData?.series.length) return [];
    const start = skuTimeStart || skuMonths[0];
    const end = skuTimeEnd || skuMonths[skuMonths.length - 1];
    return skuData.series.filter((p) => { const m = String(p.month); return m >= start && m <= end; });
  }, [skuData, skuMonths, skuTimeStart, skuTimeEnd]);

  const mergedFilteredSeries = useMemo(() => {
    const lastHistMonth = skuMonths[skuMonths.length - 1] ?? "";

    // Start with the filtered demand series
    let result = skuFilteredSeries as Record<string, unknown>[];

    // Build production forecast month map
    const prodMap = new Map<string, number | null>();
    if (prodForecastData?.forecasts.length) {
      for (const pt of prodForecastData.forecasts) {
        prodMap.set(pt.forecast_month, pt.forecast_qty);
      }
    }

    // Build staging forecast month maps: staging_{model_id} → month → qty
    const stagingMaps = new Map<string, Map<string, number | null>>();
    if (stagingForecastData?.models) {
      for (const [modelId, points] of Object.entries(stagingForecastData.models)) {
        const key = `staging_${modelId}`;
        const monthMap = new Map<string, number | null>();
        for (const pt of points) {
          monthMap.set(pt.forecast_month, pt.forecast_qty);
        }
        stagingMaps.set(key, monthMap);
      }
    }

    // Merge production + staging into existing chart points
    result = result.map((pt) => {
      const m = String(pt.month);
      const extras: Record<string, unknown> = {};
      if (prodMap.has(m)) extras.production_forecast = prodMap.get(m);
      for (const [key, monthMap] of stagingMaps) {
        if (monthMap.has(m)) extras[key] = monthMap.get(m);
      }
      return Object.keys(extras).length > 0 ? { ...pt, ...extras } : pt;
    });

    // Collect all future months from production + staging that are beyond the last historical month
    const futureMonthSet = new Set<string>();
    for (const month of prodMap.keys()) {
      if (month > lastHistMonth) futureMonthSet.add(month);
    }
    for (const monthMap of stagingMaps.values()) {
      for (const month of monthMap.keys()) {
        if (month > lastHistMonth) futureMonthSet.add(month);
      }
    }

    // Create future points with all available forecast values
    const futureMonths = Array.from(futureMonthSet).sort();
    const futurePts = futureMonths.map((month) => {
      const pt: Record<string, unknown> = { month };
      if (prodMap.has(month)) pt.production_forecast = prodMap.get(month);
      for (const [key, monthMap] of stagingMaps) {
        if (monthMap.has(month)) pt[key] = monthMap.get(month);
      }
      return pt;
    });

    return [...result, ...futurePts] as Record<string, unknown>[];
  }, [skuFilteredSeries, skuMonths, prodForecastData, stagingForecastData]);

  // Available models for the SHAP dropdown
  const shapModelOptions = useMemo(() => {
    if (!skuData?.models.length) return [];
    return skuData.models;
  }, [skuData]);

  function handleReset() {
    setSkuData(null);
    setSkuAutoSampled(false);
    setSkuItem("");
    setSkuLocation("");
    setSelectedModel(null);
  }

  const emptyMessage =
    !skuItem.trim() || !skuLocation.trim()
      ? "Enter both item and location to view analysis."
      : "No data available for the selected filters.";

  // ========================================================================
  // Inventory queries — DFU-level data for attributes
  // ========================================================================
  const invMonths = 12;
  const invKpiParams = useMemo(
    () => ({ item: debouncedSkuItem, location: debouncedSkuLocation, months: invMonths }),
    [debouncedSkuItem, debouncedSkuLocation],
  );
  const invTrendParams = useMemo(
    () => ({ item: debouncedSkuItem, location: debouncedSkuLocation, months: invMonths }),
    [debouncedSkuItem, debouncedSkuLocation],
  );
  const invPositionParams = useMemo(
    () => ({ item: debouncedSkuItem, location: debouncedSkuLocation, limit: 1, offset: 0, sort_by: "snapshot_date" as const, sort_dir: "desc" as const }),
    [debouncedSkuItem, debouncedSkuLocation],
  );

  const hasDfu = !!(debouncedSkuItem.trim() && debouncedSkuLocation.trim());

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
    () => ({ item: debouncedSkuItem, location: debouncedSkuLocation, limit: 1 }),
    [debouncedSkuItem, debouncedSkuLocation],
  );
  const { data: ltProfilePayload } = useQuery({
    queryKey: queryKeys.ltProfile(ltProfileParams),
    queryFn: () => fetchLtProfile(ltProfileParams),
    staleTime: STALE.FIVE_MIN,
    enabled: hasDfu,
  });
  const ltProfileRow = ltProfilePayload?.rows?.[0] ?? null;

  // DQ corrections query
  const { data: correctionsPayload } = useQuery({
    queryKey: correctionKeys.byItem(debouncedSkuItem, debouncedSkuLocation),
    queryFn: () => fetchCorrectionsByItem(debouncedSkuItem.trim(), debouncedSkuLocation.trim()),
    staleTime: STALE.FIVE_MIN,
    enabled: hasDfu && panels.dqCorrections,
  });
  const corrections = correctionsPayload?.corrections ?? [];

  // ========================================================================
  // Render
  // ========================================================================
  // UX-1: breadcrumb trail renders only when a DFU is selected.
  const breadcrumbItems = skuItem
    ? [
        { label: "Item Analysis", onClick: () => { setSkuItem(""); setSkuLocation(""); } },
        { label: `Item ${skuItem}`, ...(skuLocation ? { onClick: () => setSkuLocation("") } : {}) },
        ...(skuLocation ? [{ label: skuLocation }] : []),
      ]
    : [];

  return (
    <section className="mt-4 space-y-4">
      {breadcrumbItems.length > 0 && (
        <Breadcrumbs items={breadcrumbItems} />
      )}
      {/* ---- Selector (always visible) ---- */}
      <Card className="animate-fade-in">
        <SelectorPanel
          skuItem={skuItem} setSkuItem={setSkuItem}
          skuLocation={skuLocation} setSkuLocation={setSkuLocation}
          skuPoints={skuPoints} setSkuPoints={setSkuPoints}
          skuItemSuggestions={skuItemSuggestions}
          skuLocationSuggestions={skuLocationSuggestions}
          skuData={skuData}
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
            {skuData && skuData.series.length > 0 ? (
              <>
                {panels.overlay && (
                  <UnifiedChartPanel
                    skuData={skuData}
                    skuFilteredSeries={mergedFilteredSeries}
                    skuMonths={skuMonths}
                    skuTimeStart={skuTimeStart} setSkuTimeStart={setSkuTimeStart}
                    skuTimeEnd={skuTimeEnd} setSkuTimeEnd={setSkuTimeEnd}
                    skuDefaultStart={skuDefaultStart}
                    skuVisibleSeries={skuVisibleSeries}
                    setSkuVisibleSeries={setSkuVisibleSeries}
                    prodForecastData={prodForecastData}
                    stagingForecastData={stagingForecastData}
                    selectedModel={selectedModel}
                    onModelSelect={setSelectedModel}
                    trendData={trendData}
                    trendParams={trendParams2}
                    corrections={corrections}
                    showCorrections={panels.dqCorrections}
                  />
                )}
                {panels.shap && selectedModel && (
                  <SkuShapPanel
                    selectedModel={selectedModel}
                    itemNo={debouncedSkuItem}
                    loc={debouncedSkuLocation}
                    skuMode="item_location"
                    visibleMonths={mergedFilteredSeries.map((p) => String(p.month))}
                  />
                )}
                {panels.forecastKpis && (
                  <ModelKpiSection
                    skuData={skuData}
                    skuKpis={skuKpis}
                    skuKpiMonths={skuKpiMonths}
                    setSkuKpiMonths={setSkuKpiMonths}
                    skuVisibleSeries={skuVisibleSeries}
                  />
                )}
              </>
            ) : skuLoading ? (
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
