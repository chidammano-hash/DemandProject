import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { Card, CardContent } from "@/components/ui/card";
import { LoadingElement } from "@/components/LoadingElement";

import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { usePublishActiveSku } from "@/context/ActiveSkuContext";
import { useDebounce } from "@/hooks/useDebounce";
import { useItemForecastOverlays } from "@/hooks/useItemForecastOverlays";
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
  fetchSamplePair,
  fetchDomainSuggest,
  fetchSkuAnalysis,
} from "@/api/queries";
import type {
  SkuAnalysisKpis,
  SkuAnalysisPayload,
  InventoryKpis,
  InventoryTrendPoint,
} from "@/types";

// DFU Analysis panels
import { SelectorPanel } from "./dfu-analysis/SelectorPanel";
import { ModelKpiSection } from "./dfu-analysis/ModelKpiSection";
import { SkuShapPanel } from "./dfu-analysis/DfuShapPanel";

// Unified chart (demand + supply + customer blend in one view)
import { CustomerBlendUnifiedChartPanel } from "./item-analysis/CustomerBlendUnifiedChartPanel";
import { loadDefaultMeasures, buildInitialVisibleSeries } from "./item-analysis/measures";
import { itemBreadcrumbLabel } from "./item-analysis/breadcrumb";
import { ItemAnalysisToolbar } from "./item-analysis/ItemAnalysisToolbar";

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
  aiChampion: true,
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function ItemAnalysisTab() {
  const {
    panels,
    toggle,
    set: setPanel,
    setAll,
    allOn,
  } = usePanelToggles("ds:itemAnalysis:panels", PANEL_DEFAULTS);

  // ========================================================================
  // DFU Analysis state — single DFU mode only
  // ========================================================================
  const [skuItem, setSkuItem] = useState("");
  const [skuLocation, setSkuLocation] = useState("");
  const [skuPoints, setSkuPoints] = useState(36);
  const [skuKpiMonths, setSkuKpiMonths] = useState(12);
  const [skuData, setSkuData] = useState<SkuAnalysisPayload | null>(null);
  const [skuVisibleSeries, setSkuVisibleSeries] = useState<Set<string>>(() =>
    loadDefaultMeasures()
  );
  const [skuTimeStart, setSkuTimeStart] = useState("");
  const [skuTimeEnd, setSkuTimeEnd] = useState("");
  const [skuDefaultStart, setSkuDefaultStart] = useState("");
  const [skuLoading, setSkuLoading] = useState(false);
  const [skuError, setSkuError] = useState<string | null>(null);
  const [skuAutoSampled, setSkuAutoSampled] = useState(false);
  const [skuItemSuggestions, setSkuItemSuggestions] = useState<string[]>([]);
  const [skuLocationSuggestions, setSkuLocationSuggestions] = useState<string[]>([]);
  // SHAP model selection via dropdown (null = none)
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const debouncedSkuItem = useDebounce(skuItem, 500);
  const debouncedSkuLocation = useDebounce(skuLocation, 500);

  // Publish the SKU shown on this page so the global chat assistant inherits its scope.
  usePublishActiveSku(debouncedSkuItem, debouncedSkuLocation);

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
    if (skuItem.trim() || skuLocation.trim()) {
      setSkuAutoSampled(true);
      return;
    }
    let cancelled = false;
    async function loadSample() {
      try {
        const payload = await fetchSamplePair("sales");
        if (!cancelled) {
          if (payload.item) setSkuItem(String(payload.item));
          if (payload.location) setSkuLocation(String(payload.location));
        }
      } catch {
        /* non-blocking: leave inputs empty for manual entry */
      } finally {
        if (!cancelled) setSkuAutoSampled(true);
      }
    }
    loadSample();
    return () => {
      cancelled = true;
    };
  }, [skuAutoSampled, skuItem, skuLocation]);

  // Fetch DFU analysis data (always item_location mode)
  const anyDemandPanelOn = panels.overlay || panels.shap || panels.forecastKpis;
  useEffect(() => {
    if (!anyDemandPanelOn) return;
    if (!debouncedSkuItem.trim() || !debouncedSkuLocation.trim()) return;
    setSkuData(null);
    setSkuError(null);
    let cancelled = false;
    async function loadAnalysis() {
      setSkuLoading(true);
      try {
        const payload = await fetchSkuAnalysis({
          mode: "item_location",
          item: debouncedSkuItem.trim(),
          location: debouncedSkuLocation.trim(),
          points: skuPoints,
        });
        if (!cancelled) {
          setSkuData(payload);
          setSelectedModel(null);
          setSkuVisibleSeries(buildInitialVisibleSeries(payload.models));
          const fcKeys = payload.models.map((m) => `forecast_${m}`);
          let smartStart = "";
          for (const pt of payload.series) {
            if (fcKeys.some((k) => k in pt)) {
              smartStart = String(pt.month);
              break;
            }
          }
          setSkuDefaultStart(smartStart);
          setSkuTimeStart(smartStart);
          setSkuTimeEnd("");
        }
      } catch (err) {
        // U3.1: surface the sanitized error instead of silently leaving a blank chart.
        if (!cancelled) {
          setSkuData(null);
          setSkuError(err instanceof Error ? err.message : "Failed to load analysis");
        }
      } finally {
        if (!cancelled) setSkuLoading(false);
      }
    }
    loadAnalysis();
    return () => {
      cancelled = true;
    };
  }, [debouncedSkuItem, debouncedSkuLocation, skuPoints, anyDemandPanelOn]);

  // Item typeahead
  useEffect(() => {
    if (!skuItem.trim()) {
      setSkuItemSuggestions([]);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const filters = debouncedSkuLocation.trim()
          ? { loc: `=${debouncedSkuLocation.trim()}` }
          : undefined;
        const values = await fetchDomainSuggest("sales", "item_id", skuItem.trim(), filters, 12);
        if (!cancelled) setSkuItemSuggestions(values);
      } catch {
        if (!cancelled) setSkuItemSuggestions([]);
      }
    }, 180);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [skuItem, debouncedSkuLocation]);

  // Location typeahead
  useEffect(() => {
    if (!skuLocation.trim()) {
      setSkuLocationSuggestions([]);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const filters = debouncedSkuItem.trim()
          ? { item_id: `=${debouncedSkuItem.trim()}` }
          : undefined;
        const values = await fetchDomainSuggest("sales", "loc", skuLocation.trim(), filters, 12);
        if (!cancelled) setSkuLocationSuggestions(values);
      } catch {
        if (!cancelled) setSkuLocationSuggestions([]);
      }
    }, 180);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
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
      let sumForecast = 0,
        sumActual = 0,
        sumAbsErr = 0;
      for (const r of window) {
        sumForecast += r.forecast;
        sumActual += r.actual;
        sumAbsErr += Math.abs(r.forecast - r.actual);
      }
      const absActual = Math.abs(sumActual);
      const wape = absActual > 0 ? (100 * sumAbsErr) / absActual : null;
      const accuracy = wape !== null ? 100 - wape : null;
      const bias = absActual > 0 ? sumForecast / sumActual - 1 : null;
      result[modelId] = {
        accuracy_pct: accuracy !== null ? Math.round(accuracy * 10000) / 10000 : null,
        wape: wape !== null ? Math.round(wape * 10000) / 10000 : null,
        bias: bias !== null ? Math.round(bias * 10000) / 10000 : null,
        sum_forecast: sumForecast,
        sum_actual: sumActual,
        months_covered: window.length,
      };
    }
    return result;
  }, [skuData, skuKpiMonths]);

  const skuMonths = useMemo(
    () => (!skuData?.series.length ? [] : skuData.series.map((p) => String(p.month))),
    [skuData]
  );

  const skuFilteredSeries = useMemo(() => {
    if (!skuData?.series.length) return [];
    const start = skuTimeStart || skuMonths[0];
    const end = skuTimeEnd || skuMonths[skuMonths.length - 1];
    return skuData.series.filter((p) => {
      const m = String(p.month);
      return m >= start && m <= end;
    });
  }, [skuData, skuMonths, skuTimeStart, skuTimeEnd]);

  const {
    prodForecastData,
    stagingForecastData,
    candidateForecastData,
    skuFutureMonths,
    mergedFilteredSeries,
    aiChampionFetched,
    hasAiChampion,
    aiChampionLead,
  } = useItemForecastOverlays({
    itemId: debouncedSkuItem,
    locationId: debouncedSkuLocation,
    historyMonths: skuMonths,
    filteredSeries: skuFilteredSeries as Record<string, unknown>[],
    timeEnd: skuTimeEnd,
    aiEnabled: panels.aiChampion,
  });

  // Available models for the SHAP dropdown
  const shapModelOptions = useMemo(() => {
    if (!skuData?.models.length) return [];
    return skuData.models;
  }, [skuData]);

  // A selected item/location can aggregate multiple customer groups. Pass the
  // DFU grain to SHAP only when it is unambiguous; the API returns a useful 422
  // for ambiguous selections instead of silently choosing the first group.
  const shapCustomerGroups = new Set(
    (skuData?.dfu_attributes ?? [])
      .map((attrs) => attrs.customer_group?.trim())
      .filter((group): group is string => Boolean(group))
  );
  const shapCustomerGroup =
    shapCustomerGroups.size === 1 ? shapCustomerGroups.values().next().value : undefined;

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
    [debouncedSkuItem, debouncedSkuLocation]
  );
  const invTrendParams = useMemo(
    () => ({ item: debouncedSkuItem, location: debouncedSkuLocation, months: invMonths }),
    [debouncedSkuItem, debouncedSkuLocation]
  );
  const invPositionParams = useMemo(
    () => ({
      item: debouncedSkuItem,
      location: debouncedSkuLocation,
      limit: 1,
      offset: 0,
      sort_by: "snapshot_date" as const,
      sort_dir: "desc" as const,
    }),
    [debouncedSkuItem, debouncedSkuLocation]
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
    [debouncedSkuItem, debouncedSkuLocation]
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
        {
          label: "Item Analysis",
          onClick: () => {
            setSkuItem("");
            setSkuLocation("");
          },
        },
        {
          label: itemBreadcrumbLabel(skuItem, skuData?.item_desc),
          ...(skuLocation ? { onClick: () => setSkuLocation("") } : {}),
        },
        ...(skuLocation ? [{ label: skuLocation }] : []),
      ]
    : [];

  return (
    <section className="mt-4 space-y-4">
      {breadcrumbItems.length > 0 && <Breadcrumbs items={breadcrumbItems} />}
      {/* ---- Selector (always visible) ---- */}
      <Card className="animate-fade-in">
        <SelectorPanel
          skuItem={skuItem}
          setSkuItem={setSkuItem}
          skuLocation={skuLocation}
          setSkuLocation={setSkuLocation}
          skuPoints={skuPoints}
          setSkuPoints={setSkuPoints}
          skuItemSuggestions={skuItemSuggestions}
          skuLocationSuggestions={skuLocationSuggestions}
          skuData={skuData}
          onReset={handleReset}
          kpiData={kpiData as InventoryKpis | undefined}
          trendParams={trendParams2}
          positionRow={positionRow}
          ltProfile={ltProfileRow}
        />

        <ItemAnalysisToolbar
          panels={panels}
          allOn={allOn}
          onSetAll={setAll}
          onToggle={toggle}
          shapModels={shapModelOptions}
          selectedModel={selectedModel}
          onSelectedModelChange={setSelectedModel}
        />
      </Card>

      {/* ---- Demand panels ---- */}
      {anyDemandPanelOn && (
        <Card>
          <CardContent className="space-y-4 pt-4">
            {skuData && skuData.series.length > 0 ? (
              <>
                {panels.overlay && (
                  <CustomerBlendUnifiedChartPanel
                    skuData={skuData}
                    skuFilteredSeries={mergedFilteredSeries}
                    skuMonths={skuMonths}
                    skuFutureMonths={skuFutureMonths}
                    skuTimeStart={skuTimeStart}
                    setSkuTimeStart={setSkuTimeStart}
                    skuTimeEnd={skuTimeEnd}
                    setSkuTimeEnd={setSkuTimeEnd}
                    skuDefaultStart={skuDefaultStart}
                    skuVisibleSeries={skuVisibleSeries}
                    setSkuVisibleSeries={setSkuVisibleSeries}
                    prodForecastData={prodForecastData}
                    stagingForecastData={stagingForecastData}
                    candidateForecastData={candidateForecastData}
                    selectedModel={selectedModel}
                    onModelSelect={setSelectedModel}
                    trendData={trendData}
                    trendParams={trendParams2}
                    corrections={corrections}
                    showCorrections={panels.dqCorrections}
                    hasAiChampion={hasAiChampion}
                    aiChampionRecCode={aiChampionLead?.recommendation_code ?? null}
                    aiChampionRationale={aiChampionLead?.rationale ?? null}
                  />
                )}
                {panels.shap && selectedModel && (
                  <SkuShapPanel
                    selectedModel={selectedModel}
                    itemNo={debouncedSkuItem}
                    loc={debouncedSkuLocation}
                    customerGroup={shapCustomerGroup}
                    skuMode="item_location"
                    visibleMonths={mergedFilteredSeries.map((p) => String(p.month))}
                  />
                )}
                {/* The SHAP/AI Champion toggles must never be silent no-ops —
                    say why nothing rendered instead of rendering nothing. */}
                {panels.shap && !selectedModel && (
                  <p className="rounded-md border border-dashed px-3 py-2 text-xs text-muted-foreground">
                    {shapModelOptions.length > 0
                      ? "SHAP is on - choose a model in the SHAP dropdown above to see feature contributions."
                      : "No SHAP data is available for this DFU."}
                  </p>
                )}
                {panels.aiChampion && aiChampionFetched && !hasAiChampion && (
                  <p className="rounded-md border border-dashed px-3 py-2 text-xs text-muted-foreground">
                    No saved AI Champion adjustment exists for this DFU yet.
                  </p>
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
            ) : skuError ? (
              <div className="flex h-[320px] flex-col items-center justify-center gap-1 text-sm">
                <span className="font-medium text-red-600">Could not load analysis</span>
                <span className="text-muted-foreground">{skuError}</span>
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
