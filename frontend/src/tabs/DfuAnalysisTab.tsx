import { useEffect, useMemo, useRef, useState } from "react";

import { Card, CardContent } from "@/components/ui/card";
import { LoadingElement } from "@/components/LoadingElement";

import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { useDebounce } from "@/hooks/useDebounce";
import { fetchSeasonalityProfileNames } from "@/api/queries";
import { fetchProductionForecast } from "@/api/queries/production-forecast";
import type { ProductionForecastPayload } from "@/api/queries/production-forecast";
import type {
  SkuAnalysisMode,
  SkuAnalysisKpis,
  SkuAnalysisPayload,
  SamplePairPayload,
  SuggestPayload,
} from "@/types";

import { SelectorPanel } from "./dfu-analysis/SelectorPanel";
import { OverlayChartPanel } from "./dfu-analysis/OverlayChartPanel";
import { ModelKpiSection } from "./dfu-analysis/ModelKpiSection";
import { SkuShapPanel } from "./dfu-analysis/DfuShapPanel";

export function SkuAnalysisTab() {
  // ---- state ----
  const [skuMode, setSkuMode] = useState<SkuAnalysisMode>("item_location");
  const [skuItem, setSkuItem] = useState("");
  const [skuLocation, setSkuLocation] = useState("");
  const [skuPoints, setSkuPoints] = useState(36);
  const [skuKpiMonths, setSkuKpiMonths] = useState(12);
  const [skuData, setSkuData] = useState<SkuAnalysisPayload | null>(null);
  const [skuVisibleSeries, setSkuVisibleSeries] = useState<Set<string>>(
    new Set(["tothist_dmd", "qty_shipped", "qty_ordered"]),
  );
  const [skuTimeStart, setSkuTimeStart] = useState("");
  const [skuTimeEnd, setSkuTimeEnd] = useState("");
  const [skuDefaultStart, setSkuDefaultStart] = useState("");
  const [skuLoading, setSkuLoading] = useState(false);
  const [skuAutoSampled, setSkuAutoSampled] = useState(false);
  const [skuItemSuggestions, setSkuItemSuggestions] = useState<string[]>([]);
  const [skuLocationSuggestions, setSkuLocationSuggestions] = useState<string[]>([]);
  const [seasonalityProfile, setSeasonalityProfile] = useState("");
  const [seasonalityProfiles, setSeasonalityProfiles] = useState<string[]>([]);
  const [prodForecastData, setProdForecastData] = useState<ProductionForecastPayload | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const { filters: globalFilters } = useGlobalFilterContext();
  const debouncedSkuItem = useDebounce(skuItem, 500);
  const debouncedSkuLocation = useDebounce(skuLocation, 500);

  // ---- sync global item/location filter into local inputs (one-time) ----
  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setSkuItem(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setSkuLocation(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  // ---- fetch seasonality profile names once (Feature 32) ----
  useEffect(() => {
    let cancelled = false;
    fetchSeasonalityProfileNames()
      .then((profiles) => { if (!cancelled) setSeasonalityProfiles(profiles); })
      .catch(() => { /* non-blocking */ });
    return () => { cancelled = true; };
  }, []);

  // ---- auto-sample on first visit ----
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

  // ---- fetch DFU analysis data ----
  useEffect(() => {
    const needsItem = skuMode !== "all_items_at_location";
    const needsLoc = skuMode !== "item_at_all_locations";
    if (needsItem && !debouncedSkuItem.trim()) return;
    if (needsLoc && !debouncedSkuLocation.trim()) return;
    setSkuData(null);
    let cancelled = false;
    async function loadAnalysis() {
      setSkuLoading(true);
      try {
        const params = new URLSearchParams({ mode: skuMode, item: debouncedSkuItem.trim(), location: debouncedSkuLocation.trim(), points: String(skuPoints) });
        if (seasonalityProfile) params.set("seasonality_profile", seasonalityProfile);
        const res = await fetch(`/sku/analysis?${params}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = (await res.json()) as SkuAnalysisPayload;
        if (!cancelled) {
          setSkuData(payload);
          setSelectedModel(null);
          const allKeys = new Set(["tothist_dmd", "qty_shipped", "qty_ordered", "production_forecast", ...payload.models.map((m) => `forecast_${m}`)]);
          setSkuVisibleSeries(allKeys);
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
  }, [skuMode, debouncedSkuItem, debouncedSkuLocation, skuPoints, seasonalityProfile]);

  // ---- fetch production forecast (future months) ----
  useEffect(() => {
    if (skuMode !== "item_location") { setProdForecastData(null); return; }
    if (!debouncedSkuItem.trim() || !debouncedSkuLocation.trim()) { setProdForecastData(null); return; }
    let cancelled = false;
    fetchProductionForecast({ item_id: debouncedSkuItem.trim(), loc: debouncedSkuLocation.trim() })
      .then((payload) => { if (!cancelled) setProdForecastData(payload); })
      .catch(() => { if (!cancelled) setProdForecastData(null); });
    return () => { cancelled = true; };
  }, [skuMode, debouncedSkuItem, debouncedSkuLocation]);

  // ---- item typeahead suggestions ----
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

  // ---- location typeahead suggestions ----
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

  // ---- computed: KPIs per model ----
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

  // Merge production forecast (future months) into filtered series
  const mergedFilteredSeries = useMemo(() => {
    if (!prodForecastData?.forecasts.length) return skuFilteredSeries as Record<string, unknown>[];
    const lastHistMonth = skuMonths[skuMonths.length - 1] ?? "";
    const prodMap = new Map<string, number | null>(
      prodForecastData.forecasts.map((pt) => [pt.forecast_month, pt.forecast_qty])
    );
    // Add production_forecast field to overlapping historical points
    const enhanced = skuFilteredSeries.map((pt) => {
      const m = String(pt.month);
      return prodMap.has(m) ? { ...pt, production_forecast: prodMap.get(m) } : pt;
    });
    // Append pure-future months not in historical series
    const futurePts = prodForecastData.forecasts
      .filter((pt) => pt.forecast_month > lastHistMonth)
      .sort((a, b) => a.forecast_month.localeCompare(b.forecast_month))
      .map((pt) => ({ month: pt.forecast_month, production_forecast: pt.forecast_qty }));
    return [...enhanced, ...futurePts] as Record<string, unknown>[];
  }, [skuFilteredSeries, skuMonths, prodForecastData]);

  function handleReset() {
    setSkuData(null);
    setSkuAutoSampled(false);
    setSkuItem("");
    setSkuLocation("");
  }

  const emptyMessage =
    skuMode === "item_location" && (!skuItem.trim() || !skuLocation.trim())
      ? "Enter both item and location to view DFU analysis."
      : skuMode === "all_items_at_location" && !skuLocation.trim()
        ? "Enter a location to view aggregated analysis."
        : skuMode === "item_at_all_locations" && !skuItem.trim()
          ? "Enter an item to view aggregated analysis."
          : "No data available for the selected filters.";

  return (
    <section className="mt-4 grid gap-4 [&>*]:min-w-0 xl:grid-cols-1">
      <Card className="animate-fade-in">
        <SelectorPanel
          skuMode={skuMode} setSkuMode={setSkuMode}
          skuItem={skuItem} setSkuItem={setSkuItem}
          skuLocation={skuLocation} setSkuLocation={setSkuLocation}
          skuPoints={skuPoints} setSkuPoints={setSkuPoints}
          skuKpiMonths={skuKpiMonths} setSkuKpiMonths={setSkuKpiMonths}
          skuItemSuggestions={skuItemSuggestions}
          skuLocationSuggestions={skuLocationSuggestions}
          seasonalityProfile={seasonalityProfile} setSeasonalityProfile={setSeasonalityProfile}
          seasonalityProfiles={seasonalityProfiles}
          skuData={skuData}
          onReset={handleReset}
        />
        <CardContent className="space-y-4">
          {skuData && skuData.series.length > 0 ? (
            <>
              <OverlayChartPanel
                skuData={skuData}
                skuFilteredSeries={mergedFilteredSeries}
                skuMonths={skuMonths}
                skuTimeStart={skuTimeStart} setSkuTimeStart={setSkuTimeStart}
                skuTimeEnd={skuTimeEnd} setSkuTimeEnd={setSkuTimeEnd}
                skuDefaultStart={skuDefaultStart}
                skuVisibleSeries={skuVisibleSeries}
                setSkuVisibleSeries={setSkuVisibleSeries}
                prodForecastData={prodForecastData}
                selectedModel={selectedModel}
                onModelSelect={setSelectedModel}
              />
              {skuMode === "item_location" && (
                <SkuShapPanel
                  selectedModel={selectedModel}
                  itemNo={debouncedSkuItem}
                  loc={debouncedSkuLocation}
                  skuMode={skuMode}
                  visibleMonths={mergedFilteredSeries.map((p) => String(p.month))}
                />
              )}
              <ModelKpiSection
                skuData={skuData}
                skuKpis={skuKpis}
                skuKpiMonths={skuKpiMonths}
                skuVisibleSeries={skuVisibleSeries}
              />
            </>
          ) : skuLoading ? (
            <div className="flex h-[320px] items-center justify-center">
              <LoadingElement message="Fetching DFU analysis..." />
            </div>
          ) : (
            <div className="flex h-[320px] items-center justify-center text-sm text-muted-foreground">
              {emptyMessage}
            </div>
          )}
        </CardContent>
      </Card>
    </section>
  );
}
