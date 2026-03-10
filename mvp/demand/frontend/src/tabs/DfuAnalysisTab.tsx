import { useEffect, useMemo, useRef, useState } from "react";

import { Card, CardContent } from "@/components/ui/card";
import { LoadingElement } from "@/components/LoadingElement";

import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { useDebounce } from "@/hooks/useDebounce";
import { fetchSeasonalityProfileNames } from "@/api/queries";
import { fetchProductionForecast } from "@/api/queries/production-forecast";
import type { ProductionForecastPayload } from "@/api/queries/production-forecast";
import { ELEMENT_CONFIG } from "@/constants/elements";
import type {
  DfuAnalysisMode,
  DfuAnalysisKpis,
  DfuAnalysisPayload,
  SamplePairPayload,
  SuggestPayload,
} from "@/types";

import { SelectorPanel } from "./dfu-analysis/SelectorPanel";
import { OverlayChartPanel } from "./dfu-analysis/OverlayChartPanel";
import { ModelKpiSection } from "./dfu-analysis/ModelKpiSection";
import { DfuShapPanel } from "./dfu-analysis/DfuShapPanel";

export function DfuAnalysisTab() {
  // ---- state ----
  const [dfuMode, setDfuMode] = useState<DfuAnalysisMode>("item_location");
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
  const [seasonalityProfile, setSeasonalityProfile] = useState("");
  const [seasonalityProfiles, setSeasonalityProfiles] = useState<string[]>([]);
  const [prodForecastData, setProdForecastData] = useState<ProductionForecastPayload | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const { filters: globalFilters } = useGlobalFilterContext();
  const debouncedDfuItem = useDebounce(dfuItem, 500);
  const debouncedDfuLocation = useDebounce(dfuLocation, 500);

  // ---- sync global item/location filter into local inputs (one-time) ----
  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setDfuItem(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setDfuLocation(globalFilters.location[0]);
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

  // ---- fetch DFU analysis data ----
  useEffect(() => {
    const needsItem = dfuMode !== "all_items_at_location";
    const needsLoc = dfuMode !== "item_at_all_locations";
    if (needsItem && !debouncedDfuItem.trim()) return;
    if (needsLoc && !debouncedDfuLocation.trim()) return;
    setDfuData(null);
    let cancelled = false;
    async function loadAnalysis() {
      setDfuLoading(true);
      try {
        const params = new URLSearchParams({ mode: dfuMode, item: debouncedDfuItem.trim(), location: debouncedDfuLocation.trim(), points: String(dfuPoints) });
        if (seasonalityProfile) params.set("seasonality_profile", seasonalityProfile);
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
  }, [dfuMode, debouncedDfuItem, debouncedDfuLocation, dfuPoints, seasonalityProfile]);

  // ---- fetch production forecast (future months) ----
  useEffect(() => {
    if (dfuMode !== "item_location") { setProdForecastData(null); return; }
    if (!debouncedDfuItem.trim() || !debouncedDfuLocation.trim()) { setProdForecastData(null); return; }
    let cancelled = false;
    fetchProductionForecast({ item_no: debouncedDfuItem.trim(), loc: debouncedDfuLocation.trim() })
      .then((payload) => { if (!cancelled) setProdForecastData(payload); })
      .catch(() => { if (!cancelled) setProdForecastData(null); });
    return () => { cancelled = true; };
  }, [dfuMode, debouncedDfuItem, debouncedDfuLocation]);

  // ---- item typeahead suggestions ----
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

  // ---- location typeahead suggestions ----
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

  // ---- computed: KPIs per model ----
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

  // Merge production forecast (future months) into filtered series
  const mergedFilteredSeries = useMemo(() => {
    if (!prodForecastData?.forecasts.length) return dfuFilteredSeries as Record<string, unknown>[];
    const lastHistMonth = dfuMonths[dfuMonths.length - 1] ?? "";
    const prodMap = new Map<string, number | null>(
      prodForecastData.forecasts.map((pt) => [pt.forecast_month, pt.forecast_qty])
    );
    // Add production_forecast field to overlapping historical points
    const enhanced = dfuFilteredSeries.map((pt) => {
      const m = String(pt.month);
      return prodMap.has(m) ? { ...pt, production_forecast: prodMap.get(m) } : pt;
    });
    // Append pure-future months not in historical series
    const futurePts = prodForecastData.forecasts
      .filter((pt) => pt.forecast_month > lastHistMonth)
      .sort((a, b) => a.forecast_month.localeCompare(b.forecast_month))
      .map((pt) => ({ month: pt.forecast_month, production_forecast: pt.forecast_qty }));
    return [...enhanced, ...futurePts] as Record<string, unknown>[];
  }, [dfuFilteredSeries, dfuMonths, prodForecastData]);

  function handleReset() {
    setDfuData(null);
    setDfuAutoSampled(false);
    setDfuItem("");
    setDfuLocation("");
  }

  const emptyMessage =
    dfuMode === "item_location" && (!dfuItem.trim() || !dfuLocation.trim())
      ? "Enter both item and location to view DFU analysis."
      : dfuMode === "all_items_at_location" && !dfuLocation.trim()
        ? "Enter a location to view aggregated analysis."
        : dfuMode === "item_at_all_locations" && !dfuItem.trim()
          ? "Enter an item to view aggregated analysis."
          : "No data available for the selected filters.";

  return (
    <section className="mt-4 grid gap-4 [&>*]:min-w-0 xl:grid-cols-1">
      <Card className="animate-fade-in">
        <SelectorPanel
          dfuMode={dfuMode} setDfuMode={setDfuMode}
          dfuItem={dfuItem} setDfuItem={setDfuItem}
          dfuLocation={dfuLocation} setDfuLocation={setDfuLocation}
          dfuPoints={dfuPoints} setDfuPoints={setDfuPoints}
          dfuKpiMonths={dfuKpiMonths} setDfuKpiMonths={setDfuKpiMonths}
          dfuItemSuggestions={dfuItemSuggestions}
          dfuLocationSuggestions={dfuLocationSuggestions}
          seasonalityProfile={seasonalityProfile} setSeasonalityProfile={setSeasonalityProfile}
          seasonalityProfiles={seasonalityProfiles}
          dfuData={dfuData}
          onReset={handleReset}
        />
        <CardContent className="space-y-4">
          {dfuData && dfuData.series.length > 0 ? (
            <>
              <OverlayChartPanel
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
              />
              {dfuMode === "item_location" && (
                <DfuShapPanel
                  selectedModel={selectedModel}
                  itemNo={debouncedDfuItem}
                  loc={debouncedDfuLocation}
                  dfuMode={dfuMode}
                  visibleMonths={mergedFilteredSeries.map((p) => String(p.month))}
                />
              )}
              <ModelKpiSection
                dfuData={dfuData}
                dfuKpis={dfuKpis}
                dfuKpiMonths={dfuKpiMonths}
                dfuVisibleSeries={dfuVisibleSeries}
              />
            </>
          ) : dfuLoading ? (
            <div className="flex h-[320px] items-center justify-center">
              <LoadingElement config={ELEMENT_CONFIG.dfuAnalysis} message="Fetching DFU analysis..." />
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
