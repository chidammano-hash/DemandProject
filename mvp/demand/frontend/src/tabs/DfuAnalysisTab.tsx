import { useEffect, useMemo, useRef, useState } from "react";
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
import { ChartColumn, RefreshCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { LoadingElement } from "@/components/LoadingElement";

import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { useDebounce } from "@/hooks/useDebounce";
import {
  DFU_SALES_COLORS,
  dfuModelColor,
} from "@/constants/colors";
import { fetchSeasonalityProfileNames } from "@/api/queries";
import { useChartColors } from "@/hooks/useChartColors";
import { ELEMENT_CONFIG } from "@/constants/elements";
import {
  formatNumber,
  formatPercent,
  formatCompactNumber,
  titleCase,
} from "@/lib/formatters";
import type {
  DfuAnalysisMode,
  DfuAnalysisPayload,
  DfuAnalysisKpis,
  SamplePairPayload,
  SuggestPayload,
} from "@/types";

// ---------------------------------------------------------------------------
// Hoisted constants for Recharts inline props
// ---------------------------------------------------------------------------
const DFU_CHART_MARGIN = { top: 8, right: 16, left: 18, bottom: 8 };
const ACTIVE_DOT_SM = { r: 4 };
const ACTIVE_DOT_MD = { r: 5 };

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
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
  const [dfuLocationSuggestions, setDfuLocationSuggestions] = useState<
    string[]
  >([]);

  // Seasonality profile filter (Feature 32)
  const [seasonalityProfile, setSeasonalityProfile] = useState("");
  const [seasonalityProfiles, setSeasonalityProfiles] = useState<string[]>([]);

  const { filters: globalFilters } = useGlobalFilterContext();
  const { chartColors, trendColors } = useChartColors();

  const debouncedDfuItem = useDebounce(dfuItem, 500);
  const debouncedDfuLocation = useDebounce(dfuLocation, 500);

  // ---- sync global item/location filter into local inputs (one-time when global changes) ----
  const syncedGlobalRef = useRef<string>("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedGlobalRef.current) return;
    syncedGlobalRef.current = key;
    if (globalFilters.item.length === 1) setDfuItem(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setDfuLocation(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  // ---- fetch seasonality profile names once for filter dropdown (Feature 32) ----
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
    if (dfuItem.trim() || dfuLocation.trim()) {
      setDfuAutoSampled(true);
      return;
    }
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
      } catch {
        /* non-blocking */
      } finally {
        if (!cancelled) setDfuAutoSampled(true);
      }
    }
    loadSample();
    return () => {
      cancelled = true;
    };
  }, [dfuAutoSampled, dfuItem, dfuLocation]);

  // ---- fetch DFU analysis data ----
  useEffect(() => {
    const needsItem = dfuMode !== "all_items_at_location";
    const needsLoc = dfuMode !== "item_at_all_locations";
    if (needsItem && !debouncedDfuItem.trim()) return;
    if (needsLoc && !debouncedDfuLocation.trim()) return;

    // Clear stale data so the loading indicator shows during mode switches
    setDfuData(null);

    let cancelled = false;
    async function loadAnalysis() {
      setDfuLoading(true);
      try {
        const params = new URLSearchParams({
          mode: dfuMode,
          item: debouncedDfuItem.trim(),
          location: debouncedDfuLocation.trim(),
          points: String(dfuPoints),
        });
        if (seasonalityProfile) params.set("seasonality_profile", seasonalityProfile);
        const res = await fetch(`/dfu/analysis?${params}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = (await res.json()) as DfuAnalysisPayload;
        if (!cancelled) {
          setDfuData(payload);
          // Select all measures by default
          const allKeys = new Set([
            "tothist_dmd",
            "qty_shipped",
            "qty_ordered",
            ...payload.models.map((m) => `forecast_${m}`),
          ]);
          setDfuVisibleSeries(allKeys);
          // Default "from" to the first month where all measures have data
          const measureKeys = new Set<string>();
          for (const pt of payload.series) {
            for (const k of Object.keys(pt)) {
              if (k !== "month") measureKeys.add(k);
            }
          }
          let smartStart = "";
          if (measureKeys.size > 0 && payload.series.length > 0) {
            const keys = Array.from(measureKeys);
            for (const pt of payload.series) {
              if (keys.every((k) => k in pt)) {
                smartStart = String(pt.month);
                break;
              }
            }
          }
          setDfuDefaultStart(smartStart);
          setDfuTimeStart(smartStart);
          setDfuTimeEnd("");
        }
      } catch {
        if (!cancelled) setDfuData(null);
      } finally {
        if (!cancelled) setDfuLoading(false);
      }
    }
    loadAnalysis();
    return () => {
      cancelled = true;
    };
  }, [dfuMode, debouncedDfuItem, debouncedDfuLocation, dfuPoints, seasonalityProfile]);

  // ---- item typeahead suggestions ----
  useEffect(() => {
    if (!dfuItem.trim()) {
      setDfuItemSuggestions([]);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({
          field: "dmdunit",
          q: dfuItem.trim(),
          limit: "12",
        });
        if (debouncedDfuLocation.trim()) {
          params.set(
            "filters",
            JSON.stringify({ loc: `=${debouncedDfuLocation.trim()}` }),
          );
        }
        const res = await fetch(`/domains/sales/suggest?${params}`);
        if (!res.ok) throw new Error("HTTP error");
        const payload = (await res.json()) as SuggestPayload;
        if (!cancelled) setDfuItemSuggestions(payload.values || []);
      } catch {
        if (!cancelled) setDfuItemSuggestions([]);
      }
    }, 180);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [dfuItem, debouncedDfuLocation]);

  // ---- location typeahead suggestions ----
  useEffect(() => {
    if (!dfuLocation.trim()) {
      setDfuLocationSuggestions([]);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({
          field: "loc",
          q: dfuLocation.trim(),
          limit: "12",
        });
        if (debouncedDfuItem.trim()) {
          params.set(
            "filters",
            JSON.stringify({ dmdunit: `=${debouncedDfuItem.trim()}` }),
          );
        }
        const res = await fetch(`/domains/sales/suggest?${params}`);
        if (!res.ok) throw new Error("HTTP error");
        const payload = (await res.json()) as SuggestPayload;
        if (!cancelled) setDfuLocationSuggestions(payload.values || []);
      } catch {
        if (!cancelled) setDfuLocationSuggestions([]);
      }
    }, 180);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [dfuLocation, debouncedDfuItem]);

  // ---- computed: KPIs per model ----
  const dfuKpis = useMemo<Record<string, DfuAnalysisKpis>>(() => {
    if (!dfuData?.model_monthly) return {};
    const result: Record<string, DfuAnalysisKpis> = {};
    for (const [modelId, rows] of Object.entries(dfuData.model_monthly)) {
      // rows are sorted month desc; take the most recent kpi_months entries
      const window = rows.slice(0, dfuKpiMonths);
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
        accuracy_pct:
          accuracy !== null
            ? Math.round(accuracy * 10000) / 10000
            : null,
        wape: wape !== null ? Math.round(wape * 10000) / 10000 : null,
        bias: bias !== null ? Math.round(bias * 10000) / 10000 : null,
        sum_forecast: sumForecast,
        sum_actual: sumActual,
        months_covered: window.length,
      };
    }
    return result;
  }, [dfuData, dfuKpiMonths]);

  // ---- computed: available months ----
  const dfuMonths = useMemo(() => {
    if (!dfuData?.series.length) return [] as string[];
    return dfuData.series.map((p) => String(p.month));
  }, [dfuData]);

  // ---- computed: time-range-filtered series ----
  const dfuFilteredSeries = useMemo(() => {
    if (!dfuData?.series.length) return [];
    const start = dfuTimeStart || dfuMonths[0];
    const end = dfuTimeEnd || dfuMonths[dfuMonths.length - 1];
    return dfuData.series.filter((p) => {
      const m = String(p.month);
      return m >= start && m <= end;
    });
  }, [dfuData, dfuMonths, dfuTimeStart, dfuTimeEnd]);

  // ---- render ----
  return (
    <section className="mt-4 grid gap-4 [&>*]:min-w-0 xl:grid-cols-1">
      <Card className="animate-fade-in">
        <CardHeader className="space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <CardTitle className="text-base">DFU Analysis</CardTitle>
              <CardDescription>
                {dfuMode === "item_location"
                  ? "Sales + multi-model forecasts for a specific DFU (item @ location)"
                  : dfuMode === "all_items_at_location"
                    ? "Aggregated sales + forecasts across all items at a location"
                    : "Aggregated sales + forecasts for an item across all locations"}
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setDfuData(null);
                setDfuAutoSampled(false);
                setDfuItem("");
                setDfuLocation("");
              }}
            >
              <RefreshCcw className="mr-1 h-4 w-4" /> Reset
            </Button>
          </div>

          {/* Row 1: Analysis scope selector */}
          <div className="grid gap-2 md:grid-cols-3">
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Analysis Scope
              <select
                className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                value={dfuMode}
                onChange={(e) =>
                  setDfuMode(e.target.value as DfuAnalysisMode)
                }
              >
                <option value="item_location">
                  Item @ Location (single DFU)
                </option>
                <option value="all_items_at_location">
                  All Items @ Location
                </option>
                <option value="item_at_all_locations">
                  Item @ All Locations
                </option>
              </select>
            </label>
            <div className="grid grid-cols-2 gap-2">
              <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Points
                <select
                  className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  value={dfuPoints}
                  onChange={(e) => setDfuPoints(Number(e.target.value))}
                >
                  {[12, 24, 36, 48, 60].map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
              <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                KPI Window
                <select
                  className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  value={dfuKpiMonths}
                  onChange={(e) =>
                    setDfuKpiMonths(Number(e.target.value))
                  }
                >
                  {Array.from({ length: 12 }, (_, i) => i + 1).map(
                    (m) => (
                      <option key={m} value={m}>
                        {m} mo
                      </option>
                    ),
                  )}
                </select>
              </label>
            </div>
          </div>

          {/* Row 2: Item + Location + Seasonality Profile filters */}
          <div className="grid gap-2 md:grid-cols-2">
            {dfuMode !== "all_items_at_location" ? (
              <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Item (dmdunit)
                <Input
                  className="h-9"
                  placeholder="Type to search items..."
                  list="dfu-analysis-item-suggest"
                  value={dfuItem}
                  onChange={(e) => setDfuItem(e.target.value)}
                />
                <datalist id="dfu-analysis-item-suggest">
                  {dfuItemSuggestions.map((val) => (
                    <option key={val} value={val} />
                  ))}
                </datalist>
              </label>
            ) : (
              <div className="flex items-end">
                <p className="pb-2 text-xs text-muted-foreground italic">
                  Item: All (aggregated at location level)
                </p>
              </div>
            )}
            {dfuMode !== "item_at_all_locations" ? (
              <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Location (loc)
                <Input
                  className="h-9"
                  placeholder="Type to search locations..."
                  list="dfu-analysis-loc-suggest"
                  value={dfuLocation}
                  onChange={(e) => setDfuLocation(e.target.value)}
                />
                <datalist id="dfu-analysis-loc-suggest">
                  {dfuLocationSuggestions.map((val) => (
                    <option key={val} value={val} />
                  ))}
                </datalist>
              </label>
            ) : (
              <div className="flex items-end">
                <p className="pb-2 text-xs text-muted-foreground italic">
                  Location: All (aggregated at item level)
                </p>
              </div>
            )}
          </div>

          {/* Row 3: Seasonality Profile filter (Feature 32) */}
          {seasonalityProfiles.length > 0 ? (
            <div className="grid gap-2 md:grid-cols-3">
              <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Seasonality Profile
                <select
                  className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  value={seasonalityProfile}
                  onChange={(e) => setSeasonalityProfile(e.target.value)}
                >
                  <option value="">All Profiles</option>
                  {seasonalityProfiles.map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </label>
            </div>
          ) : null}
        </CardHeader>

        <CardContent className="space-y-4">
          {/* DFU Attributes */}
          {dfuData &&
            dfuData.dfu_attributes &&
            dfuData.dfu_attributes.length > 0 && (
              <details className="group rounded-md border border-input bg-background">
                <summary className="cursor-pointer select-none px-3 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground">
                  DFU Attributes ({dfuData.dfu_attributes.length}{" "}
                  {dfuData.dfu_attributes.length === 1
                    ? "record"
                    : "records"}
                  )
                  <span className="ml-1 text-xs text-muted-foreground group-open:hidden">
                    + expand
                  </span>
                </summary>
                <div className="border-t border-input px-3 py-2 space-y-3">
                  {dfuData.dfu_attributes.map((attrs, dfuIdx) => (
                    <div key={dfuIdx}>
                      {dfuData.dfu_attributes.length > 1 && (
                        <p className="mb-1 text-xs font-medium text-foreground">
                          {attrs.dmdunit} / {attrs.dmdgroup} @ {attrs.loc}
                        </p>
                      )}
                      <div className="grid grid-cols-2 gap-x-6 gap-y-0.5 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                        {Object.entries(attrs).map(([key, val]) => (
                          <div
                            key={key}
                            className="flex items-baseline gap-1 text-xs truncate"
                          >
                            <span className="font-medium text-muted-foreground shrink-0">
                              {titleCase(key)}:
                            </span>
                            <span
                              className="text-foreground truncate"
                              title={val ?? "\u2014"}
                            >
                              {val ?? "\u2014"}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </details>
            )}

          {/* Main content: chart + controls, loading, or placeholder */}
          {dfuData && dfuData.series.length > 0 ? (
            <>
              {/* Measure toggles */}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Visible Measures
                  </span>
                  {(() => {
                    const allKeys = new Set([
                      "tothist_dmd",
                      "qty_shipped",
                      "qty_ordered",
                      ...dfuData.models.map((m) => `forecast_${m}`),
                    ]);
                    const allSelected = [...allKeys].every((k) =>
                      dfuVisibleSeries.has(k),
                    );
                    return (
                      <button
                        className="text-xs font-medium text-primary hover:underline"
                        onClick={() =>
                          setDfuVisibleSeries(
                            allSelected ? new Set() : allKeys,
                          )
                        }
                      >
                        {allSelected ? "Deselect All" : "Select All"}
                      </button>
                    );
                  })()}
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1 rounded-md border border-input bg-background p-2">
                  {(
                    [
                      {
                        key: "tothist_dmd",
                        label: "Sale Qty (external)",
                        color: DFU_SALES_COLORS.tothist_dmd,
                      },
                      {
                        key: "qty_shipped",
                        label: "Qty Shipped",
                        color: DFU_SALES_COLORS.qty_shipped,
                      },
                      {
                        key: "qty_ordered",
                        label: "Qty Ordered",
                        color: DFU_SALES_COLORS.qty_ordered,
                      },
                    ] as { key: string; label: string; color: string }[]
                  ).map(({ key, label, color }) => (
                    <label
                      key={key}
                      className="flex items-center gap-2 text-xs font-medium"
                    >
                      <Checkbox
                        checked={dfuVisibleSeries.has(key)}
                        onCheckedChange={(v) => {
                          setDfuVisibleSeries((prev) => {
                            const next = new Set(prev);
                            if (v) next.add(key);
                            else next.delete(key);
                            return next;
                          });
                        }}
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
                  {dfuData.models.map((model, idx) => {
                    const seriesKey = `forecast_${model}`;
                    return (
                      <label
                        key={model}
                        className="flex items-center gap-2 text-xs font-medium"
                      >
                        <Checkbox
                          checked={dfuVisibleSeries.has(seriesKey)}
                          onCheckedChange={(v) => {
                            setDfuVisibleSeries((prev) => {
                              const next = new Set(prev);
                              if (v) next.add(seriesKey);
                              else next.delete(seriesKey);
                              return next;
                            });
                          }}
                        />
                        <span className="flex items-center gap-1">
                          <span
                            className="inline-block h-2.5 w-2.5 rounded-full"
                            style={{
                              backgroundColor: dfuModelColor(model, idx),
                            }}
                          />
                          {model}
                        </span>
                      </label>
                    );
                  })}
                </div>
              </div>

              {/* Time range */}
              <div className="flex flex-wrap items-end gap-3">
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  From
                  <select
                    className="h-8 w-36 rounded-md border border-input bg-background px-2 text-sm"
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
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  To
                  <select
                    className="h-8 w-36 rounded-md border border-input bg-background px-2 text-sm"
                    value={
                      dfuTimeEnd || dfuMonths[dfuMonths.length - 1] || ""
                    }
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
                  className="h-8 rounded-md border border-input bg-background px-3 text-xs font-medium text-muted-foreground hover:text-foreground"
                  onClick={() => {
                    setDfuTimeStart("");
                    setDfuTimeEnd("");
                  }}
                >
                  Show All
                </button>
                <button
                  className="h-8 rounded-md border border-input bg-background px-3 text-xs font-medium text-muted-foreground hover:text-foreground"
                  onClick={() => {
                    setDfuTimeStart(dfuDefaultStart);
                    setDfuTimeEnd("");
                  }}
                >
                  Default
                </button>
              </div>

              {/* Chart */}
              <Card className="min-w-0 border-muted shadow-none">
                <CardHeader className="pb-0">
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <ChartColumn className="h-4 w-4" /> Sales vs Forecast
                    Overlay
                    {dfuData.scope_count != null && (
                      <span className="ml-2 rounded bg-muted px-2 py-0.5 text-xs font-normal text-muted-foreground">
                        {dfuData.mode === "item_at_all_locations"
                          ? `${dfuData.scope_count} locations aggregated`
                          : `${dfuData.scope_count} items aggregated`}
                      </span>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="h-[380px] pt-2">
                  <div className="h-full overflow-x-scroll overflow-y-hidden pb-2 [scrollbar-gutter:stable]">
                    <div
                      className="h-full"
                      style={{
                        minWidth: `${Math.max(1200, dfuFilteredSeries.length * 100)}px`,
                      }}
                    >
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={dfuFilteredSeries}
                          margin={DFU_CHART_MARGIN}
                        >
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
                              backgroundColor:
                                chartColors.tooltip_bg,
                              borderColor:
                                chartColors.tooltip_border,
                            }}
                            formatter={(
                              value: number,
                              name: string,
                            ) => [
                              formatNumber(
                                Number.isFinite(Number(value))
                                  ? Number(value)
                                  : null,
                              ),
                              String(name),
                            ]}
                          />
                          <Legend />
                          {dfuVisibleSeries.has("tothist_dmd") ? (
                            <Line
                              type="monotone"
                              dataKey="tothist_dmd"
                              yAxisId="left"
                              name="Sale Qty (external)"
                              stroke={DFU_SALES_COLORS.tothist_dmd}
                              strokeWidth={2.5}
                              dot={false}
                              activeDot={ACTIVE_DOT_MD}
                            />
                          ) : null}
                          {dfuVisibleSeries.has("qty_shipped") ? (
                            <Line
                              type="monotone"
                              dataKey="qty_shipped"
                              yAxisId="left"
                              name="Qty Shipped"
                              stroke={DFU_SALES_COLORS.qty_shipped}
                              strokeWidth={2}
                              dot={false}
                              activeDot={ACTIVE_DOT_SM}
                            />
                          ) : null}
                          {dfuVisibleSeries.has("qty_ordered") ? (
                            <Line
                              type="monotone"
                              dataKey="qty_ordered"
                              yAxisId="left"
                              name="Qty Ordered"
                              stroke={DFU_SALES_COLORS.qty_ordered}
                              strokeWidth={2}
                              dot={false}
                              activeDot={ACTIVE_DOT_SM}
                            />
                          ) : null}
                          {dfuData.models
                            .filter((m) =>
                              dfuVisibleSeries.has(`forecast_${m}`),
                            )
                            .map((model, idx) => (
                              <Line
                                key={model}
                                type="monotone"
                                dataKey={`forecast_${model}`}
                                yAxisId="left"
                                name={model}
                                stroke={dfuModelColor(model, idx)}
                                strokeWidth={
                                  model === "champion" ? 2.5 : 1.5
                                }
                                strokeDasharray={
                                  model === "champion"
                                    ? undefined
                                    : "5 3"
                                }
                                dot={false}
                                activeDot={ACTIVE_DOT_SM}
                              />
                            ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* KPI Cards per model */}
              {Object.keys(dfuKpis).length > 0 ? (
                <div className="space-y-2">
                  <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Model KPIs ({dfuKpiMonths}-month window)
                  </span>
                  <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                    {dfuData.models
                      .filter(
                        (m) =>
                          dfuVisibleSeries.has(`forecast_${m}`) &&
                          dfuKpis[m],
                      )
                      .map((model) => {
                        const kpi = dfuKpis[model];
                        const colorIdx =
                          dfuData!.models.indexOf(model) + 1;
                        return (
                          <Card
                            key={model}
                            className="border-muted bg-muted/20 shadow-none"
                          >
                            <CardContent className="pt-4">
                              <div className="flex items-center gap-2 mb-2">
                                <span
                                  className="inline-block h-3 w-3 rounded-full"
                                  style={{
                                    backgroundColor:
                                      trendColors[
                                        colorIdx %
                                          trendColors.length
                                      ],
                                  }}
                                />
                                <p className="text-xs uppercase tracking-wide text-muted-foreground font-semibold">
                                  {model}
                                </p>
                                <span className="text-xs text-muted-foreground ml-auto">
                                  {kpi.months_covered} mo
                                </span>
                              </div>
                              <div className="grid grid-cols-5 gap-2">
                                <div>
                                  <p className="text-xs uppercase text-muted-foreground">
                                    Accuracy
                                  </p>
                                  <p className="text-sm font-semibold tabular-nums">
                                    {formatPercent(kpi.accuracy_pct)}
                                  </p>
                                </div>
                                <div>
                                  <p className="text-xs uppercase text-muted-foreground">
                                    WAPE
                                  </p>
                                  <p className="text-sm font-semibold tabular-nums">
                                    {formatPercent(kpi.wape)}
                                  </p>
                                </div>
                                <div>
                                  <p className="text-xs uppercase text-muted-foreground">
                                    Bias
                                  </p>
                                  <p className="text-sm font-semibold tabular-nums">
                                    {formatNumber(kpi.bias)}
                                  </p>
                                </div>
                                <div>
                                  <p className="text-xs uppercase text-muted-foreground">
                                    Fcst
                                  </p>
                                  <p className="text-sm font-semibold tabular-nums">
                                    {formatCompactNumber(
                                      kpi.sum_forecast,
                                    )}
                                  </p>
                                </div>
                                <div>
                                  <p className="text-xs uppercase text-muted-foreground">
                                    Actual
                                  </p>
                                  <p className="text-sm font-semibold tabular-nums">
                                    {formatCompactNumber(
                                      kpi.sum_actual,
                                    )}
                                  </p>
                                </div>
                              </div>
                            </CardContent>
                          </Card>
                        );
                      })}
                  </div>
                </div>
              ) : null}
            </>
          ) : dfuLoading ? (
            <div className="flex h-[320px] items-center justify-center">
              <LoadingElement
                config={ELEMENT_CONFIG.dfuAnalysis}
                message="Fetching DFU analysis..."
              />
            </div>
          ) : (
            <div className="flex h-[320px] items-center justify-center text-sm text-muted-foreground">
              {dfuMode === "item_location" &&
              (!dfuItem.trim() || !dfuLocation.trim())
                ? "Enter both item and location to view DFU analysis."
                : dfuMode === "all_items_at_location" &&
                    !dfuLocation.trim()
                  ? "Enter a location to view aggregated analysis."
                  : dfuMode === "item_at_all_locations" &&
                      !dfuItem.trim()
                    ? "Enter an item to view aggregated analysis."
                    : "No data available for the selected filters."}
            </div>
          )}
        </CardContent>
      </Card>
    </section>
  );
}
