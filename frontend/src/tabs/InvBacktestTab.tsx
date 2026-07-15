import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity } from "lucide-react";

import {
  queryKeys,
  STALE,
  fetchForecastModels,
  fetchInvBacktestSummary,
  fetchInvBacktestTrend,
  fetchInvBacktestRootCause,
  fetchInvBacktestDetail,
} from "@/api/queries";
import type { InvBacktestModelMetrics } from "@/types";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { KpiCard } from "@/components/KpiCard";
import { LoadingElement } from "@/components/LoadingElement";
import { cn } from "@/lib/utils";
import { useDebounce } from "@/hooks/useDebounce";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { useChartColors } from "@/hooks/useChartColors";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";

import {
  DetailTablePanel,
  ModelComparisonChart,
  RootCauseChart,
  TrendChart,
  PAGE_SIZE,
  type DetailSortCol,
} from "./inv-backtest";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function InvBacktestTab() {
  const { chartColors, trendColors } = useChartColors();

  // ---- State ---------------------------------------------------------------
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [itemFilter, setItemFilter] = useState("");
  const [locationFilter, setLocationFilter] = useState("");
  const [clusterFilter, setClusterFilter] = useState("");
  const [rootCauseModel, setRootCauseModel] = useState("");
  const [trendMetric, setTrendMetric] = useState<string>("stockout_rate");
  const [eventType, setEventType] = useState<string>("all");
  const [detailOffset, setDetailOffset] = useState(0);
  const [detailSort, setDetailSort] = useState<DetailSortCol>("month_start");
  const [detailDir, setDetailDir] = useState<"asc" | "desc">("desc");

  // ---- Global filter sync --------------------------------------------------
  const { filters: globalFilters } = useGlobalFilterContext();
  const syncedRef = useRef("");
  useEffect(() => {
    const key = `${globalFilters.item.join(",")}_${globalFilters.location.join(",")}`;
    if (key === syncedRef.current) return;
    syncedRef.current = key;
    if (globalFilters.item.length === 1) setItemFilter(globalFilters.item[0]);
    if (globalFilters.location.length === 1) setLocationFilter(globalFilters.location[0]);
  }, [globalFilters.item, globalFilters.location]);

  const debouncedItem = useDebounce(itemFilter, 400);
  const debouncedLocation = useDebounce(locationFilter, 400);

  // ---- Fetch available models -----------------------------------------------
  const { data: availableModels } = useQuery({
    queryKey: queryKeys.forecastModels(),
    queryFn: fetchForecastModels,
    staleTime: STALE.TEN_MIN,
  });

  useEffect(() => {
    if (availableModels && availableModels.length > 0 && selectedModels.size === 0) {
      setSelectedModels(new Set(availableModels.slice(0, 5)));
    }
  }, [availableModels, selectedModels.size]);

  useEffect(() => {
    if (!rootCauseModel && selectedModels.size > 0) {
      setRootCauseModel([...selectedModels][0]);
    }
  }, [rootCauseModel, selectedModels]);

  // ---- Derived params -------------------------------------------------------
  const modelsStr = useMemo(() => [...selectedModels].join(","), [selectedModels]);

  const filterParams = useMemo(
    () => ({
      models: modelsStr,
      item: debouncedItem,
      location: debouncedLocation,
      cluster_assignment: clusterFilter,
    }),
    [modelsStr, debouncedItem, debouncedLocation, clusterFilter],
  );

  // ---- Queries --------------------------------------------------------------
  const { data: summaryData, isLoading: loadingSummary } = useQuery({
    queryKey: queryKeys.invBacktestSummary(filterParams),
    queryFn: () => fetchInvBacktestSummary(filterParams),
    staleTime: STALE.TWO_MIN,
    enabled: selectedModels.size > 0,
  });

  const { data: trendData, isLoading: loadingTrend } = useQuery({
    queryKey: queryKeys.invBacktestTrend(filterParams),
    queryFn: () => fetchInvBacktestTrend(filterParams),
    staleTime: STALE.TWO_MIN,
    enabled: selectedModels.size > 0,
  });

  const { data: rootCauseData, isLoading: loadingRootCause } = useQuery({
    queryKey: queryKeys.invBacktestRootCause({ ...filterParams, model_id: rootCauseModel }),
    queryFn: () => fetchInvBacktestRootCause({ ...filterParams, model_id: rootCauseModel }),
    staleTime: STALE.TWO_MIN,
    enabled: !!rootCauseModel,
  });

  const detailParams = useMemo(
    () => ({
      ...filterParams,
      event_type: eventType,
      limit: PAGE_SIZE,
      offset: detailOffset,
      sort_by: detailSort,
      sort_dir: detailDir,
    }),
    [filterParams, eventType, detailOffset, detailSort, detailDir],
  );

  const { data: detailData, isLoading: loadingDetail } = useQuery({
    queryKey: queryKeys.invBacktestDetail(detailParams),
    queryFn: () => fetchInvBacktestDetail(detailParams),
    staleTime: STALE.TWO_MIN,
    enabled: selectedModels.size > 0,
  });

  // ---- Computed data --------------------------------------------------------
  const bestModel = useMemo<{
    id: string;
    metrics: InvBacktestModelMetrics;
  } | null>(() => {
    if (!summaryData?.by_model) return null;
    let best: { id: string; metrics: InvBacktestModelMetrics } | null = null;
    for (const [id, m] of Object.entries(summaryData.by_model)) {
      if (!best || m.cycle_service_level > best.metrics.cycle_service_level) {
        best = { id, metrics: m };
      }
    }
    return best;
  }, [summaryData]);

  const comparisonData = useMemo(() => {
    if (!summaryData?.by_model) return [];
    return Object.entries(summaryData.by_model).map(([id, m]) => ({
      model: id,
      stockout_rate: m.stockout_rate,
      excess_rate: m.excess_rate,
      wape: m.wape ?? 0,
    }));
  }, [summaryData]);

  const rootCauseChartData = useMemo(() => {
    if (!rootCauseData) return [];
    return [
      {
        event: "Stockout",
        under_forecast: rootCauseData.stockout_under_forecast,
        over_forecast: rootCauseData.stockout_over_forecast,
        exact: rootCauseData.stockout_exact,
      },
      {
        event: "Excess",
        under_forecast: rootCauseData.excess_under_forecast,
        over_forecast: rootCauseData.excess_over_forecast,
        exact: rootCauseData.excess_exact,
      },
    ];
  }, [rootCauseData]);

  const trendChartData = useMemo(() => {
    if (!trendData?.trend) return [];
    const models = summaryData?.models ?? [];
    return trendData.trend.map((pt) => {
      const row: Record<string, string | number | null> = { month: pt.month };
      for (const mid of models) {
        const val = pt.by_model[mid];
        row[mid] = val ? (val[trendMetric as keyof typeof val] ?? null) : null;
      }
      return row;
    });
  }, [trendData, summaryData, trendMetric]);

  // ---- Handlers -------------------------------------------------------------
  const toggleModel = useCallback((model: string) => {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(model)) next.delete(model);
      else next.add(model);
      return next;
    });
    setDetailOffset(0);
  }, []);

  const handleDetailSort = useCallback(
    (col: DetailSortCol) => {
      if (detailSort === col) {
        setDetailDir((prev) => (prev === "asc" ? "desc" : "asc"));
      } else {
        setDetailSort(col);
        setDetailDir("desc");
      }
      setDetailOffset(0);
    },
    [detailSort],
  );

  // ---- Render ---------------------------------------------------------------
  return (
    <section className="mt-4 space-y-4">
      {/* ---- KPI Cards --------------------------------------------------- */}
      {loadingSummary ? (
        <LoadingElement tabKey="invBacktest" message="Loading backtest summary..." />
      ) : bestModel ? (
        <div className="flex flex-wrap gap-3">
          <KpiCard
            label="Best Cycle Service Level (CSL)"
            value={`${formatNumber(bestModel.metrics.cycle_service_level)}%`}
            sublabel={bestModel.id}
            severity={
              bestModel.metrics.cycle_service_level >= 95
                ? "best"
                : bestModel.metrics.cycle_service_level < 90
                  ? "warning"
                  : "neutral"
            }
          />
          <KpiCard
            label="Lowest Stockout Rate"
            value={`${formatNumber(
              Math.min(...Object.values(summaryData!.by_model).map((m) => m.stockout_rate)),
            )}%`}
            severity="best"
          />
          <KpiCard
            label="Lowest Excess Rate"
            value={`${formatNumber(
              Math.min(...Object.values(summaryData!.by_model).map((m) => m.excess_rate)),
            )}%`}
            severity="neutral"
          />
          <KpiCard
            label="Models Compared"
            value={String(summaryData!.models.length)}
          />
          <KpiCard
            label="DFU-Months"
            value={formatCompactNumber(bestModel.metrics.dfu_months)}
          />
        </div>
      ) : null}

      {/* ---- Main Card with filters + charts ----------------------------- */}
      <Card className="animate-fade-in">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            <CardTitle className="text-base">Inventory Planning Backtest</CardTitle>
          </div>
          <CardDescription>
            Connect forecast accuracy to inventory outcomes. Compare which
            algorithms lead to fewer stockouts and less excess inventory.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* ---- Filter Controls ----------------------------------------- */}
          <div className="flex flex-wrap items-end gap-3">
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Item
              <input
                className="h-9 w-40 rounded-md border border-input bg-background px-3 text-sm"
                placeholder="Filter item..."
                value={itemFilter}
                onChange={(e) => { setItemFilter(e.target.value); setDetailOffset(0); }}
              />
            </label>
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Location
              <input
                className="h-9 w-40 rounded-md border border-input bg-background px-3 text-sm"
                placeholder="Filter location..."
                value={locationFilter}
                onChange={(e) => { setLocationFilter(e.target.value); setDetailOffset(0); }}
              />
            </label>
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Cluster
              <input
                className="h-9 w-40 rounded-md border border-input bg-background px-3 text-sm"
                placeholder="Filter cluster..."
                value={clusterFilter}
                onChange={(e) => setClusterFilter(e.target.value)}
              />
            </label>
          </div>

          {/* ---- Model Selector ------------------------------------------ */}
          {availableModels && availableModels.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Models
              </p>
              <div className="flex flex-wrap gap-2">
                {availableModels.map((m) => (
                  <button
                    key={m}
                    onClick={() => toggleModel(m)}
                    className={cn(
                      "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                      selectedModels.has(m)
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-input text-muted-foreground hover:bg-muted",
                    )}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* ---- Charts (extracted sub-components) ----------------------- */}
          <ModelComparisonChart
            comparisonData={comparisonData}
            chartColors={chartColors}
            trendColors={trendColors}
          />

          <RootCauseChart
            rootCauseChartData={rootCauseChartData}
            loadingRootCause={loadingRootCause}
            rootCauseModel={rootCauseModel}
            models={summaryData?.models}
            onRootCauseModelChange={setRootCauseModel}
            chartColors={chartColors}
          />

          <TrendChart
            trendChartData={trendChartData}
            loadingTrend={loadingTrend}
            trendMetric={trendMetric}
            models={summaryData?.models ?? []}
            onTrendMetricChange={setTrendMetric}
            chartColors={chartColors}
          />

          {/* ---- Detail Table -------------------------------------------- */}
          <DetailTablePanel
            detailData={detailData}
            loadingDetail={loadingDetail}
            eventType={eventType}
            detailOffset={detailOffset}
            detailSort={detailSort}
            detailDir={detailDir}
            onEventTypeChange={(t) => { setEventType(t); setDetailOffset(0); }}
            onDetailSort={handleDetailSort}
            onOffsetChange={setDetailOffset}
          />
        </CardContent>
      </Card>
    </section>
  );
}
