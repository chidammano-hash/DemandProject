/**
 * Portfolio Analysis — aggregate-level forecast performance analytics.
 * The analytical equivalent of Item Analysis, but at portfolio levels:
 * forecast vs actuals charts, accuracy KPIs, performance heatmaps,
 * and model comparisons — all sliceable by brand/category/location/cluster.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { RotateCcw, CalendarClock, ChartColumn } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { KpiCard } from "@/components/KpiCard";
import { Skeleton } from "@/components/Skeleton";
import { ForecastTrendChart } from "@/components/ForecastTrendChart";
import { HeatmapGrid, makeHeatmapScale } from "@/components/HeatmapGrid";
import { CollapsibleSection } from "@/components/CollapsibleSection";

import { useDebounce } from "@/hooks/useDebounce";
import { usePanelToggles } from "@/hooks/usePanelToggles";
import { useChartColors } from "@/hooks/useChartColors";
import { cn } from "@/lib/utils";

import {
  queryKeys,
  STALE,
  fetchDashboardKpis,
  fetchDashboardTrend,
  fetchDashboardHeatmap,
  fetchForecastModels,
  fetchAccuracySlice,
  fetchLagCurve,
  fetchCompetitionConfig,
  fetchCompetitionSummary,
  fetchShapModels,
  fetchShapSummary,
  fetchShapTimeframes,
  fetchShapTimeframeDetail,
  fetchShapClusters,
  fetchSeasonalityProfileNames,
  saveCompetitionConfig,
  runCompetition,
  fetchPlanningDate,
  fetchDfuCount,
  filterMetaKeys,
  type CompetitionConfig,
  type SliceParams,
  type LagCurveParams,
  type DashboardFilterParams,
} from "@/api/queries";

import type { AccuracySliceRow, LagPoint } from "@/types";

// Accuracy sub-panels
import { SliceTablePanel } from "./accuracy/SliceTablePanel";
import { TrendChartPanel } from "./accuracy/TrendChartPanel";
import { ChampionPanel } from "./accuracy/ChampionPanel";
import { ShapPanel } from "./accuracy/ShapPanel";
import { BiasCorrectionsPanel } from "./accuracy/BiasCorrectionsPanel";

// Extracted sub-components
import {
  FilterDropdown,
  SearchableFilterDropdown,
  TimeGrainToggle,
  PANEL_DEFAULTS,
  PANELS,
  FILTERS,
  EMPTY_FILTERS,
  HEATMAP_SCALE,
  buildCascade,
  hasActiveFilters,
  formatNumberCompact,
  trendDirection,
  type LocalFilters,
} from "./aggregate-analysis";

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
interface AggregateAnalysisTabProps {
  onNavigate?: (tab: string) => void;
}

export function AggregateAnalysisTab(_props: AggregateAnalysisTabProps) {
  const queryClient = useQueryClient();

  // --------------- local filter state ---------------
  const [filters, setFilters] = useState<LocalFilters>(EMPTY_FILTERS);
  const debouncedFilters = useDebounce(filters, 300);

  const updateFilter = useCallback((key: (typeof FILTERS)[number]["key"], vals: string[]) => {
    setFilters((prev) => ({ ...prev, [key]: vals }));
  }, []);

  const dashFilters = useMemo<DashboardFilterParams>(() => ({
    brand: debouncedFilters.brand,
    category: debouncedFilters.category,
    item: debouncedFilters.item,
    location: debouncedFilters.location,
    market: debouncedFilters.market,
    channel: debouncedFilters.channel,
    cluster: debouncedFilters.cluster,
    time_grain: debouncedFilters.timeGrain,
  }), [debouncedFilters]);

  // --------------- panel toggles ---------------
  const { panels: visible, toggle } = usePanelToggles("ds:aggregateAnalysis:panels", PANEL_DEFAULTS);

  // --------------- KPI window ---------------
  const [kpiWindow, setKpiWindow] = useState(3);
  const KPI_OPTIONS = [1, 3, 6, 12];

  // --------------- Forecast chart state ---------------
  const [trendWindow, setTrendWindow] = useState(12);
  const TREND_OPTIONS = [6, 12, 18, 24];

  // --------------- Heatmap state ---------------
  type HmGrain = "category" | "brand" | "location" | "class" | "sub_class" | "date";
  const [heatmapRowGrain, setHeatmapRowGrain] = useState<HmGrain>("category");
  const [heatmapColGrain, setHeatmapColGrain] = useState<HmGrain>("date");
  const [heatmapModel, setHeatmapModel] = useState("external");

  // --------------- theme ---------------
  const { theme, chartColors, trendColors } = useChartColors();

  // --------------- Accuracy slice / filter state ---------------
  const [sliceGroupBy, setSliceGroupBy] = useState("cluster_assignment");
  const [sliceLag, setSliceLag] = useState(-1);
  const [sliceModels, setSliceModels] = useState("");
  const [sliceKpis, setSliceKpis] = useState<string[]>(["accuracy_pct", "wape", "bias"]);
  const [lagCurveMetric, setLagCurveMetric] = useState("accuracy_pct");
  const [sliceMonths, setSliceMonths] = useState(12);
  const [commonDfus, setCommonDfus] = useState(false);
  const [seasonalityProfile, setSeasonalityProfile] = useState("");
  const [seasonalityProfiles, setSeasonalityProfiles] = useState<string[]>([]);

  // --------------- Competition config state ---------------
  const [competitionConfig, setCompetitionConfig] = useState<CompetitionConfig | null>(null);

  // --------------- SHAP panel state ---------------
  const [shapOpen, setShapOpen] = useState(false);
  const [shapModelId, setShapModelId] = useState<string>("");
  const [shapTimeframeIdx, setShapTimeframeIdx] = useState<number | null>(null);
  const [shapCluster, setShapCluster] = useState<string>("all");

  // --------------- Seasonality profile names (Feature 32) ---------------
  useEffect(() => {
    let cancelled = false;
    fetchSeasonalityProfileNames()
      .then((profiles) => { if (!cancelled) setSeasonalityProfiles(profiles); })
      .catch(() => { /* non-blocking */ });
    return () => { cancelled = true; };
  }, []);

  // --------------- Dashboard queries ---------------
  const { data: planDate } = useQuery({
    queryKey: queryKeys.planningDate(),
    queryFn: fetchPlanningDate,
    staleTime: STALE.TEN_MIN,
  });

  const dashFilterRecord = useMemo(() => dashFilters as unknown as Record<string, unknown>, [dashFilters]);

  const kpiQ = useQuery({
    queryKey: queryKeys.dashboardKpis({ window: kpiWindow, ...dashFilterRecord }),
    queryFn: () => fetchDashboardKpis(kpiWindow, dashFilters),
    staleTime: STALE.THIRTY_SEC,
  });

  const trendQ = useQuery({
    queryKey: queryKeys.dashboardTrend({ window: trendWindow, ...dashFilterRecord }),
    queryFn: () => fetchDashboardTrend(trendWindow, dashFilters),
    staleTime: STALE.THIRTY_SEC,
    enabled: visible.forecastChart,
  });

  const { data: heatmapModels } = useQuery({
    queryKey: queryKeys.forecastModels(),
    queryFn: fetchForecastModels,
    staleTime: STALE.TEN_MIN,
    enabled: visible.heatmap,
  });

  const heatmapQ = useQuery({
    queryKey: queryKeys.dashboardHeatmap({ grain: heatmapRowGrain, col_grain: heatmapColGrain, model: heatmapModel, ...dashFilterRecord }),
    queryFn: () => fetchDashboardHeatmap(heatmapRowGrain, 6, dashFilters, heatmapColGrain, heatmapModel),
    staleTime: STALE.THIRTY_SEC,
    enabled: visible.heatmap,
  });

  const dfuCountQ = useQuery({
    queryKey: filterMetaKeys.dfuCount(debouncedFilters),
    queryFn: () => fetchDfuCount(debouncedFilters),
    staleTime: STALE.THIRTY_SEC,
    enabled: hasActiveFilters(filters),
  });

  // --------------- Derived filter params for accuracy queries ---------------
  const globalItem = debouncedFilters.item.length > 0 ? debouncedFilters.item.join(",") : undefined;
  const globalLocation = debouncedFilters.location.length > 0 ? debouncedFilters.location.join(",") : undefined;
  const brandParam = debouncedFilters.brand.length > 0 ? debouncedFilters.brand.join(",") : undefined;
  const categoryParam = debouncedFilters.category.length > 0 ? debouncedFilters.category.join(",") : undefined;
  const marketParam = debouncedFilters.market.length > 0 ? debouncedFilters.market.join(",") : undefined;
  const clusterParam = debouncedFilters.cluster.length > 0 ? debouncedFilters.cluster.join(",") : undefined;
  const needDfuCount = sliceKpis.includes("dfu_count");

  const monthFrom = useMemo(() => {
    if (sliceGroupBy === "month_start") return "";
    const anchor = planDate?.planning_date ? new Date(planDate.planning_date + "T00:00:00") : new Date();
    const from = new Date(anchor.getFullYear(), anchor.getMonth() - sliceMonths, 1);
    return from.toISOString().slice(0, 10);
  }, [sliceGroupBy, sliceMonths, planDate]);

  const sliceParams: SliceParams = useMemo(() => ({
    group_by: sliceGroupBy, lag: sliceLag, models: sliceModels, month_from: monthFrom,
    common_dfus: commonDfus, include_dfu_count: needDfuCount,
    item: globalItem, location: globalLocation, seasonality_profile: seasonalityProfile || undefined,
    time_grain: debouncedFilters.timeGrain,
    brand: brandParam, category: categoryParam, market: marketParam, cluster_assignment: clusterParam,
  }), [sliceGroupBy, sliceLag, sliceModels, monthFrom, commonDfus, needDfuCount, globalItem, globalLocation, seasonalityProfile, debouncedFilters.timeGrain, brandParam, categoryParam, marketParam, clusterParam]);

  const lagCurveParams: LagCurveParams = useMemo(() => ({
    models: sliceModels, month_from: monthFrom, common_dfus: commonDfus,
    include_dfu_count: needDfuCount, item: globalItem, location: globalLocation,
    seasonality_profile: seasonalityProfile || undefined,
    time_grain: debouncedFilters.timeGrain,
    brand: brandParam, category: categoryParam, market: marketParam, cluster_assignment: clusterParam,
  }), [sliceModels, monthFrom, commonDfus, needDfuCount, globalItem, globalLocation, seasonalityProfile, debouncedFilters.timeGrain, brandParam, categoryParam, marketParam, clusterParam]);

  // --------------- Accuracy queries ---------------
  const { data: slicePayload, isLoading: loadingSlice } = useQuery({
    queryKey: queryKeys.accuracySlice(sliceParams as unknown as Record<string, unknown>),
    queryFn: () => fetchAccuracySlice(sliceParams),
    staleTime: STALE.TWO_MIN,
    enabled: visible.accuracy,
  });
  const { data: lagPayload } = useQuery({
    queryKey: queryKeys.lagCurve(lagCurveParams as unknown as Record<string, unknown>),
    queryFn: () => fetchLagCurve(lagCurveParams),
    staleTime: STALE.TWO_MIN,
    enabled: visible.lagCurve,
  });
  const { data: configPayload } = useQuery({
    queryKey: queryKeys.competitionConfig(),
    queryFn: fetchCompetitionConfig,
    staleTime: STALE.FIVE_MIN,
    enabled: visible.champion,
    select: (data) => { if (data?.config && competitionConfig === null) setCompetitionConfig(data.config); return data; },
  });
  const { data: summaryPayload } = useQuery({
    queryKey: queryKeys.competitionSummary(),
    queryFn: fetchCompetitionSummary,
    staleTime: STALE.FIVE_MIN,
    enabled: visible.champion,
  });
  const { data: shapModelsData } = useQuery({
    queryKey: queryKeys.shapModels(),
    queryFn: fetchShapModels,
    staleTime: STALE.TEN_MIN,
    enabled: shapOpen && visible.shap,
  });
  const shapModels = shapModelsData?.models ?? [];
  const activeShapModel = shapModelId || shapModels[0] || "";
  const { data: shapTimeframesData } = useQuery({
    queryKey: queryKeys.shapTimeframes(activeShapModel),
    queryFn: () => fetchShapTimeframes(activeShapModel),
    staleTime: STALE.TEN_MIN,
    enabled: shapOpen && !!activeShapModel && visible.shap,
  });
  const { data: shapSummaryData, isLoading: loadingShapSummary } = useQuery({
    queryKey: queryKeys.shapSummary(activeShapModel, 15),
    queryFn: () => fetchShapSummary(activeShapModel, 15),
    staleTime: STALE.TEN_MIN,
    enabled: shapOpen && !!activeShapModel && shapTimeframeIdx === null && visible.shap,
  });
  const { data: shapDetailData, isLoading: loadingShapDetail } = useQuery({
    queryKey: queryKeys.shapTimeframeDetail(activeShapModel, shapTimeframeIdx ?? 0, 15, shapCluster),
    queryFn: () => fetchShapTimeframeDetail(activeShapModel, shapTimeframeIdx!, 15, shapCluster),
    staleTime: STALE.TEN_MIN,
    enabled: shapOpen && !!activeShapModel && shapTimeframeIdx !== null && visible.shap,
  });
  const { data: shapClustersData } = useQuery({
    queryKey: queryKeys.shapClusters(activeShapModel),
    queryFn: () => fetchShapClusters(activeShapModel),
    staleTime: STALE.TEN_MIN,
    enabled: shapOpen && !!activeShapModel && visible.shap,
  });

  // --------------- Mutations ---------------
  const saveConfigMutation = useMutation({ mutationFn: saveCompetitionConfig });
  const runCompetitionMutation = useMutation({
    mutationFn: async (config: CompetitionConfig) => { await saveCompetitionConfig(config); return runCompetition(); },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.competitionSummary() });
      queryClient.invalidateQueries({ queryKey: queryKeys.accuracySlice(sliceParams as unknown as Record<string, unknown>) });
    },
  });

  // --------------- Derived accuracy data ---------------
  const sliceData: AccuracySliceRow[] = slicePayload?.rows ?? [];
  const lagCurveData: LagPoint[] = lagPayload?.by_lag ?? [];
  const allModels = useMemo(() => Array.from(new Set(sliceData.flatMap((r) => Object.keys(r.by_model)))).sort(), [sliceData]);
  const lagModels = useMemo(() => Array.from(new Set(lagCurveData.flatMap((p) => Object.keys(p.by_model)))).sort(), [lagCurveData]);
  const activeLagMetric = useMemo(() => (sliceKpis.includes(lagCurveMetric) ? lagCurveMetric : sliceKpis[0]), [sliceKpis, lagCurveMetric]);
  const shapFeatures = useMemo(() => {
    if (shapTimeframeIdx === null)
      return (shapSummaryData?.features ?? []).map((f) => ({ feature: f.feature, value: f.mean_abs_shap_across_timeframes, selected: f.selected_count === f.n_timeframes }));
    return (shapDetailData?.features ?? []).map((f) => ({ feature: f.feature, value: f.mean_abs_shap, selected: f.selected }));
  }, [shapTimeframeIdx, shapSummaryData, shapDetailData]);

  // --------------- Derived dashboard data ---------------
  const kpi = kpiQ.data;
  const colorScale = useMemo(() => makeHeatmapScale(HEATMAP_SCALE), []);
  const heatmapRows = heatmapQ.data?.rows ?? [];
  const heatmapLabels = heatmapQ.data?.period_labels ?? [];

  // --------------- Accuracy callbacks ---------------
  const handleSliceGroupByChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => setSliceGroupBy(e.target.value), []);
  const handleSliceLagChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => setSliceLag(Number(e.target.value)), []);
  const handleSliceModelsChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => setSliceModels(e.target.value), []);
  const handleSliceMonthsChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => setSliceMonths(Number(e.target.value)), []);
  const handleCommonDfusToggle = useCallback(() => setCommonDfus((v) => !v), []);
  const handleLagCurveMetricChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => setLagCurveMetric(e.target.value), []);
  const handleKpiToggle = useCallback((key: string) => setSliceKpis((prev) => prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]), []);
  const handleSeasonalityProfileChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => setSeasonalityProfile(e.target.value), []);
  const handleCompetingModelToggle = useCallback((model: string) => {
    setCompetitionConfig((prev) => { if (!prev) return prev; const checked = prev.models.includes(model); return { ...prev, models: checked ? prev.models.filter((x) => x !== model) : [...prev.models, model] }; });
  }, []);
  const handleMetricChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => setCompetitionConfig((prev) => (prev ? { ...prev, metric: e.target.value } : prev)), []);
  const handleLagChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => setCompetitionConfig((prev) => (prev ? { ...prev, lag: e.target.value } : prev)), []);
  const handleSaveConfig = useCallback(() => { if (!competitionConfig) return; saveConfigMutation.mutate(competitionConfig); }, [competitionConfig, saveConfigMutation]);
  const handleRunCompetition = useCallback(() => { if (!competitionConfig || competitionConfig.models.length < 2) return; runCompetitionMutation.mutate(competitionConfig); }, [competitionConfig, runCompetitionMutation]);
  const handleShapModelChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => { setShapModelId(e.target.value); setShapTimeframeIdx(null); setShapCluster("all"); }, []);
  const handleShapTimeframeChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => setShapTimeframeIdx(e.target.value === "summary" ? null : Number(e.target.value)), []);
  const handleShapClusterChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => setShapCluster(e.target.value), []);

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* -------- Header -------- */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Portfolio Analysis</h2>
          <p className="text-xs text-muted-foreground">
            Forecast performance and accuracy analytics across your portfolio.
            Filter by brand, category, item, location.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {planDate?.planning_date && (
            <span className="flex items-center gap-1 rounded bg-muted/50 px-2 py-1 text-[10px] text-muted-foreground">
              <CalendarClock className="h-3 w-3" />
              {planDate.planning_date}
            </span>
          )}
          {hasActiveFilters(filters) && dfuCountQ.data && (
            <span className="rounded bg-primary/10 px-2 py-1 text-[10px] font-medium text-primary">
              {dfuCountQ.data.count?.toLocaleString() ?? "?"} DFUs
            </span>
          )}
        </div>
      </div>

      {/* -------- Filter bar -------- */}
      <div className="flex flex-wrap items-center gap-2">
        {FILTERS.map((fc) =>
          fc.searchable ? (
            <SearchableFilterDropdown key={fc.key} config={fc} selected={filters[fc.key]} onSelect={(v) => updateFilter(fc.key, v)} cascade={buildCascade(filters, fc.key)} />
          ) : (
            <FilterDropdown key={fc.key} config={fc} selected={filters[fc.key]} onSelect={(v) => updateFilter(fc.key, v)} cascade={buildCascade(filters, fc.key)} />
          ),
        )}
        <TimeGrainToggle value={filters.timeGrain} onChange={(v) => setFilters((prev) => ({ ...prev, timeGrain: v }))} />
        {hasActiveFilters(filters) && (
          <Button variant="ghost" size="sm" className="h-7 gap-1 text-xs" onClick={() => setFilters(EMPTY_FILTERS)}>
            <RotateCcw className="h-3 w-3" /> Reset
          </Button>
        )}
      </div>

      {/* -------- Panel toggle toolbar -------- */}
      <div className="flex flex-wrap items-center gap-3 rounded-md border border-border bg-muted/30 px-3 py-2">
        {PANELS.map((p) => (
          <label key={p.key} className="flex items-center gap-1.5 text-xs">
            <Checkbox checked={visible[p.key]} onCheckedChange={() => toggle(p.key)} className="h-3.5 w-3.5" />
            <span className={cn(visible[p.key] ? "text-foreground" : "text-muted-foreground")}>{p.label}</span>
          </label>
        ))}
      </div>

      {/* ================================================================ */}
      {/* KPI Cards                                                        */}
      {/* ================================================================ */}
      {visible.kpis && (
        <CollapsibleSection
          title="Performance KPIs"
          headerRight={
            <div className="flex items-center gap-1">
              {KPI_OPTIONS.map((w) => (
                <button key={w} onClick={() => setKpiWindow(w)} className={cn("rounded px-2 py-0.5 text-[10px] transition-colors", kpiWindow === w ? "bg-primary/10 text-primary font-medium" : "text-muted-foreground hover:bg-muted/50")}>
                  {w}mo
                </button>
              ))}
            </div>
          }
        >
          {kpiQ.isLoading ? (
            <div className="grid grid-cols-5 gap-3">
              {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-20" />)}
            </div>
          ) : kpi ? (
            <div className="grid grid-cols-5 gap-3">
              <KpiCard
                label="Accuracy %"
                value={kpi.accuracy_pct != null ? `${kpi.accuracy_pct.toFixed(1)}%` : "N/A"}
                trend={kpi.deltas?.accuracy_pct != null ? { delta: kpi.deltas.accuracy_pct, direction: trendDirection(kpi.deltas.accuracy_pct), unit: "pp", period: `prev ${kpiWindow}mo` } : undefined}
                severity={kpi.accuracy_pct != null ? (kpi.accuracy_pct >= 90 ? "best" : kpi.accuracy_pct >= 80 ? "neutral" : "warning") : "neutral"}
              />
              <KpiCard
                label="WAPE %"
                value={kpi.wape_pct != null ? `${kpi.wape_pct.toFixed(1)}%` : "N/A"}
                trend={kpi.deltas?.wape_pct != null ? { delta: -kpi.deltas.wape_pct, direction: trendDirection(-kpi.deltas.wape_pct), unit: "pp", period: `prev ${kpiWindow}mo` } : undefined}
                severity={kpi.wape_pct != null ? (kpi.wape_pct <= 10 ? "best" : kpi.wape_pct <= 20 ? "neutral" : "warning") : "neutral"}
              />
              <KpiCard
                label="Bias %"
                value={kpi.bias_pct != null ? `${kpi.bias_pct.toFixed(1)}%` : "N/A"}
                trend={kpi.deltas?.bias_pct != null ? { delta: -Math.abs(kpi.deltas.bias_pct), direction: trendDirection(-Math.abs(kpi.deltas.bias_pct)), unit: "pp", period: `prev ${kpiWindow}mo` } : undefined}
                severity={kpi.bias_pct != null ? (Math.abs(kpi.bias_pct) <= 5 ? "best" : Math.abs(kpi.bias_pct) <= 15 ? "neutral" : "warning") : "neutral"}
              />
              <KpiCard
                label="Forecast Vol"
                value={formatNumberCompact(kpi.total_forecast)}
                severity="neutral"
              />
              <KpiCard
                label="Actual Vol"
                value={formatNumberCompact(kpi.total_actual)}
                severity="neutral"
              />
            </div>
          ) : null}
        </CollapsibleSection>
      )}

      {/* ================================================================ */}
      {/* Forecast vs Actual Chart                                         */}
      {/* ================================================================ */}
      {visible.forecastChart && (
        <CollapsibleSection
          title="Forecast vs Actual"
          headerRight={
            <div className="flex items-center gap-1">
              {TREND_OPTIONS.map((w) => (
                <button key={w} onClick={() => setTrendWindow(w)} className={cn("rounded px-2 py-0.5 text-[10px] transition-colors", trendWindow === w ? "bg-primary/10 text-primary font-medium" : "text-muted-foreground hover:bg-muted/50")}>
                  {w}mo
                </button>
              ))}
            </div>
          }
        >
          {trendQ.isLoading ? (
            <Skeleton className="h-[260px]" />
          ) : (
            <ForecastTrendChart
              data={trendQ.data?.months ?? []}
              theme={theme === "soft" ? "light" : theme}
              chartColors={{ grid: chartColors.grid, axis: chartColors.axis, tooltip: chartColors.tooltip_bg }}
              seriesColors={trendColors}
            />
          )}
        </CollapsibleSection>
      )}

      {/* ================================================================ */}
      {/* Accuracy Heatmap                                                 */}
      {/* ================================================================ */}
      {visible.heatmap && (
        <CollapsibleSection
          title="Accuracy Heatmap"
          headerRight={
            <div className="flex items-center gap-2 text-[10px]">
              <span className="text-muted-foreground font-medium">Model</span>
              <select
                className="h-6 rounded border border-input bg-background px-1.5 text-[10px]"
                value={heatmapModel}
                onChange={(e) => setHeatmapModel(e.target.value)}
              >
                {(heatmapModels ?? ["external"]).map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
              <span className="text-muted-foreground font-medium">Rows</span>
              <select
                className="h-6 rounded border border-input bg-background px-1.5 text-[10px]"
                value={heatmapRowGrain}
                onChange={(e) => {
                  const v = e.target.value as HmGrain;
                  setHeatmapRowGrain(v);
                  if (v === heatmapColGrain) setHeatmapColGrain(v === "date" ? "category" : "date");
                }}
              >
                <option value="category">Category</option>
                <option value="brand">Brand</option>
                <option value="class">Class</option>
                <option value="sub_class">Sub-class</option>
                <option value="location">Location</option>
                <option value="date">Date</option>
              </select>
              <span className="text-muted-foreground font-medium">Columns</span>
              <select
                className="h-6 rounded border border-input bg-background px-1.5 text-[10px]"
                value={heatmapColGrain}
                onChange={(e) => {
                  const v = e.target.value as HmGrain;
                  setHeatmapColGrain(v);
                  if (v === heatmapRowGrain) setHeatmapRowGrain(v === "date" ? "category" : "date");
                }}
              >
                <option value="category">Category</option>
                <option value="brand">Brand</option>
                <option value="class">Class</option>
                <option value="sub_class">Sub-class</option>
                <option value="location">Location</option>
                <option value="date">Date</option>
              </select>
            </div>
          }
        >
          {heatmapQ.isLoading ? (
            <Skeleton className="h-[200px]" />
          ) : (
            <HeatmapGrid
              rows={heatmapRows}
              columnLabels={heatmapLabels}
              colorScale={colorScale}
            />
          )}
        </CollapsibleSection>
      )}

      {/* ================================================================ */}
      {/* Accuracy Comparison (slice table) + Lag Curve                   */}
      {/* ================================================================ */}
      {(visible.accuracy || visible.lagCurve) && (
        <CollapsibleSection title="Accuracy Comparison">
          <div className="space-y-5">
            {visible.accuracy && (
              <SliceTablePanel
                sliceGroupBy={sliceGroupBy} sliceLag={sliceLag} sliceModels={sliceModels}
                sliceKpis={sliceKpis} sliceMonths={sliceMonths} commonDfus={commonDfus}
                seasonalityProfile={seasonalityProfile} seasonalityProfiles={seasonalityProfiles}
                loadingSlice={loadingSlice} sliceData={sliceData} allModels={allModels}
                commonDfuCount={slicePayload?.common_dfu_count ?? null}
                dfuCounts={slicePayload?.dfu_counts ?? null}
                onSliceGroupByChange={handleSliceGroupByChange}
                onSliceLagChange={handleSliceLagChange}
                onSliceModelsChange={handleSliceModelsChange}
                onSliceMonthsChange={handleSliceMonthsChange}
                onCommonDfusToggle={handleCommonDfusToggle}
                onKpiToggle={handleKpiToggle}
                onSeasonalityProfileChange={handleSeasonalityProfileChange}
              />
            )}
            {visible.lagCurve && (
              <TrendChartPanel
                lagCurveData={lagCurveData} lagModels={lagModels}
                sliceKpis={sliceKpis} activeLagMetric={activeLagMetric}
                onLagCurveMetricChange={handleLagCurveMetricChange}
              />
            )}
          </div>
        </CollapsibleSection>
      )}

      {/* ================================================================ */}
      {/* Champion Panel                                                   */}
      {/* ================================================================ */}
      {visible.champion && (
        <ChampionPanel
          competitionConfig={competitionConfig}
          availableModels={configPayload?.available_models ?? []}
          championSummary={summaryPayload?.summary ?? null}
          savingConfig={saveConfigMutation.isPending}
          runningCompetition={runCompetitionMutation.isPending}
          onCompetingModelToggle={handleCompetingModelToggle}
          onMetricChange={handleMetricChange}
          onLagChange={handleLagChange}
          onSaveConfig={handleSaveConfig}
          onRunCompetition={handleRunCompetition}
        />
      )}

      {/* ================================================================ */}
      {/* SHAP Panel                                                       */}
      {/* ================================================================ */}
      {visible.shap && (
        <ShapPanel
          shapOpen={shapOpen} shapModels={shapModels} activeShapModel={activeShapModel}
          shapTimeframes={shapTimeframesData?.timeframes ?? []}
          shapTimeframeIdx={shapTimeframeIdx} shapFeatures={shapFeatures}
          loadingShap={shapTimeframeIdx === null ? loadingShapSummary : loadingShapDetail}
          shapClusters={shapClustersData?.clusters ?? []}
          shapCluster={shapCluster}
          onToggleOpen={() => setShapOpen((v) => !v)}
          onModelChange={handleShapModelChange}
          onTimeframeChange={handleShapTimeframeChange}
          onClusterChange={handleShapClusterChange}
        />
      )}

      {/* ================================================================ */}
      {/* Bias Corrections Panel                                           */}
      {/* ================================================================ */}
      {visible.bias && (
        <CollapsibleSection title="Bias Corrections" defaultOpen={false}>
          <BiasCorrectionsPanel />
        </CollapsibleSection>
      )}
    </div>
  );
}
