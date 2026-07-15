/**
 * Portfolio Analysis — aggregate-level forecast performance analytics.
 * The analytical equivalent of Item Analysis, but at portfolio levels:
 * forecast vs actuals charts, accuracy KPIs, performance heatmaps,
 * and model comparisons — all sliceable by brand/category/location/cluster.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { makeHeatmapScale } from "@/components/HeatmapGrid";
import { CollapsibleSection } from "@/components/CollapsibleSection";

import { useDebounce } from "@/hooks/useDebounce";
import { usePanelToggles } from "@/hooks/usePanelToggles";

import {
  queryKeys,
  STALE,
  fetchDashboardKpis,
  fetchDashboardTrend,
  fetchDashboardHeatmap,
  fetchForecastModels,
  fetchAccuracySlice,
  fetchLagCurve,
  fetchCompetitionSummary,
  fetchShapModels,
  fetchShapSummary,
  fetchShapTimeframes,
  fetchShapTimeframeDetail,
  fetchShapClusters,
  fetchSeasonalityProfileNames,
  fetchPlanningDate,
  fetchSkuCount,
  filterMetaKeys,
  SLICE_DEFAULT_LIMIT,
  type SliceParams,
  type LagCurveParams,
  type DashboardFilterParams,
} from "@/api/queries";

import type { AccuracySliceRow, LagPoint } from "@/types";

// Accuracy sub-panels
import { SliceTablePanel } from "./accuracy/SliceTablePanel";
import { TrendChartPanel } from "./accuracy/TrendChartPanel";
import { LagLeaderboardPanel } from "./aggregate-analysis/LagLeaderboardPanel";
import { PortfolioForecastComparison } from "./aggregate-analysis/PortfolioForecastComparison";
import { PortfolioHeaderControls } from "./aggregate-analysis/PortfolioHeaderControls";
import { detailShapFeatureRows } from "./accuracy/shapFeatureRows";
import { ChampionPanel } from "./accuracy/ChampionPanel";
import { ShapPanel } from "./accuracy/ShapPanel";
import { BiasCorrectionsPanel } from "./accuracy/BiasCorrectionsPanel";
import { ErrorDecompositionPanel } from "./accuracy/ErrorDecompositionPanel";

// Extracted sub-components
import {
  KpiCardsSection,
  AccuracyHeatmapSection,
  PANEL_DEFAULTS,
  FILTERS,
  EMPTY_FILTERS,
  HEATMAP_SCALE,
  hasActiveFilters,
  type HmGrain,
  type LocalFilters,
} from "./aggregate-analysis";

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
interface AggregateAnalysisTabProps {
  onNavigate?: (tab: string) => void;
}

export function AggregateAnalysisTab(_props: AggregateAnalysisTabProps) {
  // --------------- local filter state ---------------
  const [filters, setFilters] = useState<LocalFilters>(EMPTY_FILTERS);
  const debouncedFilters = useDebounce(filters, 300);

  const updateFilter = useCallback((key: (typeof FILTERS)[number]["key"], vals: string[]) => {
    setFilters((prev) => ({ ...prev, [key]: vals }));
  }, []);

  const dashFilters = useMemo<DashboardFilterParams>(
    () => ({
      brand: debouncedFilters.brand,
      category: debouncedFilters.category,
      item: debouncedFilters.item,
      location: debouncedFilters.location,
      market: debouncedFilters.market,
      channel: debouncedFilters.channel,
      cluster: debouncedFilters.cluster,
      time_grain: debouncedFilters.timeGrain,
    }),
    [debouncedFilters]
  );

  // --------------- panel toggles ---------------
  const { panels: visible, toggle } = usePanelToggles(
    "ds:aggregateAnalysis:panels",
    PANEL_DEFAULTS
  );

  // --------------- KPI window + model ---------------
  const [kpiWindow, setKpiWindow] = useState(3);
  const KPI_OPTIONS = [1, 3, 6, 12];
  const [kpiModel, setKpiModel] = useState("external");

  // --------------- Forecast chart state ---------------
  const [trendWindow, setTrendWindow] = useState(12);

  // --------------- Heatmap state ---------------
  const [heatmapRowGrain, setHeatmapRowGrain] = useState<HmGrain>("category");
  const [heatmapColGrain, setHeatmapColGrain] = useState<HmGrain>("date");
  const [heatmapModel, setHeatmapModel] = useState("external");

  // theme/colors are now consumed directly by leaf chart components via context.

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

  // --------------- SHAP panel state ---------------
  const [shapOpen, setShapOpen] = useState(false);
  const [shapModelId, setShapModelId] = useState<string>("");
  const [shapTimeframeIdx, setShapTimeframeIdx] = useState<number | null>(null);
  const [shapCluster, setShapCluster] = useState<string>("all");

  // --------------- Seasonality profile names (Feature 32) ---------------
  useEffect(() => {
    let cancelled = false;
    fetchSeasonalityProfileNames()
      .then((profiles) => {
        if (!cancelled) setSeasonalityProfiles(profiles);
      })
      .catch(() => {
        /* non-blocking */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // --------------- Dashboard queries ---------------
  const { data: planDate } = useQuery({
    queryKey: queryKeys.planningDate(),
    queryFn: fetchPlanningDate,
    staleTime: STALE.TEN_MIN,
  });

  const dashFilterRecord = useMemo(
    () => dashFilters as unknown as Record<string, unknown>,
    [dashFilters]
  );

  const kpiQ = useQuery({
    queryKey: queryKeys.dashboardKpis({ window: kpiWindow, model: kpiModel, ...dashFilterRecord }),
    queryFn: () => fetchDashboardKpis(kpiWindow, dashFilters, kpiModel),
    staleTime: STALE.THIRTY_SEC,
  });

  const trendQ = useQuery({
    queryKey: queryKeys.dashboardTrend({
      window: trendWindow,
      model: kpiModel,
      ...dashFilterRecord,
    }),
    queryFn: () => fetchDashboardTrend(trendWindow, dashFilters, kpiModel),
    staleTime: STALE.THIRTY_SEC,
    enabled: visible.forecastChart || visible.kpis,
  });

  const { data: heatmapModels } = useQuery({
    queryKey: queryKeys.forecastModels(),
    queryFn: fetchForecastModels,
    staleTime: STALE.TEN_MIN,
    enabled: visible.kpis || visible.heatmap,
  });

  const heatmapQ = useQuery({
    queryKey: queryKeys.dashboardHeatmap({
      grain: heatmapRowGrain,
      col_grain: heatmapColGrain,
      model: heatmapModel,
      ...dashFilterRecord,
    }),
    queryFn: () =>
      fetchDashboardHeatmap(heatmapRowGrain, 6, dashFilters, heatmapColGrain, heatmapModel),
    staleTime: STALE.THIRTY_SEC,
    enabled: visible.heatmap,
  });

  const skuCountQ = useQuery({
    queryKey: filterMetaKeys.skuCount(debouncedFilters),
    queryFn: () => fetchSkuCount(debouncedFilters),
    staleTime: STALE.THIRTY_SEC,
    enabled: hasActiveFilters(filters),
  });

  // --------------- Derived filter params for accuracy queries ---------------
  const globalItem = debouncedFilters.item.length > 0 ? debouncedFilters.item.join(",") : undefined;
  const globalLocation =
    debouncedFilters.location.length > 0 ? debouncedFilters.location.join(",") : undefined;
  const brandParam =
    debouncedFilters.brand.length > 0 ? debouncedFilters.brand.join(",") : undefined;
  const categoryParam =
    debouncedFilters.category.length > 0 ? debouncedFilters.category.join(",") : undefined;
  const marketParam =
    debouncedFilters.market.length > 0 ? debouncedFilters.market.join(",") : undefined;
  const clusterParam =
    debouncedFilters.cluster.length > 0 ? debouncedFilters.cluster.join(",") : undefined;
  const needDfuCount = sliceKpis.includes("sku_count");

  const monthFrom = useMemo(() => {
    if (sliceGroupBy === "month_start") return "";
    const anchor = planDate?.planning_date
      ? new Date(planDate.planning_date + "T00:00:00")
      : new Date();
    const from = new Date(anchor.getFullYear(), anchor.getMonth() - sliceMonths, 1);
    return from.toISOString().slice(0, 10);
  }, [sliceGroupBy, sliceMonths, planDate]);

  const sliceParams: SliceParams = useMemo(
    () => ({
      group_by: sliceGroupBy,
      lag: sliceLag,
      models: sliceModels,
      month_from: monthFrom,
      common_skus: commonDfus,
      include_sku_count: needDfuCount,
      item: globalItem,
      location: globalLocation,
      seasonality_profile: seasonalityProfile || undefined,
      time_grain: debouncedFilters.timeGrain,
      brand: brandParam,
      category: categoryParam,
      market: marketParam,
      cluster_assignment: clusterParam,
    }),
    [
      sliceGroupBy,
      sliceLag,
      sliceModels,
      monthFrom,
      commonDfus,
      needDfuCount,
      globalItem,
      globalLocation,
      seasonalityProfile,
      debouncedFilters.timeGrain,
      brandParam,
      categoryParam,
      marketParam,
      clusterParam,
    ]
  );

  const lagCurveParams: LagCurveParams = useMemo(
    () => ({
      models: sliceModels,
      month_from: monthFrom,
      common_skus: commonDfus,
      include_sku_count: needDfuCount,
      item: globalItem,
      location: globalLocation,
      seasonality_profile: seasonalityProfile || undefined,
      time_grain: debouncedFilters.timeGrain,
      brand: brandParam,
      category: categoryParam,
      market: marketParam,
      cluster_assignment: clusterParam,
    }),
    [
      sliceModels,
      monthFrom,
      commonDfus,
      needDfuCount,
      globalItem,
      globalLocation,
      seasonalityProfile,
      debouncedFilters.timeGrain,
      brandParam,
      categoryParam,
      marketParam,
      clusterParam,
    ]
  );

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
  const {
    data: shapDetailData,
    isLoading: loadingShapDetail,
    isFetching: fetchingShapDetail,
  } = useQuery({
    queryKey: queryKeys.shapTimeframeDetail(
      activeShapModel,
      shapTimeframeIdx ?? 0,
      15,
      shapCluster
    ),
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

  // --------------- Derived accuracy data ---------------
  const sliceData = useMemo<AccuracySliceRow[]>(() => slicePayload?.rows ?? [], [slicePayload]);
  const lagCurveData = useMemo<LagPoint[]>(() => lagPayload?.by_lag ?? [], [lagPayload]);
  const allModels = useMemo(
    () => Array.from(new Set(sliceData.flatMap((r) => Object.keys(r.by_model)))).sort(),
    [sliceData]
  );
  const lagModels = useMemo(
    () => Array.from(new Set(lagCurveData.flatMap((p) => Object.keys(p.by_model)))).sort(),
    [lagCurveData]
  );
  const activeLagMetric = useMemo(
    () => (sliceKpis.includes(lagCurveMetric) ? lagCurveMetric : sliceKpis[0]),
    [sliceKpis, lagCurveMetric]
  );
  const shapFeatures = useMemo(() => {
    if (shapTimeframeIdx === null)
      return (shapSummaryData?.features ?? []).map((f) => ({
        feature: f.feature,
        value: f.mean_abs_shap_across_timeframes,
        selected: f.selected_count === f.n_timeframes,
      }));
    return detailShapFeatureRows(shapDetailData, shapCluster);
  }, [shapTimeframeIdx, shapSummaryData, shapDetailData, shapCluster]);

  // --------------- Derived dashboard data ---------------
  const kpi = kpiQ.data;
  const colorScale = useMemo(() => makeHeatmapScale(HEATMAP_SCALE), []);
  const heatmapRows = heatmapQ.data?.rows ?? [];
  const heatmapLabels = heatmapQ.data?.period_labels ?? [];

  // --------------- Accuracy callbacks ---------------
  const handleSliceGroupByChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => setSliceGroupBy(e.target.value),
    []
  );
  const handleSliceLagChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => setSliceLag(Number(e.target.value)),
    []
  );
  const handleSliceModelsChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => setSliceModels(e.target.value),
    []
  );
  const handleSliceMonthsChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => setSliceMonths(Number(e.target.value)),
    []
  );
  const handleCommonDfusToggle = useCallback(() => setCommonDfus((v) => !v), []);
  const handleLagCurveMetricChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => setLagCurveMetric(e.target.value),
    []
  );
  const handleKpiToggle = useCallback(
    (key: string) =>
      setSliceKpis((prev) => (prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key])),
    []
  );
  const handleSeasonalityProfileChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => setSeasonalityProfile(e.target.value),
    []
  );
  const handleShapModelChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setShapModelId(e.target.value);
    setShapTimeframeIdx(null);
    setShapCluster("all");
  }, []);
  const handleShapTimeframeChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) =>
      setShapTimeframeIdx(e.target.value === "summary" ? null : Number(e.target.value)),
    []
  );
  const handleShapClusterChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => setShapCluster(e.target.value),
    []
  );

  return (
    <div className="flex flex-col gap-4 p-4">
      <PortfolioHeaderControls
        filters={filters}
        setFilters={setFilters}
        onFilterChange={updateFilter}
        planningDate={planDate?.planning_date}
        skuCount={skuCountQ.data?.count}
        visiblePanels={visible}
        onTogglePanel={toggle}
      />

      {/* ================================================================ */}
      {/* KPI Cards                                                        */}
      {/* ================================================================ */}
      {visible.kpis && (
        <KpiCardsSection
          kpi={kpi}
          isLoading={kpiQ.isLoading}
          kpiModel={kpiModel}
          kpiWindow={kpiWindow}
          kpiOptions={KPI_OPTIONS}
          heatmapModels={heatmapModels}
          trendData={trendQ.data}
          onKpiModelChange={setKpiModel}
          onKpiWindowChange={setKpiWindow}
        />
      )}

      {/* ================================================================ */}
      {/* Forecast vs Actual Chart                                         */}
      {/* ================================================================ */}
      {visible.forecastChart && (
        <PortfolioForecastComparison
          kpiModel={kpiModel}
          trendWindow={trendWindow}
          onTrendWindowChange={setTrendWindow}
          dashboardFilters={dashFilters}
          standardMonths={trendQ.data?.months ?? []}
          standardLoading={trendQ.isLoading}
        />
      )}

      {/* ================================================================ */}
      {/* Accuracy Heatmap                                                 */}
      {/* ================================================================ */}
      {visible.heatmap && (
        <AccuracyHeatmapSection
          isLoading={heatmapQ.isLoading}
          rows={heatmapRows}
          columnLabels={heatmapLabels}
          colorScale={colorScale}
          heatmapModel={heatmapModel}
          heatmapModels={heatmapModels}
          heatmapRowGrain={heatmapRowGrain}
          heatmapColGrain={heatmapColGrain}
          onHeatmapModelChange={setHeatmapModel}
          onRowGrainChange={setHeatmapRowGrain}
          onColGrainChange={setHeatmapColGrain}
        />
      )}

      {/* ================================================================ */}
      {/* Accuracy Comparison (slice table) + Lag Curve                   */}
      {/* ================================================================ */}
      {(visible.accuracy || visible.lagCurve) && (
        <CollapsibleSection title="Accuracy Comparison">
          <div className="space-y-5">
            {visible.accuracy && (
              <SliceTablePanel
                sliceGroupBy={sliceGroupBy}
                sliceLag={sliceLag}
                sliceModels={sliceModels}
                sliceKpis={sliceKpis}
                sliceMonths={sliceMonths}
                commonDfus={commonDfus}
                seasonalityProfile={seasonalityProfile}
                seasonalityProfiles={seasonalityProfiles}
                loadingSlice={loadingSlice}
                sliceData={sliceData}
                allModels={allModels}
                commonDfuCount={slicePayload?.common_sku_count ?? null}
                skuCounts={slicePayload?.sku_counts ?? null}
                truncated={slicePayload?.truncated ?? false}
                sliceLimit={slicePayload?.limit ?? SLICE_DEFAULT_LIMIT}
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
              <>
                <TrendChartPanel
                  lagCurveData={lagCurveData}
                  lagModels={lagModels}
                  sliceKpis={sliceKpis}
                  activeLagMetric={activeLagMetric}
                  onLagCurveMetricChange={handleLagCurveMetricChange}
                />
                <LagLeaderboardPanel />
              </>
            )}
          </div>
        </CollapsibleSection>
      )}

      {/* ================================================================ */}
      {/* Error Decomposition (diagnostic: where the accuracy gap lives)  */}
      {/* ================================================================ */}
      {visible.decomposition && (
        <CollapsibleSection title="Error Decomposition" defaultOpen={false}>
          <ErrorDecompositionPanel
            models={sliceModels}
            lag={sliceLag}
            monthFrom={monthFrom}
            clusterAssignment={clusterParam}
            seasonalityProfile={seasonalityProfile || undefined}
            enabled={visible.decomposition}
          />
        </CollapsibleSection>
      )}

      {/* ================================================================ */}
      {/* Champion Panel                                                   */}
      {/* ================================================================ */}
      {visible.champion && <ChampionPanel championSummary={summaryPayload?.summary ?? null} />}

      {/* ================================================================ */}
      {/* SHAP Panel                                                       */}
      {/* ================================================================ */}
      {visible.shap && (
        <ShapPanel
          shapOpen={shapOpen}
          shapModels={shapModels}
          activeShapModel={activeShapModel}
          shapTimeframes={shapTimeframesData?.timeframes ?? []}
          shapTimeframeIdx={shapTimeframeIdx}
          shapFeatures={shapFeatures}
          loadingShap={
            shapTimeframeIdx === null ? loadingShapSummary : loadingShapDetail || fetchingShapDetail
          }
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
