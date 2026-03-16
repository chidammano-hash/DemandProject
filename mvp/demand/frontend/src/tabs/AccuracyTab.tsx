import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { ChartColumn } from "lucide-react";

import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import {
  queryKeys,
  STALE,
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
  type CompetitionConfig,
  type SliceParams,
  type LagCurveParams,
} from "@/api/queries";
import type { AccuracySliceRow, LagPoint } from "@/types";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

import { SliceTablePanel } from "./accuracy/SliceTablePanel";
import { TrendChartPanel } from "./accuracy/TrendChartPanel";
import { ChampionPanel } from "./accuracy/ChampionPanel";
import { ShapPanel } from "./accuracy/ShapPanel";
import { BiasCorrectionsPanel } from "./accuracy/BiasCorrectionsPanel";

export function AccuracyTab() {
  const queryClient = useQueryClient();
  const { filters, planningDate } = useGlobalFilterContext();

  // ── Slice / filter state ────────────────────────────────────────────────
  const [sliceGroupBy, setSliceGroupBy] = useState("cluster_assignment");
  const [sliceLag, setSliceLag] = useState(-1);
  const [sliceModels, setSliceModels] = useState("");
  const [sliceKpis, setSliceKpis] = useState<string[]>(["accuracy_pct", "wape", "bias"]);
  const [lagCurveMetric, setLagCurveMetric] = useState("accuracy_pct");
  const [sliceMonths, setSliceMonths] = useState(12);
  const [commonDfus, setCommonDfus] = useState(false);
  const [seasonalityProfile, setSeasonalityProfile] = useState("");
  const [seasonalityProfiles, setSeasonalityProfiles] = useState<string[]>([]);

  // ── Competition config state ────────────────────────────────────────────
  const [competitionConfig, setCompetitionConfig] = useState<CompetitionConfig | null>(null);

  // ── SHAP panel state ────────────────────────────────────────────────────
  const [shapOpen, setShapOpen] = useState(false);
  const [shapModelId, setShapModelId] = useState<string>("");
  const [shapTimeframeIdx, setShapTimeframeIdx] = useState<number | null>(null);
  const [shapCluster, setShapCluster] = useState<string>("all");

  // ── Seasonality profile names (Feature 32) ──────────────────────────────
  useEffect(() => {
    let cancelled = false;
    fetchSeasonalityProfileNames()
      .then((profiles) => { if (!cancelled) setSeasonalityProfiles(profiles); })
      .catch(() => { /* non-blocking */ });
    return () => { cancelled = true; };
  }, []);

  // ── Derived params ──────────────────────────────────────────────────────
  const globalItem = filters.item.length > 0 ? filters.item.join(",") : undefined;
  const globalLocation = filters.location.length > 0 ? filters.location.join(",") : undefined;
  const brandParam = filters.brand.length > 0 ? filters.brand.join(",") : undefined;
  const categoryParam = filters.category.length > 0 ? filters.category.join(",") : undefined;
  const marketParam = filters.market.length > 0 ? filters.market.join(",") : undefined;
  const needDfuCount = sliceKpis.includes("dfu_count");

  const monthFrom = useMemo(() => {
    if (sliceGroupBy === "month_start") return "";
    const anchor = planningDate ? new Date(planningDate + "T00:00:00") : new Date();
    const from = new Date(anchor.getFullYear(), anchor.getMonth() - sliceMonths, 1);
    return from.toISOString().slice(0, 10);
  }, [sliceGroupBy, sliceMonths, planningDate]);

  const sliceParams: SliceParams = useMemo(() => ({
    group_by: sliceGroupBy, lag: sliceLag, models: sliceModels, month_from: monthFrom,
    common_dfus: commonDfus, include_dfu_count: needDfuCount,
    item: globalItem, location: globalLocation, seasonality_profile: seasonalityProfile || undefined,
    time_grain: filters.timeGrain,
    brand: brandParam, category: categoryParam, market: marketParam,
  }), [sliceGroupBy, sliceLag, sliceModels, monthFrom, commonDfus, needDfuCount, globalItem, globalLocation, seasonalityProfile, filters.timeGrain, brandParam, categoryParam, marketParam]);

  const lagCurveParams: LagCurveParams = useMemo(() => ({
    models: sliceModels, month_from: monthFrom, common_dfus: commonDfus,
    include_dfu_count: needDfuCount, item: globalItem, location: globalLocation,
    seasonality_profile: seasonalityProfile || undefined,
    time_grain: filters.timeGrain,
    brand: brandParam, category: categoryParam, market: marketParam,
  }), [sliceModels, monthFrom, commonDfus, needDfuCount, globalItem, globalLocation, seasonalityProfile, filters.timeGrain, brandParam, categoryParam, marketParam]);

  // ── Queries ─────────────────────────────────────────────────────────────
  const { data: slicePayload, isLoading: loadingSlice } = useQuery({
    queryKey: queryKeys.accuracySlice(sliceParams as unknown as Record<string, unknown>), queryFn: () => fetchAccuracySlice(sliceParams), staleTime: STALE.TWO_MIN,
  });
  const { data: lagPayload } = useQuery({
    queryKey: queryKeys.lagCurve(lagCurveParams as unknown as Record<string, unknown>), queryFn: () => fetchLagCurve(lagCurveParams), staleTime: STALE.TWO_MIN,
  });
  const { data: configPayload } = useQuery({
    queryKey: queryKeys.competitionConfig(), queryFn: fetchCompetitionConfig, staleTime: STALE.FIVE_MIN,
    select: (data) => { if (data?.config && competitionConfig === null) setCompetitionConfig(data.config); return data; },
  });
  const { data: summaryPayload } = useQuery({
    queryKey: queryKeys.competitionSummary(), queryFn: fetchCompetitionSummary, staleTime: STALE.FIVE_MIN,
  });
  const { data: shapModelsData } = useQuery({
    queryKey: queryKeys.shapModels(), queryFn: fetchShapModels, staleTime: STALE.TEN_MIN, enabled: shapOpen,
  });
  const shapModels = shapModelsData?.models ?? [];
  const activeShapModel = shapModelId || shapModels[0] || "";
  const { data: shapTimeframesData } = useQuery({
    queryKey: queryKeys.shapTimeframes(activeShapModel), queryFn: () => fetchShapTimeframes(activeShapModel),
    staleTime: STALE.TEN_MIN, enabled: shapOpen && !!activeShapModel,
  });
  const { data: shapSummaryData, isLoading: loadingShapSummary } = useQuery({
    queryKey: queryKeys.shapSummary(activeShapModel, 15), queryFn: () => fetchShapSummary(activeShapModel, 15),
    staleTime: STALE.TEN_MIN, enabled: shapOpen && !!activeShapModel && shapTimeframeIdx === null,
  });
  const { data: shapDetailData, isLoading: loadingShapDetail } = useQuery({
    queryKey: queryKeys.shapTimeframeDetail(activeShapModel, shapTimeframeIdx ?? 0, 15, shapCluster),
    queryFn: () => fetchShapTimeframeDetail(activeShapModel, shapTimeframeIdx!, 15, shapCluster),
    staleTime: STALE.TEN_MIN, enabled: shapOpen && !!activeShapModel && shapTimeframeIdx !== null,
  });
  const { data: shapClustersData } = useQuery({
    queryKey: queryKeys.shapClusters(activeShapModel),
    queryFn: () => fetchShapClusters(activeShapModel),
    staleTime: STALE.TEN_MIN, enabled: shapOpen && !!activeShapModel,
  });

  // ── Mutations ───────────────────────────────────────────────────────────
  const saveConfigMutation = useMutation({ mutationFn: saveCompetitionConfig });
  const runCompetitionMutation = useMutation({
    mutationFn: async (config: CompetitionConfig) => { await saveCompetitionConfig(config); return runCompetition(); },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.competitionSummary() });
      queryClient.invalidateQueries({ queryKey: queryKeys.accuracySlice(sliceParams as unknown as Record<string, unknown>) });
    },
  });

  // ── Derived data ────────────────────────────────────────────────────────
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

  // ── Callbacks ───────────────────────────────────────────────────────────
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

  // ── Render ──────────────────────────────────────────────────────────────
  return (
    <section className="mt-4">
      <Card className="animate-fade-in">
        <CardHeader>
          <div className="flex items-center gap-2">
            <ChartColumn className="h-5 w-5" />
            <CardTitle className="text-base">Accuracy Comparison</CardTitle>
          </div>
          <CardDescription className="max-w-3xl">
            Compare forecast accuracy across models sliced by DFU attribute (cluster, item, location, brand, etc.).
            Use the <strong>Group By</strong> dropdown to change the slice dimension, <strong>Lag</strong> to select
            the forecast horizon (0 = same month, 4 = 4 months ahead), and <strong>Window</strong> to set the
            trailing months of data. The table shows WAPE, accuracy %, and bias for each model per group.
            Below, the <strong>Lag Curve</strong> chart visualizes how accuracy degrades across horizons.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
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
          <TrendChartPanel
            lagCurveData={lagCurveData} lagModels={lagModels}
            sliceKpis={sliceKpis} activeLagMetric={activeLagMetric}
            onLagCurveMetricChange={handleLagCurveMetricChange}
          />
        </CardContent>
      </Card>

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

      <BiasCorrectionsPanel />
    </section>
  );
}
