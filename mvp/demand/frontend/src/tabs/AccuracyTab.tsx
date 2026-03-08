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
  const { filters } = useGlobalFilterContext();

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
  const needDfuCount = sliceKpis.includes("dfu_count");

  const monthFrom = useMemo(() => {
    if (sliceGroupBy === "month_start") return "";
    const now = new Date();
    const from = new Date(now.getFullYear(), now.getMonth() - sliceMonths, 1);
    return from.toISOString().slice(0, 10);
  }, [sliceGroupBy, sliceMonths]);

  const sliceParams: SliceParams = useMemo(() => ({
    group_by: sliceGroupBy, lag: sliceLag, models: sliceModels, month_from: monthFrom,
    common_dfus: commonDfus, include_dfu_count: needDfuCount,
    item: globalItem, location: globalLocation, seasonality_profile: seasonalityProfile || undefined,
  }), [sliceGroupBy, sliceLag, sliceModels, monthFrom, commonDfus, needDfuCount, globalItem, globalLocation, seasonalityProfile]);

  const lagCurveParams: LagCurveParams = useMemo(() => ({
    models: sliceModels, month_from: monthFrom, common_dfus: commonDfus,
    include_dfu_count: needDfuCount, item: globalItem, location: globalLocation,
    seasonality_profile: seasonalityProfile || undefined,
  }), [sliceModels, monthFrom, commonDfus, needDfuCount, globalItem, globalLocation, seasonalityProfile]);

  // ── Queries ─────────────────────────────────────────────────────────────
  const { data: slicePayload, isLoading: loadingSlice } = useQuery({
    queryKey: queryKeys.accuracySlice(sliceParams), queryFn: () => fetchAccuracySlice(sliceParams), staleTime: STALE.TWO_MIN,
  });
  const { data: lagPayload } = useQuery({
    queryKey: queryKeys.lagCurve(lagCurveParams), queryFn: () => fetchLagCurve(lagCurveParams), staleTime: STALE.TWO_MIN,
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
    queryKey: queryKeys.shapTimeframeDetail(activeShapModel, shapTimeframeIdx ?? 0, 15),
    queryFn: () => fetchShapTimeframeDetail(activeShapModel, shapTimeframeIdx!, 15),
    staleTime: STALE.TEN_MIN, enabled: shapOpen && !!activeShapModel && shapTimeframeIdx !== null,
  });

  // ── Mutations ───────────────────────────────────────────────────────────
  const saveConfigMutation = useMutation({ mutationFn: saveCompetitionConfig });
  const runCompetitionMutation = useMutation({
    mutationFn: async (config: CompetitionConfig) => { await saveCompetitionConfig(config); return runCompetition(); },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.competitionSummary() });
      queryClient.invalidateQueries({ queryKey: queryKeys.accuracySlice(sliceParams) });
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
  const handleShapModelChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => { setShapModelId(e.target.value); setShapTimeframeIdx(null); }, []);
  const handleShapTimeframeChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => setShapTimeframeIdx(e.target.value === "summary" ? null : Number(e.target.value)), []);

  // ── Render ──────────────────────────────────────────────────────────────
  return (
    <section className="mt-4">
      <Card className="animate-fade-in">
        <CardHeader>
          <div className="flex items-center gap-2">
            <ChartColumn className="h-5 w-5" />
            <CardTitle className="text-base">Accuracy Comparison</CardTitle>
          </div>
          <CardDescription>
            Compare forecast accuracy across models by DFU attribute. Uses pre-aggregated views for fast results.
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
        onToggleOpen={() => setShapOpen((v) => !v)}
        onModelChange={handleShapModelChange}
        onTimeframeChange={handleShapTimeframeChange}
      />

      <BiasCorrectionsPanel />
    </section>
  );
}
