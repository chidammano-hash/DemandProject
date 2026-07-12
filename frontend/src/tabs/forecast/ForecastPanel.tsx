/**
 * ForecastPanel -- 5th tab in the Model Experimentation Studio.
 *
 * Three-step flow:
 *   Step 1 — Model Readiness: train, generate, promote per model
 *   Step 2 — Algorithm Selection: pick champion or single model
 *   Step 3 — Generate Forecast: configure horizon, generate production forecast
 *
 * Shows production forecast configuration, algorithm picker,
 * generation controls, and recent forecast job history.
 */
import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Zap, Calendar, Clock, Settings2 } from "lucide-react";

import { fetchPipelineConfig, pipelineConfigKeys } from "@/api/queries/unified-model-tuning";
import {
  fetchBacktestSummary,
  fetchTrainingStatus,
  submitTraining,
  fetchStagingSummary,
  submitGenerateForecast,
  fetchPromotionStatus,
  submitPromote,
  backtestMgmtKeys,
  BACKTEST_MGMT_STALE,
  type StagingSummaryMap,
} from "@/api/queries/backtest-management";
import {
  fetchProductionForecastVersions,
  type ProductionForecastVersion,
} from "@/api/queries/production-forecast";
import {
  fetchPromotedChampionExperiment,
  championExperimentKeys,
  CHAMPION_EXP_STALE,
} from "@/api/queries/champion-experiments";
import { fetchJobs } from "@/api/queries/jobs";
import type { Job } from "@/types/jobs";
import { modelLabel } from "@/lib/model-labels";
import { toast } from "@/components/Toaster";
import { formatApiError } from "@/lib/formatApiError";

import { KpiCard } from "@/components/KpiCard";
import { LoadingElement } from "@/components/LoadingElement";
import { timeAgo } from "@/components/shared-tuning-utils";

import {
  forecastPanelKeys,
  STALE,
  deriveForecastAlgos,
  requiresTraining,
  type ForecastAlgorithm,
} from "./forecastPanelShared";
import { ModelReadinessCard } from "./ModelReadinessCard";
import { AlgorithmSelectionCard } from "./AlgorithmSelectionCard";
import { GenerateForecastCard } from "./GenerateForecastCard";
import { RecentJobsCard } from "./RecentJobsCard";

export function isExpectedStagingRunReady(
  candidate: { source_run_id: string; run_status: string } | undefined,
  expectedRunId: string | undefined
): boolean {
  return Boolean(
    expectedRunId &&
    candidate?.source_run_id === expectedRunId &&
    candidate.run_status === "ready"
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ForecastPanel() {
  const queryClient = useQueryClient();

  // -- State ---------------------------------------------------------------
  const [selectedModel, setSelectedModel] = useState<string>("champion");
  const [horizon, setHorizon] = useState<number | null>(null); // null = use config default
  const [includeCI, setIncludeCI] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingModelId, setTrainingModelId] = useState<string | null>(null);
  const [generatingModelId, setGeneratingModelId] = useState<string | null>(null);
  const [pendingRunIds, setPendingRunIds] = useState<Record<string, string>>({});
  const [promotingModelId, setPromotingModelId] = useState<string | null>(null);

  const isGenerating = generatingModelId !== null;

  // -- Queries -------------------------------------------------------------

  // Pipeline config (algorithms + production_forecast section)
  const { data: pipelineConfig, isLoading: configLoading } = useQuery({
    queryKey: pipelineConfigKeys.config,
    queryFn: fetchPipelineConfig,
    staleTime: STALE.CONFIG,
  });

  // Backtest summary (model readiness)
  const { data: backtestSummary } = useQuery({
    queryKey: backtestMgmtKeys.summary,
    queryFn: fetchBacktestSummary,
    staleTime: BACKTEST_MGMT_STALE.SUMMARY,
  });

  // Training status (production training readiness per model)
  const { data: trainingStatus } = useQuery({
    queryKey: backtestMgmtKeys.trainingStatus,
    queryFn: fetchTrainingStatus,
    staleTime: 30_000,
    refetchInterval: isTraining ? 5_000 : false,
  });

  // Staging summary (generated forecast staging data per model)
  const { data: stagingData } = useQuery({
    queryKey: backtestMgmtKeys.stagingSummary,
    queryFn: fetchStagingSummary,
    staleTime: 30_000,
    refetchInterval: isGenerating ? 5_000 : false,
  });

  // Production forecast versions (last generated info)
  const { data: versionsData } = useQuery({
    queryKey: forecastPanelKeys.versions,
    queryFn: fetchProductionForecastVersions,
    staleTime: STALE.VERSIONS,
  });

  // Recent forecast jobs
  const { data: jobsData } = useQuery({
    queryKey: forecastPanelKeys.jobs(0),
    queryFn: () => fetchJobs({ job_type: "generate_production_forecast", limit: 10 }),
    staleTime: STALE.JOBS,
    refetchInterval: isSubmitting ? 5_000 : false,
  });

  // Promoted champion experiment (to know which models are in the active champion)
  const { data: promotedData } = useQuery({
    queryKey: championExperimentKeys.promoted(),
    queryFn: fetchPromotedChampionExperiment,
    staleTime: CHAMPION_EXP_STALE.PROMOTED,
  });

  // Promotion status (which model is currently production)
  const { data: promotionData } = useQuery({
    queryKey: backtestMgmtKeys.promotionStatus,
    queryFn: fetchPromotionStatus,
    staleTime: BACKTEST_MGMT_STALE.PROMOTION ?? 30_000,
    refetchInterval: promotingModelId !== null ? 5_000 : false,
  });

  // -- Derived data --------------------------------------------------------

  const prodConfig = pipelineConfig?.production_forecast;
  const effectiveHorizon = horizon ?? prodConfig?.horizon_months ?? 24;
  const promotedModel = promotionData?.promoted ?? null;
  const staging: StagingSummaryMap = stagingData ?? {};

  const forecastAlgos = useMemo(
    () => deriveForecastAlgos(pipelineConfig?.algorithms, backtestSummary),
    [pipelineConfig?.algorithms, backtestSummary]
  );

  const versions: ProductionForecastVersion[] = versionsData?.versions ?? [];
  const latestVersion = versions.length > 0 ? versions[0] : null;

  const recentJobs: Job[] = jobsData?.jobs ?? [];
  const isForecastRunning = recentJobs.some((j) => j.status === "running" || j.status === "queued");

  // Tree models that are forecastable
  const treeAlgos = forecastAlgos.filter((a) => requiresTraining(a.type));

  // Models ready to generate right now: non-tree models (zero-shot) plus tree
  // models that are production-trained. Drives the "Generate All" button.
  const generatableAlgos = useMemo(
    () =>
      forecastAlgos.filter((a) => {
        if (!requiresTraining(a.type)) return true;
        const st = trainingStatus?.[a.id];
        return Boolean(st?.trained && st?.training_mode === "production");
      }),
    [forecastAlgos, trainingStatus]
  );

  // Count of tree models that are production-trained
  const trainedTreeCount = treeAlgos.filter((a) => {
    const status = trainingStatus?.[a.id];
    return status?.trained && status?.training_mode === "production";
  }).length;

  // Are all tree models production-ready?
  const allTreesTrained = treeAlgos.length > 0 && trainedTreeCount === treeAlgos.length;

  // Promoted champion experiment — use its model list for the champion display
  const promotedExperiment = promotedData?.promoted ?? null;

  // Champion models: from the promoted experiment (may include models not in forecastAlgos)
  // If no promoted experiment, fall back to all algos with compete: true
  const championCompetingAlgos = useMemo(() => {
    const allAlgos = pipelineConfig?.algorithms ?? {};
    const promotedModelIds = new Set(promotedExperiment?.models ?? []);
    if (promotedModelIds.size > 0) {
      return Array.from(promotedModelIds).map((id) => {
        const a = allAlgos[id];
        const summary = backtestSummary?.[id];
        return {
          id,
          type: a?.type ?? "unknown",
          enabled: a?.enabled ?? false,
          forecast: a?.forecast ?? false,
          compete: a?.compete ?? true,
          hasPredictions: summary?.has_predictions ?? false,
          accuracy: summary?.current_accuracy ?? null,
        } as ForecastAlgorithm;
      });
    }
    return forecastAlgos.filter((a) => a.compete);
  }, [pipelineConfig?.algorithms, backtestSummary, promotedExperiment?.models, forecastAlgos]);

  // Champion generation needs current production artifacts; promotion needs one
  // completed immutable champion candidate run returned by the generation API.
  const championConstituents: string[] = promotedExperiment?.models ?? [];
  const championMissingModels = championConstituents.filter((id) => {
    const algorithm = pipelineConfig?.algorithms?.[id];
    if (!algorithm || !requiresTraining(algorithm.type)) return false;
    const status = trainingStatus?.[id];
    return !(status?.trained && status.training_mode === "production");
  });
  const championCanGenerate = promotedExperiment != null && championMissingModels.length === 0;
  const championCandidate = staging.champion;
  const championReady =
    championCandidate?.run_status === "ready" && championCandidate.promotion_eligible;
  const championDfuCount = championCandidate?.dfu_count ?? 0;
  const isChampionPromoted = Boolean(
    championCandidate?.source_run_id &&
    promotedModel?.source_run_id === championCandidate.source_run_id
  );

  // -- Handlers ------------------------------------------------------------

  async function handleTrain(modelId: string) {
    setIsTraining(true);
    setTrainingModelId(modelId);
    try {
      await submitTraining(modelId);
      // Polling via refetchInterval will pick up completion
    } catch (err) {
      console.error("Training failed:", err);
      setIsTraining(false);
      setTrainingModelId(null);
    }
  }

  async function handleTrainAll() {
    setIsTraining(true);
    setTrainingModelId("__all__");
    try {
      // Submit training for all untrained tree models sequentially
      for (const algo of treeAlgos) {
        const status = trainingStatus?.[algo.id];
        if (!status?.trained || status?.training_mode !== "production") {
          await submitTraining(algo.id);
        }
      }
    } catch (err) {
      console.error("Train All failed:", err);
    }
    // Keep polling — will auto-clear when all trained
  }

  async function handleGenerate(modelId: string) {
    setGeneratingModelId(modelId);
    try {
      const submitted = await submitGenerateForecast(modelId, {
        horizon: effectiveHorizon,
        confidenceIntervals: includeCI,
      });
      setPendingRunIds((current) => ({
        ...current,
        [modelId]: submitted.source_run_id,
      }));
      toast.success(`Generating forecast for ${modelLabel(modelId)}…`);
    } catch (err) {
      toast.error(formatApiError(err));
      setGeneratingModelId(null);
    }
  }

  // Generate staging forecasts for every ready model in one click. Submissions
  // run sequentially; the staging-poll effect clears the "__all__" spinner once
  // all models have staged rows.
  async function handleGenerateAll() {
    if (generatableAlgos.length === 0) {
      toast.info("No models are ready to generate — train the tree models first.");
      return;
    }
    setGeneratingModelId("__all__");
    toast.info(
      `Generating forecasts for ${generatableAlgos.length} model${generatableAlgos.length === 1 ? "" : "s"}…`
    );
    let failures = 0;
    const submittedRuns: Record<string, string> = {};
    for (const algo of generatableAlgos) {
      try {
        const submitted = await submitGenerateForecast(algo.id, {
          horizon: effectiveHorizon,
          confidenceIntervals: includeCI,
        });
        submittedRuns[algo.id] = submitted.source_run_id;
      } catch (err) {
        failures += 1;
        toast.error(`${modelLabel(algo.id)}: ${formatApiError(err)}`);
      }
    }
    setPendingRunIds((current) => ({ ...current, ...submittedRuns }));
    queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.stagingSummary });
    if (failures === generatableAlgos.length) {
      // Nothing was queued — clear the spinner now (poll effect won't fire).
      setGeneratingModelId(null);
    }
  }

  async function handlePromote(modelId: string) {
    const candidate = staging[modelId];
    if (
      !candidate?.source_run_id ||
      candidate.run_status !== "ready" ||
      !candidate.promotion_eligible
    ) {
      toast.error("Generate a new release candidate before promoting.");
      return;
    }
    setPromotingModelId(modelId);
    try {
      await submitPromote(modelId, candidate.source_run_id);
      toast.success(
        modelId === "champion"
          ? "Champion promoted to production."
          : `${modelLabel(modelId)} promoted to production.`
      );
      queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.promotionStatus });
      queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.stagingSummary });
      queryClient.invalidateQueries({ queryKey: forecastPanelKeys.versions });
    } catch (err) {
      // Surfaces the promotion gate's 409 (WAPE/coverage) or 400 (no staged
      // rows / missing champion winners) detail instead of failing silently.
      toast.error(formatApiError(err));
    } finally {
      setPromotingModelId(null);
    }
  }

  // Generate champion forecasts via the legacy production forecast job — routes
  // per-DFU using champion assignments.
  async function handleGenerateChampion() {
    setGeneratingModelId("champion");
    try {
      await submitChampionJob();
      toast.success("Champion forecast generation queued.");
      queryClient.invalidateQueries({ queryKey: forecastPanelKeys.jobs(0) });
      queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.stagingSummary });
    } catch (err) {
      toast.error(formatApiError(err));
      setGeneratingModelId(null);
    }
  }

  // Effect: stop polling once training completes
  // Check if training is done when trainingStatus updates
  useEffect(() => {
    if (!isTraining || !trainingStatus) return;

    if (trainingModelId === "__all__") {
      // All tree models — check if every one is now production-trained
      const allDone = treeAlgos.every((a) => {
        const st = trainingStatus[a.id];
        return st?.trained && st?.training_mode === "production";
      });
      if (allDone) {
        setIsTraining(false);
        setTrainingModelId(null);
      }
    } else if (trainingModelId) {
      const st = trainingStatus[trainingModelId];
      if (st?.trained && st?.training_mode === "production") {
        setIsTraining(false);
        setTrainingModelId(null);
      }
    }
  }, [trainingStatus, isTraining, trainingModelId, treeAlgos]);

  // Effect: stop polling once generate completes (staging row appears)
  useEffect(() => {
    if (!generatingModelId || !stagingData) return;
    if (generatingModelId === "__all__") {
      // Clear only when each newly submitted run—not an older staged run—is ready.
      const allGenerated =
        generatableAlgos.length > 0 &&
        generatableAlgos.every((a) => {
          const expectedRun = pendingRunIds[a.id];
          const current = stagingData[a.id];
          return isExpectedStagingRunReady(current, expectedRun);
        });
      if (allGenerated) setGeneratingModelId(null);
      return;
    }
    const s = stagingData[generatingModelId];
    if (isExpectedStagingRunReady(s, pendingRunIds[generatingModelId])) {
      setGeneratingModelId(null);
    }
  }, [stagingData, generatingModelId, generatableAlgos, pendingRunIds]);

  /** Submit a run-scoped champion job using the promoted DFU routing artifact. */
  async function submitChampionJob(): Promise<string> {
    const submitted = await submitGenerateForecast("champion", {
      horizon: effectiveHorizon,
      confidenceIntervals: includeCI,
    });
    setPendingRunIds((current) => ({
      ...current,
      champion: submitted.source_run_id,
    }));
    return submitted.source_run_id;
  }

  async function handleGenerateForecast() {
    setIsSubmitting(true);
    setGeneratingModelId(selectedModel);
    try {
      if (selectedModel === "champion") {
        await submitChampionJob();
      } else {
        // For a specific model: use the new generate endpoint, threading the
        // panel's horizon + CI toggle (previously dropped here).
        const submitted = await submitGenerateForecast(selectedModel, {
          horizon: effectiveHorizon,
          confidenceIntervals: includeCI,
        });
        setPendingRunIds((current) => ({
          ...current,
          [selectedModel]: submitted.source_run_id,
        }));
      }
      toast.success(
        selectedModel === "champion"
          ? "Champion forecast generation queued."
          : `Forecast generation queued for ${modelLabel(selectedModel)}.`
      );
      queryClient.invalidateQueries({ queryKey: forecastPanelKeys.jobs(0) });
      queryClient.invalidateQueries({ queryKey: forecastPanelKeys.versions });
      queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.stagingSummary });
    } catch (err) {
      toast.error(formatApiError(err));
      setGeneratingModelId(null);
    } finally {
      setIsSubmitting(false);
    }
  }

  // -- Loading state -------------------------------------------------------

  if (configLoading) {
    return <LoadingElement message="Loading forecast configuration..." />;
  }

  // -- Render --------------------------------------------------------------

  return (
    <div className="space-y-4">
      {/* KPI Row */}
      <div className="grid grid-cols-4 gap-3">
        <KpiCard
          label="Last Plan Version"
          value={latestVersion?.plan_version ?? "--"}
          icon={Calendar}
        />
        <KpiCard label="Forecast Horizon" value={`${effectiveHorizon} months`} icon={Zap} />
        <KpiCard
          label="Default Model"
          value={
            prodConfig?.cold_start_model_id
              ? modelLabel(pipelineConfig?.champion?.fallback_model_id ?? "lgbm_cluster")
              : "--"
          }
          sublabel="(champion)"
          icon={Settings2}
        />
        <KpiCard
          label="Last Generated"
          value={latestVersion?.generated_at ? timeAgo(latestVersion.generated_at) : "--"}
          icon={Clock}
        />
      </div>

      {/* ══════ STEP 1: Model Readiness ══════ */}
      <ModelReadinessCard
        forecastAlgos={forecastAlgos}
        trainingStatus={trainingStatus}
        staging={staging}
        treeAlgos={treeAlgos}
        trainedTreeCount={trainedTreeCount}
        allTreesTrained={allTreesTrained}
        isTraining={isTraining}
        trainingModelId={trainingModelId}
        generatingModelId={generatingModelId}
        isGenerating={isGenerating}
        promotingModelId={promotingModelId}
        isSubmitting={isSubmitting}
        promotedModel={promotedModel}
        promotedExperiment={promotedExperiment}
        championConstituents={championConstituents}
        championMissingModels={championMissingModels}
        championCanGenerate={championCanGenerate}
        championReady={championReady}
        championDfuCount={championDfuCount}
        isChampionPromoted={isChampionPromoted}
        onTrain={handleTrain}
        onTrainAll={handleTrainAll}
        onGenerate={handleGenerate}
        onGenerateAll={handleGenerateAll}
        generatableCount={generatableAlgos.length}
        onPromote={handlePromote}
        onGenerateChampion={handleGenerateChampion}
      />

      {/* ══════ STEP 2: Algorithm Selection ══════ */}
      <div className="grid grid-cols-3 gap-4">
        {/* Left: Algorithm Picker (2 cols) */}
        <AlgorithmSelectionCard
          forecastAlgos={forecastAlgos}
          championCompetingAlgos={championCompetingAlgos}
          selectedModel={selectedModel}
          onSelectModel={setSelectedModel}
          trainingStatus={trainingStatus}
          staging={staging}
          promotedExperiment={promotedExperiment}
          fallbackModelId={pipelineConfig?.champion?.fallback_model_id ?? "lgbm_cluster"}
          prodConfig={prodConfig}
        />

        {/* Right: Action panel (1 col) */}
        <GenerateForecastCard
          selectedModel={selectedModel}
          effectiveHorizon={effectiveHorizon}
          onHorizonChange={setHorizon}
          includeCI={includeCI}
          onIncludeCIChange={setIncludeCI}
          isSubmitting={isSubmitting}
          isForecastRunning={isForecastRunning}
          onGenerateForecast={handleGenerateForecast}
          latestVersion={latestVersion}
        />
      </div>

      {/* Recent Forecast Jobs */}
      <RecentJobsCard recentJobs={recentJobs} />
    </div>
  );
}
