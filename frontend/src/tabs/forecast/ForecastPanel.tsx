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
  fetchStagingSummary,
  submitGenerateForecast,
  fetchPromotionStatus,
  submitStageForecast,
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
import { isForecastModelId, modelLabel } from "@/lib/model-labels";
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
import { useForecastTraining } from "./useForecastTraining";

const EMPTY_JOBS: Job[] = [];

export function isExpectedStagingRunReady(
  candidate: { source_run_id: string; run_status: string } | undefined,
  expectedRunId: string | undefined
): boolean {
  return Boolean(
    expectedRunId && candidate?.source_run_id === expectedRunId && candidate.run_status === "ready"
  );
}

export function resolveConfidenceIntervals(
  configEnabled: boolean | undefined,
  override: boolean | undefined
): boolean {
  return override ?? configEnabled ?? true;
}

export function buildForecastGenerationOptions(
  horizon: number,
  confidenceIntervalsOverride: boolean | undefined
): { horizon: number; confidenceIntervals?: boolean } {
  return confidenceIntervalsOverride === undefined
    ? { horizon }
    : { horizon, confidenceIntervals: confidenceIntervalsOverride };
}

export function findFailedGenerationModels(
  jobs: Job[],
  pendingRunIds: Record<string, string>,
  modelIds: string[]
): Array<{ modelId: string; job: Job }> {
  return modelIds.flatMap((modelId) => {
    const expectedRunId = pendingRunIds[modelId];
    if (!expectedRunId) return [];
    const failedJob = jobs.find(
      (job) =>
        (job.status === "failed" || job.status === "cancelled") &&
        job.params.run_id === expectedRunId
    );
    return failedJob ? [{ modelId, job: failedJob }] : [];
  });
}

export function generationFailureMessage(modelId: string): string {
  return `${modelLabel(modelId)} generation failed. Open Jobs for details.`;
}

export function productionPromotionBlockedReason(
  isForecastRunning: boolean,
  candidateStaged: boolean
): string | undefined {
  if (isForecastRunning) {
    return "Wait for the active forecast generation job to finish before promoting.";
  }
  if (!candidateStaged) {
    return "Promote the selected generated candidate to staging first.";
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ForecastPanel() {
  const queryClient = useQueryClient();

  // -- State ---------------------------------------------------------------
  const [selectedModel, setSelectedModel] = useState<string>("champion");
  const [horizon, setHorizon] = useState<number | null>(null); // null = use config default
  const [confidenceIntervalsOverride, setConfidenceIntervalsOverride] = useState<
    boolean | undefined
  >(undefined);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [generatingModelId, setGeneratingModelId] = useState<string | null>(null);
  const [pendingRunIds, setPendingRunIds] = useState<Record<string, string>>({});
  const [pendingBatchModelIds, setPendingBatchModelIds] = useState<string[]>([]);
  const [promotingModelId, setPromotingModelId] = useState<string | null>(null);
  const [stagingModelId, setStagingModelId] = useState<string | null>(null);
  const { isTraining, trainingModelId, train, trainAll } = useForecastTraining();

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
    refetchInterval: isSubmitting || isGenerating ? 5_000 : false,
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
  const includeCI = resolveConfidenceIntervals(
    prodConfig?.confidence_interval?.enabled,
    confidenceIntervalsOverride
  );
  const generationOptions = buildForecastGenerationOptions(
    effectiveHorizon,
    confidenceIntervalsOverride
  );
  const promotedModel = promotionData?.promoted ?? null;
  const staging: StagingSummaryMap = stagingData ?? {};

  const forecastAlgos = useMemo(
    () => deriveForecastAlgos(pipelineConfig?.algorithms, backtestSummary),
    [pipelineConfig?.algorithms, backtestSummary]
  );

  const versions: ProductionForecastVersion[] = versionsData?.versions ?? [];
  const latestVersion = versions.length > 0 ? versions[0] : null;

  const recentJobs = jobsData?.jobs ?? EMPTY_JOBS;
  const isForecastRunning = recentJobs.some((j) => j.status === "running" || j.status === "queued");

  const trainableAlgos = useMemo(
    () => forecastAlgos.filter((algo) => requiresTraining(algo.type)),
    [forecastAlgos]
  );
  const trainedArtifactCount = trainableAlgos.filter((algo) => {
    const status = trainingStatus?.[algo.id];
    return status?.ready === true;
  }).length;
  const allRequiredArtifactsReady =
    trainableAlgos.length > 0 && trainedArtifactCount === trainableAlgos.length;

  // MSTL and Chronos 2E infer directly. LightGBM, N-HiTS, and N-BEATS become
  // generatable after their immutable production artifacts are ready.
  const generatableAlgos = useMemo(
    () =>
      forecastAlgos.filter((a) => {
        if (!requiresTraining(a.type)) return true;
        const st = trainingStatus?.[a.id];
        return st?.ready === true;
      }),
    [forecastAlgos, trainingStatus]
  );

  // Promoted champion experiment — use its model list for the champion display
  const promotedExperiment = promotedData?.promoted ?? null;

  // Champion models come only from the atomically assigned experiment.
  const championCompetingAlgos = useMemo(() => {
    const allAlgos = pipelineConfig?.algorithms ?? {};
    const promotedModelIds = new Set((promotedExperiment?.models ?? []).filter(isForecastModelId));
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
    return [];
  }, [pipelineConfig?.algorithms, backtestSummary, promotedExperiment?.models]);

  // Champion generation needs current production artifacts; promotion needs one
  // completed immutable champion candidate run returned by the generation API.
  const championConstituents: string[] = (promotedExperiment?.models ?? []).filter(
    isForecastModelId
  );
  const championCandidate = staging.champion;
  const championReady = Boolean(
    championCandidate?.run_status === "ready" && championCandidate.promotion_eligible
  );
  const championDfuCount = championCandidate?.dfu_count ?? 0;
  const isChampionPromoted = Boolean(
    championCandidate?.source_run_id &&
    promotedModel?.model_id === "champion" &&
    promotedModel?.source_run_id === championCandidate.source_run_id
  );
  const selectedCandidate = staging[selectedModel];
  const isSelectedPromoted = Boolean(
    selectedCandidate?.source_run_id &&
    promotedModel?.model_id === selectedModel &&
    promotedModel.source_run_id === selectedCandidate.source_run_id
  );
  const selectedCandidateGenerated = Boolean(
    selectedCandidate?.source_run_id &&
    (selectedCandidate.run_status === "ready" || selectedCandidate.run_status === "promoted")
  );
  const selectedCandidateStaged = Boolean(
    selectedCandidateGenerated && (selectedCandidate?.promotion_eligible || isSelectedPromoted)
  );
  const promotionBlockedReason = productionPromotionBlockedReason(
    isForecastRunning,
    selectedCandidateStaged
  );

  // -- Handlers ------------------------------------------------------------

  async function handleGenerate(modelId: string) {
    setGeneratingModelId(modelId);
    try {
      const submitted = await submitGenerateForecast(modelId, generationOptions);
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
      toast.info("No models are ready to generate — train the required production models first.");
      return;
    }
    setGeneratingModelId("__all__");
    setPendingBatchModelIds([]);
    toast.info(
      `Generating forecasts for ${generatableAlgos.length} model${generatableAlgos.length === 1 ? "" : "s"}…`
    );
    const submittedRuns: Record<string, string> = {};
    for (const algo of generatableAlgos) {
      try {
        const submitted = await submitGenerateForecast(algo.id, generationOptions);
        submittedRuns[algo.id] = submitted.source_run_id;
      } catch (err) {
        toast.error(`${modelLabel(algo.id)}: ${formatApiError(err)}`);
      }
    }
    setPendingRunIds((current) => ({ ...current, ...submittedRuns }));
    const submittedModelIds = Object.keys(submittedRuns);
    setPendingBatchModelIds(submittedModelIds);
    queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.stagingSummary });
    if (submittedModelIds.length === 0) {
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

  async function handleStage(modelId: string) {
    const candidate = staging[modelId];
    if (!candidate?.source_run_id || candidate.run_status !== "ready") {
      toast.error("Generate a draft candidate before promoting it to staging.");
      return;
    }
    setStagingModelId(modelId);
    try {
      await submitStageForecast(modelId, candidate.source_run_id);
      toast.success(`${modelLabel(modelId)} promoted to staging.`);
      queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.stagingSummary });
    } catch (err) {
      toast.error(formatApiError(err));
    } finally {
      setStagingModelId(null);
    }
  }

  // Effect: stop polling once generate completes (staging row appears)
  useEffect(() => {
    if (!generatingModelId || !stagingData) return;
    if (generatingModelId === "__all__") {
      // Clear only when each newly submitted run—not an older staged run—is ready.
      const allGenerated =
        pendingBatchModelIds.length > 0 &&
        pendingBatchModelIds.every((modelId) => {
          const expectedRun = pendingRunIds[modelId];
          const current = stagingData[modelId];
          return isExpectedStagingRunReady(current, expectedRun);
        });
      if (allGenerated) {
        setGeneratingModelId(null);
        setPendingBatchModelIds([]);
      }
      return;
    }
    const s = stagingData[generatingModelId];
    if (isExpectedStagingRunReady(s, pendingRunIds[generatingModelId])) {
      setGeneratingModelId(null);
    }
  }, [stagingData, generatingModelId, pendingBatchModelIds, pendingRunIds]);

  // A failed/cancelled job never produces a ready staging row. Stop waiting for
  // that exact run and keep a Generate All batch polling only its successful
  // submissions, so one failure cannot leave the UI spinning forever.
  useEffect(() => {
    if (!generatingModelId) return;
    const activeModelIds =
      generatingModelId === "__all__" ? pendingBatchModelIds : [generatingModelId];
    const failures = findFailedGenerationModels(recentJobs, pendingRunIds, activeModelIds);
    if (failures.length === 0) return;

    failures.forEach(({ modelId }) => {
      toast.error(generationFailureMessage(modelId));
    });

    if (generatingModelId === "__all__") {
      const failedModelIds = new Set(failures.map(({ modelId }) => modelId));
      const remainingModelIds = pendingBatchModelIds.filter(
        (modelId) => !failedModelIds.has(modelId)
      );
      setPendingBatchModelIds(remainingModelIds);
      if (remainingModelIds.length === 0) setGeneratingModelId(null);
    } else {
      setGeneratingModelId(null);
    }
  }, [generatingModelId, pendingBatchModelIds, pendingRunIds, recentJobs]);

  /** Submit a run-scoped champion job using the promoted DFU routing artifact. */
  async function submitChampionJob(): Promise<string> {
    const submitted = await submitGenerateForecast("champion", generationOptions);
    setPendingRunIds((current) => ({
      ...current,
      champion: submitted.source_run_id,
    }));
    return submitted.source_run_id;
  }

  async function handleGenerateForecast() {
    if (selectedModel === "champion" && !promotedExperiment) {
      toast.error("Select and assign a completed experiment in Champion first.");
      return;
    }
    setIsSubmitting(true);
    setGeneratingModelId(selectedModel);
    try {
      if (selectedModel === "champion") {
        await submitChampionJob();
      } else {
        // For a specific model: use the new generate endpoint, threading the
        // panel's horizon + CI toggle (previously dropped here).
        const submitted = await submitGenerateForecast(selectedModel, generationOptions);
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
        trainableAlgos={trainableAlgos}
        trainedArtifactCount={trainedArtifactCount}
        allRequiredArtifactsReady={allRequiredArtifactsReady}
        isTraining={isTraining}
        trainingModelId={trainingModelId}
        generatingModelId={generatingModelId}
        isGenerating={isGenerating}
        promotedExperiment={promotedExperiment}
        championConstituents={championConstituents}
        championReady={championReady}
        championDfuCount={championDfuCount}
        isChampionPromoted={isChampionPromoted}
        activeProductionModelId={promotedModel?.model_id ?? null}
        onTrain={train}
        onTrainAll={trainAll}
        onGenerate={handleGenerate}
        onGenerateAll={handleGenerateAll}
        generatableCount={generatableAlgos.length}
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
          onIncludeCIChange={setConfidenceIntervalsOverride}
          isSubmitting={isSubmitting}
          isForecastRunning={isForecastRunning}
          candidateGenerated={selectedCandidateGenerated}
          candidateStaged={selectedCandidateStaged}
          candidateDfuCount={selectedCandidate?.dfu_count}
          isStaging={stagingModelId === selectedModel}
          isPromoting={promotingModelId === selectedModel}
          isSelectedPromoted={isSelectedPromoted}
          blockedReason={
            selectedModel === "champion" && !promotedExperiment
              ? "Select and assign a completed experiment in Champion first."
              : undefined
          }
          promotionBlockedReason={promotionBlockedReason}
          onGenerateForecast={handleGenerateForecast}
          onStage={() => handleStage(selectedModel)}
          onPromote={() => handlePromote(selectedModel)}
          latestVersion={latestVersion}
        />
      </div>

      {/* Recent Forecast Jobs */}
      <RecentJobsCard recentJobs={recentJobs} />
    </div>
  );
}
