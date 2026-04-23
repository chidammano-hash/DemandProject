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
import { useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Zap,
  Calendar,
  Clock,
  Settings2,
  Loader2,
  CheckCircle2,
  XCircle,
  Play,
  AlertTriangle,
  Dumbbell,
  BarChart3,
  Crown,
  RotateCcw,
} from "lucide-react";

import {
  fetchPipelineConfig,
  pipelineConfigKeys,
  type PipelineAlgorithm,
} from "@/api/queries/unified-model-tuning";
import {
  fetchBacktestSummary,
  fetchTrainingStatus,
  submitTraining,
  fetchStagingSummary,
  submitGenerateForecast,
  fetchPromotionStatus,
  submitPromote,
  submitBacktestLoad,
  backtestMgmtKeys,
  BACKTEST_MGMT_STALE,
  type BacktestSummary,
  type TrainingStatusMap,
  type StagingSummaryMap,
  type PromotionStatus,
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
import { fetchJobs, submitJob } from "@/api/queries/jobs";
import type { Job } from "@/types/jobs";
import { modelLabel, MODEL_TYPE_COLORS } from "@/lib/model-labels";
import { cn } from "@/lib/utils";
import { timeAgo, StatusBadge, formatDuration } from "@/components/shared-tuning-utils";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { KpiCard } from "@/components/KpiCard";
import { LoadingElement } from "@/components/LoadingElement";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

// ---------------------------------------------------------------------------
// Query keys & stale times
// ---------------------------------------------------------------------------

const forecastPanelKeys = {
  versions: ["forecast-panel", "versions"] as const,
  jobs: (offset: number) => ["forecast-panel", "jobs", offset] as const,
};

const STALE = {
  VERSIONS: 30_000,
  CONFIG: 60_000,
  JOBS: 10_000,
} as const;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface ForecastAlgorithm {
  id: string;
  type: string;
  enabled: boolean;
  forecast: boolean;
  compete: boolean;
  hasPredictions: boolean;
  accuracy: number | null;
}

function deriveForecastAlgos(
  algorithms: Record<string, PipelineAlgorithm> | undefined,
  backtestSummary: BacktestSummary | undefined,
): ForecastAlgorithm[] {
  if (!algorithms) return [];
  return Object.entries(algorithms)
    .filter(([, a]) => a.forecast)
    .map(([id, a]) => {
      const summary = backtestSummary?.[id];
      return {
        id,
        type: a.type,
        enabled: a.enabled,
        forecast: a.forecast,
        compete: a.compete,
        hasPredictions: summary?.has_predictions ?? false,
        accuracy: summary?.current_accuracy ?? null,
      };
    });
}

/** True for model types that need explicit .pkl training before inference. */
function requiresTraining(type: string): boolean {
  return type === "tree";
}

// ---------------------------------------------------------------------------
// Training status badge
// ---------------------------------------------------------------------------

function TrainingStatusIndicator({
  trained,
  trainingMode,
  trainedAt,
  needsTraining,
}: {
  trained: boolean;
  trainingMode: string | null;
  trainedAt: string | null;
  needsTraining: boolean;
}) {
  if (!needsTraining) {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
        <CheckCircle2 className="h-3 w-3 text-blue-500" />
        No training needed
      </span>
    );
  }

  if (!trained) {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-red-600 dark:text-red-400">
        <XCircle className="h-3 w-3" />
        Not Trained
      </span>
    );
  }

  if (trainingMode === "production") {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-emerald-600 dark:text-emerald-400">
        <CheckCircle2 className="h-3 w-3" />
        Production Ready
        {trainedAt && (
          <span className="text-muted-foreground ml-1">({timeAgo(trainedAt)})</span>
        )}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400">
      <AlertTriangle className="h-3 w-3" />
      Backtest Only
      {trainedAt && (
        <span className="text-muted-foreground ml-1">({timeAgo(trainedAt)})</span>
      )}
    </span>
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
    [pipelineConfig?.algorithms, backtestSummary],
  );

  const versions: ProductionForecastVersion[] = versionsData?.versions ?? [];
  const latestVersion = versions.length > 0 ? versions[0] : null;

  const recentJobs: Job[] = jobsData?.jobs ?? [];
  const isForecastRunning = recentJobs.some(
    (j) => j.status === "running" || j.status === "queued",
  );

  // Tree models that are forecastable
  const treeAlgos = forecastAlgos.filter((a) => requiresTraining(a.type));

  // Count of tree models that are production-trained
  const trainedTreeCount = treeAlgos.filter((a) => {
    const status = trainingStatus?.[a.id];
    return status?.trained && status?.training_mode === "production";
  }).length;

  // Are all tree models production-ready?
  const allTreesTrained = treeAlgos.length > 0 && trainedTreeCount === treeAlgos.length;

  // Promoted champion experiment — use its model list for the champion display
  const promotedExperiment = promotedData?.promoted ?? null;
  const promotedModelIds = new Set(promotedExperiment?.models ?? []);

  // Champion models: from the promoted experiment (may include models not in forecastAlgos)
  // If no promoted experiment, fall back to all algos with compete: true
  const championCompetingAlgos = useMemo(() => {
    const allAlgos = pipelineConfig?.algorithms ?? {};
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
  }, [pipelineConfig?.algorithms, backtestSummary, promotedModelIds, forecastAlgos]);

  // Count models with staged forecasts (for Promote Champion button)
  const stagedModelCount = Object.keys(staging).filter(
    (id) => staging[id].row_count > 0,
  ).length;

  // Champion row prerequisites: all constituent models need staged forecasts.
  const championConstituents: string[] = promotedExperiment?.models ?? [];
  const championMissingModels = championConstituents.filter(
    (id) => (staging[id]?.row_count ?? 0) === 0,
  );
  const championReady =
    championConstituents.length > 0 && championMissingModels.length === 0;
  const championDfuCount = championReady
    ? Math.max(0, ...championConstituents.map((id) => staging[id]?.dfu_count ?? 0))
    : 0;
  const isChampionPromoted = promotedModel?.model_id === "champion";

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
      await submitGenerateForecast(modelId);
    } catch (err) {
      console.error("Generate failed:", err);
      setGeneratingModelId(null);
    }
  }

  async function handlePromote(modelId: string) {
    setPromotingModelId(modelId);
    try {
      await submitPromote(modelId);
      queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.promotionStatus });
      queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.stagingSummary });
      queryClient.invalidateQueries({ queryKey: forecastPanelKeys.versions });
    } catch (err) {
      console.error("Promote failed:", err);
    } finally {
      setPromotingModelId(null);
    }
  }

  // Effect: stop polling once training completes
  // Check if training is done when trainingStatus updates
  useMemo(() => {
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
  useMemo(() => {
    if (generatingModelId && stagingData) {
      const s = stagingData[generatingModelId];
      if (s && s.row_count > 0) {
        setGeneratingModelId(null);
      }
    }
  }, [stagingData, generatingModelId]);

  /** Submit the legacy champion job (per-DFU routing via assignments). */
  async function submitChampionJob() {
    const params: Record<string, unknown> = { horizon: effectiveHorizon };
    if (includeCI) params.confidence_intervals = true;
    await submitJob("generate_production_forecast", params, "Production Forecast");
  }

  async function handleGenerateForecast() {
    setIsSubmitting(true);
    try {
      if (selectedModel === "champion") {
        await submitChampionJob();
      } else {
        // For a specific model: use the new generate endpoint
        await submitGenerateForecast(selectedModel);
      }
      queryClient.invalidateQueries({ queryKey: forecastPanelKeys.jobs(0) });
      queryClient.invalidateQueries({ queryKey: forecastPanelKeys.versions });
      queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.stagingSummary });
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
        <KpiCard
          label="Forecast Horizon"
          value={`${effectiveHorizon} months`}
          icon={Zap}
        />
        <KpiCard
          label="Default Model"
          value={prodConfig?.cold_start_model_id
            ? modelLabel(pipelineConfig?.champion?.fallback_model_id ?? "lgbm_cluster")
            : "--"}
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
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <Dumbbell className="h-4 w-4" />
              Step 1: Model Readiness
            </CardTitle>
            <div className="flex items-center gap-3">
              <span className="text-xs text-muted-foreground">
                {trainedTreeCount}/{treeAlgos.length} tree models production-ready
              </span>
              {!allTreesTrained && treeAlgos.length > 0 && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleTrainAll}
                  disabled={isTraining || allTreesTrained}
                >
                  {isTraining && trainingModelId === "__all__" ? (
                    <>
                      <Loader2 className="mr-1.5 h-3 w-3 animate-spin" />
                      Training All...
                    </>
                  ) : (
                    <>
                      <Dumbbell className="mr-1.5 h-3 w-3" />
                      Train All Tree Models
                    </>
                  )}
                </Button>
              )}
              {allTreesTrained && (
                <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200 text-[10px]">
                  All Ready
                </Badge>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs">Model</TableHead>
                <TableHead className="text-xs">Type</TableHead>
                <TableHead className="text-xs">Accuracy</TableHead>
                <TableHead className="text-xs">Training</TableHead>
                <TableHead className="text-xs">Staging</TableHead>
                <TableHead className="text-xs text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {promotedExperiment && (
                <TableRow key="champion" className="bg-amber-50/40 dark:bg-amber-950/20">
                  <TableCell>
                    <div className="flex items-center gap-1.5">
                      <Crown className="h-3.5 w-3.5 text-amber-600" />
                      <div>
                        <div className="text-sm font-medium">Champion</div>
                        <div className="text-[10px] text-muted-foreground">
                          {championConstituents.join(", ") || "no promoted experiment"}
                        </div>
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge className="bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200 text-[10px] px-1.5 py-0">
                      ensemble
                    </Badge>
                  </TableCell>
                  <TableCell className="text-sm tabular-nums">
                    {promotedExperiment.champion_accuracy != null ? (
                      `${promotedExperiment.champion_accuracy.toFixed(1)}%`
                    ) : (
                      <span className="text-muted-foreground">--</span>
                    )}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className="text-[10px] gap-0.5 text-blue-600 border-blue-200">
                      <CheckCircle2 className="h-3 w-3" /> No training needed
                    </Badge>
                  </TableCell>
                  <TableCell className="text-xs tabular-nums">
                    {championReady ? (
                      <span className="text-emerald-600 dark:text-emerald-400">
                        {championDfuCount.toLocaleString()} DFUs ready
                      </span>
                    ) : (
                      <span className="text-amber-600" title={`Generate forecasts for: ${championMissingModels.join(", ")}`}>
                        Waiting: {championMissingModels.join(", ")}
                      </span>
                    )}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex items-center justify-end gap-1">
                      <Badge variant="outline" className="text-[10px] gap-0.5 text-blue-600 border-blue-200">
                        <CheckCircle2 className="h-3 w-3" /> N/A
                      </Badge>
                      {/* Generate uses the legacy production forecast job — routes per-DFU using champion assignments. */}
                      <Button
                        size="sm" variant="outline"
                        className="h-7 px-2 text-[11px] gap-1"
                        onClick={async () => {
                          setIsSubmitting(true);
                          try {
                            const params: Record<string, unknown> = { horizon: effectiveHorizon };
                            if (includeCI) params.confidence_intervals = true;
                            await submitJob("generate_production_forecast", params, "Production Forecast");
                            queryClient.invalidateQueries({ queryKey: forecastPanelKeys.jobs(0) });
                            queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.stagingSummary });
                          } finally {
                            setIsSubmitting(false);
                          }
                        }}
                        disabled={isSubmitting || !championReady}
                        title={championReady ? "Run champion inference" : `Generate first: ${championMissingModels.join(", ")}`}
                      >
                        <BarChart3 className="h-3 w-3" />
                        Generate
                      </Button>
                      {promotingModelId === "champion" ? (
                        <Button size="sm" variant="outline" className="h-7 px-2 text-[11px] gap-1" disabled>
                          <Loader2 className="h-3 w-3 animate-spin" />
                          Promoting...
                        </Button>
                      ) : isChampionPromoted ? (
                        <Button
                          size="sm" variant="outline"
                          className="h-7 px-2 text-[11px] gap-1 text-amber-700 border-amber-200 bg-amber-50 hover:bg-amber-100"
                          onClick={() => handlePromote("champion")}
                          disabled={promotingModelId !== null}
                          title="Click to re-promote"
                        >
                          <Crown className="h-3 w-3" /> Promoted
                          <RotateCcw className="h-3 w-3 opacity-50" />
                        </Button>
                      ) : (
                        <Button
                          size="sm" variant="outline"
                          className="h-7 px-2 text-[11px] gap-1"
                          onClick={() => handlePromote("champion")}
                          disabled={!championReady || promotingModelId !== null}
                          title={championReady ? "Promote champion forecasts to production" : `Generate first: ${championMissingModels.join(", ")}`}
                        >
                          <Crown className="h-3 w-3" />
                          Promote
                        </Button>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              )}
              {forecastAlgos.map((algo) => {
                const status = trainingStatus?.[algo.id];
                const needsTraining = requiresTraining(algo.type);
                const isTrained = status?.trained ?? false;
                const isProductionTrained =
                  isTrained && status?.training_mode === "production";
                const isCurrentlyTraining =
                  isTraining &&
                  (trainingModelId === algo.id || trainingModelId === "__all__");

                const staged = staging[algo.id];
                const hasStagedForecast = staged != null && staged.row_count > 0;
                const isCurrentPromoted = promotedModel?.model_id === algo.id;

                return (
                  <TableRow key={algo.id}>
                    <TableCell className="text-sm font-medium">
                      {modelLabel(algo.id)}
                    </TableCell>
                    <TableCell>
                      <Badge
                        className={cn(
                          "text-[10px] px-1.5 py-0",
                          MODEL_TYPE_COLORS[algo.type] ?? "bg-gray-100 text-gray-700",
                        )}
                      >
                        {algo.type}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm tabular-nums">
                      {algo.accuracy != null ? (
                        `${algo.accuracy.toFixed(1)}%`
                      ) : (
                        <span className="text-muted-foreground">--</span>
                      )}
                    </TableCell>
                    <TableCell>
                      <TrainingStatusIndicator
                        trained={isTrained}
                        trainingMode={status?.training_mode ?? null}
                        trainedAt={status?.trained_at ?? null}
                        needsTraining={needsTraining}
                      />
                    </TableCell>
                    {/* Staging column - shows staged forecast DFU count */}
                    <TableCell className="text-xs tabular-nums">
                      {hasStagedForecast ? (
                        <span className="text-emerald-600 dark:text-emerald-400">
                          {staged.dfu_count.toLocaleString()} DFUs
                        </span>
                      ) : (
                        <span className="text-muted-foreground">--</span>
                      )}
                    </TableCell>

                    {/* Actions column - 3 buttons: Train, Generate, Promote */}
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-1">
                        {/* Train (tree models only) */}
                        {needsTraining ? (
                          isCurrentlyTraining ? (
                            <Button
                              size="sm" variant="outline"
                              className="h-7 px-2 text-[11px] gap-1"
                              disabled
                            >
                              <Loader2 className="h-3 w-3 animate-spin" />
                              Training...
                            </Button>
                          ) : isProductionTrained ? (
                            <Button
                              size="sm" variant="outline"
                              className="h-7 px-2 text-[11px] gap-1 text-emerald-600 border-emerald-200 hover:bg-emerald-50"
                              onClick={() => handleTrain(algo.id)}
                              disabled={isTraining}
                              title="Click to re-train"
                            >
                              <CheckCircle2 className="h-3 w-3" /> Trained
                              <RotateCcw className="h-3 w-3 opacity-50" />
                            </Button>
                          ) : (
                            <Button
                              size="sm" variant="outline"
                              className="h-7 px-2 text-[11px] gap-1"
                              onClick={() => handleTrain(algo.id)}
                              disabled={isTraining}
                            >
                              <Dumbbell className="h-3 w-3" />
                              Train
                            </Button>
                          )
                        ) : (
                          <Badge variant="outline" className="text-[10px] gap-0.5 text-blue-600 border-blue-200">
                            <CheckCircle2 className="h-3 w-3" /> N/A
                          </Badge>
                        )}

                        {/* Generate */}
                        {generatingModelId === algo.id ? (
                          <Button
                            size="sm" variant="outline"
                            className="h-7 px-2 text-[11px] gap-1"
                            disabled
                          >
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Generating...
                          </Button>
                        ) : hasStagedForecast ? (
                          <Button
                            size="sm" variant="outline"
                            className="h-7 px-2 text-[11px] gap-1 text-emerald-600 border-emerald-200 hover:bg-emerald-50"
                            onClick={() => handleGenerate(algo.id)}
                            disabled={isGenerating}
                            title="Click to re-generate"
                          >
                            <CheckCircle2 className="h-3 w-3" /> Generated
                            <RotateCcw className="h-3 w-3 opacity-50" />
                          </Button>
                        ) : (
                          <Button
                            size="sm" variant="outline"
                            className="h-7 px-2 text-[11px] gap-1"
                            onClick={() => handleGenerate(algo.id)}
                            disabled={isGenerating || (algo.type === "tree" && !isProductionTrained)}
                          >
                            <BarChart3 className="h-3 w-3" />
                            Generate
                          </Button>
                        )}

                        {/* Promote */}
                        {promotingModelId === algo.id ? (
                          <Button
                            size="sm" variant="outline"
                            className="h-7 px-2 text-[11px] gap-1"
                            disabled
                          >
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Promoting...
                          </Button>
                        ) : isCurrentPromoted ? (
                          <Button
                            size="sm" variant="outline"
                            className="h-7 px-2 text-[11px] gap-1 text-amber-700 border-amber-200 bg-amber-50 hover:bg-amber-100"
                            onClick={() => handlePromote(algo.id)}
                            disabled={promotingModelId !== null}
                            title="Click to re-promote"
                          >
                            <Crown className="h-3 w-3" /> Promoted
                            <RotateCcw className="h-3 w-3 opacity-50" />
                          </Button>
                        ) : (
                          <Button
                            size="sm" variant="outline"
                            className="h-7 px-2 text-[11px] gap-1"
                            onClick={() => handlePromote(algo.id)}
                            disabled={!hasStagedForecast || promotingModelId !== null}
                          >
                            {promotingModelId === algo.id ? (
                              <Loader2 className="h-3 w-3 animate-spin" />
                            ) : (
                              <Crown className="h-3 w-3" />
                            )}
                            Promote
                          </Button>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
              {forecastAlgos.length === 0 && (
                <TableRow>
                  <TableCell
                    colSpan={6}
                    className="text-center text-sm text-muted-foreground py-6"
                  >
                    No forecastable algorithms configured.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* ══════ STEP 2: Algorithm Selection ══════ */}
      <div className="grid grid-cols-3 gap-4">
        {/* Left: Algorithm Picker (2 cols) */}
        <div className="col-span-2 space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold flex items-center gap-2">
                <Play className="h-4 w-4" />
                Step 2: Algorithm Selection
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {/* Champion option */}
              <label
                className={cn(
                  "flex items-center gap-3 rounded-md border p-3 cursor-pointer transition-colors",
                  selectedModel === "champion"
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50",
                )}
              >
                <input
                  type="radio"
                  name="forecast-model"
                  value="champion"
                  checked={selectedModel === "champion"}
                  onChange={() => setSelectedModel("champion")}
                  className="accent-primary"
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">Use Champion</span>
                    <Badge variant="secondary" className="text-[10px]">Recommended</Badge>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    Uses the champion model selected by the meta-learner per DFU.
                    Falls back to {modelLabel(pipelineConfig?.champion?.fallback_model_id ?? "lgbm_cluster")}.
                  </p>
                  {/* Champion participating models with trained/not-trained status */}
                  {!promotedExperiment && (
                    <p className="text-xs text-amber-600 dark:text-amber-400 mt-1 flex items-center gap-1">
                      <AlertTriangle className="h-3 w-3 shrink-0" />
                      No promoted champion experiment. Run champion selection first.
                    </p>
                  )}
                  {championCompetingAlgos.length > 0 && (
                    <div className="mt-2 space-y-1">
                      <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                        {promotedExperiment
                          ? `Promoted: "${promotedExperiment.label}" (${championCompetingAlgos.length} models)`
                          : `Champion Algorithms (${championCompetingAlgos.length})`}
                      </p>
                      <div className="flex flex-wrap gap-x-3 gap-y-1">
                        {championCompetingAlgos.map((algo) => {
                          // Green = forecast generated in staging; Red = not yet generated
                          const hasStaged = (staging[algo.id]?.row_count ?? 0) > 0;
                          return (
                            <span
                              key={algo.id}
                              className={cn(
                                "inline-flex items-center gap-1 text-xs",
                                hasStaged
                                  ? "text-emerald-600 dark:text-emerald-400"
                                  : "text-red-600 dark:text-red-400",
                              )}
                            >
                              {hasStaged ? (
                                <CheckCircle2 className="h-3 w-3" />
                              ) : (
                                <XCircle className="h-3 w-3" />
                              )}
                              {modelLabel(algo.id)}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </label>

              {/* Algorithm options */}
              {forecastAlgos.map((algo) => {
                const isTree = requiresTraining(algo.type);
                const isTrained = trainingStatus?.[algo.id]?.trained ?? false;
                const isProductionTrained =
                  isTrained &&
                  trainingStatus?.[algo.id]?.training_mode === "production";
                // Tree models must be production-trained; non-tree always allowed
                const isDisabled = isTree && !isProductionTrained;

                let disabledReason: string | undefined;
                if (isDisabled) {
                  disabledReason =
                    "Tree model not trained for production -- train it in Step 1 above";
                }

                return (
                  <label
                    key={algo.id}
                    className={cn(
                      "flex items-center gap-3 rounded-md border p-3 transition-colors",
                      isDisabled
                        ? "opacity-50 cursor-not-allowed"
                        : "cursor-pointer",
                      !isDisabled && selectedModel === algo.id
                        ? "border-primary bg-primary/5"
                        : "border-border",
                      !isDisabled && selectedModel !== algo.id && "hover:border-primary/50",
                    )}
                    title={disabledReason}
                  >
                    <input
                      type="radio"
                      name="forecast-model"
                      value={algo.id}
                      checked={selectedModel === algo.id}
                      onChange={() => setSelectedModel(algo.id)}
                      disabled={isDisabled}
                      className="accent-primary"
                    />
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">{modelLabel(algo.id)}</span>
                        <Badge
                          className={cn(
                            "text-[10px] px-1.5 py-0",
                            MODEL_TYPE_COLORS[algo.type] ?? "bg-gray-100 text-gray-700",
                          )}
                        >
                          {algo.type}
                        </Badge>
                        {algo.accuracy != null && (
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                            {algo.accuracy.toFixed(1)}% acc
                          </Badge>
                        )}
                        {/* Status icons */}
                        {isTree ? (
                          isProductionTrained ? (
                            <CheckCircle2 className="h-3 w-3 text-emerald-500" />
                          ) : isTrained ? (
                            <AlertTriangle className="h-3 w-3 text-amber-500" />
                          ) : (
                            <XCircle className="h-3 w-3 text-muted-foreground" />
                          )
                        ) : algo.hasPredictions ? (
                          <CheckCircle2 className="h-3 w-3 text-emerald-500" />
                        ) : (
                          <XCircle className="h-3 w-3 text-muted-foreground" />
                        )}
                      </div>
                      {isDisabled && disabledReason && (
                        <p className="text-[10px] text-muted-foreground mt-0.5">
                          {disabledReason}
                        </p>
                      )}
                    </div>
                  </label>
                );
              })}
            </CardContent>
          </Card>

          {/* Configuration display (read-only) */}
          {prodConfig && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold">Pipeline Configuration</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
                  <ConfigRow label="Horizon Months" value={String(prodConfig.horizon_months)} />
                  <ConfigRow label="Min History Months" value={String(prodConfig.min_history_months)} />
                  <ConfigRow label="Cold-Start Model" value={modelLabel(prodConfig.cold_start_model_id)} />
                  <ConfigRow label="Cold-Start Min Months" value={String(prodConfig.cold_start_min_months)} />
                  <ConfigRow
                    label="Confidence Intervals"
                    value={
                      (prodConfig as ProdConfigExtended).confidence_interval?.enabled
                        ? "Enabled"
                        : "Disabled"
                    }
                  />
                  <ConfigRow
                    label="Recursive Mode"
                    value={
                      (prodConfig as ProdConfigExtended).recursive !== undefined
                        ? String((prodConfig as ProdConfigExtended).recursive)
                        : "--"
                    }
                  />
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right: Action panel (1 col) */}
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold">Step 3: Generate Forecast</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Horizon input */}
              <div>
                <label htmlFor="forecast-horizon" className="text-xs text-muted-foreground block mb-1">
                  Horizon (months)
                </label>
                <input
                  id="forecast-horizon"
                  type="number"
                  min={1}
                  max={60}
                  value={effectiveHorizon}
                  onChange={(e) => setHorizon(Math.max(1, parseInt(e.target.value, 10) || 1))}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                />
              </div>

              {/* Selected model badge */}
              <div className="text-xs text-muted-foreground">
                Model:{" "}
                <span className="font-medium text-foreground">
                  {selectedModel === "champion" ? "Champion (meta-learner)" : modelLabel(selectedModel)}
                </span>
              </div>

              {/* Confidence intervals checkbox */}
              <div className="flex items-center gap-2">
                <Checkbox
                  id="include-ci"
                  checked={includeCI}
                  onCheckedChange={(checked) => setIncludeCI(checked === true)}
                />
                <label
                  htmlFor="include-ci"
                  className="text-xs cursor-pointer select-none"
                >
                  Include Confidence Intervals (P10/P90)
                </label>
              </div>

              {/* Generate button */}
              <Button
                className="w-full"
                size="lg"
                onClick={handleGenerateForecast}
                disabled={isSubmitting || isForecastRunning}
              >
                {isSubmitting || isForecastRunning ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {isForecastRunning ? "Forecast Running..." : "Submitting..."}
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Generate Forecast
                  </>
                )}
              </Button>

              {isForecastRunning && (
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  A forecast generation job is currently running. Wait for it to complete before submitting another.
                </p>
              )}
            </CardContent>
          </Card>

          {/* Latest version info card */}
          {latestVersion && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold">Latest Version</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <ConfigRow label="Version" value={latestVersion.plan_version} />
                <ConfigRow label="DFUs" value={(latestVersion.sku_count ?? latestVersion.dfu_count)?.toLocaleString() ?? "--"} />
                <ConfigRow label="Rows" value={latestVersion.total_rows?.toLocaleString() ?? "--"} />
                <ConfigRow
                  label="Generated"
                  value={latestVersion.generated_at ? timeAgo(latestVersion.generated_at) : "--"}
                />
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Recent Forecast Jobs */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold">Recent Forecast Jobs</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {recentJobs.length === 0 ? (
            <div className="py-8 text-center text-sm text-muted-foreground">
              No forecast jobs found. Click "Generate Forecast" to start.
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Job ID</TableHead>
                  <TableHead className="text-xs">Label</TableHead>
                  <TableHead className="text-xs">Status</TableHead>
                  <TableHead className="text-xs">Submitted</TableHead>
                  <TableHead className="text-xs">Duration</TableHead>
                  <TableHead className="text-xs">Progress</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recentJobs.map((job) => (
                  <TableRow key={job.job_id}>
                    <TableCell className="text-xs font-mono max-w-[120px] truncate">
                      {job.job_id.slice(0, 8)}
                    </TableCell>
                    <TableCell className="text-xs">
                      {job.job_label || "--"}
                    </TableCell>
                    <TableCell>
                      <StatusBadge status={job.status} />
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {timeAgo(job.submitted_at)}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {formatDuration(job.started_at, job.completed_at)}
                    </TableCell>
                    <TableCell className="text-xs">
                      {job.status === "running"
                        ? `${job.progress_pct}%`
                        : job.status === "completed"
                          ? "Done"
                          : "--"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ConfigRow -- simple key/value pair for config display
// ---------------------------------------------------------------------------

function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Extended types for fields present in YAML but not in PipelineConfig interface
// ---------------------------------------------------------------------------

interface ProdConfigExtended {
  horizon_months: number;
  min_history_months: number;
  cold_start_model_id: string;
  cold_start_min_months: number;
  recursive?: boolean;
  confidence_interval?: {
    enabled: boolean;
    source_model_ids?: string[];
  };
}
