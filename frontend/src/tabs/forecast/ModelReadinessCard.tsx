/**
 * ModelReadinessCard -- Step 1 of the ForecastPanel.
 *
 * Per-model readiness table for training and staging generation. Production
 * selection and promotion live in Steps 2-3 so every candidate follows the
 * same explicit workflow.
 */
import {
  Loader2,
  CheckCircle2,
  Dumbbell,
  BarChart3,
  Crown,
  RotateCcw,
  ShieldCheck,
  AlertTriangle,
} from "lucide-react";

import type { ChampionExperiment } from "@/api/queries/champion-experiments";
import type {
  TrainingStatusMap,
  StagingSummaryMap,
  SnapshotRosterReadiness,
} from "@/api/queries/backtest-management";
import { modelLabel, MODEL_TYPE_COLORS } from "@/lib/model-labels";
import { cn } from "@/lib/utils";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import type { ForecastAlgorithm } from "./forecastPanelShared";
import { requiresTraining } from "./forecastPanelShared";
import { TrainingStatusIndicator } from "./TrainingStatusIndicator";

interface ModelReadinessCardProps {
  forecastAlgos: ForecastAlgorithm[];
  trainingStatus: TrainingStatusMap | undefined;
  staging: StagingSummaryMap;
  trainableAlgos: ForecastAlgorithm[];
  trainedArtifactCount: number;
  allRequiredArtifactsReady: boolean;
  isTraining: boolean;
  trainingModelId: string | null;
  generatingModelId: string | null;
  isGenerating: boolean;
  promotedExperiment: ChampionExperiment | null;
  championConstituents: string[];
  championReady: boolean;
  championDfuCount: number;
  isChampionPromoted: boolean;
  activeProductionModelId: string | null;
  snapshotReadiness: SnapshotRosterReadiness | undefined;
  isPreparingPublish: boolean;
  onTrain: (modelId: string) => void;
  onTrainAll: () => void;
  onGenerate: (modelId: string) => void;
  onGenerateAll: () => void;
  generatableCount: number;
  onPreparePublish: (pipeline: "model-refresh" | "champion-refresh" | "forecast-publish") => void;
}

export function ModelReadinessCard({
  forecastAlgos,
  trainingStatus,
  staging,
  trainableAlgos,
  trainedArtifactCount,
  allRequiredArtifactsReady,
  isTraining,
  trainingModelId,
  generatingModelId,
  isGenerating,
  promotedExperiment,
  championConstituents,
  championReady,
  championDfuCount,
  isChampionPromoted,
  activeProductionModelId,
  snapshotReadiness,
  isPreparingPublish,
  onTrain,
  onTrainAll,
  onGenerate,
  onGenerateAll,
  generatableCount,
  onPreparePublish,
}: ModelReadinessCardProps) {
  const isGeneratingAll = generatingModelId === "__all__";
  const snapshotEvidenceReady = snapshotReadiness?.ready === true;
  const releasePublished = activeProductionModelId !== null;
  const publishEvidenceReady = releasePublished || snapshotEvidenceReady;
  const shouldPrepareRelease =
    !releasePublished && snapshotReadiness?.action_pipeline != null && !snapshotEvidenceReady;
  const readinessPipeline = snapshotReadiness?.action_pipeline ?? "forecast-publish";
  const championSelectionRequired = readinessPipeline === "champion-refresh";
  const canLaunchPreparation = shouldPrepareRelease && !championSelectionRequired;
  const preparationLabel =
    readinessPipeline === "model-refresh" ? "Refresh Models" : "Prepare Release";
  const preparationProgressLabel =
    readinessPipeline === "model-refresh" ? "Refreshing Models..." : "Preparing Release...";
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Dumbbell className="h-4 w-4" />
            Step 1: Model Readiness
          </CardTitle>
          <div className="flex items-center gap-3">
            <span className="text-xs text-muted-foreground">
              {trainingStatus
                ? `${trainedArtifactCount}/${trainableAlgos.length} production artifacts ready`
                : "Checking production artifacts..."}
            </span>
            {trainingStatus && !allRequiredArtifactsReady && trainableAlgos.length > 0 ? (
              <Button size="sm" variant="outline" onClick={onTrainAll} disabled={isTraining}>
                {isTraining && trainingModelId === "__all__" ? (
                  <>
                    <Loader2 className="mr-1.5 h-3 w-3 animate-spin" />
                    Training Models...
                  </>
                ) : (
                  <>
                    <Dumbbell className="mr-1.5 h-3 w-3" />
                    Train Production Models
                  </>
                )}
              </Button>
            ) : null}
            {trainingStatus && allRequiredArtifactsReady ? (
              <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200 text-[10px]">
                Required Artifacts Ready
              </Badge>
            ) : null}
            {/* Generate staging forecasts for every ready model in one click. */}
            <Button
              size="sm"
              variant="outline"
              onClick={onGenerateAll}
              disabled={isGenerating || generatableCount === 0}
              title={
                generatableCount === 0
                  ? "No retained model is ready; check model configuration and artifacts."
                  : `Generate staging forecasts for ${generatableCount} ready model(s)`
              }
            >
              {isGeneratingAll ? (
                <>
                  <Loader2 className="mr-1.5 h-3 w-3 animate-spin" />
                  Generating All...
                </>
              ) : (
                <>
                  <BarChart3 className="mr-1.5 h-3 w-3" />
                  Generate All ({generatableCount})
                </>
              )}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div
          className={cn(
            "mx-4 mb-3 flex min-h-14 items-center justify-between gap-4 rounded-lg border px-3 py-2",
            publishEvidenceReady
              ? "border-emerald-200 bg-emerald-50/70 dark:border-emerald-800 dark:bg-emerald-950/20"
              : "border-amber-200 bg-amber-50/70 dark:border-amber-800 dark:bg-amber-950/20"
          )}
          role="status"
          aria-live="polite"
        >
          <div className="flex min-w-0 items-start gap-2">
            {publishEvidenceReady ? (
              <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
            ) : (
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
            )}
            <div className="min-w-0">
              <p className="text-xs font-semibold">
                {releasePublished
                  ? "Production release published"
                  : snapshotReadiness
                    ? `Champion + ${snapshotReadiness.ready_contender_count}/${snapshotReadiness.required_contender_count} contenders ready`
                    : "Checking champion + top-three publish evidence..."}
              </p>
              <p className="mt-0.5 text-[11px] text-muted-foreground">
                {releasePublished
                  ? `${modelLabel(activeProductionModelId ?? "champion")} is active in production. Generate All creates ${generatableCount} staged comparison forecast${generatableCount === 1 ? "" : "s"} without changing production.`
                  : publishEvidenceReady
                    ? "The exact snapshot roster is ready; select any staged candidate in Step 2 to publish."
                    : (snapshotReadiness?.stale_reason ??
                      "Waiting for current release evidence before promotion is enabled.")}
              </p>
            </div>
          </div>
          {championSelectionRequired ? (
            <p className="max-w-64 text-right text-xs font-medium text-amber-700 dark:text-amber-300">
              Go to Champion and select a completed experiment to assign.
            </p>
          ) : canLaunchPreparation ? (
            <Button
              size="sm"
              onClick={() => onPreparePublish(readinessPipeline)}
              disabled={isPreparingPublish}
              className="h-9 shrink-0"
              title="Run the canonical Forecast Publish workflow and monitor each job"
            >
              {isPreparingPublish ? (
                <>
                  <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  {preparationProgressLabel}
                </>
              ) : (
                <>
                  <ShieldCheck className="mr-1.5 h-3.5 w-3.5" />
                  {preparationLabel}
                </>
              )}
            </Button>
          ) : null}
        </div>
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
                  <Badge
                    variant="outline"
                    className="text-[10px] gap-0.5 text-blue-600 border-blue-200"
                  >
                    <CheckCircle2 className="h-3 w-3" /> No training needed
                  </Badge>
                </TableCell>
                <TableCell className="text-xs tabular-nums">
                  {releasePublished ? (
                    <span className="text-emerald-600 dark:text-emerald-400">
                      {championDfuCount.toLocaleString()} DFUs in production
                    </span>
                  ) : championReady ? (
                    <span className="text-emerald-600 dark:text-emerald-400">
                      {championDfuCount.toLocaleString()} DFUs ready
                    </span>
                  ) : (
                    <span className="text-amber-600">Generate a release candidate</span>
                  )}
                </TableCell>
                <TableCell className="text-right">
                  <div className="flex items-center justify-end gap-1">
                    <Badge
                      variant="outline"
                      className="text-[10px] gap-0.5 text-blue-600 border-blue-200"
                    >
                      <CheckCircle2 className="h-3 w-3" /> N/A
                    </Badge>
                    <Badge
                      variant="outline"
                      className={cn(
                        "text-[10px]",
                        isChampionPromoted && "border-emerald-200 text-emerald-700"
                      )}
                    >
                      {isChampionPromoted
                        ? "In production"
                        : championReady
                          ? "Candidate staged"
                          : "Select in Step 2"}
                    </Badge>
                  </div>
                </TableCell>
              </TableRow>
            )}
            {forecastAlgos.map((algo) => {
              const status = trainingStatus?.[algo.id];
              const needsTraining = requiresTraining(algo.type);
              const isProductionReady = status?.ready === true;
              const isCurrentlyTraining =
                isTraining && (trainingModelId === algo.id || trainingModelId === "__all__");

              const staged = staging[algo.id];
              const hasStagedForecast = staged != null && staged.row_count > 0;

              return (
                <TableRow key={algo.id}>
                  <TableCell className="text-sm font-medium">{modelLabel(algo.id)}</TableCell>
                  <TableCell>
                    <Badge
                      className={cn(
                        "text-[10px] px-1.5 py-0",
                        MODEL_TYPE_COLORS[algo.type] ?? "bg-gray-100 text-gray-700"
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
                      trained={isProductionReady}
                      trainingMode={status?.training_mode ?? null}
                      trainedAt={status?.trained_at ?? null}
                      needsTraining={needsTraining}
                      staleReason={status?.stale_reason}
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

                  {/* Actions column: prepare artifacts and generate staging candidates. */}
                  <TableCell className="text-right">
                    <div className="flex items-center justify-end gap-1">
                      {/* Persisted-artifact models can be trained or retrained here. */}
                      {needsTraining ? (
                        isCurrentlyTraining ? (
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-7 px-2 text-[11px] gap-1"
                            disabled
                          >
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Training...
                          </Button>
                        ) : isProductionReady ? (
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-7 px-2 text-[11px] gap-1 text-emerald-600 border-emerald-200 hover:bg-emerald-50"
                            onClick={() => onTrain(algo.id)}
                            disabled={isTraining}
                            title="Click to re-train"
                          >
                            <CheckCircle2 className="h-3 w-3" /> Trained
                            <RotateCcw className="h-3 w-3 opacity-50" />
                          </Button>
                        ) : (
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-7 px-2 text-[11px] gap-1"
                            onClick={() => onTrain(algo.id)}
                            disabled={isTraining}
                          >
                            <Dumbbell className="h-3 w-3" />
                            Train
                          </Button>
                        )
                      ) : (
                        <Badge
                          variant="outline"
                          className="text-[10px] gap-0.5 text-blue-600 border-blue-200"
                        >
                          <CheckCircle2 className="h-3 w-3" /> N/A
                        </Badge>
                      )}

                      {/* Generate */}
                      {generatingModelId === algo.id ? (
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 px-2 text-[11px] gap-1"
                          disabled
                        >
                          <Loader2 className="h-3 w-3 animate-spin" />
                          Generating...
                        </Button>
                      ) : hasStagedForecast ? (
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 px-2 text-[11px] gap-1 text-emerald-600 border-emerald-200 hover:bg-emerald-50"
                          onClick={() => onGenerate(algo.id)}
                          disabled={isGenerating || (needsTraining && !isProductionReady)}
                          title={
                            needsTraining && !isProductionReady
                              ? (status?.stale_reason ??
                                "Prepare a current production artifact before generating")
                              : "Click to re-generate"
                          }
                        >
                          <CheckCircle2 className="h-3 w-3" /> Generated
                          <RotateCcw className="h-3 w-3 opacity-50" />
                        </Button>
                      ) : (
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 px-2 text-[11px] gap-1"
                          onClick={() => onGenerate(algo.id)}
                          disabled={isGenerating || (needsTraining && !isProductionReady)}
                          title={
                            needsTraining && !isProductionReady
                              ? (status?.stale_reason ??
                                "Prepare a current production artifact before generating")
                              : undefined
                          }
                        >
                          <BarChart3 className="h-3 w-3" />
                          Generate
                        </Button>
                      )}

                      {hasStagedForecast ? (
                        <Badge variant="outline" className="text-[10px] text-muted-foreground">
                          Candidate staged
                        </Badge>
                      ) : null}
                    </div>
                  </TableCell>
                </TableRow>
              );
            })}
            {forecastAlgos.length === 0 && (
              <TableRow>
                <TableCell colSpan={6} className="text-center text-sm text-muted-foreground py-6">
                  No forecastable algorithms configured.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
