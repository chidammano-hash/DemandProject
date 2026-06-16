/**
 * ModelReadinessCard -- Step 1 of the ForecastPanel.
 *
 * Per-model readiness table: train, generate, and promote each model
 * (plus the champion ensemble row). Pure presentation; all state and
 * handlers are lifted in from ForecastPanel.
 */
import {
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Dumbbell,
  BarChart3,
  Crown,
  RotateCcw,
} from "lucide-react";

import type { ChampionExperiment } from "@/api/queries/champion-experiments";
import type {
  TrainingStatusMap,
  StagingSummaryMap,
  PromotionStatus,
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
  treeAlgos: ForecastAlgorithm[];
  trainedTreeCount: number;
  allTreesTrained: boolean;
  isTraining: boolean;
  trainingModelId: string | null;
  generatingModelId: string | null;
  isGenerating: boolean;
  promotingModelId: string | null;
  isSubmitting: boolean;
  promotedModel: PromotionStatus | null;
  promotedExperiment: ChampionExperiment | null;
  championConstituents: string[];
  championMissingModels: string[];
  championReady: boolean;
  championDfuCount: number;
  isChampionPromoted: boolean;
  onTrain: (modelId: string) => void;
  onTrainAll: () => void;
  onGenerate: (modelId: string) => void;
  onPromote: (modelId: string) => void;
  onGenerateChampion: () => void;
}

export function ModelReadinessCard({
  forecastAlgos,
  trainingStatus,
  staging,
  treeAlgos,
  trainedTreeCount,
  allTreesTrained,
  isTraining,
  trainingModelId,
  generatingModelId,
  isGenerating,
  promotingModelId,
  isSubmitting,
  promotedModel,
  promotedExperiment,
  championConstituents,
  championMissingModels,
  championReady,
  championDfuCount,
  isChampionPromoted,
  onTrain,
  onTrainAll,
  onGenerate,
  onPromote,
  onGenerateChampion,
}: ModelReadinessCardProps) {
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
              {trainedTreeCount}/{treeAlgos.length} tree models production-ready
            </span>
            {!allTreesTrained && treeAlgos.length > 0 && (
              <Button
                size="sm"
                variant="outline"
                onClick={onTrainAll}
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
                      onClick={onGenerateChampion}
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
                        onClick={() => onPromote("champion")}
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
                        onClick={() => onPromote("champion")}
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
                            onClick={() => onTrain(algo.id)}
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
                            onClick={() => onTrain(algo.id)}
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
                          onClick={() => onGenerate(algo.id)}
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
                          onClick={() => onGenerate(algo.id)}
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
                          onClick={() => onPromote(algo.id)}
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
                          onClick={() => onPromote(algo.id)}
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
  );
}
