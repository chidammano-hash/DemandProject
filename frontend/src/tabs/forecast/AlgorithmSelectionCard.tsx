/**
 * AlgorithmSelectionCard -- Step 2 of the ForecastPanel.
 *
 * Champion / single-model radio picker plus the read-only pipeline
 * configuration display. Pure presentation; selection state is lifted in.
 */
import { Play, CheckCircle2, XCircle, AlertTriangle } from "lucide-react";

import type { ChampionExperiment } from "@/api/queries/champion-experiments";
import type {
  TrainingStatusMap,
  StagingSummaryMap,
} from "@/api/queries/backtest-management";
import type { PipelineConfig } from "@/api/queries/unified-model-tuning";
import { modelLabel, MODEL_TYPE_COLORS } from "@/lib/model-labels";
import { cn } from "@/lib/utils";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

import type {
  ForecastAlgorithm,
  ProdConfigExtended,
} from "./forecastPanelShared";
import { requiresTraining } from "./forecastPanelShared";
import { ConfigRow } from "./ConfigRow";

interface AlgorithmSelectionCardProps {
  forecastAlgos: ForecastAlgorithm[];
  championCompetingAlgos: ForecastAlgorithm[];
  selectedModel: string;
  onSelectModel: (modelId: string) => void;
  trainingStatus: TrainingStatusMap | undefined;
  staging: StagingSummaryMap;
  promotedExperiment: ChampionExperiment | null;
  fallbackModelId: string;
  prodConfig: PipelineConfig["production_forecast"] | undefined;
}

export function AlgorithmSelectionCard({
  forecastAlgos,
  championCompetingAlgos,
  selectedModel,
  onSelectModel,
  trainingStatus,
  staging,
  promotedExperiment,
  fallbackModelId,
  prodConfig,
}: AlgorithmSelectionCardProps) {
  return (
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
              onChange={() => onSelectModel("champion")}
              className="accent-primary"
            />
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Use Champion</span>
                <Badge variant="secondary" className="text-[10px]">Recommended</Badge>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                Uses the champion model selected by the meta-learner per DFU.
                Falls back to {modelLabel(fallbackModelId)}.
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
                  onChange={() => onSelectModel(algo.id)}
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
  );
}
