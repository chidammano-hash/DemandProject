/**
 * ChampionPromoteModal — 2-stage promotion dialog.
 *
 * Stage 1: Promote config to forecast_pipeline_config.yaml
 * Stage 2: Load champion results into fact_external_forecast_monthly
 */
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Crown, Database, Check, Loader2 } from "lucide-react";

import {
  championExperimentKeys,
  promoteChampionExperiment,
  promoteChampionResults,
  fetchChampionResultsStatus,
  type ChampionExperiment,
} from "@/api/queries";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface Props {
  experiment: ChampionExperiment;
  open: boolean;
  onClose: () => void;
}

export function ChampionPromoteModal({ experiment, open, onClose }: Props) {
  const queryClient = useQueryClient();
  const [stage, setStage] = useState<1 | 2>(1);

  // Stage 1: Promote config
  const promoteMutation = useMutation({
    mutationFn: () => promoteChampionExperiment(experiment.experiment_id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: championExperimentKeys.all });
      setStage(2);
    },
  });

  // Stage 2: Load results
  const loadResultsMutation = useMutation({
    mutationFn: () => promoteChampionResults(experiment.experiment_id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: championExperimentKeys.all });
    },
  });

  // Poll for results status (after Stage 2 submitted)
  const { data: resultsStatus } = useQuery({
    queryKey: ["champion-results-status", experiment.experiment_id],
    queryFn: () => fetchChampionResultsStatus(experiment.experiment_id),
    enabled: loadResultsMutation.isSuccess,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed") return false;
      return 3000;
    },
  });

  if (!open) return null;

  const isResultsDone = resultsStatus?.is_results_promoted || experiment.is_results_promoted;
  const isAlreadyPromoted = experiment.is_promoted;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-lg rounded-lg border bg-background shadow-lg">
        {/* Header */}
        <div className="flex items-center gap-2 border-b px-6 py-4">
          <Crown className="h-5 w-5 text-amber-500" />
          <h2 className="text-lg font-semibold">Promote Champion Experiment</h2>
        </div>

        {/* Body */}
        <div className="px-6 py-4 space-y-4">
          {/* Summary */}
          <Card>
            <CardContent className="py-3 space-y-1 text-sm">
              <div>
                <span className="text-muted-foreground">Experiment:</span>{" "}
                <span className="font-medium">{experiment.label}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Strategy:</span>{" "}
                <span className="font-mono">{experiment.strategy}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Accuracy:</span>{" "}
                <span className="font-medium">
                  {experiment.champion_accuracy?.toFixed(2) ?? "--"}%
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Gap to Ceiling:</span>{" "}
                <span className="font-medium">
                  {experiment.gap_bps?.toFixed(0) ?? "--"} bps
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Stage 1 */}
          <div className="flex items-start gap-3">
            <div
              className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
                isAlreadyPromoted || stage === 2
                  ? "bg-emerald-100 text-emerald-800"
                  : "bg-blue-100 text-blue-800"
              }`}
            >
              {isAlreadyPromoted || stage === 2 ? (
                <Check className="h-3.5 w-3.5" />
              ) : (
                "1"
              )}
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium">Promote Config</div>
              <div className="text-xs text-muted-foreground">
                Write strategy and params to forecast_pipeline_config.yaml (with backup)
              </div>
              {stage === 1 && !isAlreadyPromoted && (
                <Button
                  size="sm"
                  className="mt-2"
                  disabled={promoteMutation.isPending}
                  onClick={() => promoteMutation.mutate()}
                >
                  {promoteMutation.isPending ? (
                    <>
                      <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                      Promoting...
                    </>
                  ) : (
                    "Promote Config"
                  )}
                </Button>
              )}
              {promoteMutation.isError && (
                <p className="text-xs text-red-500 mt-1">
                  {(promoteMutation.error as Error).message}
                </p>
              )}
            </div>
          </div>

          {/* Stage 2 */}
          <div className="flex items-start gap-3">
            <div
              className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
                isResultsDone
                  ? "bg-emerald-100 text-emerald-800"
                  : stage === 2
                    ? "bg-blue-100 text-blue-800"
                    : "bg-gray-100 text-gray-400"
              }`}
            >
              {isResultsDone ? <Check className="h-3.5 w-3.5" /> : "2"}
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium flex items-center gap-1.5">
                <Database className="h-3.5 w-3.5" />
                Load Results
              </div>
              <div className="text-xs text-muted-foreground">
                Run champion selection and insert champion rows into forecast tables
              </div>
              {(stage === 2 || isAlreadyPromoted) && !isResultsDone && (
                <>
                  {!loadResultsMutation.isSuccess && (
                    <Button
                      size="sm"
                      className="mt-2"
                      disabled={loadResultsMutation.isPending}
                      onClick={() => loadResultsMutation.mutate()}
                    >
                      {loadResultsMutation.isPending ? (
                        <>
                          <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                          Submitting...
                        </>
                      ) : (
                        "Load Results"
                      )}
                    </Button>
                  )}
                  {loadResultsMutation.isSuccess && resultsStatus && (
                    <div className="mt-2 text-xs">
                      <span className="text-muted-foreground">Status: </span>
                      <span className="font-medium">{resultsStatus.status}</span>
                      {resultsStatus.progress_msg && (
                        <span className="ml-2 text-muted-foreground">
                          — {resultsStatus.progress_msg}
                        </span>
                      )}
                    </div>
                  )}
                </>
              )}
              {isResultsDone && (
                <p className="text-xs text-emerald-600 mt-1">Results loaded successfully</p>
              )}
              {loadResultsMutation.isError && (
                <p className="text-xs text-red-500 mt-1">
                  {(loadResultsMutation.error as Error).message}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end border-t px-6 py-3">
          <Button variant="outline" onClick={onClose}>
            {isResultsDone ? "Done" : "Close"}
          </Button>
        </div>
      </div>
    </div>
  );
}
