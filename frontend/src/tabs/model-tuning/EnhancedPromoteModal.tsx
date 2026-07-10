/**
 * EnhancedPromoteModal -- Dual-action promotion dialog.
 *
 * Two independent promotions:
 * 1. Promote Parameters — writes hyperparams to algorithm_config.yaml
 * 2. Promote Results — loads backtest predictions into DB for portfolio visibility
 */
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Crown, Database, Loader2, X, Check, AlertTriangle, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatPct, formatFixed } from "@/lib/formatters";
import {
  fetchModelLagAccuracy,
  fetchPromoteResultsStatus,
  modelTuningKeys,
  promoteModelExperiment,
  promoteModelResults,
  type ModelLagAccuracy,
  type ModelType,
  type TuningRun,
} from "@/api/queries";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface EnhancedPromoteModalProps {
  model: ModelType;
  run: TuningRun;
  open: boolean;
  onClose: () => void;
  onPromoted: () => void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const MODEL_LABELS: Record<ModelType, string> = {
  lgbm: "LightGBM",
};

const PARAM_STEPS = [
  "Write hyperparameters to algorithm_config.yaml",
  "Mark this run as the promoted champion",
  "Next forecast-generate uses these parameters",
];

const RESULTS_STEPS = [
  "Load predictions into fact_external_forecast_monthly",
  "Load all-lags into backtest_lag_archive",
  "Refresh accuracy materialized views",
  "Accuracy visible in all UI screens",
  "Champion selection can compare this model",
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function EnhancedPromoteModal({
  model,
  run,
  open,
  onClose,
  onPromoted,
}: EnhancedPromoteModalProps) {
  const queryClient = useQueryClient();
  const [paramsConfirmed, setParamsConfirmed] = useState(false);
  const [resultsConfirmed, setResultsConfirmed] = useState(false);

  const isParamsPromoted = run.is_promoted ?? false;
  const isResultsPromoted = run.is_results_promoted === true;

  // Fetch per-lag accuracy
  const { data: lagData } = useQuery<ModelLagAccuracy[]>({
    queryKey: modelTuningKeys.lags(model, run.run_id),
    queryFn: () => fetchModelLagAccuracy(model, run.run_id),
    enabled: open,
    staleTime: 60_000,
  });

  // Poll results promotion status while job is running
  const { data: resultsStatus } = useQuery({
    queryKey: modelTuningKeys.promoteResultsStatus(model, run.run_id),
    queryFn: () => fetchPromoteResultsStatus(model, run.run_id),
    enabled: open && !isResultsPromoted,
    refetchInterval: (query) => {
      const st = query.state.data?.status;
      return st === "running" || st === "queued" ? 2000 : false;
    },
    staleTime: 2_000,
  });

  const invalidateAll = () => {
    queryClient.invalidateQueries({ queryKey: ["model-tuning"] });
    queryClient.invalidateQueries({ queryKey: ["model-tuning-runs"] });
  };

  // Promote params mutation
  const paramsMut = useMutation({
    mutationFn: () => promoteModelExperiment(model, run.run_id),
    onSuccess: () => {
      invalidateAll();
      onPromoted();
    },
  });

  // Promote results mutation
  const resultsMut = useMutation({
    mutationFn: () => promoteModelResults(model, run.run_id),
    onSuccess: () => {
      invalidateAll();
    },
  });

  if (!open) return null;

  const params = run.params ?? {};
  const paramKeys = Object.keys(params).sort();
  const lags = lagData ?? [];

  const resultsJobStatus = resultsStatus?.status;
  const resultsJobRunning = resultsJobStatus === "running" || resultsJobStatus === "queued";
  const resultsJobCompleted = isResultsPromoted || resultsJobStatus === "completed";
  const resultsJobFailed = resultsJobStatus === "failed";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm">
      <div className="w-full max-w-2xl rounded-xl border bg-card shadow-xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-5 py-4 shrink-0">
          <div>
            <p className="text-sm font-semibold text-foreground">
              Promote Run #{run.run_id} — {run.run_label}
            </p>
            <p className="text-xs text-muted-foreground">
              {MODEL_LABELS[model]} · Accuracy: {formatPct(run.accuracy_pct)} · WAPE:{" "}
              {formatFixed(run.wape, 2)}
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {/* Per-lag accuracy row */}
          {lags.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {lags.map((lag) => (
                <div
                  key={lag.exec_lag}
                  className="rounded-md border border-border/60 px-3 py-1.5 text-center"
                >
                  <p className="text-[10px] text-muted-foreground">Lag {lag.exec_lag}</p>
                  <p className="text-sm font-semibold tabular-nums">
                    {formatPct(lag.accuracy_pct)}
                  </p>
                </div>
              ))}
            </div>
          )}

          {/* Config target note */}
          <div className="rounded-md bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
            <span className="font-medium">Target:</span> Hyperparameters will be written to{" "}
            <code className="font-mono">algorithm_config.yaml</code>. Champion strategy changes go
            to <code className="font-mono">forecast_pipeline_config.yaml</code>.
          </div>

          {/* Two promotion cards */}
          <div className="grid gap-4 md:grid-cols-2">
            {/* Card 1: Promote Parameters */}
            <div className="rounded-lg border p-4 space-y-3">
              <div className="flex items-center gap-2">
                <Crown className="h-4 w-4 text-amber-500" />
                <p className="text-sm font-semibold">Promote Parameters</p>
                {isParamsPromoted && (
                  <Badge className="ml-auto bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300 text-[10px]">
                    Promoted
                  </Badge>
                )}
              </div>

              <ul className="space-y-1">
                {PARAM_STEPS.map((step, i) => (
                  <li key={i} className="flex items-start gap-1.5 text-xs text-muted-foreground">
                    <Check className="h-3 w-3 mt-0.5 text-emerald-500 shrink-0" />
                    <span>{step}</span>
                  </li>
                ))}
              </ul>

              {/* Mini param preview */}
              {paramKeys.length > 0 && (
                <div className="rounded-md border bg-muted/30 p-2 max-h-32 overflow-y-auto">
                  <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
                    {paramKeys.slice(0, 10).map((key) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-muted-foreground font-mono truncate">{key}</span>
                        <span className="font-mono font-medium">{String(params[key])}</span>
                      </div>
                    ))}
                    {paramKeys.length > 10 && (
                      <span className="text-muted-foreground col-span-2">
                        +{paramKeys.length - 10} more
                      </span>
                    )}
                  </div>
                </div>
              )}

              {!isParamsPromoted && (
                <>
                  <label className="flex items-start gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={paramsConfirmed}
                      onChange={(e) => setParamsConfirmed(e.target.checked)}
                      className="mt-0.5 rounded border-border"
                    />
                    <span className="text-[11px] text-muted-foreground">
                      Overwrite production hyperparameters
                    </span>
                  </label>

                  {paramsMut.isError && (
                    <div className="rounded-md border border-red-200 bg-red-50 px-2 py-1.5 text-[11px] text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-400 flex items-center gap-1">
                      <AlertTriangle className="h-3 w-3 shrink-0" />
                      {(paramsMut.error as Error).message}
                    </div>
                  )}

                  <Button
                    size="sm"
                    onClick={() => paramsMut.mutate()}
                    disabled={!paramsConfirmed || paramsMut.isPending}
                    className={cn(
                      "w-full gap-1.5",
                      paramsConfirmed ? "bg-amber-600 hover:bg-amber-700 text-white" : "opacity-60"
                    )}
                  >
                    {paramsMut.isPending ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Crown className="h-3.5 w-3.5" />
                    )}
                    Promote Parameters
                  </Button>
                </>
              )}

              {isParamsPromoted && (
                <div className="flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Parameters promoted to production
                </div>
              )}
            </div>

            {/* Card 2: Promote Results */}
            <div className="rounded-lg border p-4 space-y-3">
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-blue-500" />
                <p className="text-sm font-semibold">Promote Results</p>
                {resultsJobCompleted && (
                  <Badge className="ml-auto bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300 text-[10px]">
                    Loaded
                  </Badge>
                )}
              </div>

              <ul className="space-y-1">
                {RESULTS_STEPS.map((step, i) => (
                  <li key={i} className="flex items-start gap-1.5 text-xs text-muted-foreground">
                    <Check className="h-3 w-3 mt-0.5 text-blue-500 shrink-0" />
                    <span>{step}</span>
                  </li>
                ))}
              </ul>

              {/* Progress bar when running */}
              {resultsJobRunning && (
                <div className="space-y-1.5">
                  <div className="h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all"
                      style={{ width: `${resultsStatus?.progress_pct ?? 0}%` }}
                    />
                  </div>
                  <p className="text-[11px] text-muted-foreground truncate">
                    {resultsStatus?.progress_msg || "Loading..."}
                  </p>
                </div>
              )}

              {resultsJobFailed && (
                <div className="rounded-md border border-red-200 bg-red-50 px-2 py-1.5 text-[11px] text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-400 flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3 shrink-0" />
                  {resultsStatus?.error ?? "Results load failed"}
                </div>
              )}

              {!resultsJobCompleted && !resultsJobRunning && (
                <>
                  <label className="flex items-start gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={resultsConfirmed}
                      onChange={(e) => setResultsConfirmed(e.target.checked)}
                      className="mt-0.5 rounded border-border"
                    />
                    <span className="text-[11px] text-muted-foreground">
                      Replace existing {MODEL_LABELS[model]} predictions in database
                    </span>
                  </label>

                  {resultsMut.isError && (
                    <div className="rounded-md border border-red-200 bg-red-50 px-2 py-1.5 text-[11px] text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-400 flex items-center gap-1">
                      <AlertTriangle className="h-3 w-3 shrink-0" />
                      {(resultsMut.error as Error).message}
                    </div>
                  )}

                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => resultsMut.mutate()}
                    disabled={!resultsConfirmed || resultsMut.isPending}
                    className={cn(
                      "w-full gap-1.5",
                      resultsConfirmed
                        ? "border-blue-300 text-blue-700 hover:bg-blue-50 dark:border-blue-700 dark:text-blue-300 dark:hover:bg-blue-950/30"
                        : "opacity-60"
                    )}
                  >
                    {resultsMut.isPending ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Database className="h-3.5 w-3.5" />
                    )}
                    Load Results
                  </Button>
                </>
              )}

              {resultsJobCompleted && (
                <div className="flex items-center gap-1.5 text-xs text-blue-600 dark:text-blue-400">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Predictions loaded into database
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end border-t px-5 py-3 shrink-0">
          <Button variant="ghost" size="sm" onClick={onClose}>
            {isParamsPromoted || resultsJobCompleted ? "Done" : "Cancel"}
          </Button>
        </div>
      </div>
    </div>
  );
}
