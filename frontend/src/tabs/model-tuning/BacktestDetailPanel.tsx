/**
 * BacktestDetailPanel -- detail view for non-tunable models showing
 * current KPIs, run/load actions, and run history table.
 */
import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Play, Database, CheckCircle2, Clock, XCircle, Loader2 } from "lucide-react";

import {
  backtestMgmtKeys,
  BACKTEST_MGMT_STALE,
  fetchBacktestRuns,
  fetchBacktestCurrent,
  submitBacktestRun,
  submitBacktestLoad,
  type BacktestRun,
} from "@/api/queries/backtest-management";
import { formatPct, formatFixed, formatInt } from "@/lib/formatters";
import { modelLabel } from "@/lib/model-labels";
import { cn } from "@/lib/utils";
import {
  fetchPipelineConfig,
  pipelineConfigKeys,
} from "@/api/queries/unified-model-tuning";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { KpiCard } from "@/components/KpiCard";
import { LoadingElement } from "@/components/LoadingElement";
import { StatusBadge, timeAgo } from "@/components/shared-tuning-utils";

// ---------------------------------------------------------------------------
// Status icon helper
// ---------------------------------------------------------------------------

function statusIcon(status: BacktestRun["status"]) {
  switch (status) {
    case "completed":
      return <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />;
    case "running":
    case "queued":
      return <Loader2 className="h-3.5 w-3.5 text-amber-500 animate-spin" />;
    case "failed":
      return <XCircle className="h-3.5 w-3.5 text-destructive" />;
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface BacktestDetailPanelProps {
  modelId: string;
}

export function BacktestDetailPanel({ modelId }: BacktestDetailPanelProps) {
  const queryClient = useQueryClient();
  const [isRunning, setIsRunning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingRunId, setLoadingRunId] = useState<number | null>(null);

  // Fetch run history — poll while running or loading
  const {
    data: runs,
    isLoading: runsLoading,
    isError: runsError,
  } = useQuery({
    queryKey: backtestMgmtKeys.runs(modelId),
    queryFn: () => fetchBacktestRuns(modelId),
    staleTime: BACKTEST_MGMT_STALE.RUNS,
    refetchInterval: isRunning || isLoading ? 5_000 : false,
  });

  // Fetch current metadata from disk
  const { data: current } = useQuery({
    queryKey: backtestMgmtKeys.current(modelId),
    queryFn: () => fetchBacktestCurrent(modelId),
    staleTime: BACKTEST_MGMT_STALE.CURRENT,
  });

  // Derive KPIs from current metadata or latest completed run
  const latestCompleted = runs?.find(
    (r) => r.status === "completed" && r.accuracy_pct != null,
  );
  const accuracy =
    (current?.accuracy_pct as number | undefined) ??
    latestCompleted?.accuracy_pct ??
    null;
  const wape =
    (current?.wape as number | undefined) ?? latestCompleted?.wape ?? null;
  const bias =
    (current?.bias as number | undefined) ?? latestCompleted?.bias ?? null;
  const nPredictions =
    (current?.n_predictions as number | undefined) ??
    latestCompleted?.n_predictions ??
    null;
  const nDfus =
    (current?.n_dfus as number | undefined) ??
    latestCompleted?.n_dfus ??
    null;
  const hasAnyRun = (runs?.length ?? 0) > 0;
  const hasActiveRun = runs?.some(
    (r) => r.status === "running" || r.status === "queued",
  );
  // A load is "active" if any run has a load_job_id but hasn't been marked loaded yet
  const hasActiveLoad = runs?.some(
    (r) => r.load_job_id != null && !r.is_loaded_to_db,
  ) ?? false;

  // On mount, if a load is already active (e.g. page refresh), sync isLoading state
  useEffect(() => {
    if (hasActiveLoad && !isLoading) {
      setIsLoading(true);
    }
  }, [hasActiveLoad]); // eslint-disable-line react-hooks/exhaustive-deps

  // Handlers
  const invalidateAll = () => {
    queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.summary });
    queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.runs(modelId) });
    queryClient.invalidateQueries({
      queryKey: backtestMgmtKeys.current(modelId),
    });
  };

  const handleRun = async () => {
    setIsRunning(true);
    try {
      await submitBacktestRun(modelId);
      invalidateAll();
    } catch (err) {
      console.error("Failed to submit backtest:", err);
    } finally {
      // Keep isRunning true briefly so the refetchInterval picks up
      setTimeout(() => setIsRunning(false), 10_000);
    }
  };

  // Track which run ID we submitted a load for, to know when it's done
  const [loadTargetRunId, setLoadTargetRunId] = useState<number | null>(null);

  const handleLoad = async (runId?: number) => {
    if (runId != null) {
      setLoadingRunId(runId);
      setLoadTargetRunId(runId);
    }
    setIsLoading(true);
    try {
      await submitBacktestLoad(modelId, runId);
      invalidateAll();
      // Keep isLoading true — cleared by the effect below when the run shows loaded
    } catch (err) {
      console.error("Failed to load backtest:", err);
      setIsLoading(false);
      setLoadingRunId(null);
      setLoadTargetRunId(null);
    }
  };

  // Clear loading state once the target run's is_loaded_to_db becomes true
  useEffect(() => {
    if (loadTargetRunId == null || !runs) return;
    const targetRun = runs.find((r) => r.id === loadTargetRunId);
    if (targetRun?.is_loaded_to_db) {
      setIsLoading(false);
      setLoadingRunId(null);
      setLoadTargetRunId(null);
    }
  }, [runs, loadTargetRunId]);

  return (
    <div className="space-y-4">
      {/* KPI row */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <KpiCard
          label="Accuracy"
          value={accuracy != null ? formatPct(accuracy) : "--"}
          severity={accuracy != null && accuracy >= 80 ? "best" : "neutral"}
          size="md"
        />
        <KpiCard
          label="WAPE"
          value={wape != null ? formatFixed(wape, 2) : "--"}
          size="md"
        />
        <KpiCard
          label="Bias"
          value={bias != null ? formatFixed(bias, 2) : "--"}
          size="md"
        />
        <KpiCard
          label="Predictions"
          value={nPredictions != null ? formatInt(nPredictions) : "--"}
          size="md"
        />
        <KpiCard
          label="DFUs"
          value={nDfus != null ? formatInt(nDfus) : "--"}
          size="md"
        />
      </div>

      {/* Backtest config (tree models only) */}
      <BacktestConfigPanel modelId={modelId} />

      {/* Action buttons */}
      <div className="flex gap-3">
        <Button
          onClick={handleRun}
          disabled={isRunning || !!hasActiveRun}
          className="gap-2"
        >
          {isRunning || hasActiveRun ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Play className="h-4 w-4" />
          )}
          {isRunning || hasActiveRun ? "Backtest Running..." : "Run Backtest"}
        </Button>
        {/* Backtests auto-load into the DB on completion, so the manual Load is
            a recovery affordance only — shown when a completed run never loaded
            (e.g. the best-effort auto-load failed). */}
        {((isLoading || hasActiveLoad) ||
          (latestCompleted && !latestCompleted.is_loaded_to_db)) && (
          <Button
            variant="outline"
            onClick={() => handleLoad(latestCompleted?.id)}
            disabled={isLoading || hasActiveLoad || !latestCompleted}
            title="Backtests load into the database automatically on completion. Use this only if the automatic load didn't run."
            className="gap-2"
          >
            {isLoading || hasActiveLoad ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Database className="h-4 w-4" />
            )}
            {isLoading || hasActiveLoad ? "Loading to DB..." : "Load to DB"}
          </Button>
        )}
      </div>

      {/* Run history table */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">
            Run History -- {modelLabel(modelId)}
          </CardTitle>
          <CardDescription className="text-xs">
            {hasAnyRun
              ? `${runs?.length ?? 0} run${(runs?.length ?? 0) !== 1 ? "s" : ""} recorded`
              : "No backtest runs yet. Click 'Run Backtest' to get started."}
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          {runsLoading ? (
            <LoadingElement message="Loading run history..." size="md" />
          ) : runsError ? (
            <div className="py-8 text-center text-sm text-destructive">
              Failed to load run history.
            </div>
          ) : !hasAnyRun ? (
            <div className="flex flex-col items-center justify-center py-12 text-center px-6">
              <Clock className="h-8 w-8 text-muted-foreground/30 mb-2" />
              <p className="text-sm text-muted-foreground">
                No runs recorded for this model yet.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-12">#</TableHead>
                    <TableHead className="w-24">Date</TableHead>
                    <TableHead className="w-20">Status</TableHead>
                    <TableHead className="text-right w-20">Acc%</TableHead>
                    <TableHead className="text-right w-16">WAPE</TableHead>
                    <TableHead className="text-right w-16">Bias</TableHead>
                    <TableHead className="w-20 text-center">Loaded</TableHead>
                    <TableHead className="w-20 text-center">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {runs?.map((run) => (
                    <TableRow key={run.id}>
                      <TableCell className="font-mono text-xs">
                        #{run.id}
                      </TableCell>
                      <TableCell
                        className="text-xs text-muted-foreground"
                        title={
                          run.created_at
                            ? new Date(run.created_at).toLocaleString()
                            : ""
                        }
                      >
                        {timeAgo(run.created_at)}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1">
                          {statusIcon(run.status)}
                          <StatusBadge status={run.status} />
                        </div>
                      </TableCell>
                      <TableCell className="text-right tabular-nums text-sm">
                        {formatPct(run.accuracy_pct)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums text-sm">
                        {formatFixed(run.wape, 2)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums text-sm">
                        {formatFixed(run.bias, 2)}
                      </TableCell>
                      <TableCell className="text-center">
                        {run.is_loaded_to_db ? (
                          <Badge
                            variant="outline"
                            className="text-[9px] bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300"
                          >
                            Loaded
                          </Badge>
                        ) : (
                          <span className="text-[10px] text-muted-foreground">
                            --
                          </span>
                        )}
                      </TableCell>
                      <TableCell className="text-center">
                        {run.status === "completed" && !run.is_loaded_to_db && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2 text-[10px] gap-1"
                            disabled={loadingRunId === run.id || isLoading || hasActiveLoad}
                            onClick={() => handleLoad(run.id)}
                          >
                            {loadingRunId === run.id ? (
                              <Loader2 className="h-3 w-3 animate-spin" />
                            ) : (
                              <Database className="h-3 w-3" />
                            )}
                            Load
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// BacktestConfigPanel — inline config for tree-model backtest options
// ---------------------------------------------------------------------------

const TREE_MODELS = new Set([
  "lgbm_cluster", "catboost_cluster", "xgboost_cluster",
  "lgbm_cust_enriched", "catboost_cust_enriched", "xgboost_cust_enriched",
]);

function BacktestConfigPanel({ modelId }: { modelId: string }) {
  const isTree = TREE_MODELS.has(modelId);

  const { data: pipelineConfig } = useQuery({
    queryKey: pipelineConfigKeys.config,
    queryFn: fetchPipelineConfig,
    staleTime: 60_000,
    enabled: isTree,
  });

  const algo = pipelineConfig?.algorithms?.[modelId];
  const params = algo?.params ?? {};

  if (!isTree || !algo) return null;

  const strategy = algo.cluster_strategy ?? "per_cluster";
  const recursive = params.recursive as boolean ?? true;
  const shapSelect = params.shap_select as boolean ?? false;
  const correlationFilter = params.correlation_filter as boolean ?? false;
  const varianceFilter = params.variance_filter as boolean ?? false;
  const tuneInline = params.tune_inline as boolean ?? false;
  const nEstimators = params.n_estimators as number ?? 2000;
  const learningRate = params.learning_rate as number ?? 0.015;
  const numLeaves = params.num_leaves as number ?? 63;
  const maxDepth = params.max_depth as number ?? 8;
  const objective = params.objective as string ?? "regression_l1";

  return (
    <Card>
      <CardHeader className="py-3">
        <CardTitle className="text-sm">Backtest Configuration</CardTitle>
        <CardDescription className="text-xs">
          From forecast_pipeline_config.yaml — edit config file to change values
        </CardDescription>
      </CardHeader>
      <CardContent className="pb-3 space-y-3">
        {/* Strategy & Mode */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <ConfigItem label="Strategy" value={strategy} />
          <ConfigItem label="Prediction" value={recursive ? "Recursive" : "Direct"} />
          <ConfigItem label="Objective" value={objective} />
          <ConfigItem label="Inline Tune" value={tuneInline ? "ON" : "OFF"} highlight={tuneInline} />
        </div>

        {/* Hyperparams */}
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
            Hyperparameters {" "}
            <span className="font-normal normal-case">(base — may be overridden by cluster profiles)</span>
          </p>
          <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
            <ConfigItem label="n_estimators" value={nEstimators} />
            <ConfigItem label="learning_rate" value={learningRate} />
            <ConfigItem label="num_leaves" value={numLeaves} />
            <ConfigItem label="max_depth" value={maxDepth} />
            <ConfigItem label="subsample" value={params.subsample as number ?? 0.75} />
          </div>
        </div>

        {/* Feature Selection */}
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
            Feature Selection
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <ConfigItem label="SHAP Select" value={shapSelect ? "ON" : "OFF"} highlight={shapSelect} />
            <ConfigItem label="SHAP Threshold" value={params.shap_threshold as number ?? 0.9} />
            <ConfigItem label="Corr. Filter" value={correlationFilter ? "ON" : "OFF"} highlight={correlationFilter} />
            <ConfigItem label="Var. Filter" value={varianceFilter ? "ON" : "OFF"} highlight={varianceFilter} />
          </div>
        </div>

        {/* Cluster Tuning Profiles */}
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
            Cluster Tuning Profiles
          </p>
          <p className="text-xs text-muted-foreground">
            Per-cluster hyperparameter overrides from <code className="text-[10px]">cluster_tuning_profiles.yaml</code>.
            Run <code className="text-[10px]">make tune-lgbm-clusters</code> to auto-tune per cluster.
            Set <code className="text-[10px]">enabled: false</code> in the YAML to use global params only.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

// Small display-only config item
function ConfigItem({ label, value, highlight }: { label: string; value: string | number | boolean; highlight?: boolean }) {
  const display = typeof value === "boolean" ? (value ? "ON" : "OFF") : String(value);
  return (
    <div className="rounded border px-2 py-1.5">
      <p className="text-[9px] font-medium text-muted-foreground uppercase tracking-wider">{label}</p>
      <p className={cn("text-xs font-semibold tabular-nums", highlight && "text-primary")}>{display}</p>
    </div>
  );
}
