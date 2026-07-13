/**
 * ChampionExperimentsPanel — Main panel for the Champion sub-tab.
 *
 * KPI cards, experiment list table, comparison panel, and log viewer.
 * Production champion mutation is reserved for the governed champion-refresh flow.
 */
import { useMemo, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Crown,
  Database,
  Plus,
  FileText,
  Trash2,
  XCircle,
  Target,
  TrendingUp,
  Activity,
  Trophy,
} from "lucide-react";

import {
  championExperimentKeys,
  CHAMPION_EXP_STALE,
  fetchChampionExperiments,
  fetchChampionExperimentLogs,
  assignChampionExperiment,
  cancelChampionExperiment,
  deleteChampionExperiment,
  type ChampionExperiment,
} from "@/api/queries";
import { Button } from "@/components/ui/button";
import { KpiCard } from "@/components/KpiCard";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { StatusBadge, formatDuration } from "@/components/shared-tuning-utils";
import { cn } from "@/lib/utils";
import { formatApiError } from "@/lib/formatApiError";
import { toast } from "@/components/Toaster";

import { LagFilterBar } from "@/tabs/model-tuning/LagFilterBar";
import { ChampionExperimentBuilder } from "./ChampionExperimentBuilder";
import { ChampionComparisonPanel } from "./ChampionComparisonPanel";
import { ChampionPromoteModal } from "./ChampionPromoteModal";
import { SweepBuilder } from "./SweepBuilder";
import { SweepResultsPanel } from "./SweepResultsPanel";
import {
  championSweepKeys,
  fetchChampionSweeps,
  type ChampionSweep,
} from "@/api/queries";

// ---------------------------------------------------------------------------
// Status filter
// ---------------------------------------------------------------------------
type StatusFilter = "all" | "running" | "completed" | "failed";

const ASSIGNABLE_MODELS = new Set([
  "lgbm_cluster",
  "chronos2_enriched",
  "mstl",
  "nbeats",
  "nhits",
]);

function isAssignmentEligible(experiment: ChampionExperiment): boolean {
  return (
    experiment.models.length === ASSIGNABLE_MODELS.size &&
    new Set(experiment.models).size === ASSIGNABLE_MODELS.size &&
    experiment.models.every((modelId) => ASSIGNABLE_MODELS.has(modelId))
  );
}

// ---------------------------------------------------------------------------
// Strategy labels
// ---------------------------------------------------------------------------
const STRATEGY_SHORT: Record<string, string> = {
  expanding: "Expanding",
  rolling: "Rolling",
  decay: "Decay",
  ensemble: "Ensemble",
  meta_learner: "Meta-Learner",
  hybrid_warmup: "Hybrid Warmup",
  adaptive_ensemble: "Adaptive Ens.",
  ensemble_rolling: "Ens. Rolling",
  optimized_decay: "Opt. Decay",
  learned_blend: "Learned Blend",
  ridge_blend: "Ridge Blend",
  shrinkage_blend: "Shrinkage",
  bayesian_model_avg: "Bayesian Avg",
  per_segment: "Per-Segment",
  per_cluster: "Per-Cluster",
  seasonal: "Seasonal",
  diverse_ensemble: "Diverse Ens.",
  uncertainty_aware: "Uncertainty",
  cascade_ensemble: "Cascade Ens.",
  adversarial_filter: "Adversarial",
  dynamic_window: "Dynamic Win.",
  regime_adaptive: "Regime Adapt.",
  error_correcting: "Error Corr.",
  thompson_sampling: "Thompson",
  thompson_ensemble: "Thompson Ens.",
  linucb: "LinUCB",
  exp3: "EXP3",
  dfu_strategy_router: "DFU Router",
  stacked_strategies: "Stacked",
  cluster_regime_hybrid: "Cluster-Regime",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ChampionExperimentsPanel() {
  const queryClient = useQueryClient();

  // State
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [execLag, setExecLag] = useState<number | undefined>(undefined);
  const [showBuilder, setShowBuilder] = useState(false);
  const [showSweepBuilder, setShowSweepBuilder] = useState(false);
  const [selectedSweepId] = useState<number | null>(null);
  const [baselineId, setBaselineId] = useState<number | null>(null);
  const [candidateId, setCandidateId] = useState<number | null>(null);
  const [logExpId, setLogExpId] = useState<number | null>(null);
  const [logOffset, setLogOffset] = useState(0);
  const [assignmentExperiment, setAssignmentExperiment] =
    useState<ChampionExperiment | null>(null);

  // Data
  const { data, isLoading, isError } = useQuery({
    queryKey: championExperimentKeys.experiments({ status: statusFilter, exec_lag: execLag }),
    queryFn: () =>
      fetchChampionExperiments({
        status: statusFilter !== "all" ? statusFilter : undefined,
        exec_lag: execLag,
      }),
    staleTime: CHAMPION_EXP_STALE.EXPERIMENTS,
  });

  const experiments = useMemo(() => data?.experiments ?? [], [data?.experiments]);

  // Sweeps — most recent first; auto-select the newest so results show after launch.
  const { data: sweepData } = useQuery({
    queryKey: championSweepKeys.list(),
    queryFn: () => fetchChampionSweeps({ limit: 20 }),
    refetchInterval: (q) => {
      const sweeps = (q.state.data as { sweeps: ChampionSweep[] } | undefined)?.sweeps ?? [];
      return sweeps.some((s) => s.status === "queued" || s.status === "running") ? 3000 : false;
    },
  });
  const sweeps = sweepData?.sweeps ?? [];
  const activeSweepId = selectedSweepId ?? sweeps[0]?.sweep_id ?? null;

  // Log fetcher
  const { data: logData } = useQuery({
    queryKey: championExperimentKeys.logs(logExpId ?? 0, logOffset),
    queryFn: () => fetchChampionExperimentLogs(logExpId!, logOffset),
    enabled: logExpId !== null,
    refetchInterval: (query) => {
      if (query.state.data?.has_more) return 5000;
      return false;
    },
  });

  // Mutations
  const cancelMutation = useMutation({
    mutationFn: cancelChampionExperiment,
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: championExperimentKeys.all }),
  });

  const deleteMutation = useMutation({
    mutationFn: deleteChampionExperiment,
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: championExperimentKeys.all }),
  });

  const assignmentMutation = useMutation({
    mutationFn: assignChampionExperiment,
    onSuccess: (result) => {
      setAssignmentExperiment(null);
      queryClient.invalidateQueries({ queryKey: championExperimentKeys.all });
      toast.success(
        `Champion assignment queued as ${result.job_id}. Monitor Jobs, then continue to Forecast.`,
      );
    },
    onError: (error) => toast.error(formatApiError(error)),
  });

  // KPIs
  const kpis = useMemo(() => {
    const completed = experiments.filter(
      (e) => e.status === "completed" && e.champion_accuracy != null,
    );
    const running = experiments.filter((e) => e.status === "running");
    const promoted = experiments.find((e) => e.is_promoted);
    const best = completed.reduce<ChampionExperiment | null>(
      (acc, e) =>
        !acc || (e.champion_accuracy ?? 0) > (acc.champion_accuracy ?? 0) ? e : acc,
      null,
    );

    return {
      bestAccuracy: best?.champion_accuracy ?? null,
      productionStrategy: promoted?.strategy ?? null,
      gapToCeiling: best?.gap_bps ?? null,
      activeRuns: running.length,
    };
  }, [experiments]);

  // Auto-rank completed experiments by accuracy (highest = rank 1)
  const rankMap = useMemo(() => {
    const completed = experiments
      .filter((e) => e.status === "completed" && e.champion_accuracy != null)
      .sort((a, b) => (b.champion_accuracy ?? 0) - (a.champion_accuracy ?? 0));
    const map = new Map<number, number>();
    completed.forEach((e, i) => map.set(e.experiment_id, i + 1));
    return map;
  }, [experiments]);

  // Row click (baseline/candidate selection)
  function handleRowClick(id: number) {
    if (baselineId === null) {
      setBaselineId(id);
      setCandidateId(null);
    } else if (baselineId === id) {
      setBaselineId(null);
      setCandidateId(null);
    } else if (candidateId === id) {
      setCandidateId(null);
    } else {
      setCandidateId(id);
    }
  }

  const baseline = experiments.find((e) => e.experiment_id === baselineId) ?? null;
  const candidate = experiments.find((e) => e.experiment_id === candidateId) ?? null;

  return (
    <div className="space-y-4">
      {/* KPI Row */}
      <div className="grid grid-cols-4 gap-3">
        <KpiCard
          label="Best Champion Accuracy"
          value={kpis.bestAccuracy != null ? `${kpis.bestAccuracy.toFixed(2)}%` : "--"}
          icon={Target}
        />
        <KpiCard
          label="Production Strategy"
          value={
            kpis.productionStrategy
              ? STRATEGY_SHORT[kpis.productionStrategy] ?? kpis.productionStrategy
              : "--"
          }
          icon={Crown}
        />
        <KpiCard
          label="Gap to Ceiling"
          value={kpis.gapToCeiling != null ? `${kpis.gapToCeiling.toFixed(0)} bps` : "--"}
          icon={TrendingUp}
        />
        <KpiCard
          label="Active Runs"
          value={String(kpis.activeRuns)}
          icon={Activity}
        />
      </div>

      {/* Toolbar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Select
            value={statusFilter}
            onValueChange={(v) => setStatusFilter(v as StatusFilter)}
          >
            <SelectTrigger className="w-36 h-8 text-xs">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="running">Running</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
            </SelectContent>
          </Select>
          <LagFilterBar value={execLag} onChange={setExecLag} />
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={() => setShowSweepBuilder(true)}>
            <Trophy className="mr-1 h-3.5 w-3.5" />
            Run Sweep
          </Button>
          <Button size="sm" onClick={() => setShowBuilder(true)}>
            <Plus className="mr-1 h-3.5 w-3.5" />
            New Experiment
          </Button>
        </div>
      </div>

      {/* Sweep results (tournament) — shows the latest/selected sweep */}
      {activeSweepId != null ? (
        <SweepResultsPanel sweepId={activeSweepId} />
      ) : null}

      {showSweepBuilder ? <SweepBuilder onClose={() => setShowSweepBuilder(false)} /> : null}

      {/* Layout: table + comparison */}
      <div className="flex gap-4">
        {/* Table */}
        <div className={cn("flex-1 min-w-0", baseline && candidate ? "w-1/2" : "w-full")}>
          <Card>
            <CardContent className="p-0">
              {isLoading ? (
                <div className="py-12 text-center text-sm text-muted-foreground">
                  Loading experiments...
                </div>
              ) : isError ? (
                <div className="py-12 text-center text-sm text-red-500">
                  Failed to load experiments.
                </div>
              ) : experiments.length === 0 ? (
                <div className="py-12 text-center text-sm text-muted-foreground">
                  No experiments yet. Click "New Experiment" to start.
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="text-xs w-8">#</TableHead>
                      <TableHead className="text-xs w-12 text-center">Rank</TableHead>
                      <TableHead className="text-xs">Label</TableHead>
                      <TableHead className="text-xs">Strategy</TableHead>
                      <TableHead className="text-xs text-right">Acc%</TableHead>
                      <TableHead className="text-xs text-right">Ceiling%</TableHead>
                      <TableHead className="text-xs text-right">Gap</TableHead>
                      <TableHead className="text-xs">Status</TableHead>
                      <TableHead className="text-xs">Duration</TableHead>
                      <TableHead className="text-xs w-24">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {experiments.map((exp) => {
                      const isBaseline = baselineId === exp.experiment_id;
                      const isCandidate = candidateId === exp.experiment_id;
                      return (
                        <TableRow
                          key={exp.experiment_id}
                          className={cn(
                            "cursor-pointer transition-colors",
                            isBaseline && "bg-blue-50 dark:bg-blue-900/20",
                            isCandidate && "bg-emerald-50 dark:bg-emerald-900/20",
                          )}
                          onClick={() => handleRowClick(exp.experiment_id)}
                        >
                          <TableCell className="text-xs font-mono">
                            {exp.experiment_id}
                          </TableCell>
                          <TableCell className="text-center">
                            {(() => {
                              const rank = rankMap.get(exp.experiment_id);
                              if (!rank) return <span className="text-xs text-muted-foreground">--</span>;
                              if (rank === 1) return <span className="text-sm" title={`Rank #${rank}`}>&#x1F947;</span>;
                              if (rank === 2) return <span className="text-sm" title={`Rank #${rank}`}>&#x1F948;</span>;
                              if (rank === 3) return <span className="text-sm" title={`Rank #${rank}`}>&#x1F949;</span>;
                              return <span className="text-xs font-mono text-muted-foreground" title={`Rank #${rank}`}>#{rank}</span>;
                            })()}
                          </TableCell>
                          <TableCell className="text-xs font-medium max-w-[200px]">
                            <div className="flex items-center gap-1">
                              <span className="truncate">{exp.label}</span>
                              {exp.is_promoted && (
                                <span title="Promoted to production">
                                  <Crown className="shrink-0 h-3 w-3 text-amber-500" />
                                </span>
                              )}
                              {exp.is_results_promoted && (
                                <span title="Results loaded">
                                  <Database className="shrink-0 h-3 w-3 text-blue-500" />
                                </span>
                              )}
                            </div>
                          </TableCell>
                          <TableCell className="text-xs font-mono">
                            {STRATEGY_SHORT[exp.strategy] ?? exp.strategy}
                          </TableCell>
                          <TableCell className="text-xs text-right font-medium">
                            {exp.champion_accuracy?.toFixed(2) ?? "--"}
                          </TableCell>
                          <TableCell className="text-xs text-right">
                            {exp.ceiling_accuracy?.toFixed(2) ?? "--"}
                          </TableCell>
                          <TableCell className="text-xs text-right">
                            {exp.gap_bps != null ? `${exp.gap_bps.toFixed(0)}` : "--"}
                          </TableCell>
                          <TableCell>
                            <StatusBadge status={exp.status} />
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground">
                            {formatDuration(exp.started_at, exp.completed_at)}
                          </TableCell>
                          <TableCell>
                            <div
                              className="flex gap-1"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6"
                                title="View Logs"
                                onClick={() => {
                                  setLogExpId(exp.experiment_id);
                                  setLogOffset(0);
                                }}
                              >
                                <FileText className="h-3 w-3" />
                              </Button>
                              {exp.status === "completed" && !exp.is_promoted && (
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-6 w-6"
                                  title={
                                    isAssignmentEligible(exp)
                                      ? "Select & Assign Champion"
                                      : "Assignment requires all five governed models"
                                  }
                                  onClick={() => setAssignmentExperiment(exp)}
                                  disabled={!isAssignmentEligible(exp)}
                                >
                                  <Crown className="h-3 w-3 text-amber-500" />
                                </Button>
                              )}
                              {(exp.status === "running" || exp.status === "queued") && (
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-6 w-6"
                                  title="Cancel"
                                  onClick={() =>
                                    cancelMutation.mutate(exp.experiment_id)
                                  }
                                >
                                  <XCircle className="h-3 w-3" />
                                </Button>
                              )}
                              {(exp.status === "completed" ||
                                exp.status === "failed" ||
                                exp.status === "cancelled") &&
                                !exp.is_promoted && (
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-6 w-6"
                                    title="Delete"
                                    onClick={() =>
                                      deleteMutation.mutate(exp.experiment_id)
                                    }
                                  >
                                    <Trash2 className="h-3 w-3" />
                                  </Button>
                                )}
                            </div>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>

          {/* Selection hint */}
          {baselineId && !candidateId && (
            <p className="mt-2 text-xs text-muted-foreground">
              Baseline selected (blue). Click another row to select candidate.
            </p>
          )}
        </div>

        {/* Comparison panel (right side) */}
        {baseline && candidate && (
          <div className="w-1/2 min-w-[300px]">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold">Comparison</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setBaselineId(null);
                  setCandidateId(null);
                }}
              >
                Clear
              </Button>
            </div>
            <ChampionComparisonPanel baseline={baseline} candidate={candidate} execLag={execLag} />
          </div>
        )}
      </div>

      {/* Log viewer slide-over */}
      {logExpId !== null && (
        <div className="fixed inset-y-0 right-0 z-40 w-[480px] border-l bg-background shadow-xl flex flex-col">
          <div className="flex items-center justify-between border-b px-4 py-3">
            <h3 className="text-sm font-semibold">
              Experiment #{logExpId} Logs
            </h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setLogExpId(null)}
            >
              Close
            </Button>
          </div>
          <pre className="flex-1 overflow-auto p-4 text-xs font-mono whitespace-pre-wrap">
            {logData?.log || "No logs yet..."}
          </pre>
          {logData?.has_more && (
            <div className="border-t px-4 py-2 text-xs text-muted-foreground">
              Streaming...
            </div>
          )}
        </div>
      )}

      {/* Builder modal */}
      <ChampionExperimentBuilder
        open={showBuilder}
        onClose={() => setShowBuilder(false)}
        onSubmitted={() =>
          queryClient.invalidateQueries({ queryKey: championExperimentKeys.all })
        }
      />

      {assignmentExperiment ? (
        <ChampionPromoteModal
          experiment={assignmentExperiment}
          open
          onClose={() => setAssignmentExperiment(null)}
          onAssign={(experimentId) => assignmentMutation.mutate(experimentId)}
          isPending={assignmentMutation.isPending}
        />
      ) : null}

    </div>
  );
}
