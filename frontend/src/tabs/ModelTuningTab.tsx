/**
 * ModelTuningTab -- Unified Model Tuning Studio.
 *
 * Main tab component for the demand planning model tuning workflow.
 * Supports LGBM, CatBoost, and XGBoost via model selector pills.
 * Provides experiment management, comparison, cluster EDA, and feature lab.
 */
import { useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  FlaskConical,
  MessageSquare,
  X,
  BarChart3,
  Microscope,
  Target,
  Crown,
  Database,
  Plus,
  FileText,
} from "lucide-react";

import {
  STALE,
  type TuningRun,
  type ModelType,
} from "@/api/queries";
import { useChartColors } from "@/hooks/useChartColors";
import { formatPct, formatFixed, formatInt } from "@/lib/formatters";
import { cn } from "@/lib/utils";

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
import { KpiCard } from "@/components/KpiCard";
import { LoadingElement } from "@/components/LoadingElement";
import { StatusBadge, formatDuration, timeAgo } from "@/components/shared-tuning-utils";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";

import { TuningChatPanel } from "./lgbm-tuning/TuningChatPanel";
import { ClusterEDAPanel } from "./lgbm-tuning/ClusterEDAPanel";
import { FeatureLabPanel } from "./lgbm-tuning/FeatureLabPanel";
import { LagFilterBar } from "./model-tuning/LagFilterBar";
import { EnhancedComparisonPanel } from "./model-tuning/EnhancedComparisonPanel";
import { ExperimentBuilder } from "./model-tuning/ExperimentBuilder";
import { EnhancedPromoteModal } from "./model-tuning/EnhancedPromoteModal";
import { LogViewer } from "./model-tuning/LogViewer";
import { ClusterExperimentsPanel } from "./clusters/ClusterExperimentsPanel";
import { ChampionExperimentsPanel } from "./champion/ChampionExperimentsPanel";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const MODEL_LABELS: Record<ModelType, string> = {
  lgbm: "LightGBM",
  catboost: "CatBoost",
  xgboost: "XGBoost",
};

const MODEL_PREFIX: Record<ModelType, string> = {
  lgbm: "/model-tuning/lgbm",
  catboost: "/model-tuning/catboost",
  xgboost: "/model-tuning/xgboost",
};

// ---------------------------------------------------------------------------
// Sub-tab definitions
// ---------------------------------------------------------------------------
type SubTab = "experiments" | "cluster-experiments" | "champion" | "cluster-eda" | "feature-lab";

const SUB_TAB_LABELS: Record<SubTab, { label: string; icon: typeof FlaskConical }> = {
  experiments: { label: "Algorithm Experiments", icon: FlaskConical },
  "cluster-experiments": { label: "Cluster Experiments", icon: Target },
  champion: { label: "Champion", icon: Crown },
  "cluster-eda": { label: "Cluster EDA", icon: BarChart3 },
  "feature-lab": { label: "Feature Lab", icon: Microscope },
};

// ---------------------------------------------------------------------------
// Status filter options
// ---------------------------------------------------------------------------
type StatusFilter = "all" | "running" | "completed" | "failed";

// ---------------------------------------------------------------------------
// Fetchers
// ---------------------------------------------------------------------------
async function fetchExperiments(
  model: ModelType,
  opts?: { limit?: number; status?: string; exec_lag?: number },
): Promise<{ experiments: TuningRun[]; total: number }> {
  const sp = new URLSearchParams();
  if (opts?.limit) sp.set("limit", String(opts.limit));
  if (opts?.status && opts.status !== "all") sp.set("status", opts.status);
  if (opts?.exec_lag !== undefined) sp.set("exec_lag", String(opts.exec_lag));
  const res = await fetch(`${MODEL_PREFIX[model]}/experiments?${sp}`, {
    cache: "no-cache",
  });
  if (!res.ok) throw new Error(`Failed to fetch runs: ${res.status}`);
  return res.json();
}

// StatusBadge, formatDuration, timeAgo imported from @/components/shared-tuning-utils

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function ModelTuningTab() {
  useChartColors(); // ensure theme context
  const queryClient = useQueryClient();

  // ---- State ---------------------------------------------------------------
  const [selectedModel, setSelectedModel] = useState<ModelType>("lgbm");
  const [execLag, setExecLag] = useState<number | undefined>(undefined);
  const [subTab, setSubTab] = useState<SubTab>("experiments");
  const [baselineId, setBaselineId] = useState<number | null>(null);
  const [candidateId, setCandidateId] = useState<number | null>(null);
  const [showBuilder, setShowBuilder] = useState(false);
  const [selectedRunForLogs, setSelectedRunForLogs] = useState<number | null>(
    null,
  );
  const [selectedRunForPromote, setSelectedRunForPromote] =
    useState<TuningRun | null>(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [page, setPage] = useState(0);
  const [sortCol, setSortCol] = useState<string>("run_id");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const pageSize = 25;

  // ---- Reset on model change -----------------------------------------------
  function handleModelChange(model: ModelType) {
    if (model === selectedModel) return;
    setSelectedModel(model);
    setSubTab("experiments");
    setBaselineId(null);
    setCandidateId(null);
    setSelectedRunForPromote(null);
    setSelectedRunForLogs(null);
    setPage(0);
  }

  // ---- Data fetching -------------------------------------------------------
  const {
    data: runsPayload,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: [
      "model-tuning-runs",
      selectedModel,
      statusFilter,
      execLag,
    ],
    queryFn: () =>
      fetchExperiments(selectedModel, {
        status: statusFilter !== "all" ? statusFilter : undefined,
        exec_lag: execLag,
      }),
    staleTime: STALE.TWO_MIN,
  });

  const allRuns = runsPayload?.experiments ?? [];

  // ---- Sort ----------------------------------------------------------------
  const sortedRuns = useMemo(() => {
    const runs = [...allRuns];
    runs.sort((a, b) => {
      let aVal: unknown = (a as Record<string, unknown>)[sortCol];
      let bVal: unknown = (b as Record<string, unknown>)[sortCol];

      // Numeric sort for certain columns
      if (
        sortCol === "run_id" ||
        sortCol === "accuracy_pct" ||
        sortCol === "wape" ||
        sortCol === "bias"
      ) {
        aVal = Number(aVal) || 0;
        bVal = Number(bVal) || 0;
        return sortDir === "asc"
          ? (aVal as number) - (bVal as number)
          : (bVal as number) - (aVal as number);
      }

      // String sort
      const aStr = String(aVal ?? "");
      const bStr = String(bVal ?? "");
      return sortDir === "asc"
        ? aStr.localeCompare(bStr)
        : bStr.localeCompare(aStr);
    });
    return runs;
  }, [allRuns, sortCol, sortDir]);

  // Paginate
  const pagedRuns = sortedRuns.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(sortedRuns.length / pageSize);

  // ---- KPIs ----------------------------------------------------------------
  const kpis = useMemo(() => {
    const completed = allRuns.filter(
      (r) => r.status === "completed" && r.accuracy_pct != null,
    );
    const running = allRuns.filter((r) => r.status === "running");
    const best = completed.reduce<TuningRun | null>(
      (acc, r) =>
        !acc || (r.accuracy_pct ?? 0) > (acc.accuracy_pct ?? 0) ? r : acc,
      null,
    );
    const promoted = allRuns.find((r) => r.is_promoted);

    return {
      bestAccuracy: best?.accuracy_pct ?? null,
      productionAccuracy: promoted?.accuracy_pct ?? null,
      totalRuns: allRuns.length,
      activeRuns: running.length,
    };
  }, [allRuns]);

  // ---- Row click handler ---------------------------------------------------
  function handleRowClick(runId: number) {
    if (baselineId === null) {
      setBaselineId(runId);
    } else if (candidateId === null) {
      if (runId === baselineId) return;
      setCandidateId(runId);
    } else {
      setBaselineId(runId);
      setCandidateId(null);
    }
  }

  function clearSelection() {
    setBaselineId(null);
    setCandidateId(null);
  }

  // ---- Column sort handler -------------------------------------------------
  function handleSort(col: string) {
    if (sortCol === col) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortCol(col);
      setSortDir("desc");
    }
  }

  function SortIndicator({ col }: { col: string }) {
    if (sortCol !== col) return null;
    return (
      <span className="ml-0.5 text-[10px]">
        {sortDir === "asc" ? "\u25B2" : "\u25BC"}
      </span>
    );
  }

  return (
    <div className="flex flex-col gap-5 p-4 md:p-6">
      {/* ---- Header ---- */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <FlaskConical className="h-5 w-5 text-muted-foreground" />
            Model Tuning Studio
          </h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            Configure experiments, compare results, and promote champions.
          </p>
        </div>
      </div>

      {/* ---- Sub-tab navigation ---- */}
      <div className="flex gap-1 border-b border-border pb-1">
        {(Object.keys(SUB_TAB_LABELS) as SubTab[]).map((key) => {
          const { label, icon: Icon } = SUB_TAB_LABELS[key];
          return (
            <button
              key={key}
              onClick={() => setSubTab(key)}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md transition-colors",
                subTab === key
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted",
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </button>
          );
        })}
      </div>

      {/* ---- Sub-tab content ---- */}
      {subTab === "cluster-experiments" && <ClusterExperimentsPanel />}
      {subTab === "champion" && <ChampionExperimentsPanel />}
      {subTab === "cluster-eda" && <ClusterEDAPanel />}
      {subTab === "feature-lab" && <FeatureLabPanel />}

      {/* ---- Experiments sub-tab ---- */}
      {subTab === "experiments" && (
        <>
          {/* Model selector pills + Lag filter */}
          <div className="flex items-center justify-between gap-4">
            <div className="flex gap-1 rounded-lg border border-border bg-muted/30 p-1">
              {(Object.keys(MODEL_LABELS) as ModelType[]).map((model) => (
                <button
                  key={model}
                  onClick={() => handleModelChange(model)}
                  className={cn(
                    "px-3 py-1 text-xs font-medium rounded-md transition-colors",
                    selectedModel === model
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted",
                  )}
                >
                  {MODEL_LABELS[model]}
                </button>
              ))}
            </div>
            <LagFilterBar value={execLag} onChange={setExecLag} />
          </div>

          {/* Loading / error states */}
          {isLoading ? (
            <LoadingElement message={`Loading ${MODEL_LABELS[selectedModel]} experiments...`} size="md" />
          ) : isError ? (
            <div className="py-8 text-center text-sm text-destructive">
              Failed to load experiments: {(error as Error).message}
            </div>
          ) : (
          <>
          {/* KPI cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <KpiCard
              label="Best Accuracy"
              value={
                kpis.bestAccuracy != null
                  ? formatPct(kpis.bestAccuracy)
                  : "--"
              }
              severity={
                kpis.bestAccuracy != null && kpis.bestAccuracy >= 80
                  ? "best"
                  : "neutral"
              }
              size="md"
              icon={Target}
            />
            <KpiCard
              label="Production Accuracy"
              value={
                kpis.productionAccuracy != null
                  ? formatPct(kpis.productionAccuracy)
                  : "Not promoted"
              }
              severity={
                kpis.productionAccuracy != null ? "best" : "neutral"
              }
              size="md"
              icon={Crown}
            />
            <KpiCard
              label="Total Runs"
              value={formatInt(kpis.totalRuns)}
              size="md"
            />
            <KpiCard
              label="Active Runs"
              value={formatInt(kpis.activeRuns)}
              severity={kpis.activeRuns > 0 ? "warning" : "neutral"}
              size="md"
            />
          </div>

          {/* Table toolbar */}
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Select
                value={statusFilter}
                onValueChange={(v) => {
                  setStatusFilter(v as StatusFilter);
                  setPage(0);
                }}
              >
                <SelectTrigger className="h-8 text-xs w-[120px]">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="running">Running</SelectItem>
                  <SelectItem value="completed">Completed</SelectItem>
                  <SelectItem value="failed">Failed</SelectItem>
                </SelectContent>
              </Select>
              <span className="text-xs text-muted-foreground">
                {sortedRuns.length} run{sortedRuns.length !== 1 ? "s" : ""}
              </span>
            </div>

            <Button
              size="sm"
              className="gap-1.5"
              onClick={() => setShowBuilder(true)}
            >
              <Plus className="h-3.5 w-3.5" />
              New Experiment
            </Button>
          </div>

          {/* Main content: table + comparison */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
            {/* Run history table */}
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">Run History</CardTitle>
                    <CardDescription className="text-xs">
                      Click two rows to compare (baseline then candidate)
                    </CardDescription>
                  </div>
                  {(baselineId !== null || candidateId !== null) && (
                    <Button variant="ghost" size="sm" onClick={clearSelection}>
                      Clear
                    </Button>
                  )}
                </div>
              </CardHeader>
              <CardContent className="p-0">
                {allRuns.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-16 text-center px-6">
                    <FlaskConical className="h-10 w-10 text-muted-foreground/30 mb-3" />
                    <p className="text-sm font-medium text-foreground mb-1">
                      No experiments yet for {MODEL_LABELS[selectedModel]}
                    </p>
                    <p className="text-xs text-muted-foreground mb-4">
                      Click "New Experiment" to configure and launch your first
                      tuning run.
                    </p>
                    <Button
                      size="sm"
                      className="gap-1.5"
                      onClick={() => setShowBuilder(true)}
                    >
                      <Plus className="h-3.5 w-3.5" />
                      New Experiment
                    </Button>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead
                            className="w-14 cursor-pointer"
                            onClick={() => handleSort("run_id")}
                          >
                            #<SortIndicator col="run_id" />
                          </TableHead>
                          <TableHead
                            className="cursor-pointer"
                            onClick={() => handleSort("run_label")}
                          >
                            Label
                            <SortIndicator col="run_label" />
                          </TableHead>
                          <TableHead
                            className="w-20 cursor-pointer"
                            onClick={() => handleSort("status")}
                          >
                            Status
                            <SortIndicator col="status" />
                          </TableHead>
                          <TableHead
                            className="text-right w-20 cursor-pointer"
                            onClick={() => handleSort("accuracy_pct")}
                          >
                            Acc%
                            <SortIndicator col="accuracy_pct" />
                          </TableHead>
                          <TableHead
                            className="text-right w-16 cursor-pointer"
                            onClick={() => handleSort("wape")}
                          >
                            WAPE
                            <SortIndicator col="wape" />
                          </TableHead>
                          <TableHead
                            className="text-right w-16 cursor-pointer"
                            onClick={() => handleSort("bias")}
                          >
                            Bias
                            <SortIndicator col="bias" />
                          </TableHead>
                          <TableHead className="w-20">Duration</TableHead>
                          <TableHead
                            className="w-24 cursor-pointer"
                            onClick={() => handleSort("started_at")}
                          >
                            Started
                            <SortIndicator col="started_at" />
                          </TableHead>
                          <TableHead className="w-20 text-center">
                            Actions
                          </TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {pagedRuns.map((run) => {
                          const isBaseline = baselineId === run.run_id;
                          const isCandidate = candidateId === run.run_id;
                          const isSelected = isBaseline || isCandidate;

                          return (
                            <TableRow
                              key={run.run_id}
                              className={cn(
                                "cursor-pointer transition-colors",
                                isBaseline &&
                                  "bg-blue-50 dark:bg-blue-950/30",
                                isCandidate &&
                                  "bg-emerald-50 dark:bg-emerald-950/30",
                                !isSelected && "hover:bg-muted/50",
                              )}
                              onClick={() => handleRowClick(run.run_id)}
                            >
                              <TableCell className="font-mono text-xs">
                                #{run.run_id}
                                {isBaseline && (
                                  <span className="ml-1 text-[10px] text-blue-600 dark:text-blue-400">
                                    (B)
                                  </span>
                                )}
                                {isCandidate && (
                                  <span className="ml-1 text-[10px] text-emerald-600 dark:text-emerald-400">
                                    (C)
                                  </span>
                                )}
                              </TableCell>
                              <TableCell className="text-sm max-w-[200px]">
                                <div className="flex items-center gap-1">
                                  <span className="truncate">{run.run_label}</span>
                                  {run.is_promoted && (
                                    <Crown className="shrink-0 h-3 w-3 text-amber-500" title="Parameters promoted" />
                                  )}
                                  {(run as Record<string, unknown>).is_results_promoted === true && (
                                    <Database className="shrink-0 h-3 w-3 text-blue-500" title="Results loaded" />
                                  )}
                                  {(run as Record<string, unknown>).cluster_source === "experimental" && (
                                    <span
                                      className="shrink-0 px-1.5 py-0 text-[9px] font-medium rounded-full bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300"
                                      title={
                                        (run as Record<string, unknown>).cluster_experiment_label
                                          ? `Cluster: ${(run as Record<string, unknown>).cluster_experiment_label}`
                                          : `Cluster experiment #${(run as Record<string, unknown>).cluster_experiment_id}`
                                      }
                                    >
                                      Exp #{String((run as Record<string, unknown>).cluster_experiment_id ?? "?")}
                                    </span>
                                  )}
                                </div>
                              </TableCell>
                              <TableCell>
                                <StatusBadge status={run.status} />
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
                              <TableCell className="text-xs text-muted-foreground tabular-nums">
                                {formatDuration(
                                  run.started_at,
                                  run.completed_at ?? null,
                                )}
                              </TableCell>
                              <TableCell
                                className="text-xs text-muted-foreground"
                                title={
                                  run.started_at
                                    ? new Date(
                                        run.started_at,
                                      ).toLocaleString()
                                    : ""
                                }
                              >
                                {timeAgo(run.started_at)}
                              </TableCell>
                              <TableCell className="text-center">
                                <div
                                  className="flex items-center justify-center gap-0.5"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  <button
                                    title="View Logs"
                                    className="rounded p-1 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
                                    onClick={() =>
                                      setSelectedRunForLogs(run.run_id)
                                    }
                                  >
                                    <FileText className="h-3.5 w-3.5" />
                                  </button>
                                  {run.status === "completed" && (
                                      <button
                                        title="Promote"
                                        className="rounded p-1 hover:bg-muted text-muted-foreground hover:text-amber-600 dark:hover:text-amber-400 transition-colors"
                                        onClick={() =>
                                          setSelectedRunForPromote(run)
                                        }
                                      >
                                        <Crown className="h-3.5 w-3.5" />
                                      </button>
                                    )}
                                </div>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>

                    {/* Pagination */}
                    {totalPages > 1 && (
                      <div className="flex items-center justify-between px-4 py-2 border-t border-border/40">
                        <span className="text-xs text-muted-foreground">
                          Page {page + 1} of {totalPages}
                        </span>
                        <div className="flex gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2 text-xs"
                            disabled={page === 0}
                            onClick={() => setPage((p) => Math.max(0, p - 1))}
                          >
                            Prev
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2 text-xs"
                            disabled={page >= totalPages - 1}
                            onClick={() =>
                              setPage((p) =>
                                Math.min(totalPages - 1, p + 1),
                              )
                            }
                          >
                            Next
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Comparison panel */}
            <div>
              {baselineId !== null && candidateId !== null ? (
                <EnhancedComparisonPanel
                  model={selectedModel}
                  baselineId={baselineId}
                  candidateId={candidateId}
                  execLag={execLag}
                  onPromote={(run) => setSelectedRunForPromote(run)}
                />
              ) : (
                <Card className="h-full">
                  <CardContent className="flex flex-col items-center justify-center py-20 text-center">
                    <FlaskConical className="h-10 w-10 text-muted-foreground/40 mb-3" />
                    <p className="text-sm text-muted-foreground">
                      {baselineId !== null
                        ? "Now click a second row to select the candidate run."
                        : "Click a row to select the baseline, then click another for the candidate."}
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </>
          )}
        </>
      )}

      {/* ---- Experiment Builder Modal ---- */}
      <ExperimentBuilder
        model={selectedModel}
        open={showBuilder}
        onClose={() => setShowBuilder(false)}
        onSubmitted={() => {
          setShowBuilder(false);
          queryClient.invalidateQueries({
            queryKey: ["model-tuning-runs", selectedModel],
          });
        }}
      />

      {/* ---- Log Viewer Slide-Over ---- */}
      <LogViewer
        model={selectedModel}
        runId={selectedRunForLogs ?? 0}
        open={selectedRunForLogs !== null}
        onClose={() => setSelectedRunForLogs(null)}
      />

      {/* ---- Enhanced Promote Modal ---- */}
      {selectedRunForPromote && (
        <EnhancedPromoteModal
          model={selectedModel}
          run={selectedRunForPromote}
          open={true}
          onClose={() => setSelectedRunForPromote(null)}
          onPromoted={() => {
            setSelectedRunForPromote(null);
            queryClient.invalidateQueries({
              queryKey: ["model-tuning-runs", selectedModel],
            });
          }}
        />
      )}

      {/* ---- Floating AI Tuning Advisor ---- */}
      {createPortal(
        <div className="fixed top-4 right-6 z-50 flex flex-col items-end gap-3">
          {!chatOpen && (
            <button
              onClick={() => setChatOpen(true)}
              title="AI Tuning Advisor"
              className="h-10 w-10 rounded-full bg-primary text-primary-foreground shadow-lg hover:bg-primary/90 transition-all hover:scale-105 flex items-center justify-center"
            >
              <MessageSquare className="h-4.5 w-4.5" />
            </button>
          )}

          {chatOpen && (
            <div className="w-[420px] max-h-[80vh] animate-in slide-in-from-top-4 fade-in duration-200 rounded-2xl border border-border bg-card shadow-2xl overflow-hidden flex flex-col">
              <div className="flex items-center justify-between px-4 py-2.5 border-b border-border bg-muted/30">
                <div className="flex items-center gap-2">
                  <MessageSquare className="h-4 w-4 text-primary" />
                  <span className="text-sm font-semibold">
                    AI Tuning Advisor
                  </span>
                </div>
                <button
                  onClick={() => setChatOpen(false)}
                  className="rounded-md p-1 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="flex-1 overflow-hidden">
                <TuningChatPanel />
              </div>
            </div>
          )}
        </div>,
        document.body,
      )}
    </div>
  );
}
