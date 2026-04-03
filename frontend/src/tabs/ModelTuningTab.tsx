/**
 * ModelTuningTab -- Model Experimentation Studio.
 *
 * Pipeline-aligned layout:
 *   Clustering → Backtest & Tune → Champion → Forecast → Config
 *
 * The "Backtest & Tune" stage shows a 12-model grid with per-model
 * experiment history, feature lab, and cluster EDA nested underneath.
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
  Settings2,
  Workflow,
  TrendingUp,
  Layers,
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
import { Badge } from "@/components/ui/badge";
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
import { PipelineConfigPanel } from "./model-tuning/PipelineConfigPanel";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** All 12 models in the pipeline roster */
const ALL_MODELS: { id: string; label: string; type: string; tunable: boolean; modelType?: ModelType }[] = [
  { id: "lgbm_cluster",       label: "LightGBM",       type: "tree",          tunable: true,  modelType: "lgbm" },
  { id: "catboost_cluster",   label: "CatBoost",       type: "tree",          tunable: true,  modelType: "catboost" },
  { id: "xgboost_cluster",    label: "XGBoost",        type: "tree",          tunable: true,  modelType: "xgboost" },
  { id: "chronos",            label: "Chronos T5",     type: "foundation",    tunable: false },
  { id: "chronos_bolt",       label: "Chronos Bolt",   type: "foundation",    tunable: false },
  { id: "chronos2",           label: "Chronos 2",      type: "foundation",    tunable: false },
  { id: "chronos2_enriched",  label: "Chronos 2E",     type: "foundation",    tunable: false },
  { id: "mstl",               label: "MSTL",           type: "statistical",   tunable: false },
  { id: "nbeats",             label: "N-BEATS",        type: "deep_learning", tunable: false },
  { id: "nhits",              label: "N-HiTS",         type: "deep_learning", tunable: false },
  { id: "seasonal_naive",     label: "Seasonal Naive", type: "statistical",   tunable: false },
  { id: "rolling_mean",       label: "Rolling Mean",   type: "statistical",   tunable: false },
];

const TYPE_COLORS: Record<string, string> = {
  tree: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
  foundation: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
  statistical: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
  deep_learning: "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300",
};

const MODEL_PREFIX: Record<ModelType, string> = {
  lgbm: "/model-tuning/lgbm",
  catboost: "/model-tuning/catboost",
  xgboost: "/model-tuning/xgboost",
};

// ---------------------------------------------------------------------------
// Pipeline stage tabs
// ---------------------------------------------------------------------------
type PipelineStage = "clustering" | "backtest" | "champion" | "forecast" | "config";

const STAGE_TABS: { key: PipelineStage; label: string; icon: typeof FlaskConical }[] = [
  { key: "clustering", label: "Clustering",       icon: Target },
  { key: "backtest",   label: "Backtest & Tune",  icon: FlaskConical },
  { key: "champion",   label: "Champion",         icon: Crown },
  { key: "forecast",   label: "Forecast",         icon: TrendingUp },
  { key: "config",     label: "Pipeline Config",  icon: Settings2 },
];

// Model detail sub-tabs (shown when a model is selected)
type ModelDetailTab = "experiments" | "feature-lab" | "cluster-eda";

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

// Fetch summary for all tunable models (for the model grid KPIs)
async function fetchModelSummary(model: ModelType): Promise<{ best: number | null; runs: number; active: number; promoted: number | null }> {
  try {
    const res = await fetch(`${MODEL_PREFIX[model]}/experiments?limit=100`, { cache: "no-cache" });
    if (!res.ok) return { best: null, runs: 0, active: 0, promoted: null };
    const data = await res.json();
    const exps: TuningRun[] = data.experiments ?? [];
    const completed = exps.filter(e => e.status === "completed" && e.accuracy_pct != null);
    const best = completed.reduce<number | null>((acc, e) => !acc || (e.accuracy_pct ?? 0) > acc ? (e.accuracy_pct ?? 0) : acc, null);
    const promoted = exps.find(e => e.is_promoted)?.accuracy_pct ?? null;
    return { best, runs: exps.length, active: exps.filter(e => e.status === "running").length, promoted };
  } catch { return { best: null, runs: 0, active: 0, promoted: null }; }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function ModelTuningTab() {
  useChartColors(); // ensure theme context
  const queryClient = useQueryClient();

  // ---- State ---------------------------------------------------------------
  const [stage, setStage] = useState<PipelineStage>("backtest");
  const [selectedModelId, setSelectedModelId] = useState<string>("lgbm_cluster");
  const [modelDetailTab, setModelDetailTab] = useState<ModelDetailTab>("experiments");
  const [execLag, setExecLag] = useState<number | undefined>(undefined);
  const [baselineId, setBaselineId] = useState<number | null>(null);
  const [candidateId, setCandidateId] = useState<number | null>(null);
  const [showBuilder, setShowBuilder] = useState(false);
  const [selectedRunForLogs, setSelectedRunForLogs] = useState<number | null>(null);
  const [selectedRunForPromote, setSelectedRunForPromote] = useState<TuningRun | null>(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [page, setPage] = useState(0);
  const [sortCol, setSortCol] = useState<string>("run_id");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const pageSize = 25;

  // Resolve selected model info
  const selectedModelInfo = ALL_MODELS.find(m => m.id === selectedModelId) ?? ALL_MODELS[0];
  const selectedModel: ModelType = selectedModelInfo.modelType ?? "lgbm";
  const isTunable = selectedModelInfo.tunable;

  // ---- Model grid summaries (for tunable models) ---------------------------
  const { data: lgbmSummary } = useQuery({
    queryKey: ["model-summary", "lgbm"], queryFn: () => fetchModelSummary("lgbm"), staleTime: STALE.TWO_MIN, enabled: stage === "backtest",
  });
  const { data: catboostSummary } = useQuery({
    queryKey: ["model-summary", "catboost"], queryFn: () => fetchModelSummary("catboost"), staleTime: STALE.TWO_MIN, enabled: stage === "backtest",
  });
  const { data: xgboostSummary } = useQuery({
    queryKey: ["model-summary", "xgboost"], queryFn: () => fetchModelSummary("xgboost"), staleTime: STALE.TWO_MIN, enabled: stage === "backtest",
  });
  const modelSummaries: Record<string, { best: number | null; runs: number; active: number; promoted: number | null }> = {
    lgbm_cluster: lgbmSummary ?? { best: null, runs: 0, active: 0, promoted: null },
    catboost_cluster: catboostSummary ?? { best: null, runs: 0, active: 0, promoted: null },
    xgboost_cluster: xgboostSummary ?? { best: null, runs: 0, active: 0, promoted: null },
  };

  // ---- Reset on model change -----------------------------------------------
  function handleModelSelect(modelId: string) {
    if (modelId === selectedModelId) return;
    setSelectedModelId(modelId);
    setModelDetailTab("experiments");
    setBaselineId(null);
    setCandidateId(null);
    setSelectedRunForPromote(null);
    setSelectedRunForLogs(null);
    setPage(0);
  }

  // ---- Data fetching (for selected tunable model) --------------------------
  const {
    data: runsPayload,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: ["model-tuning-runs", selectedModel, statusFilter, execLag],
    queryFn: () =>
      fetchExperiments(selectedModel, {
        status: statusFilter !== "all" ? statusFilter : undefined,
        exec_lag: execLag,
      }),
    staleTime: STALE.TWO_MIN,
    enabled: stage === "backtest" && isTunable,
  });

  const allRuns = runsPayload?.experiments ?? [];

  // ---- Sort ----------------------------------------------------------------
  const sortedRuns = useMemo(() => {
    const runs = [...allRuns];
    runs.sort((a, b) => {
      let aVal: unknown = (a as Record<string, unknown>)[sortCol];
      let bVal: unknown = (b as Record<string, unknown>)[sortCol];
      if (sortCol === "run_id" || sortCol === "accuracy_pct" || sortCol === "wape" || sortCol === "bias") {
        aVal = Number(aVal) || 0;
        bVal = Number(bVal) || 0;
        return sortDir === "asc" ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number);
      }
      const aStr = String(aVal ?? "");
      const bStr = String(bVal ?? "");
      return sortDir === "asc" ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
    });
    return runs;
  }, [allRuns, sortCol, sortDir]);

  const pagedRuns = sortedRuns.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(sortedRuns.length / pageSize);

  // ---- KPIs ----------------------------------------------------------------
  const kpis = useMemo(() => {
    const completed = allRuns.filter(r => r.status === "completed" && r.accuracy_pct != null);
    const running = allRuns.filter(r => r.status === "running");
    const best = completed.reduce<TuningRun | null>((acc, r) => !acc || (r.accuracy_pct ?? 0) > (acc.accuracy_pct ?? 0) ? r : acc, null);
    const promoted = allRuns.find(r => r.is_promoted);
    return {
      bestAccuracy: best?.accuracy_pct ?? null,
      productionAccuracy: promoted?.accuracy_pct ?? null,
      totalRuns: allRuns.length,
      activeRuns: running.length,
    };
  }, [allRuns]);

  // ---- Row click handler ---------------------------------------------------
  function handleRowClick(runId: number) {
    if (baselineId === null) { setBaselineId(runId); }
    else if (candidateId === null) { if (runId === baselineId) return; setCandidateId(runId); }
    else { setBaselineId(runId); setCandidateId(null); }
  }

  function clearSelection() { setBaselineId(null); setCandidateId(null); }

  // ---- Column sort handler -------------------------------------------------
  function handleSort(col: string) {
    if (sortCol === col) { setSortDir(d => d === "asc" ? "desc" : "asc"); }
    else { setSortCol(col); setSortDir("desc"); }
  }

  function SortIndicator({ col }: { col: string }) {
    if (sortCol !== col) return null;
    return <span className="ml-0.5 text-[10px]">{sortDir === "asc" ? "\u25B2" : "\u25BC"}</span>;
  }

  return (
    <div className="flex flex-col gap-5 p-4 md:p-6">
      {/* ---- Header ---- */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <Layers className="h-5 w-5 text-muted-foreground" />
            Model Experimentation Studio
          </h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            Run experiments, compare results, and promote to production.
          </p>
        </div>
      </div>

      {/* ---- Pipeline Stage Tabs ---- */}
      <div className="flex gap-1 border-b border-border pb-1">
        {STAGE_TABS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setStage(key)}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md transition-colors",
              stage === key
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-muted",
            )}
          >
            <Icon className="h-3.5 w-3.5" />
            {label}
          </button>
        ))}
      </div>

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* CLUSTERING stage                                                    */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {stage === "clustering" && <ClusterExperimentsPanel />}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* CHAMPION stage                                                      */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {stage === "champion" && <ChampionExperimentsPanel />}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* FORECAST stage — production forecast summary                        */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {stage === "forecast" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Production Forecast</CardTitle>
            <CardDescription>Production forecast settings are managed in Pipeline Config.</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Switch to the <button className="underline text-primary" onClick={() => setStage("config")}>Pipeline Config</button> tab
              to configure horizon (24 months), cold-start routing, confidence intervals, and model registry.
            </p>
          </CardContent>
        </Card>
      )}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* CONFIG stage                                                        */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {stage === "config" && <PipelineConfigPanel />}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* BACKTEST & TUNE stage — 12-model grid + selected model detail       */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {stage === "backtest" && (
        <>
          {/* ---- Model Grid ---- */}
          <div>
            <h3 className="text-sm font-medium text-muted-foreground mb-2">Select a model to view experiments</h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
              {ALL_MODELS.map((m) => {
                const summary = modelSummaries[m.id];
                const isSelected = selectedModelId === m.id;
                return (
                  <button
                    key={m.id}
                    onClick={() => handleModelSelect(m.id)}
                    className={cn(
                      "rounded-lg border p-3 text-left transition-all",
                      isSelected
                        ? "border-primary bg-primary/5 ring-1 ring-primary shadow-sm"
                        : "hover:bg-muted/50 hover:border-foreground/20",
                    )}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-semibold truncate">{m.label}</span>
                      {m.tunable && summary && summary.promoted != null && (
                        <Crown className="h-3 w-3 text-amber-500 shrink-0" />
                      )}
                    </div>
                    <Badge variant="outline" className={`text-[9px] px-1.5 py-0 ${TYPE_COLORS[m.type] ?? ""}`}>
                      {m.type.replace("_", " ")}
                    </Badge>
                    {m.tunable && summary ? (
                      <div className="mt-2 space-y-0.5">
                        <div className="text-xs tabular-nums">
                          {summary.best != null ? (
                            <span className="font-semibold">{summary.best.toFixed(1)}%</span>
                          ) : (
                            <span className="text-muted-foreground">--</span>
                          )}
                          <span className="text-muted-foreground ml-1 text-[10px]">best</span>
                        </div>
                        <div className="text-[10px] text-muted-foreground">
                          {summary.runs} run{summary.runs !== 1 ? "s" : ""}
                          {summary.active > 0 && (
                            <span className="text-amber-500 ml-1">({summary.active} active)</span>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className="mt-2 text-[10px] text-muted-foreground">
                        {m.tunable ? "No runs yet" : "Zero-shot / statistical"}
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* ---- Selected Model Detail ---- */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CardTitle className="text-base">{selectedModelInfo.label}</CardTitle>
                  <Badge variant="outline" className={`text-[10px] ${TYPE_COLORS[selectedModelInfo.type] ?? ""}`}>
                    {selectedModelInfo.type.replace("_", " ")}
                  </Badge>
                </div>
                {isTunable && (
                  <div className="flex gap-1 rounded-lg border border-border bg-muted/30 p-0.5">
                    {([
                      { key: "experiments" as const, label: "Experiments", icon: FlaskConical },
                      { key: "feature-lab" as const, label: "Feature Lab", icon: Microscope },
                      { key: "cluster-eda" as const, label: "Cluster EDA", icon: BarChart3 },
                    ]).map(({ key, label, icon: Icon }) => (
                      <button
                        key={key}
                        onClick={() => setModelDetailTab(key)}
                        className={cn(
                          "flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-md transition-colors",
                          modelDetailTab === key
                            ? "bg-background text-foreground shadow-sm"
                            : "text-muted-foreground hover:text-foreground",
                        )}
                      >
                        <Icon className="h-3 w-3" />
                        {label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {/* Feature Lab / Cluster EDA sub-tabs */}
              {isTunable && modelDetailTab === "feature-lab" && <FeatureLabPanel />}
              {isTunable && modelDetailTab === "cluster-eda" && <ClusterEDAPanel />}

              {/* Non-tunable model info */}
              {!isTunable && (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  <p className="font-medium mb-1">{selectedModelInfo.label} is a {selectedModelInfo.type.replace("_", " ")} model</p>
                  <p>
                    {selectedModelInfo.type === "foundation"
                      ? "Foundation models are zero-shot — no hyperparameters to tune. Backtest results are available in the Champion tab."
                      : "Statistical models use fixed algorithms. Backtest results are available in the Champion tab."}
                  </p>
                </div>
              )}

              {/* Experiments sub-tab (tunable models only) */}
              {isTunable && modelDetailTab === "experiments" && (
                <>
                  {/* Lag filter + status filter + actions */}
                  <div className="flex items-center justify-between gap-4 mb-4">
                    <div className="flex items-center gap-2">
                      <LagFilterBar value={execLag} onChange={setExecLag} />
                      <Select
                        value={statusFilter}
                        onValueChange={(v) => { setStatusFilter(v as StatusFilter); setPage(0); }}
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
                    <Button size="sm" className="gap-1.5" onClick={() => setShowBuilder(true)}>
                      <Plus className="h-3.5 w-3.5" />
                      New Experiment
                    </Button>
                  </div>

                  {/* KPI cards */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                    <KpiCard label="Best Accuracy" value={kpis.bestAccuracy != null ? formatPct(kpis.bestAccuracy) : "--"} severity={kpis.bestAccuracy != null && kpis.bestAccuracy >= 80 ? "best" : "neutral"} size="md" icon={Target} />
                    <KpiCard label="Production Accuracy" value={kpis.productionAccuracy != null ? formatPct(kpis.productionAccuracy) : "Not promoted"} severity={kpis.productionAccuracy != null ? "best" : "neutral"} size="md" icon={Crown} />
                    <KpiCard label="Total Runs" value={formatInt(kpis.totalRuns)} size="md" />
                    <KpiCard label="Active Runs" value={formatInt(kpis.activeRuns)} severity={kpis.activeRuns > 0 ? "warning" : "neutral"} size="md" />
                  </div>

                  {/* Loading / Error */}
                  {isLoading ? (
                    <LoadingElement message={`Loading ${selectedModelInfo.label} experiments...`} size="md" />
                  ) : isError ? (
                    <div className="py-8 text-center text-sm text-destructive">
                      Failed to load experiments: {(error as Error).message}
                    </div>
                  ) : (
                    /* Table + Comparison */
                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
                      {/* Run history table */}
                      <Card>
                        <CardHeader className="pb-3">
                          <div className="flex items-center justify-between">
                            <div>
                              <CardTitle className="text-sm">Run History</CardTitle>
                              <CardDescription className="text-xs">Click two rows to compare</CardDescription>
                            </div>
                            {(baselineId !== null || candidateId !== null) && (
                              <Button variant="ghost" size="sm" onClick={clearSelection}>Clear</Button>
                            )}
                          </div>
                        </CardHeader>
                        <CardContent className="p-0">
                          {allRuns.length === 0 ? (
                            <div className="flex flex-col items-center justify-center py-16 text-center px-6">
                              <FlaskConical className="h-10 w-10 text-muted-foreground/30 mb-3" />
                              <p className="text-sm font-medium mb-1">No experiments yet for {selectedModelInfo.label}</p>
                              <p className="text-xs text-muted-foreground mb-4">Click "New Experiment" to launch your first tuning run.</p>
                              <Button size="sm" className="gap-1.5" onClick={() => setShowBuilder(true)}>
                                <Plus className="h-3.5 w-3.5" /> New Experiment
                              </Button>
                            </div>
                          ) : (
                            <div className="overflow-x-auto">
                              <Table>
                                <TableHeader>
                                  <TableRow>
                                    <TableHead className="w-14 cursor-pointer" onClick={() => handleSort("run_id")}>#<SortIndicator col="run_id" /></TableHead>
                                    <TableHead className="cursor-pointer" onClick={() => handleSort("run_label")}>Label<SortIndicator col="run_label" /></TableHead>
                                    <TableHead className="w-20 cursor-pointer" onClick={() => handleSort("status")}>Status<SortIndicator col="status" /></TableHead>
                                    <TableHead className="text-right w-20 cursor-pointer" onClick={() => handleSort("accuracy_pct")}>Acc%<SortIndicator col="accuracy_pct" /></TableHead>
                                    <TableHead className="text-right w-16 cursor-pointer" onClick={() => handleSort("wape")}>WAPE<SortIndicator col="wape" /></TableHead>
                                    <TableHead className="text-right w-16 cursor-pointer" onClick={() => handleSort("bias")}>Bias<SortIndicator col="bias" /></TableHead>
                                    <TableHead className="w-20">Duration</TableHead>
                                    <TableHead className="w-24 cursor-pointer" onClick={() => handleSort("started_at")}>Started<SortIndicator col="started_at" /></TableHead>
                                    <TableHead className="w-20 text-center">Actions</TableHead>
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
                                          isBaseline && "bg-blue-50 dark:bg-blue-950/30",
                                          isCandidate && "bg-emerald-50 dark:bg-emerald-950/30",
                                          !isSelected && "hover:bg-muted/50",
                                        )}
                                        onClick={() => handleRowClick(run.run_id)}
                                      >
                                        <TableCell className="font-mono text-xs">
                                          #{run.run_id}
                                          {isBaseline && <span className="ml-1 text-[10px] text-blue-600 dark:text-blue-400">(B)</span>}
                                          {isCandidate && <span className="ml-1 text-[10px] text-emerald-600 dark:text-emerald-400">(C)</span>}
                                        </TableCell>
                                        <TableCell className="text-sm max-w-[200px]">
                                          <div className="flex items-center gap-1">
                                            <span className="truncate">{run.run_label}</span>
                                            {run.is_promoted && <Crown className="shrink-0 h-3 w-3 text-amber-500" />}
                                            {(run as Record<string, unknown>).is_results_promoted === true && <Database className="shrink-0 h-3 w-3 text-blue-500" />}
                                            {(run as Record<string, unknown>).cluster_source === "experimental" && (
                                              <span className="shrink-0 px-1.5 py-0 text-[9px] font-medium rounded-full bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300">
                                                Exp #{String((run as Record<string, unknown>).cluster_experiment_id ?? "?")}
                                              </span>
                                            )}
                                          </div>
                                        </TableCell>
                                        <TableCell><StatusBadge status={run.status} /></TableCell>
                                        <TableCell className="text-right tabular-nums text-sm">{formatPct(run.accuracy_pct)}</TableCell>
                                        <TableCell className="text-right tabular-nums text-sm">{formatFixed(run.wape, 2)}</TableCell>
                                        <TableCell className="text-right tabular-nums text-sm">{formatFixed(run.bias, 2)}</TableCell>
                                        <TableCell className="text-xs text-muted-foreground tabular-nums">{formatDuration(run.started_at, run.completed_at ?? null)}</TableCell>
                                        <TableCell className="text-xs text-muted-foreground" title={run.started_at ? new Date(run.started_at).toLocaleString() : ""}>{timeAgo(run.started_at)}</TableCell>
                                        <TableCell className="text-center">
                                          <div className="flex items-center justify-center gap-0.5" onClick={(e) => e.stopPropagation()}>
                                            <button title="View Logs" className="rounded p-1 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors" onClick={() => setSelectedRunForLogs(run.run_id)}>
                                              <FileText className="h-3.5 w-3.5" />
                                            </button>
                                            {run.status === "completed" && (
                                              <button title="Promote" className="rounded p-1 hover:bg-muted text-muted-foreground hover:text-amber-600 dark:hover:text-amber-400 transition-colors" onClick={() => setSelectedRunForPromote(run)}>
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
                              {totalPages > 1 && (
                                <div className="flex items-center justify-between px-4 py-2 border-t border-border/40">
                                  <span className="text-xs text-muted-foreground">Page {page + 1} of {totalPages}</span>
                                  <div className="flex gap-1">
                                    <Button variant="ghost" size="sm" className="h-6 px-2 text-xs" disabled={page === 0} onClick={() => setPage(p => Math.max(0, p - 1))}>Prev</Button>
                                    <Button variant="ghost" size="sm" className="h-6 px-2 text-xs" disabled={page >= totalPages - 1} onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}>Next</Button>
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
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </>
      )}

      {/* ---- Experiment Builder Modal ---- */}
      {isTunable && (
        <ExperimentBuilder
          model={selectedModel}
          open={showBuilder}
          onClose={() => setShowBuilder(false)}
          onSubmitted={() => {
            setShowBuilder(false);
            queryClient.invalidateQueries({ queryKey: ["model-tuning-runs", selectedModel] });
            queryClient.invalidateQueries({ queryKey: ["model-summary", selectedModel] });
          }}
        />
      )}

      {/* ---- Log Viewer Slide-Over ---- */}
      {isTunable && (
        <LogViewer
          model={selectedModel}
          runId={selectedRunForLogs ?? 0}
          open={selectedRunForLogs !== null}
          onClose={() => setSelectedRunForLogs(null)}
        />
      )}

      {/* ---- Enhanced Promote Modal ---- */}
      {selectedRunForPromote && isTunable && (
        <EnhancedPromoteModal
          model={selectedModel}
          run={selectedRunForPromote}
          open={true}
          onClose={() => setSelectedRunForPromote(null)}
          onPromoted={() => {
            setSelectedRunForPromote(null);
            queryClient.invalidateQueries({ queryKey: ["model-tuning-runs", selectedModel] });
            queryClient.invalidateQueries({ queryKey: ["model-summary", selectedModel] });
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
                  <span className="text-sm font-semibold">AI Tuning Advisor</span>
                </div>
                <button onClick={() => setChatOpen(false)} className="rounded-md p-1 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors">
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
