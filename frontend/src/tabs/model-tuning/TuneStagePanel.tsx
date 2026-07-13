/**
 * Tune stage panel for the Model Experimentation Studio.
 * Shows tunable models, KPIs, run history table, and comparison panel.
 */
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Crown, FlaskConical, Plus, Target } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { KpiCard } from "@/components/KpiCard";
import { LoadingElement } from "@/components/LoadingElement";
import { cn } from "@/lib/utils";
import { formatPct, formatInt } from "@/lib/formatters";

import { STALE, type ModelType, type TuningRun } from "@/api/queries";
import { fetchModelExperiments, fetchModelSummary } from "@/api/queries/model-tuning";
import { modelTuningKeys } from "@/api/queries/unified-model-tuning";

import { ClusterEDAPanel } from "../lgbm-tuning/ClusterEDAPanel";
import { FeatureLabPanel } from "../lgbm-tuning/FeatureLabPanel";
import { LagFilterBar } from "./LagFilterBar";
import { EnhancedComparisonPanel } from "./EnhancedComparisonPanel";
import { RunHistoryTable } from "./RunHistoryTable";

import { MODEL_DETAIL_TABS, PAGE_SIZE, TYPE_COLORS } from "./_helpers";
import type { ModelDetailTab, ModelInfo, ModelSummaryCardData, StatusFilter } from "./_types";

interface Props {
  models: ModelInfo[];
  selectedModelId: string;
  selectedModelInfo: ModelInfo;
  selectedModel: ModelType;
  isTunable: boolean;
  modelDetailTab: ModelDetailTab;
  baselineId: number | null;
  candidateId: number | null;
  page: number;
  sortCol: string;
  sortDir: "asc" | "desc";
  statusFilter: StatusFilter;
  execLag: number | undefined;
  setExecLag: (lag: number | undefined) => void;
  onSelectModel: (modelId: string) => void;
  onSetDetailTab: (tab: ModelDetailTab) => void;
  onSetStatusFilter: (filter: StatusFilter) => void;
  onSelectRow: (runId: number) => void;
  onClearSelection: () => void;
  onToggleSort: (col: string) => void;
  onSetPage: (page: number) => void;
  onShowLogs: (runId: number) => void;
  onPromote: (run: TuningRun) => void;
  onOpenBuilder: () => void;
}

const ACTIVE_RUN_REFRESH_MS = 5_000;

function isActiveRun(status: string): boolean {
  return status === "queued" || status === "running";
}

export function TuneStagePanel({
  models,
  selectedModelId,
  selectedModelInfo,
  selectedModel,
  isTunable,
  modelDetailTab,
  baselineId,
  candidateId,
  page,
  sortCol,
  sortDir,
  statusFilter,
  execLag,
  setExecLag,
  onSelectModel,
  onSetDetailTab,
  onSetStatusFilter,
  onSelectRow,
  onClearSelection,
  onToggleSort,
  onSetPage,
  onShowLogs,
  onPromote,
  onOpenBuilder,
}: Props) {
  // Per-model summaries shown on the tunable-model grid
  const { data: lgbmSummary } = useQuery({
    queryKey: modelTuningKeys.summary("lgbm"),
    queryFn: () => fetchModelSummary("lgbm"),
    staleTime: STALE.TWO_MIN,
    refetchOnMount: "always",
    refetchInterval: (query) =>
      (query.state.data?.active ?? 0) > 0 ? ACTIVE_RUN_REFRESH_MS : false,
  });

  const emptySummary: ModelSummaryCardData = { best: null, runs: 0, active: 0, promoted: null };
  const modelSummaries: Record<string, ModelSummaryCardData> = {
    lgbm_cluster: lgbmSummary ?? emptySummary,
  };

  // Experiments for the currently selected tunable model
  const {
    data: runsPayload,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: modelTuningKeys.experiments(selectedModel, {
      status: statusFilter,
      exec_lag: execLag,
    }),
    queryFn: () =>
      fetchModelExperiments(selectedModel, {
        status: statusFilter !== "all" ? statusFilter : undefined,
        exec_lag: execLag,
      }),
    staleTime: STALE.TWO_MIN,
    refetchOnMount: "always",
    refetchInterval: (query) =>
      query.state.data?.experiments.some((run) => isActiveRun(run.status))
        ? ACTIVE_RUN_REFRESH_MS
        : false,
    enabled: isTunable,
  });

  const allRuns = useMemo(() => runsPayload?.experiments ?? [], [runsPayload]);

  // Sort runs
  const sortedRuns = useMemo(() => {
    const runs = [...allRuns];
    const readSortValue = (run: TuningRun): unknown => {
      switch (sortCol) {
        case "run_id":
        case "accuracy_pct":
        case "wape":
        case "bias":
        case "started_at":
        case "status":
        case "run_label":
          return run[sortCol];
        default:
          return run.run_id;
      }
    };
    runs.sort((a, b) => {
      let aVal = readSortValue(a);
      let bVal = readSortValue(b);
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
      const aStr = String(aVal ?? "");
      const bStr = String(bVal ?? "");
      return sortDir === "asc" ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
    });
    return runs;
  }, [allRuns, sortCol, sortDir]);

  const pagedRuns = sortedRuns.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.ceil(sortedRuns.length / PAGE_SIZE);

  // KPIs
  const kpis = useMemo(() => {
    const completed = allRuns.filter((r) => r.status === "completed" && r.accuracy_pct != null);
    const active = allRuns.filter((r) => isActiveRun(r.status));
    const best = completed.reduce<TuningRun | null>(
      (acc, r) => (!acc || (r.accuracy_pct ?? 0) > (acc.accuracy_pct ?? 0) ? r : acc),
      null
    );
    const promoted = allRuns.find((r) => r.is_promoted);
    return {
      bestAccuracy: best?.accuracy_pct ?? null,
      productionAccuracy: promoted?.accuracy_pct ?? null,
      totalRuns: allRuns.length,
      activeRuns: active.length,
    };
  }, [allRuns]);

  return (
    <>
      {/* ---- Tunable Model Grid ---- */}
      <div className="space-y-3">
        <div>
          <h3 className="text-sm font-medium text-muted-foreground mb-2">Tunable Models</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
            {models
              .filter((m) => m.tunable)
              .map((m) => {
                const summary = modelSummaries[m.id];
                const isSelected = selectedModelId === m.id;
                return (
                  <button
                    key={m.id}
                    onClick={() => onSelectModel(m.id)}
                    className={cn(
                      "rounded-lg border p-3 text-left transition-all",
                      isSelected
                        ? "border-primary bg-primary/5 ring-1 ring-primary shadow-sm"
                        : "hover:bg-muted/50 hover:border-foreground/20"
                    )}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-semibold truncate">{m.label}</span>
                      {summary && summary.promoted != null && (
                        <Crown className="h-3 w-3 text-amber-500 shrink-0" />
                      )}
                    </div>
                    <Badge
                      variant="outline"
                      className={`text-[9px] px-1.5 py-0 ${TYPE_COLORS[m.type] ?? ""}`}
                    >
                      {m.type.replace("_", " ")}
                    </Badge>
                    {summary ? (
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
                      <div className="mt-2 text-[10px] text-muted-foreground">No runs yet</div>
                    )}
                  </button>
                );
              })}
          </div>
        </div>
      </div>

      {/* ---- Selected Tunable Model Detail ---- */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CardTitle className="text-base">{selectedModelInfo.label}</CardTitle>
              <Badge
                variant="outline"
                className={`text-[10px] ${TYPE_COLORS[selectedModelInfo.type] ?? ""}`}
              >
                {selectedModelInfo.type.replace("_", " ")}
              </Badge>
            </div>
            {isTunable && (
              <div className="flex gap-1 rounded-lg border border-border bg-muted/30 p-0.5">
                {MODEL_DETAIL_TABS.map(({ key, label, icon: Icon }) => (
                  <button
                    key={key}
                    onClick={() => onSetDetailTab(key)}
                    className={cn(
                      "flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-md transition-colors",
                      modelDetailTab === key
                        ? "bg-background text-foreground shadow-sm"
                        : "text-muted-foreground hover:text-foreground"
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

          {/* Experiments sub-tab (tunable models only) */}
          {isTunable && modelDetailTab === "experiments" && (
            <>
              {/* Lag filter + status filter + actions */}
              <div className="flex items-center justify-between gap-4 mb-4">
                <div className="flex items-center gap-2">
                  <LagFilterBar value={execLag} onChange={setExecLag} />
                  <Select
                    value={statusFilter}
                    onValueChange={(v) => onSetStatusFilter(v as StatusFilter)}
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
                  onClick={onOpenBuilder}
                  disabled={kpis.activeRuns > 0}
                >
                  <Plus className="h-3.5 w-3.5" />
                  {kpis.activeRuns > 0 ? "Training…" : "New Experiment"}
                </Button>
              </div>

              {/* KPI cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                <KpiCard
                  label="Best Accuracy"
                  value={kpis.bestAccuracy != null ? formatPct(kpis.bestAccuracy) : "--"}
                  severity={
                    kpis.bestAccuracy != null && kpis.bestAccuracy >= 80 ? "best" : "neutral"
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
                  severity={kpis.productionAccuracy != null ? "best" : "neutral"}
                  size="md"
                  icon={Crown}
                />
                <KpiCard label="Total Runs" value={formatInt(kpis.totalRuns)} size="md" />
                <KpiCard
                  label="Active Runs"
                  value={formatInt(kpis.activeRuns)}
                  severity={kpis.activeRuns > 0 ? "warning" : "neutral"}
                  size="md"
                />
              </div>

              {/* Loading / Error */}
              {isLoading ? (
                <LoadingElement
                  message={`Loading ${selectedModelInfo.label} experiments...`}
                  size="md"
                />
              ) : isError ? (
                <div className="py-8 text-center text-sm text-destructive">
                  Failed to load experiments: {(error as Error).message}
                </div>
              ) : (
                /* Table + Comparison */
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
                  <RunHistoryTable
                    allRuns={allRuns}
                    pagedRuns={pagedRuns}
                    baselineId={baselineId}
                    candidateId={candidateId}
                    page={page}
                    totalPages={totalPages}
                    sortCol={sortCol}
                    sortDir={sortDir}
                    selectedModelLabel={selectedModelInfo.label}
                    activeRunsCount={kpis.activeRuns}
                    onSelectRow={onSelectRow}
                    onClearSelection={onClearSelection}
                    onToggleSort={onToggleSort}
                    onSetPage={onSetPage}
                    onShowLogs={onShowLogs}
                    onPromote={onPromote}
                    onOpenBuilder={onOpenBuilder}
                  />

                  {/* Comparison panel */}
                  <div>
                    {baselineId !== null && candidateId !== null ? (
                      <EnhancedComparisonPanel
                        model={selectedModel}
                        baselineId={baselineId}
                        candidateId={candidateId}
                        execLag={execLag}
                        onPromote={onPromote}
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
  );
}
