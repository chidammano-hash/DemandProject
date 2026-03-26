/**
 * ClusterExperimentsPanel -- Main panel for the Cluster Experimentation Studio.
 *
 * Two-column grid layout:
 * - Left: KPI cards, status filter, experiment table with row selection
 * - Right: ClusterComparisonPanel when 2 rows selected, guidance otherwise
 *
 * Follows the same pattern as ModelTuningTab experiment list.
 */
import { useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  FlaskConical,
  Plus,
  Crown,
  Database,
  Target,
  Layers,
  Activity,
  Copy,
  Trash2,
} from "lucide-react";
import {
  clusterExperimentKeys,
  CLUSTER_EXP_STALE,
  fetchClusterExperiments,
  deleteClusterExperiment,
  type ClusterExperiment,
  type ClusterExperimentStatus,
  type FeatureParams,
  type ModelParams,
  type LabelParams,
} from "@/api/queries";
import { STALE } from "@/api/queries";
import { cn } from "@/lib/utils";
import { formatFixed, formatClusterLabel } from "@/lib/formatters";

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
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { KpiCard } from "@/components/KpiCard";
import { LoadingElement } from "@/components/LoadingElement";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";

import { ClusterExperimentBuilder } from "./ClusterExperimentBuilder";
import { ClusterExperimentDetail } from "./ClusterExperimentDetail";
import { ClusterComparisonPanel } from "./ClusterComparisonPanel";
import { ClusterPromoteModal } from "./ClusterPromoteModal";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

type StatusFilter = "all" | "running" | "completed" | "failed" | "queued";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: ClusterExperimentStatus }) {
  const styles: Record<ClusterExperimentStatus, string> = {
    completed:
      "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
    running:
      "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
    failed:
      "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
    queued:
      "bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300",
    cancelled:
      "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  };
  return (
    <Badge
      className={cn(
        "text-[10px] font-medium px-2 py-0.5",
        styles[status] ?? "bg-gray-100 text-gray-700",
      )}
      aria-label={`Status: ${status}`}
    >
      {status}
    </Badge>
  );
}

function formatDuration(
  startedAt: string | null,
  completedAt: string | null,
  runtimeSeconds: number | null,
): string {
  if (runtimeSeconds != null) {
    const min = Math.floor(runtimeSeconds / 60);
    const sec = Math.round(runtimeSeconds % 60);
    if (min > 0) return `${min}m ${sec}s`;
    return `${sec}s`;
  }
  if (!startedAt) return "--";
  const start = new Date(startedAt).getTime();
  const end = completedAt ? new Date(completedAt).getTime() : Date.now();
  const elapsed = Math.max(0, end - start);
  const totalSec = Math.floor(elapsed / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  if (min > 0) return `${min}m ${sec}s`;
  return `${sec}s`;
}

function timeAgo(dateStr: string | null): string {
  if (!dateStr) return "--";
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

const CLUSTER_COLORS = [
  "#2563EB", "#f59e0b", "#10b981", "#ef4444", "#0891B2",
  "#06b6d4", "#ec4899", "#14b8a6", "#f97316", "#84cc16",
  "#8b5cf6", "#d946ef", "#64748b", "#a3e635",
];

function ClusterDistBar({
  sizes,
  profiles,
}: {
  sizes: Record<string, number> | null;
  profiles: Array<{ label: string; count: number; pct_of_total: number }> | null;
}) {
  const entries = profiles
    ? profiles.map((p) => ({ label: p.label, count: p.count, pct: p.pct_of_total }))
    : sizes
      ? (() => {
          const total = Object.values(sizes).reduce((a, b) => a + b, 0);
          return Object.entries(sizes)
            .sort(([, a], [, b]) => b - a)
            .map(([label, count]) => ({
              label,
              count,
              pct: total > 0 ? (count / total) * 100 : 0,
            }));
        })()
      : null;

  if (!entries || entries.length === 0) return <span className="text-muted-foreground text-xs">--</span>;

  return (
    <div className="flex flex-col gap-0.5 w-full min-w-[120px]">
      <div className="flex h-3 w-full rounded-sm overflow-hidden" title={entries.map(e => `${formatClusterLabel(e.label)}: ${(e.count ?? 0).toLocaleString()} (${(e.pct ?? 0).toFixed(0)}%)`).join("\n")}>
        {entries.map((e, i) => (
          <div
            key={e.label}
            className="h-full transition-all"
            style={{
              width: `${Math.max(e.pct, 1)}%`,
              backgroundColor: CLUSTER_COLORS[i % CLUSTER_COLORS.length],
            }}
          />
        ))}
      </div>
      <span className="text-[9px] text-muted-foreground truncate">
        {entries.length} clusters
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ClusterExperimentsPanel() {
  const queryClient = useQueryClient();

  // ---- State ---------------------------------------------------------------
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [baselineId, setBaselineId] = useState<number | null>(null);
  const [candidateId, setCandidateId] = useState<number | null>(null);
  const [showBuilder, setShowBuilder] = useState(false);
  const [cloneSource, setCloneSource] = useState<ClusterExperiment | null>(null);
  const [promoteTarget, setPromoteTarget] = useState<ClusterExperiment | null>(null);
  const [sortCol, setSortCol] = useState<string>("experiment_id");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  // ---- Data ----------------------------------------------------------------
  const {
    data: payload,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: clusterExperimentKeys.experiments(
      statusFilter !== "all" ? { status: statusFilter } : undefined,
    ),
    queryFn: () =>
      fetchClusterExperiments(
        statusFilter !== "all" ? { status: statusFilter } : undefined,
      ),
    staleTime: CLUSTER_EXP_STALE.EXPERIMENTS,
  });

  const allExperiments = payload?.experiments ?? [];

  // ---- Sort ----------------------------------------------------------------
  const sortedExperiments = useMemo(() => {
    const items = [...allExperiments];
    items.sort((a, b) => {
      let aVal: unknown = (a as Record<string, unknown>)[sortCol];
      let bVal: unknown = (b as Record<string, unknown>)[sortCol];

      if (
        sortCol === "experiment_id" ||
        sortCol === "optimal_k" ||
        sortCol === "silhouette_score" ||
        sortCol === "total_dfus"
      ) {
        aVal = Number(aVal) || 0;
        bVal = Number(bVal) || 0;
        return sortDir === "asc"
          ? (aVal as number) - (bVal as number)
          : (bVal as number) - (aVal as number);
      }

      const aStr = String(aVal ?? "");
      const bStr = String(bVal ?? "");
      return sortDir === "asc"
        ? aStr.localeCompare(bStr)
        : bStr.localeCompare(aStr);
    });
    return items;
  }, [allExperiments, sortCol, sortDir]);

  // ---- KPIs ----------------------------------------------------------------
  const kpis = useMemo(() => {
    const completed = allExperiments.filter(
      (e) => e.status === "completed" && e.silhouette_score != null,
    );
    const active = allExperiments.filter(
      (e) => e.status === "running" || e.status === "queued",
    );
    const promoted = allExperiments.find((e) => e.is_promoted);

    const best = completed.reduce<ClusterExperiment | null>(
      (acc, e) =>
        !acc || (e.silhouette_score ?? 0) > (acc.silhouette_score ?? 0)
          ? e
          : acc,
      null,
    );

    return {
      bestSilhouette: best?.silhouette_score ?? null,
      productionK: promoted?.optimal_k ?? null,
      totalExperiments: allExperiments.length,
      activeRuns: active.length,
    };
  }, [allExperiments]);

  // ---- Row selection -------------------------------------------------------
  function handleRowClick(id: number) {
    if (baselineId === null) {
      setBaselineId(id);
    } else if (candidateId === null) {
      if (id === baselineId) return;
      setCandidateId(id);
    } else {
      setBaselineId(id);
      setCandidateId(null);
    }
  }

  function clearSelection() {
    setBaselineId(null);
    setCandidateId(null);
  }

  // ---- Sort handler --------------------------------------------------------
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

  // ---- Clone handler -------------------------------------------------------
  function handleClone(exp: ClusterExperiment) {
    setCloneSource(exp);
    setShowBuilder(true);
  }

  // ---- Delete handler ------------------------------------------------------
  async function handleDelete(exp: ClusterExperiment) {
    if (exp.status === "running" || exp.status === "queued") return;
    try {
      await deleteClusterExperiment(exp.experiment_id);
      queryClient.invalidateQueries({ queryKey: clusterExperimentKeys.all });
    } catch {
      // Silently ignore; user can retry
    }
  }

  // ---- Loading / error -----------------------------------------------------
  if (isLoading) {
    return (
      <div className="p-6">
        <LoadingElement message="Loading cluster experiments..." size="md" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="p-6 text-center text-sm text-destructive">
        Failed to load experiments: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* KPI cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KpiCard
          label="Best Silhouette"
          value={
            kpis.bestSilhouette != null
              ? kpis.bestSilhouette.toFixed(4)
              : "--"
          }
          severity={kpis.bestSilhouette != null ? "best" : "neutral"}
          size="md"
          icon={Target}
        />
        <KpiCard
          label="Production K"
          value={kpis.productionK != null ? String(kpis.productionK) : "Not set"}
          severity={kpis.productionK != null ? "best" : "neutral"}
          size="md"
          icon={Crown}
        />
        <KpiCard
          label="Total Experiments"
          value={String(kpis.totalExperiments)}
          size="md"
          icon={Layers}
        />
        <KpiCard
          label="Active"
          value={String(kpis.activeRuns)}
          severity={kpis.activeRuns > 0 ? "warning" : "neutral"}
          size="md"
          icon={Activity}
        />
      </div>

      {/* Toolbar */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Select
            value={statusFilter}
            onValueChange={(v) => setStatusFilter(v as StatusFilter)}
          >
            <SelectTrigger className="h-8 text-xs w-[120px]">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="queued">Queued</SelectItem>
              <SelectItem value="running">Running</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
            </SelectContent>
          </Select>
          <span className="text-xs text-muted-foreground">
            {sortedExperiments.length} experiment
            {sortedExperiments.length !== 1 ? "s" : ""}
          </span>
        </div>

        <Button
          size="sm"
          className="gap-1.5"
          onClick={() => {
            setCloneSource(null);
            setShowBuilder(true);
          }}
        >
          <Plus className="h-3.5 w-3.5" />
          New Experiment
        </Button>
      </div>

      {/* Main grid: table + comparison */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        {/* Experiment table */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">Experiments</CardTitle>
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
            {allExperiments.length === 0 ? (
              /* Empty state */
              <div className="flex flex-col items-center justify-center py-16 text-center px-6">
                <FlaskConical className="h-10 w-10 text-muted-foreground/30 mb-3" />
                <p className="text-sm font-medium text-foreground mb-1">
                  No cluster experiments yet
                </p>
                <p className="text-xs text-muted-foreground mb-4 max-w-sm">
                  Cluster experiments let you test different SKU segmentation
                  configurations before committing to production. Each experiment
                  runs the full clustering pipeline with your chosen parameters.
                </p>
                <Button
                  size="sm"
                  className="gap-1.5"
                  onClick={() => {
                    setCloneSource(null);
                    setShowBuilder(true);
                  }}
                >
                  <Plus className="h-3.5 w-3.5" />
                  Create First Experiment
                </Button>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead
                        className="w-14 cursor-pointer"
                        onClick={() => handleSort("experiment_id")}
                      >
                        #<SortIndicator col="experiment_id" />
                      </TableHead>
                      <TableHead
                        className="cursor-pointer"
                        onClick={() => handleSort("label")}
                      >
                        Label
                        <SortIndicator col="label" />
                      </TableHead>
                      <TableHead
                        className="w-20 cursor-pointer"
                        onClick={() => handleSort("status")}
                      >
                        Status
                        <SortIndicator col="status" />
                      </TableHead>
                      <TableHead
                        className="text-right w-14 cursor-pointer"
                        onClick={() => handleSort("optimal_k")}
                      >
                        K
                        <SortIndicator col="optimal_k" />
                      </TableHead>
                      <TableHead
                        className="text-right w-20 cursor-pointer"
                        onClick={() => handleSort("silhouette_score")}
                      >
                        Silhouette
                        <SortIndicator col="silhouette_score" />
                      </TableHead>
                      <TableHead className="text-right w-16">SKUs</TableHead>
                      <TableHead className="w-[140px]">Clusters</TableHead>
                      <TableHead className="w-20">Duration</TableHead>
                      <TableHead
                        className="w-24 cursor-pointer"
                        onClick={() => handleSort("created_at")}
                      >
                        Created
                        <SortIndicator col="created_at" />
                      </TableHead>
                      <TableHead className="w-20 text-center">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {sortedExperiments.map((exp) => {
                      const isBaseline = baselineId === exp.experiment_id;
                      const isCandidate = candidateId === exp.experiment_id;
                      const isSelected = isBaseline || isCandidate;

                      return (
                        <TableRow
                          key={exp.experiment_id}
                          className={cn(
                            "cursor-pointer transition-colors",
                            isBaseline && "bg-blue-50 dark:bg-blue-950/30",
                            isCandidate && "bg-emerald-50 dark:bg-emerald-950/30",
                            !isSelected && "hover:bg-muted/50",
                          )}
                          onClick={() => handleRowClick(exp.experiment_id)}
                        >
                          <TableCell className="font-mono text-xs">
                            #{exp.experiment_id}
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
                              <span className="truncate">{exp.label}</span>
                              {exp.is_promoted && (
                                <Crown
                                  className="shrink-0 h-3 w-3 text-amber-500"
                                  title="Promoted to production"
                                />
                              )}
                              {exp.is_promoted && (
                                <Database
                                  className="shrink-0 h-3 w-3 text-blue-500"
                                  title="Assignments loaded to production"
                                />
                              )}
                            </div>
                          </TableCell>
                          <TableCell>
                            <StatusBadge status={exp.status} />
                          </TableCell>
                          <TableCell className="text-right tabular-nums text-sm">
                            {exp.optimal_k ?? "--"}
                          </TableCell>
                          <TableCell className="text-right tabular-nums text-sm">
                            {exp.silhouette_score != null
                              ? formatFixed(exp.silhouette_score, 4)
                              : "--"}
                          </TableCell>
                          <TableCell className="text-right tabular-nums text-sm text-muted-foreground">
                            {exp.total_dfus?.toLocaleString() ?? "--"}
                          </TableCell>
                          <TableCell>
                            {exp.status === "completed" ? (
                              <ClusterDistBar
                                sizes={exp.cluster_sizes}
                                profiles={exp.profiles}
                              />
                            ) : (
                              <span className="text-xs text-muted-foreground">--</span>
                            )}
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground tabular-nums">
                            {formatDuration(
                              exp.started_at,
                              exp.completed_at,
                              exp.runtime_seconds,
                            )}
                          </TableCell>
                          <TableCell
                            className="text-xs text-muted-foreground"
                            title={
                              exp.created_at
                                ? new Date(exp.created_at).toLocaleString()
                                : ""
                            }
                          >
                            {timeAgo(exp.created_at)}
                          </TableCell>
                          <TableCell className="text-center">
                            <div
                              className="flex items-center justify-center gap-0.5"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <button
                                title="Clone"
                                className="rounded p-1 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
                                onClick={() => handleClone(exp)}
                              >
                                <Copy className="h-3.5 w-3.5" />
                              </button>
                              {exp.status === "completed" && (
                                <button
                                  title="Promote"
                                  className="rounded p-1 hover:bg-muted text-muted-foreground hover:text-amber-600 dark:hover:text-amber-400 transition-colors"
                                  onClick={() => setPromoteTarget(exp)}
                                >
                                  <Crown className="h-3.5 w-3.5" />
                                </button>
                              )}
                              {exp.status !== "running" &&
                                exp.status !== "queued" && (
                                  <button
                                    title="Delete"
                                    className="rounded p-1 hover:bg-muted text-muted-foreground hover:text-red-600 dark:hover:text-red-400 transition-colors"
                                    onClick={() => handleDelete(exp)}
                                  >
                                    <Trash2 className="h-3.5 w-3.5" />
                                  </button>
                                )}
                            </div>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right panel: detail / comparison / guidance */}
        <div>
          {baselineId !== null && candidateId !== null ? (
            <ClusterComparisonPanel
              baselineId={baselineId}
              candidateId={candidateId}
              onPromote={(exp) => setPromoteTarget(exp)}
            />
          ) : baselineId !== null ? (
            (() => {
              const selected = allExperiments.find(
                (e) => e.experiment_id === baselineId,
              );
              if (!selected) return null;
              return (
                <div className="space-y-3">
                  <ClusterExperimentDetail
                    experiment={selected}
                    onPromote={
                      selected.status === "completed" && !selected.is_promoted
                        ? () => setPromoteTarget(selected)
                        : undefined
                    }
                  />
                  <p className="text-xs text-muted-foreground text-center">
                    Click another row to compare two experiments side by side.
                  </p>
                </div>
              );
            })()
          ) : (
            <Card className="h-full">
              <CardContent className="flex flex-col items-center justify-center py-20 text-center">
                <FlaskConical className="h-10 w-10 text-muted-foreground/40 mb-3" />
                <p className="text-sm text-muted-foreground">
                  Click a row to view experiment charts and profiles.
                  <br />
                  Click two rows to compare experiments side by side.
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Builder Modal */}
      <ClusterExperimentBuilder
        open={showBuilder}
        onClose={() => {
          setShowBuilder(false);
          setCloneSource(null);
        }}
        onSubmitted={() => {
          setShowBuilder(false);
          setCloneSource(null);
          queryClient.invalidateQueries({
            queryKey: clusterExperimentKeys.all,
          });
        }}
        cloneFrom={
          cloneSource
            ? {
                featureParams: cloneSource.feature_params ?? {
                  time_window_months: 24,
                  min_months_history: 1,
                },
                modelParams: cloneSource.model_params ?? {
                  k_range: [3, 12],
                  min_cluster_size_pct: 2.0,
                  use_pca: false,
                  pca_components: null,
                },
                labelParams: cloneSource.label_params ?? {
                  volume_high: 0.75,
                  volume_low: 0.25,
                  cv_steady: 0.3,
                  cv_volatile: 0.8,
                  seasonality_threshold: 0.5,
                  zero_demand_threshold: 0.2,
                },
                label: cloneSource.label,
                notes: cloneSource.notes ?? undefined,
              }
            : undefined
        }
      />

      {/* Promote Modal */}
      {promoteTarget && (
        <ClusterPromoteModal
          experiment={promoteTarget}
          open={true}
          onClose={() => setPromoteTarget(null)}
          onPromoted={() => {
            setPromoteTarget(null);
            queryClient.invalidateQueries({
              queryKey: clusterExperimentKeys.all,
            });
          }}
        />
      )}
    </div>
  );
}
