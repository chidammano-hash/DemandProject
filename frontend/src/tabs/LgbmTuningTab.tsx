import { useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { FlaskConical, MessageSquare, X, BarChart3, Microscope, Target, Crown } from "lucide-react";

import {
  lgbmTuningKeys,
  fetchTuningRuns,
  promoteRun,
  STALE,
  type TuningRun,
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
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { KpiCard } from "@/components/KpiCard";
import { LoadingElement } from "@/components/LoadingElement";

import { ComparisonPanel } from "./lgbm-tuning";
import { TuningChatPanel } from "./lgbm-tuning/TuningChatPanel";
import { ClusterEDAPanel } from "./lgbm-tuning/ClusterEDAPanel";
import { FeatureLabPanel } from "./lgbm-tuning/FeatureLabPanel";
import { AccuracyBudgetPanel } from "./lgbm-tuning/AccuracyBudgetPanel";
import { PromoteModal } from "./lgbm-tuning/PromoteModal";

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------
function StatusBadge({ status }: { status: TuningRun["status"] }) {
  const styles: Record<TuningRun["status"], string> = {
    completed: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
    running: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
    failed: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  };
  return (
    <Badge className={cn("text-[10px] font-medium px-2 py-0.5", styles[status])}>
      {status}
    </Badge>
  );
}

// ---------------------------------------------------------------------------
// Verdict badge (for the KPI card)
// ---------------------------------------------------------------------------
function VerdictKpiBadge({ verdict }: { verdict: string | null }) {
  if (!verdict) {
    return (
      <Badge className="bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300 text-xs px-2 py-0.5">
        N/A
      </Badge>
    );
  }
  const v = verdict.toUpperCase();
  const style =
    v === "IMPROVED"
      ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
      : v === "DEGRADED"
        ? "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300"
        : "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300";

  return (
    <Badge className={cn("text-xs px-2 py-0.5", style)}>
      {v}
    </Badge>
  );
}

// ---------------------------------------------------------------------------
// Sub-tab definitions
// ---------------------------------------------------------------------------
type SubTab = "runs" | "cluster-eda" | "feature-lab" | "accuracy-budget";

const SUB_TAB_LABELS: Record<SubTab, { label: string; icon: typeof FlaskConical }> = {
  runs: { label: "Runs", icon: FlaskConical },
  "cluster-eda": { label: "Cluster EDA", icon: BarChart3 },
  "feature-lab": { label: "Feature Lab", icon: Microscope },
  "accuracy-budget": { label: "Accuracy Budget", icon: Target },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function LgbmTuningTab() {
  useChartColors(); // ensure theme context is active
  const queryClient = useQueryClient();

  // ---- Selection state ----------------------------------------------------
  const [selectedBaseline, setSelectedBaseline] = useState<number | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<number | null>(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [activeSubTab, setActiveSubTab] = useState<SubTab>("runs");
  const [promoteTarget, setPromoteTarget] = useState<TuningRun | null>(null);

  // ---- Promote mutation ---------------------------------------------------
  const promoteMutation = useMutation({
    mutationFn: (runId: number) => promoteRun(runId),
    onSuccess: () => {
      setPromoteTarget(null);
      queryClient.invalidateQueries({ queryKey: lgbmTuningKeys.runs() });
      queryClient.invalidateQueries({ queryKey: lgbmTuningKeys.promoted() });
    },
  });

  // ---- Fetch run list -----------------------------------------------------
  const {
    data: runsPayload,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: lgbmTuningKeys.runs(),
    queryFn: () => fetchTuningRuns(),
    staleTime: STALE.TWO_MIN,
  });

  const runs = runsPayload?.runs ?? [];

  // ---- Derived KPIs -------------------------------------------------------
  const kpis = useMemo(() => {
    if (runs.length === 0) {
      return { latestAccuracy: null, bestAccuracy: null, totalRuns: 0, latestVerdict: null };
    }

    const completed = runs.filter(
      (r) => r.status === "completed" && r.accuracy_pct != null,
    );
    const latest = completed[0] ?? null;
    const best = completed.reduce<TuningRun | null>(
      (acc, r) => (!acc || (r.accuracy_pct ?? 0) > (acc.accuracy_pct ?? 0) ? r : acc),
      null,
    );

    return {
      latestAccuracy: latest?.accuracy_pct ?? null,
      bestAccuracy: best?.accuracy_pct ?? null,
      totalRuns: runs.length,
      latestVerdict: latest
        ? best && latest.run_id === best.run_id
          ? "IMPROVED"
          : "NEUTRAL"
        : null,
    };
  }, [runs]);

  // ---- Row click handler --------------------------------------------------
  function handleRowClick(runId: number) {
    if (selectedBaseline === null) {
      setSelectedBaseline(runId);
    } else if (selectedCandidate === null) {
      if (runId === selectedBaseline) return; // ignore same row
      setSelectedCandidate(runId);
    } else {
      // Reset and start new selection
      setSelectedBaseline(runId);
      setSelectedCandidate(null);
    }
  }

  function clearSelection() {
    setSelectedBaseline(null);
    setSelectedCandidate(null);
  }

  // ---- Loading / error ----------------------------------------------------
  if (isLoading) {
    return (
      <div className="p-6">
        <LoadingElement message="Loading tuning runs..." size="md" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="p-6 text-center text-sm text-destructive">
        Failed to load tuning runs: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5 p-4 md:p-6">
      {/* Header */}
      <div>
        <h2 className="flex items-center gap-2 text-lg font-semibold">
          <FlaskConical className="h-5 w-5 text-muted-foreground" />
          LGBM Tuning
        </h2>
        <p className="text-sm text-muted-foreground mt-0.5">
          Review hyperparameter tuning runs for LightGBM models. Select two runs
          to compare accuracy, WAPE, and bias across timeframes.
        </p>
      </div>

      {/* Sub-tab navigation */}
      <div className="flex gap-1 border-b border-border pb-1">
        {(Object.keys(SUB_TAB_LABELS) as SubTab[]).map((key) => {
          const { label, icon: Icon } = SUB_TAB_LABELS[key];
          return (
            <button
              key={key}
              onClick={() => setActiveSubTab(key)}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md transition-colors",
                activeSubTab === key
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

      {/* Sub-tab content */}
      {activeSubTab === "cluster-eda" && <ClusterEDAPanel />}
      {activeSubTab === "feature-lab" && <FeatureLabPanel />}
      {activeSubTab === "accuracy-budget" && <AccuracyBudgetPanel />}

      {/* Runs panel (default) */}
      {activeSubTab === "runs" && (
      <>

      {/* KPI cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KpiCard
          label="Latest Accuracy"
          value={kpis.latestAccuracy != null ? formatPct(kpis.latestAccuracy) : "--"}
          severity={kpis.latestAccuracy != null && kpis.latestAccuracy >= 80 ? "best" : "neutral"}
          size="md"
        />
        <KpiCard
          label="Best Accuracy"
          value={kpis.bestAccuracy != null ? formatPct(kpis.bestAccuracy) : "--"}
          severity="best"
          size="md"
        />
        <KpiCard
          label="Total Runs"
          value={formatInt(kpis.totalRuns)}
          size="md"
        />
        <div className="rounded-lg border border-border/60 bg-card px-4 py-3 flex flex-col justify-center">
          <p className="text-xs text-muted-foreground mb-1">Latest Verdict</p>
          <VerdictKpiBadge verdict={kpis.latestVerdict} />
        </div>
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
              {(selectedBaseline !== null || selectedCandidate !== null) && (
                <Button variant="ghost" size="sm" onClick={clearSelection}>
                  Clear
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="p-0">
            {runs.length === 0 ? (
              <p className="py-12 text-center text-sm text-muted-foreground">
                No tuning runs found.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-16">Run</TableHead>
                      <TableHead>Label</TableHead>
                      <TableHead className="w-24">Status</TableHead>
                      <TableHead className="text-right w-20">Accuracy</TableHead>
                      <TableHead className="text-right w-16">WAPE</TableHead>
                      <TableHead className="text-right w-16">Bias</TableHead>
                      <TableHead className="w-40">Started</TableHead>
                      <TableHead className="w-24 text-center">Production</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {runs.map((run) => {
                      const isBaseline = selectedBaseline === run.run_id;
                      const isCandidate = selectedCandidate === run.run_id;
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
                          <TableCell className="text-sm truncate max-w-[180px]">
                            {run.run_label}
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
                          <TableCell className="text-xs text-muted-foreground">
                            {run.started_at
                              ? new Date(run.started_at).toLocaleString(undefined, {
                                  month: "short",
                                  day: "numeric",
                                  hour: "2-digit",
                                  minute: "2-digit",
                                })
                              : "--"}
                          </TableCell>
                          <TableCell className="text-center">
                            {run.is_promoted ? (
                              <Badge className="bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300 text-[10px] gap-1">
                                <Crown className="h-3 w-3" />
                                Production
                              </Badge>
                            ) : run.status === "completed" ? (
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 px-2 text-[10px]"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setPromoteTarget(run);
                                }}
                              >
                                Promote
                              </Button>
                            ) : null}
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

        {/* Comparison panel */}
        <div>
          {selectedBaseline !== null && selectedCandidate !== null ? (
            <ComparisonPanel
              baselineId={selectedBaseline}
              candidateId={selectedCandidate}
            />
          ) : (
            <Card className="h-full">
              <CardContent className="flex flex-col items-center justify-center py-20 text-center">
                <FlaskConical className="h-10 w-10 text-muted-foreground/40 mb-3" />
                <p className="text-sm text-muted-foreground">
                  {selectedBaseline !== null
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

      {/* Promote modal */}
      {promoteTarget && (
        <PromoteModal
          run={promoteTarget}
          onConfirm={() => promoteMutation.mutate(promoteTarget.run_id)}
          onCancel={() => {
            setPromoteTarget(null);
            promoteMutation.reset();
          }}
          isPending={promoteMutation.isPending}
        />
      )}

      {/* Floating AI Tuning Advisor — portalled to body so it escapes overflow-y-auto */}
      {createPortal(
        <div className="fixed top-4 right-6 z-50 flex flex-col items-end gap-3">
          {/* FAB icon button */}
          {!chatOpen && (
            <button
              onClick={() => setChatOpen(true)}
              title="AI Tuning Advisor"
              className="h-10 w-10 rounded-full bg-primary text-primary-foreground shadow-lg hover:bg-primary/90 transition-all hover:scale-105 flex items-center justify-center"
            >
              <MessageSquare className="h-4.5 w-4.5" />
            </button>
          )}

          {/* Slide-down chat panel */}
          {chatOpen && (
            <div className="w-[420px] max-h-[80vh] animate-in slide-in-from-top-4 fade-in duration-200 rounded-2xl border border-border bg-card shadow-2xl overflow-hidden flex flex-col">
              <div className="flex items-center justify-between px-4 py-2.5 border-b border-border bg-muted/30">
                <div className="flex items-center gap-2">
                  <MessageSquare className="h-4 w-4 text-primary" />
                  <span className="text-sm font-semibold">AI Tuning Advisor</span>
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
