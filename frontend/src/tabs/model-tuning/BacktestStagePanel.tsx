/**
 * Backtest stage panel for the Model Experimentation Studio.
 *
 * Models are grouped by family (Tree / Foundation / Statistical / Deep Learning)
 * in compact tables. Run a backtest per model or per group; results AUTO-LOAD
 * into the DB on completion (server-side), so there is no separate "Load"
 * action — the Status cell flips to "Loaded" once the run + auto-load finish.
 */
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Loader2, Play } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { toast } from "@/components/Toaster";
import { formatApiError } from "@/lib/formatApiError";
import {
  backtestMgmtKeys,
  fetchBacktestSummary,
  submitBacktestRun,
  type BacktestModelSummary,
} from "@/api/queries/backtest-management";
import {
  fetchLagLeaderboard,
  lagLeaderboardKeys,
  type LagLeaderboardPayload,
} from "@/api/queries/accuracy";

import { BacktestDetailPanel } from "./BacktestDetailPanel";
import { TYPE_COLORS } from "./_helpers";
import type { ModelInfo } from "./_types";

interface Props {
  models: ModelInfo[];
  selectedModelId: string;
  selectedModelInfo: ModelInfo;
  onSelectModel: (modelId: string) => void;
}

// Display order + labels for the model-family groups. The group header conveys
// the type, so individual rows don't repeat a type badge.
const TYPE_GROUPS: { type: string; label: string }[] = [
  { type: "tree", label: "Tree Models" },
  { type: "foundation", label: "Foundation" },
  { type: "statistical", label: "Statistical" },
  { type: "deep_learning", label: "Deep Learning" },
];

export function BacktestStagePanel({
  models,
  selectedModelId,
  selectedModelInfo,
  onSelectModel,
}: Props) {
  const queryClient = useQueryClient();
  const [runningModels, setRunningModels] = useState<Set<string>>(new Set());
  const [parallel, setParallel] = useState(false);

  const { data: backtestSummary } = useQuery({
    queryKey: backtestMgmtKeys.summary,
    queryFn: fetchBacktestSummary,
    staleTime: 30_000,
    // Poll while anything is in flight (an optimistic click or a run the server
    // reports as queued/running) so the table reflects real status — including
    // the post-run auto-load that flips a row to "Loaded".
    refetchInterval: (query) => {
      const summary = query.state.data;
      const anyRunInFlight = summary
        ? Object.values(summary).some(
            (s) => s.latest_run?.status === "queued" || s.latest_run?.status === "running",
          )
        : false;
      return runningModels.size > 0 || anyRunInFlight ? 5_000 : false;
    },
  });
  const { data: lagLeaderboard } = useQuery<LagLeaderboardPayload>({
    queryKey: lagLeaderboardKeys.list({ limit: 50 }),
    queryFn: () => fetchLagLeaderboard({ limit: 50 }),
    staleTime: 30_000,
  });

  const lagAccuracyForModel = (modelId: string): Map<number, number | null> => {
    const accuracyByLag = new Map<number, number | null>();
    for (const lagBlock of lagLeaderboard?.lags ?? []) {
      const modelResult = lagBlock.rankings.find((entry) => entry.model_id === modelId);
      accuracyByLag.set(lagBlock.lag, modelResult?.accuracy_pct ?? null);
    }
    return accuracyByLag;
  };

  const isModelActive = (id: string): boolean => {
    const status = backtestSummary?.[id]?.latest_run?.status;
    return runningModels.has(id) || status === "queued" || status === "running";
  };
  const runLabel = (id: string): string => {
    const status = backtestSummary?.[id]?.latest_run?.status;
    if (status === "queued") return "Queued";
    if (status === "running" || runningModels.has(id)) return "Running";
    return "Run";
  };

  // Submit a single backtest. `silent` suppresses per-model toasts when called
  // as part of a "Run all" group submission (one summary toast covers the group).
  const runOne = async (modelId: string, opts?: { silent?: boolean }) => {
    const label = models.find((m) => m.id === modelId)?.label ?? modelId;
    setRunningModels((prev) => new Set(prev).add(modelId));
    try {
      const res = await submitBacktestRun(modelId, parallel);
      queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.summary });
      if (!opts?.silent) {
        if (res.status === "already_running") {
          toast.info(res.message ?? `A ${label} backtest is already in progress.`);
        } else {
          toast.success(`${label} backtest queued — results auto-load when it finishes.`);
        }
      }
    } catch (err) {
      toast.error(formatApiError(err));
    } finally {
      // Keep the running indicator briefly so the click registers visually.
      setTimeout(() => {
        setRunningModels((prev) => {
          const next = new Set(prev);
          next.delete(modelId);
          return next;
        });
      }, 5_000);
    }
  };

  const handleRunGroup = async (groupType: string, groupLabel: string) => {
    const targets = models.filter((m) => m.type === groupType && !isModelActive(m.id));
    if (targets.length === 0) {
      toast.info(`All ${groupLabel} backtests are already running or queued.`);
      return;
    }
    toast.success(
      `Queued ${targets.length} ${groupLabel} backtest${targets.length === 1 ? "" : "s"} — results auto-load on completion.`,
    );
    for (const m of targets) {
      await runOne(m.id, { silent: true });
    }
  };

  // Status cell: No backtest → Running/Queued → Completed → Loaded. With
  // server-side auto-load, a completed run lands on "Loaded" on its own.
  const statusCell = (id: string, bt: BacktestModelSummary | undefined) => {
    const status = bt?.latest_run?.status;
    if (runningModels.has(id) || status === "running") {
      return (
        <span className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-400">
          <Loader2 className="h-3 w-3 animate-spin" /> Running…
        </span>
      );
    }
    if (status === "queued") {
      return <span className="text-yellow-600 dark:text-yellow-400">Queued…</span>;
    }
    if (status === "failed") {
      return <span className="font-medium text-destructive">Failed</span>;
    }
    if (bt?.latest_run?.is_loaded_to_db) {
      return (
        <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
          <CheckCircle2 className="h-3 w-3" /> Loaded
        </span>
      );
    }
    // Ran but not yet loaded (auto-load pending/failed) — rare; offers a hint.
    if (bt?.has_predictions || bt?.current_accuracy != null) {
      return <span className="text-amber-600 dark:text-amber-400">Completed</span>;
    }
    return <span className="text-muted-foreground">No backtest</span>;
  };

  // Ordered groups + a catch-all for any unexpected type, dropping empties.
  const known = new Set(TYPE_GROUPS.map((g) => g.type));
  const groups = [
    ...TYPE_GROUPS.map((g) => ({ ...g, items: models.filter((m) => m.type === g.type) })),
    { type: "other", label: "Other", items: models.filter((m) => !known.has(m.type)) },
  ].filter((g) => g.items.length > 0);

  return (
    <>
      {/* ---- Grouped model tables ---- */}
      <div className="space-y-4">
        <div className="flex items-center justify-between gap-3">
          <h3 className="text-sm font-medium text-muted-foreground">All Models</h3>
          <label
            className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-muted-foreground"
            title="When on, different model types run at the same time. When off, backtests run one at a time — extra runs queue automatically. Re-running a model that's already in progress is skipped."
          >
            <input
              type="checkbox"
              checked={parallel}
              onChange={(e) => setParallel(e.target.checked)}
              className="h-3.5 w-3.5"
            />
            Run in parallel
          </label>
        </div>

        {groups.map((g) => (
          <div key={g.type} className="overflow-hidden rounded-lg border">
            <div className="flex items-center justify-between border-b bg-muted/30 px-3 py-1.5">
              <span className="inline-flex items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-foreground/80">
                  {g.label}
                </span>
                <Badge
                  variant="outline"
                  className={`text-[9px] px-1.5 py-0 ${TYPE_COLORS[g.type] ?? ""}`}
                >
                  {g.items.length}
                </Badge>
              </span>
              <Button
                size="sm"
                variant="ghost"
                className="h-6 gap-1 px-2 text-[11px]"
                onClick={() => handleRunGroup(g.type, g.label)}
                title={`Run a backtest for every model in ${g.label}`}
              >
                <Play className="h-3 w-3" /> Run all
              </Button>
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="h-8 text-xs">Model</TableHead>
                  <TableHead className="h-8 text-xs">Accuracy by forecast lag</TableHead>
                  <TableHead className="h-8 text-xs">Status</TableHead>
                  <TableHead className="h-8 w-[1%] text-right text-xs">Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {g.items.map((m) => {
                  const bt = backtestSummary?.[m.id];
                  const lagAccuracy = lagAccuracyForModel(m.id);
                  const isSelected = selectedModelId === m.id;
                  return (
                    <TableRow
                      key={m.id}
                      className={cn("cursor-pointer", isSelected && "bg-primary/5 hover:bg-primary/10")}
                      onClick={() => onSelectModel(m.id)}
                    >
                      <TableCell className="py-1.5 text-sm font-medium">{m.label}</TableCell>
                      <TableCell className="py-1.5 tabular-nums">
                        <div
                          className="flex flex-wrap items-center gap-1"
                          aria-label={`${m.label} accuracy by forecast lag`}
                        >
                          <span
                            className="rounded bg-primary/10 px-1.5 py-0.5 text-[11px] font-semibold text-primary"
                            title="Execution lag: each DFU measured at its own production-relevant lag"
                          >
                            Exec {bt?.current_accuracy != null ? `${bt.current_accuracy.toFixed(1)}%` : "—"}
                          </span>
                          {[0, 1, 2, 3, 4].map((lag) => {
                            const accuracy = lagAccuracy.get(lag);
                            return (
                              <span
                                key={lag}
                                className="rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground"
                                title={`All DFUs measured at fixed forecast lag ${lag}`}
                              >
                                L{lag} {accuracy != null ? `${accuracy.toFixed(1)}%` : "—"}
                              </span>
                            );
                          })}
                        </div>
                      </TableCell>
                      <TableCell className="py-1.5 text-xs">{statusCell(m.id, bt)}</TableCell>
                      <TableCell className="py-1.5 text-right">
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 gap-1 px-2 text-[11px]"
                          onClick={(e) => {
                            e.stopPropagation();
                            runOne(m.id);
                          }}
                          disabled={isModelActive(m.id)}
                        >
                          {isModelActive(m.id) ? (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          ) : (
                            <Play className="h-3 w-3" />
                          )}
                          {runLabel(m.id)}
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        ))}
      </div>

      {/* ---- Selected Model Backtest Detail ---- */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <CardTitle className="text-base">{selectedModelInfo.label}</CardTitle>
            <Badge
              variant="outline"
              className={`text-[10px] ${TYPE_COLORS[selectedModelInfo.type] ?? ""}`}
            >
              {selectedModelInfo.type.replace("_", " ")}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <BacktestDetailPanel modelId={selectedModelId} />
        </CardContent>
      </Card>
    </>
  );
}
