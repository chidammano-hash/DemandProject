/**
 * Run history table for the Tune stage. Click two rows to compare.
 */
import { Crown, Database, FileText, FlaskConical, Plus } from "lucide-react";
import { useMemo } from "react";

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
import { StatusBadge, formatDuration, timeAgo } from "@/components/shared-tuning-utils";
import { formatPct, formatFixed } from "@/lib/formatters";
import { cn } from "@/lib/utils";
import type { TuningRun } from "@/api/queries";

import { PAGE_SIZE } from "./_helpers";

interface Props {
  allRuns: TuningRun[];
  pagedRuns: TuningRun[];
  baselineId: number | null;
  candidateId: number | null;
  page: number;
  totalPages: number;
  sortCol: string;
  sortDir: "asc" | "desc";
  selectedModelLabel: string;
  activeRunsCount: number;
  onSelectRow: (runId: number) => void;
  onClearSelection: () => void;
  onToggleSort: (col: string) => void;
  onSetPage: (page: number) => void;
  onShowLogs: (runId: number) => void;
  onPromote: (run: TuningRun) => void;
  onOpenBuilder: () => void;
}

export function RunHistoryTable({
  allRuns,
  pagedRuns,
  baselineId,
  candidateId,
  page,
  totalPages,
  sortCol,
  sortDir,
  selectedModelLabel,
  activeRunsCount,
  onSelectRow,
  onClearSelection,
  onToggleSort,
  onSetPage,
  onShowLogs,
  onPromote,
  onOpenBuilder,
}: Props) {
  const SortIndicator = useMemo(() => {
    function Indicator({ col }: { col: string }) {
      if (sortCol !== col) return null;
      return (
        <span className="ml-0.5 text-[10px]">{sortDir === "asc" ? "▲" : "▼"}</span>
      );
    }
    return Indicator;
  }, [sortCol, sortDir]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-sm">Run History</CardTitle>
            <CardDescription className="text-xs">Click two rows to compare</CardDescription>
          </div>
          {(baselineId !== null || candidateId !== null) && (
            <Button variant="ghost" size="sm" onClick={onClearSelection}>
              Clear
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {allRuns.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center px-6">
            <FlaskConical className="h-10 w-10 text-muted-foreground/30 mb-3" />
            <p className="text-sm font-medium mb-1">No experiments yet for {selectedModelLabel}</p>
            <p className="text-xs text-muted-foreground mb-4">
              Click "New Experiment" to launch your first tuning run.
            </p>
            <Button
              size="sm"
              className="gap-1.5"
              onClick={onOpenBuilder}
              disabled={activeRunsCount > 0}
            >
              <Plus className="h-3.5 w-3.5" />
              {activeRunsCount > 0 ? "Training…" : "New Experiment"}
            </Button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-14 cursor-pointer" onClick={() => onToggleSort("run_id")}>
                    #<SortIndicator col="run_id" />
                  </TableHead>
                  <TableHead className="cursor-pointer" onClick={() => onToggleSort("run_label")}>
                    Label<SortIndicator col="run_label" />
                  </TableHead>
                  <TableHead className="w-20 cursor-pointer" onClick={() => onToggleSort("status")}>
                    Status<SortIndicator col="status" />
                  </TableHead>
                  <TableHead
                    className="text-right w-20 cursor-pointer"
                    onClick={() => onToggleSort("accuracy_pct")}
                  >
                    Acc%<SortIndicator col="accuracy_pct" />
                  </TableHead>
                  <TableHead
                    className="text-right w-16 cursor-pointer"
                    onClick={() => onToggleSort("wape")}
                  >
                    WAPE<SortIndicator col="wape" />
                  </TableHead>
                  <TableHead
                    className="text-right w-16 cursor-pointer"
                    onClick={() => onToggleSort("bias")}
                  >
                    Bias<SortIndicator col="bias" />
                  </TableHead>
                  <TableHead className="w-20">Duration</TableHead>
                  <TableHead
                    className="w-24 cursor-pointer"
                    onClick={() => onToggleSort("started_at")}
                  >
                    Started<SortIndicator col="started_at" />
                  </TableHead>
                  <TableHead className="w-20 text-center">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {pagedRuns.map((run) => {
                  const isBaseline = baselineId === run.run_id;
                  const isCandidate = candidateId === run.run_id;
                  const isSelected = isBaseline || isCandidate;
                  const runRecord = run as Record<string, unknown>;
                  return (
                    <TableRow
                      key={run.run_id}
                      className={cn(
                        "cursor-pointer transition-colors",
                        isBaseline && "bg-blue-50 dark:bg-blue-950/30",
                        isCandidate && "bg-emerald-50 dark:bg-emerald-950/30",
                        !isSelected && "hover:bg-muted/50",
                      )}
                      onClick={() => onSelectRow(run.run_id)}
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
                            <Crown className="shrink-0 h-3 w-3 text-amber-500" />
                          )}
                          {runRecord.is_results_promoted === true && (
                            <Database className="shrink-0 h-3 w-3 text-blue-500" />
                          )}
                          {runRecord.cluster_source === "experimental" && (
                            <span className="shrink-0 px-1.5 py-0 text-[9px] font-medium rounded-full bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300">
                              Exp #{String(runRecord.cluster_experiment_id ?? "?")}
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
                        {formatDuration(run.started_at, run.completed_at ?? null)}
                      </TableCell>
                      <TableCell
                        className="text-xs text-muted-foreground"
                        title={run.started_at ? new Date(run.started_at).toLocaleString() : ""}
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
                            onClick={() => onShowLogs(run.run_id)}
                          >
                            <FileText className="h-3.5 w-3.5" />
                          </button>
                          {run.status === "completed" && (
                            <button
                              title="Promote"
                              className="rounded p-1 hover:bg-muted text-muted-foreground hover:text-amber-600 dark:hover:text-amber-400 transition-colors"
                              onClick={() => onPromote(run)}
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
                    onClick={() => onSetPage(Math.max(0, page - 1))}
                  >
                    Prev
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 px-2 text-xs"
                    disabled={page >= totalPages - 1}
                    onClick={() => onSetPage(Math.min(totalPages - 1, page + 1))}
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
  );
}

export { PAGE_SIZE };
