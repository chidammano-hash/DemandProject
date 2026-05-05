/**
 * JobHistoryPanel — expandable job history table with status filters,
 * type filters, inline param/result/error details, and delete/view-results actions.
 */
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BarChart3, ChevronDown, ChevronRight, ScrollText, Trash2, Zap } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Job, JobType } from "@/types/jobs";
import { GROUP_CONFIG } from "@/types/jobs";
import { fetchJobLogs, queryKeys } from "@/api/queries";
import { purgeJobs } from "@/api/queries/jobs";
import { EmptyState } from "@/components/EmptyState";
import { formatTimestamp, jobDuration, getGroupKey, GROUP_ICONS } from "./jobsShared";
import { StatusBadge } from "./StatusBadge";

// ---------------------------------------------------------------------------
// JobHistoryRow — expandable table row
// ---------------------------------------------------------------------------
function JobHistoryRow({
  job,
  onDelete,
  onViewResults,
}: {
  job: Job;
  onDelete: (id: string) => void;
  onViewResults?: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [showPersistentLog, setShowPersistentLog] = useState(false);
  const groupKey = getGroupKey(job.job_type);
  const cfg = GROUP_CONFIG[groupKey] || GROUP_CONFIG.clustering;

  // Lazy-load persistent log only when requested
  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: queryKeys.jobLogs(job.job_id),
    queryFn: () => fetchJobLogs(job.job_id),
    enabled: showPersistentLog,
  });

  return (
    <>
      <tr
        className="border-b border-border/50 hover:bg-muted/20 cursor-pointer transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <td className="px-3 py-2.5 text-xs">
          {expanded ? (
            <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
          )}
        </td>
        <td className="px-3 py-2.5">
          <div className="flex items-center gap-2">
            <div className={cn("rounded-md p-1", cfg.iconBg)}>
              {(() => {
                const GI = GROUP_ICONS[groupKey] || Zap;
                return <GI className={cn("h-3 w-3", cfg.color)} />;
              })()}
            </div>
            <span className="text-sm font-medium text-foreground">{job.job_label}</span>
          </div>
        </td>
        <td className="px-3 py-2.5 text-xs text-muted-foreground">{job.job_type}</td>
        <td className="px-3 py-2.5">
          <StatusBadge status={job.status} />
        </td>
        <td className="px-3 py-2.5 text-xs tabular-nums text-muted-foreground">
          {jobDuration(job)}
        </td>
        <td className="px-3 py-2.5 text-xs text-muted-foreground">
          {formatTimestamp(job.submitted_at)}
        </td>
        <td className="px-3 py-2.5 flex items-center gap-1">
          {job.status === "completed" && job.job_type === "cluster_scenario" && onViewResults && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onViewResults(job.job_id);
              }}
              className="rounded-md p-1 text-muted-foreground/50 hover:text-primary hover:bg-primary/10 transition-colors"
              title="View Results in Clusters Tab"
            >
              <BarChart3 className="h-3.5 w-3.5" />
            </button>
          )}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(job.job_id);
            }}
            className="rounded-md p-1 text-muted-foreground/50 hover:text-destructive hover:bg-destructive/10 transition-colors"
            title="Delete"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </td>
      </tr>

      {expanded && (
        <tr className="border-b border-border/50 bg-muted/5">
          <td colSpan={7} className="px-6 py-4">
            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="font-semibold text-foreground/80 block mb-1">Job ID</span>
                  <code className="text-muted-foreground font-mono text-[10px] bg-muted rounded px-1.5 py-0.5">
                    {job.job_id}
                  </code>
                </div>
                <div>
                  <span className="font-semibold text-foreground/80 block mb-1">Progress</span>
                  <span className="text-muted-foreground">
                    {job.progress_pct}% — {job.progress_msg || "-"}
                  </span>
                </div>
              </div>

              {job.params && Object.keys(job.params).length > 0 && (
                <div>
                  <span className="font-semibold text-foreground/80 block mb-1">Parameters</span>
                  <pre className="text-muted-foreground bg-muted/50 rounded-lg p-2 overflow-x-auto text-[10px] font-mono">
                    {JSON.stringify(job.params, null, 2)}
                  </pre>
                </div>
              )}

              {job.error && (
                <div>
                  <span className="font-semibold text-red-500 block mb-1">Error</span>
                  <pre className="text-red-400 bg-red-50 dark:bg-red-950/30 rounded-lg p-2 overflow-x-auto text-[10px] font-mono">
                    {job.error}
                  </pre>
                </div>
              )}

              {job.result && (
                <div className="space-y-2">
                  {typeof job.result.output_log === "string" && job.result.output_log && (
                    <div>
                      <span className="font-semibold text-foreground/80 block mb-1">Output Log</span>
                      <pre className="text-green-300 bg-black/80 rounded-lg p-2 overflow-x-auto overflow-y-auto max-h-48 text-[10px] font-mono whitespace-pre-wrap">
                        {job.result.output_log}
                      </pre>
                    </div>
                  )}
                  {Object.keys(job.result).filter((k) => k !== "output_log").length > 0 && (
                    <div>
                      <span className="font-semibold text-foreground/80 block mb-1">Result</span>
                      <pre className="text-muted-foreground bg-muted/50 rounded-lg p-2 overflow-x-auto text-[10px] font-mono">
                        {JSON.stringify(
                          Object.fromEntries(Object.entries(job.result).filter(([k]) => k !== "output_log")),
                          null,
                          2,
                        )}
                      </pre>
                    </div>
                  )}
                </div>
              )}

              {/* Persistent execution log */}
              <div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowPersistentLog((v) => !v);
                  }}
                  className={cn(
                    "flex items-center gap-1 rounded-md border px-2 py-1 text-[10px] font-medium transition-colors",
                    showPersistentLog
                      ? "border-primary/40 bg-primary/10 text-primary"
                      : "border-border text-muted-foreground hover:bg-muted",
                  )}
                >
                  <ScrollText className="h-3 w-3" />
                  {showPersistentLog ? "Hide Execution Log" : "View Execution Log"}
                </button>
                {showPersistentLog && (
                  <pre className="mt-2 text-green-300 bg-black/80 rounded-lg p-2 overflow-x-auto overflow-y-auto max-h-64 text-[10px] font-mono whitespace-pre-wrap">
                    {logsLoading
                      ? "Loading..."
                      : logsData?.log
                        ? logsData.log
                        : "No execution log available"}
                  </pre>
                )}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// JobHistoryPanel
// ---------------------------------------------------------------------------
export interface JobHistoryPanelProps {
  historyJobs: Job[];
  jobTypes: JobType[];
  total: number | null;
  historyFilter: string;
  historyTypeFilter: string;
  onHistoryFilterChange: (value: string) => void;
  onHistoryTypeFilterChange: (value: string) => void;
  onDelete: (jobId: string) => void;
  onViewResults?: (jobId: string) => void;
}

export function JobHistoryPanel({
  historyJobs,
  jobTypes,
  total,
  historyFilter,
  historyTypeFilter,
  onHistoryFilterChange,
  onHistoryTypeFilterChange,
  onDelete,
  onViewResults,
}: JobHistoryPanelProps) {
  const queryClient = useQueryClient();
  // "all" = drop every terminal job; otherwise N = drop jobs older than N days
  const [purgeOlderDays, setPurgeOlderDays] = useState<"all" | number>(7);

  const purgeMutation = useMutation({
    mutationFn: purgeJobs,
    onSuccess: () => {
      // Invalidate the jobs cache so the table refreshes
      queryClient.invalidateQueries({
        predicate: (q) =>
          Array.isArray(q.queryKey) && (q.queryKey[0] === "jobs" || q.queryKey[0] === "jobsHistory"),
      });
    },
  });

  const handleClear = (): void => {
    const ageSummary =
      purgeOlderDays === "all"
        ? "ALL terminal"
        : `terminal jobs older than ${purgeOlderDays} day${purgeOlderDays === 1 ? "" : "s"}`;
    const filterSummary = historyFilter
      ? ` (status=${historyFilter})`
      : "";
    const typeSummary = historyTypeFilter
      ? ` (type=${historyTypeFilter})`
      : "";
    if (
      window.confirm(
        `Clear ${ageSummary}${filterSummary}${typeSummary} jobs? Running/queued jobs are preserved.`,
      )
    ) {
      purgeMutation.mutate({
        ...(purgeOlderDays === "all" ? {} : { older_than_hours: purgeOlderDays * 24 }),
        ...(historyFilter ? { status: historyFilter } : {}),
        ...(historyTypeFilter ? { job_type: historyTypeFilter } : {}),
      });
    }
  };

  return (
    <section>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-foreground/80 uppercase tracking-wider">
          Job History{" "}
          {total != null && (
            <span className="text-muted-foreground font-normal">({total})</span>
          )}
        </h3>
        <div className="flex items-center gap-2">
          <select
            value={historyFilter}
            onChange={(e) => onHistoryFilterChange(e.target.value)}
            className="text-xs rounded-lg border border-border bg-card px-2.5 py-1.5 text-foreground"
          >
            <option value="">All Statuses</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>
          <select
            value={historyTypeFilter}
            onChange={(e) => onHistoryTypeFilterChange(e.target.value)}
            className="text-xs rounded-lg border border-border bg-card px-2.5 py-1.5 text-foreground"
          >
            <option value="">All Types</option>
            {jobTypes.map((t) => (
              <option key={t.type_id} value={t.type_id}>
                {t.label}
              </option>
            ))}
          </select>
          <span className="text-xs text-muted-foreground pl-1">Older than</span>
          <select
            value={String(purgeOlderDays)}
            onChange={(e) => {
              const v = e.target.value;
              setPurgeOlderDays(v === "all" ? "all" : Number(v));
            }}
            className="text-xs rounded-lg border border-border bg-card px-2.5 py-1.5 text-foreground"
            aria-label="Purge age filter"
          >
            <option value="all">All terminal</option>
            <option value="1">1 day</option>
            <option value="7">7 days</option>
            <option value="30">30 days</option>
            <option value="90">90 days</option>
          </select>
          <button
            type="button"
            onClick={handleClear}
            disabled={purgeMutation.isPending}
            className="text-xs rounded-lg border border-border bg-card px-2.5 py-1.5 text-foreground hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
            title="Bulk-delete terminal jobs matching all filters above"
          >
            {purgeMutation.isPending ? "Clearing…" : "Clear"}
          </button>
        </div>
      </div>
      {purgeMutation.isSuccess && (
        <p className="mb-3 text-xs text-emerald-600 dark:text-emerald-400">
          Cleared {purgeMutation.data.deleted} job{purgeMutation.data.deleted === 1 ? "" : "s"}.
        </p>
      )}
      {purgeMutation.isError && (
        <p className="mb-3 text-xs text-destructive">
          Clear failed: {(purgeMutation.error as Error).message}
        </p>
      )}

      {historyJobs.length === 0 ? (
        (historyFilter || historyTypeFilter) ? (
          <EmptyState
            variant="filtered"
            title="No jobs match your filters"
            description="Try clearing the status or type filter to see more jobs."
            onAction={() => {
              onHistoryFilterChange("");
              onHistoryTypeFilterChange("");
            }}
            actionLabel="Clear filters"
          />
        ) : (
          <EmptyState
            variant="no-data"
            icon={BarChart3}
            title="No jobs in history yet"
            description="Submit a job above or schedule one to start building a history."
          />
        )
      ) : (
        <div className="rounded-xl border border-border overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-muted/20">
              <tr className="border-b border-border/50">
                <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground w-8" />
                <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground">
                  Job
                </th>
                <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground">
                  Type
                </th>
                <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground">
                  Status
                </th>
                <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground">
                  Duration
                </th>
                <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground">
                  Submitted
                </th>
                <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground w-10" />
              </tr>
            </thead>
            <tbody>
              {historyJobs.map((job: Job) => (
                <JobHistoryRow
                  key={job.job_id}
                  job={job}
                  onDelete={onDelete}
                  onViewResults={onViewResults}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
