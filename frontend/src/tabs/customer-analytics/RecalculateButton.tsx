/**
 * Customer Analytics — Recalculate button + job-status banner.
 *
 * Refreshes the materialized views backing the tab. Polls /jobs/active to
 * recover an in-flight `refresh_customer_analytics` job on mount, then polls
 * /jobs/{job_id} until terminal. On completion, invalidates every
 * `customer-analytics-*` query so the panels repaint with fresh data.
 */
import { useEffect, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, Loader2 } from "lucide-react";
import { triggerRecalculateCustomerAnalytics } from "@/api/queries/customer-analytics";
import { fetchActiveJobs, fetchJobDetail } from "@/api/queries/jobs";

const RECALC_JOB_TYPE = "refresh_customer_analytics";
const ACTIVE_POLL_MS = 5_000;
const JOB_POLL_MS = 3_000;
const COMPLETION_RESET_MS = 3_000;

export function RecalculateButton() {
  const queryClient = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);

  // Detect any already-running recalc job on mount.
  const { data: activeJobsData } = useQuery({
    queryKey: ["active-jobs"],
    queryFn: fetchActiveJobs,
    staleTime: ACTIVE_POLL_MS,
    refetchInterval: ACTIVE_POLL_MS,
  });

  useEffect(() => {
    if (!activeJobsData?.jobs || jobId) return;
    const running = activeJobsData.jobs.find(
      (j) => j.job_type === RECALC_JOB_TYPE && (j.status === "running" || j.status === "queued"),
    );
    if (running) setJobId(running.job_id);
  }, [activeJobsData, jobId]);

  const recalcMutation = useMutation({
    mutationFn: triggerRecalculateCustomerAnalytics,
    onSuccess: (data) => setJobId(data.job_id),
  });

  const { data: jobStatus } = useQuery({
    queryKey: ["jobs", jobId],
    queryFn: () => (jobId ? fetchJobDetail(jobId) : Promise.resolve(null)),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed" || status === "cancelled") return false;
      return JOB_POLL_MS;
    },
  });

  const jobDone = jobStatus?.status === "completed";
  const jobFailed = jobStatus?.status === "failed";

  // Refresh panel data when the refresh job completes. The CA query keys use
  // many distinct `customer-analytics-*` prefixes (no shared root segment), so
  // a predicate match is required rather than a single key prefix.
  useEffect(() => {
    if (jobDone) {
      queryClient.invalidateQueries({
        predicate: (q) =>
          typeof q.queryKey[0] === "string" &&
          q.queryKey[0].startsWith("customer-analytics"),
      });
      const timer = setTimeout(() => setJobId(null), COMPLETION_RESET_MS);
      return () => clearTimeout(timer);
    }
  }, [jobDone, queryClient]);

  const inFlight = recalcMutation.isPending || (!!jobId && !jobDone && !jobFailed);

  return (
    <div className="flex flex-col gap-2">
      <button
        onClick={() => recalcMutation.mutate()}
        disabled={inFlight}
        className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow-sm hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shrink-0"
      >
        {inFlight ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            {jobStatus?.progress_msg || "Recalculating..."}
          </>
        ) : (
          <>
            <RefreshCw className="h-4 w-4" />
            Recalculate
          </>
        )}
      </button>

      {jobId && (
        <div
          className={`rounded-md border px-4 py-2.5 text-sm ${
            jobDone
              ? "border-green-200 bg-green-50 text-green-800 dark:border-green-800 dark:bg-green-950/30 dark:text-green-300"
              : jobFailed
                ? "border-red-200 bg-red-50 text-red-800 dark:border-red-800 dark:bg-red-950/30 dark:text-red-300"
                : "border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-800 dark:bg-blue-950/30 dark:text-blue-300"
          }`}
        >
          {jobDone && "Recalculation completed. Refreshing data..."}
          {jobFailed && "Recalculation failed. Check job logs for details."}
          {!jobDone && !jobFailed && (
            <>
              Recalculating customer analytics...
              {jobStatus?.progress_pct != null && ` (${jobStatus.progress_pct}%)`}
              {jobStatus?.progress_msg && ` — ${jobStatus.progress_msg}`}
            </>
          )}
        </div>
      )}

      {recalcMutation.isError && !jobId && (
        <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2.5 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/30 dark:text-red-300">
          Failed to start recalculation: {recalcMutation.error?.message ?? "Unknown error"}
        </div>
      )}
    </div>
  );
}
