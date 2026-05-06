/**
 * SKU Features — Compute button + job-status banner.
 *
 * Polls /jobs/active to recover an in-flight `compute_sku_features` job on
 * mount, then polls /jobs/{job_id} until terminal. Invalidates the
 * `sku-features` query namespace when the job completes.
 */
import { useEffect, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Play, Loader2 } from "lucide-react";
import { triggerComputeSkuFeatures } from "@/api/queries/sku-features";
import { fetchActiveJobs, fetchJobDetail } from "@/api/queries/jobs";

const COMPUTE_JOB_TYPE = "compute_sku_features";
const TIME_WINDOW_MONTHS = 36;
const ACTIVE_POLL_MS = 5_000;
const JOB_POLL_MS = 3_000;
const COMPLETION_RESET_MS = 3_000;

export function ComputeButton() {
  const queryClient = useQueryClient();
  const [computeJobId, setComputeJobId] = useState<string | null>(null);

  // Detect any already-running compute_sku_features job on mount.
  const { data: activeJobsData } = useQuery({
    queryKey: ["active-jobs"],
    queryFn: fetchActiveJobs,
    staleTime: ACTIVE_POLL_MS,
    refetchInterval: ACTIVE_POLL_MS,
  });

  // Recover running job on mount / when active jobs change.
  useEffect(() => {
    if (!activeJobsData?.jobs || computeJobId) return;
    const running = activeJobsData.jobs.find(
      (j) => j.job_type === COMPUTE_JOB_TYPE && (j.status === "running" || j.status === "queued"),
    );
    if (running) {
      setComputeJobId(running.job_id);
    }
  }, [activeJobsData, computeJobId]);

  // Compute features mutation
  const computeMutation = useMutation({
    mutationFn: () => triggerComputeSkuFeatures(TIME_WINDOW_MONTHS),
    onSuccess: (data) => {
      setComputeJobId(data.job_id);
    },
  });

  // Poll job status while running
  const { data: jobStatus } = useQuery({
    queryKey: ["jobs", computeJobId],
    queryFn: () => (computeJobId ? fetchJobDetail(computeJobId) : Promise.resolve(null)),
    enabled: !!computeJobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed" || status === "cancelled") return false;
      return JOB_POLL_MS;
    },
  });

  const jobDone = jobStatus?.status === "completed";
  const jobFailed = jobStatus?.status === "failed";

  // Refresh data when job completes
  useEffect(() => {
    if (jobDone) {
      queryClient.invalidateQueries({ queryKey: ["sku-features"] });
      const timer = setTimeout(() => setComputeJobId(null), COMPLETION_RESET_MS);
      return () => clearTimeout(timer);
    }
  }, [jobDone, queryClient]);

  const inFlight = computeMutation.isPending || (!!computeJobId && !jobDone && !jobFailed);

  return (
    <>
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-foreground">SKU Features</h2>
          <p className="text-sm text-muted-foreground">
            Explore computed demand features across all SKUs — seasonality profiles, variability classes, trend signals, and statistical metrics.
          </p>
        </div>
        <button
          onClick={() => computeMutation.mutate()}
          disabled={inFlight}
          className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow-sm hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shrink-0"
        >
          {inFlight ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              {jobStatus?.progress_msg || "Computing..."}
            </>
          ) : (
            <>
              <Play className="h-4 w-4" />
              Compute Features
            </>
          )}
        </button>
      </div>

      {/* Job status banner */}
      {computeJobId && (
        <div className={`rounded-md border px-4 py-2.5 text-sm ${
          jobDone
            ? "border-green-200 bg-green-50 text-green-800 dark:border-green-800 dark:bg-green-950/30 dark:text-green-300"
            : jobFailed
              ? "border-red-200 bg-red-50 text-red-800 dark:border-red-800 dark:bg-red-950/30 dark:text-red-300"
              : "border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-800 dark:bg-blue-950/30 dark:text-blue-300"
        }`}>
          {jobDone && "Feature computation completed. Refreshing data..."}
          {jobFailed && "Feature computation failed. Check job logs for details."}
          {!jobDone && !jobFailed && (
            <>
              Computing features...
              {jobStatus?.progress_pct != null && ` (${jobStatus.progress_pct}%)`}
              {jobStatus?.progress_msg && ` — ${jobStatus.progress_msg}`}
            </>
          )}
        </div>
      )}

      {/* Mutation error banner */}
      {computeMutation.isError && !computeJobId && (
        <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2.5 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/30 dark:text-red-300">
          Failed to start computation: {computeMutation.error?.message ?? "Unknown error"}
        </div>
      )}
    </>
  );
}
