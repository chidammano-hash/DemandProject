import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import {
  queryKeys,
  fetchJobTypes,
  fetchJobs,
  fetchActiveJobs,
  fetchJobStats,
  fetchJobSchedules,
  submitJob,
  cancelJob,
  deleteJob,
  createSchedule,
  deleteSchedule,
  fetchNamedPipelines,
  fetchPipelineReadiness,
  pipelineReadinessKeys,
  runNamedPipeline,
  STALE,
} from "@/api/queries";
import type { Job } from "@/types/jobs";
import { useJobNotification } from "@/context/JobNotificationContext";
import { AlertCircle } from "lucide-react";
import { KpiSection } from "./jobs/KpiSection";
import { JobGroupsPanel } from "./jobs/JobGroupsPanel";
import { ActiveJobsPanel } from "./jobs/ActiveJobsPanel";
import { SchedulesPanel } from "./jobs/SchedulesPanel";
import { JobHistoryPanel } from "./jobs/JobHistoryPanel";
import { PipelineBuilderPanel } from "./jobs/PipelineBuilderPanel";
// ClusterScenarioConfigPanel removed — clustering managed via Cluster tab

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
type JobsTabProps = {
  onNavigateToScenario?: (jobId: string) => void;
  embedded?: boolean;
};

const EMPTY_JOBS: Job[] = [];

export default function JobsTab({ onNavigateToScenario, embedded = false }: JobsTabProps) {
  const queryClient = useQueryClient();
  const jobNotification = useJobNotification();
  const [historyFilter, setHistoryFilter] = useState<string>("");
  const [historyTypeFilter, setHistoryTypeFilter] = useState<string>("");
  const [scheduleDialogType, setScheduleDialogType] = useState<string | null>(null);

  // ---- queries ----
  const { data: typesData } = useQuery({
    queryKey: queryKeys.jobTypes(),
    queryFn: fetchJobTypes,
    staleTime: STALE.TEN_MIN,
  });

  const { data: statsData } = useQuery({
    queryKey: queryKeys.jobStats(),
    queryFn: fetchJobStats,
    refetchInterval: 5_000,
  });

  const { data: activeData } = useQuery({
    queryKey: queryKeys.activeJobs(),
    queryFn: fetchActiveJobs,
    refetchInterval: 2_000,
  });

  const { data: historyData } = useQuery({
    queryKey: queryKeys.jobs({
      status: historyFilter || undefined,
      job_type: historyTypeFilter || undefined,
      limit: 50,
      offset: 0,
    }),
    queryFn: () =>
      fetchJobs({
        status: historyFilter || undefined,
        job_type: historyTypeFilter || undefined,
        limit: 50,
        offset: 0,
      }),
    refetchInterval: 10_000,
  });

  // Pipeline cards must reconcile against unfiltered history. Reusing the
  // user-filtered table payload makes a completed/failed workflow appear idle
  // as soon as the history filters change.
  const { data: pipelineHistoryData } = useQuery({
    queryKey: queryKeys.jobs({ scope: "forecast-pipeline-status", limit: 200, offset: 0 }),
    queryFn: () => fetchJobs({ limit: 200, offset: 0 }),
    refetchInterval: 5_000,
  });

  const { data: schedulesData } = useQuery({
    queryKey: queryKeys.jobSchedules(),
    queryFn: fetchJobSchedules,
    staleTime: STALE.ONE_MIN,
  });

  const {
    data: namedPipelinesData,
    isLoading: namedPipelinesLoading,
    error: namedPipelinesError,
  } = useQuery({
    queryKey: queryKeys.namedPipelines(),
    queryFn: fetchNamedPipelines,
    staleTime: STALE.TEN_MIN,
  });

  const {
    data: pipelineReadiness,
    error: pipelineReadinessError,
    isLoading: pipelineReadinessLoading,
  } = useQuery({
    queryKey: pipelineReadinessKeys.readiness,
    queryFn: fetchPipelineReadiness,
    staleTime: STALE.ONE_MIN,
    refetchInterval: (query) =>
      query.state.data && !query.state.data.ready ? 30_000 : false,
  });

  // ---- Sync active jobs with notification context ----
  useEffect(() => {
    if (!activeData?.jobs) return;
    for (const job of activeData.jobs) {
      if (!jobNotification.activeJobs.has(job.job_id)) {
        jobNotification.startJob(job.job_id, job.job_type, job.job_label);
      }
    }
  }, [activeData, jobNotification]);

  // ---- shared invalidation ----
  const invalidateAll = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: queryKeys.activeJobs() });
    queryClient.invalidateQueries({ queryKey: queryKeys.jobsAll() });
    queryClient.invalidateQueries({ queryKey: queryKeys.jobStats() });
  }, [queryClient]);

  // ---- mutations ----
  const submitMutation = useMutation({
    mutationFn: ({ type, params, label }: { type: string; params: Record<string, unknown>; label: string }) =>
      submitJob(type, params, label),
    onSuccess: (data) => {
      jobNotification.startJob(data.job_id, "job", data.status);
      invalidateAll();
    },
  });

  const cancelMutation = useMutation({ mutationFn: cancelJob, onSuccess: invalidateAll });
  const deleteMutation = useMutation({ mutationFn: deleteJob, onSuccess: invalidateAll });

  const scheduleMutation = useMutation({
    mutationFn: ({ type, cron, intervalMin, label }: { type: string; cron?: string; intervalMin?: number; label?: string }) =>
      createSchedule(type, {}, label, cron, intervalMin),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.jobSchedules() }),
  });

  const deleteScheduleMutation = useMutation({
    mutationFn: deleteSchedule,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.jobSchedules() }),
  });

  const namedPipelineMutation = useMutation({
    mutationFn: (name: string) => runNamedPipeline(name),
    onSuccess: invalidateAll,
  });

  // ---- derived data ----
  const jobTypes = typesData?.types || [];
  const activeJobs = activeData?.jobs ?? EMPTY_JOBS;
  const historyJobs =
    historyData?.jobs?.filter((j: Job) => j.status !== "running" && j.status !== "queued") || [];
  const pipelineJobs = useMemo(() => {
    const byId = new Map<string, Job>();
    for (const job of pipelineHistoryData?.jobs ?? []) byId.set(job.job_id, job);
    for (const job of activeJobs) byId.set(job.job_id, job);
    return Array.from(byId.values());
  }, [activeJobs, pipelineHistoryData?.jobs]);
  const schedules = schedulesData?.schedules || [];

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      {!embedded && <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-foreground">Job Scheduler</h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            Automate, schedule, and monitor long-running operations
          </p>
        </div>
        <span className="rounded-full bg-primary/10 px-3 py-1 text-[10px] font-semibold text-primary uppercase tracking-wider">
          APScheduler Engine
        </span>
      </div>}

      {/* KPI cards */}
      {statsData && <KpiSection stats={statsData} />}

      {/* Submit error */}
      {(submitMutation.isError || namedPipelineMutation.isError) && (
        <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive flex items-center gap-2">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {(namedPipelineMutation.error as Error)?.message ||
            (submitMutation.error as Error)?.message ||
            "Failed to submit job"}
        </div>
      )}

      {cancelMutation.isError && (
        <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive flex items-center gap-2">
          <AlertCircle className="h-4 w-4 shrink-0" />
          Cancellation failed; the job status is unchanged. {(cancelMutation.error as Error)?.message}
        </div>
      )}

      {/* Active jobs */}
      <ActiveJobsPanel
        activeJobs={activeJobs}
        onCancel={(id) => cancelMutation.mutate(id)}
        cancellingJobId={cancelMutation.isPending ? cancelMutation.variables : null}
      />

      {/* Recurring schedules + dialog */}
      <SchedulesPanel
        schedules={schedules}
        jobTypes={jobTypes}
        scheduleDialogType={scheduleDialogType}
        onOpenDialog={setScheduleDialogType}
        onCloseDialog={() => setScheduleDialogType(null)}
        onCreateSchedule={(type, cron, intervalMin, label) =>
          scheduleMutation.mutate({ type, cron, intervalMin, label })
        }
        onDeleteSchedule={(id) => deleteScheduleMutation.mutate(id)}
      />

      {/* Available job types */}
      <section className="rounded-2xl border border-border bg-card/50 p-5">
        <h3 className="text-sm font-semibold text-foreground/80 uppercase tracking-wider mb-4">
          Schedule New Job
        </h3>
        <JobGroupsPanel
          jobTypes={jobTypes}
          onSubmit={(typeId, params, label) =>
            submitMutation.mutate({ type: typeId, params, label })
          }
          onSchedule={setScheduleDialogType}
          submitting={submitMutation.isPending}
          hiddenGroups={["clustering", "features", "backtest", "champion", "forecast"]}
        />
      </section>

      {/* Pipeline builder */}
      <PipelineBuilderPanel
        jobTypes={jobTypes}
        pipelines={namedPipelinesData?.pipelines ?? []}
        jobs={pipelineJobs}
        readiness={pipelineReadiness}
        readinessLoading={pipelineReadinessLoading}
        readinessError={
          pipelineReadinessError
            ? "Readiness checks are unavailable. Verify prerequisites before running a workflow."
            : null
        }
        launch={
          namedPipelineMutation.variables && !namedPipelineMutation.isError
            ? {
                name: namedPipelineMutation.variables,
                pipelineId: namedPipelineMutation.data?.pipeline_id,
                submitting: namedPipelineMutation.isPending,
                submittedAt: namedPipelineMutation.submittedAt,
              }
            : undefined
        }
        isLoading={namedPipelinesLoading}
        loadError={namedPipelinesError ? "Could not load forecast workflows." : null}
        onRun={(name) => namedPipelineMutation.mutate(name)}
      />

      {/* Job history */}
      <JobHistoryPanel
        historyJobs={historyJobs}
        jobTypes={jobTypes}
        total={historyData?.total ?? null}
        historyFilter={historyFilter}
        historyTypeFilter={historyTypeFilter}
        onHistoryFilterChange={setHistoryFilter}
        onHistoryTypeFilterChange={setHistoryTypeFilter}
        onDelete={(id) => deleteMutation.mutate(id)}
        onViewResults={onNavigateToScenario}
      />
    </div>
  );
}
