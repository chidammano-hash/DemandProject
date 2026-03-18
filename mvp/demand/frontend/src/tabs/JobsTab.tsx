import { useCallback, useEffect, useState } from "react";
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
  submitPipeline,
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
import { ChampionConfigPanel } from "./jobs/ChampionConfigPanel";
import { ClusterScenarioConfigPanel } from "./jobs/ClusterScenarioConfigPanel";

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
type JobsTabProps = { onNavigateToScenario?: (jobId: string) => void };

export default function JobsTab({ onNavigateToScenario }: JobsTabProps) {
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

  const { data: schedulesData } = useQuery({
    queryKey: queryKeys.jobSchedules(),
    queryFn: fetchJobSchedules,
    staleTime: STALE.ONE_MIN,
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
    queryClient.invalidateQueries({ queryKey: ["jobs"] });
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

  const pipelineMutation = useMutation({
    mutationFn: ({ steps, label }: { steps: { type: string; params: Record<string, unknown> }[]; label: string }) =>
      submitPipeline(steps.map((s) => ({ job_type: s.type, params: s.params })), label),
    onSuccess: invalidateAll,
  });

  // ---- derived data ----
  const jobTypes = typesData?.types || [];
  const activeJobs = activeData?.jobs || [];
  const historyJobs =
    historyData?.jobs?.filter((j: Job) => j.status !== "running" && j.status !== "queued") || [];
  const schedules = schedulesData?.schedules || [];

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-foreground">Job Scheduler</h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            Automate, schedule, and monitor long-running operations
          </p>
        </div>
        <span className="rounded-full bg-primary/10 px-3 py-1 text-[10px] font-semibold text-primary uppercase tracking-wider">
          APScheduler Engine
        </span>
      </div>

      {/* KPI cards */}
      {statsData && <KpiSection stats={statsData} />}

      {/* Submit error */}
      {submitMutation.isError && (
        <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive flex items-center gap-2">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {(submitMutation.error as Error)?.message || "Failed to submit job"}
        </div>
      )}

      {/* Active jobs */}
      <ActiveJobsPanel
        activeJobs={activeJobs}
        onCancel={(id) => cancelMutation.mutate(id)}
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
          customCards={{
            champion_select: <ChampionConfigPanel onJobSubmitted={invalidateAll} />,
            cluster_scenario: <ClusterScenarioConfigPanel onJobSubmitted={invalidateAll} />,
          }}
        />
      </section>

      {/* Pipeline builder */}
      <PipelineBuilderPanel
        jobTypes={jobTypes}
        activeJobs={activeJobs}
        onSubmit={({ steps, label }) => pipelineMutation.mutate({ steps, label })}
        isSubmitting={pipelineMutation.isPending}
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
