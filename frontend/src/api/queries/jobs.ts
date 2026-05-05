import { buildSearchParams } from "./helpers";
import { fetchJson } from "./core";
import type {
  Job, JobType, JobListPayload, JobTypesPayload, ActiveJobsPayload,
  JobStats, JobSchedule, JobSchedulesPayload, JobLogsPayload,
} from "@/types/jobs";

export type {
  Job, JobType, JobListPayload, JobTypesPayload, ActiveJobsPayload,
  JobStats, JobSchedule, JobSchedulesPayload, JobLogsPayload,
};

// ---------------------------------------------------------------------------
// Job scheduler queries (Feature 39)
// ---------------------------------------------------------------------------
export async function fetchJobTypes(): Promise<JobTypesPayload> {
  return fetchJson("/jobs/types");
}

export async function fetchJobs(params: {
  status?: string;
  job_type?: string;
  limit?: number;
  offset?: number;
}): Promise<JobListPayload> {
  const qs = buildSearchParams({
    status: params.status,
    job_type: params.job_type,
    limit: params.limit,
    offset: params.offset,
  });
  return fetchJson(`/jobs?${qs}`);
}

export async function fetchJobDetail(jobId: string): Promise<Job> {
  return fetchJson(`/jobs/${encodeURIComponent(jobId)}`);
}

export async function fetchActiveJobs(): Promise<ActiveJobsPayload> {
  return fetchJson("/jobs/active");
}

export async function submitJob(
  jobType: string,
  params: Record<string, unknown> = {},
  label?: string,
): Promise<{ job_id: string; status: string }> {
  return fetchJson("/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_type: jobType, params, label }),
  });
}

export async function cancelJob(jobId: string): Promise<{ job_id: string; status: string }> {
  return fetchJson(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
}

export async function deleteJob(jobId: string): Promise<{ deleted: boolean }> {
  return fetchJson(`/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
}

/** Bulk-delete terminal jobs. Running/queued jobs are always preserved server-side. */
export async function purgeJobs(filter: {
  older_than_hours?: number;
  status?: string;
  job_type?: string;
} = {}): Promise<{ deleted: number }> {
  const qs = new URLSearchParams();
  if (filter.older_than_hours !== undefined) qs.set("older_than_hours", String(filter.older_than_hours));
  if (filter.status) qs.set("status", filter.status);
  if (filter.job_type) qs.set("job_type", filter.job_type);
  const path = qs.toString() ? `/jobs?${qs.toString()}` : "/jobs";
  return fetchJson(path, { method: "DELETE" });
}

export async function fetchScenarioHistory(limit = 10): Promise<Job[]> {
  const data = await fetchJson<JobListPayload>(`/jobs?job_type=cluster_scenario&status=completed&limit=${limit}`);
  return data.jobs;
}

export async function fetchJobLogs(
  jobId: string,
  offset = 0,
): Promise<JobLogsPayload> {
  return fetchJson(`/jobs/${encodeURIComponent(jobId)}/logs?offset=${offset}`);
}

export async function fetchJobStats(): Promise<JobStats> {
  return fetchJson("/jobs/stats");
}

export async function fetchJobSchedules(): Promise<JobSchedulesPayload> {
  return fetchJson("/jobs/schedules");
}

export async function createSchedule(
  jobType: string,
  params: Record<string, unknown> = {},
  label?: string,
  cron?: string,
  intervalMinutes?: number,
): Promise<{ schedule_id: string; status: string }> {
  return fetchJson("/jobs/schedule", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      job_type: jobType,
      params,
      label,
      cron,
      interval_minutes: intervalMinutes,
    }),
  });
}

export async function deleteSchedule(scheduleId: string): Promise<{ deleted: boolean }> {
  return fetchJson(`/jobs/schedules/${encodeURIComponent(scheduleId)}`, { method: "DELETE" });
}

export async function submitPipeline(
  steps: { job_type: string; params?: Record<string, unknown>; label?: string }[],
  label?: string,
): Promise<{ pipeline_id: string; status: string; steps: number }> {
  return fetchJson("/jobs/pipeline", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ steps, label }),
  });
}
