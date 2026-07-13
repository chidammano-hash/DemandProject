/**
 * Forecast pipeline launcher backed by the server's canonical named presets.
 *
 * The server owns workflow steps and ordering in
 * config/forecasting/pipelines.yaml. This panel intentionally exposes only
 * the forecasting lifecycle; general ETL and inventory work stays in its
 * dedicated product surfaces.
 */
import { useMemo, useState } from "react";
import { AlertCircle, ChevronDown, ChevronUp, Loader2, Play, RotateCcw } from "lucide-react";

import type { PipelineReadiness } from "@/api/queries/dashboard";
import type { NamedPipelinePreset } from "@/api/queries/jobs";
import type { Job, JobType } from "@/types/jobs";
import { GROUP_CONFIG } from "@/types/jobs";

const FORECAST_PIPELINE_ORDER = [
  "clustering-refresh",
  "model-refresh",
  "forecast-publish",
  "forecast-snapshot-bundle",
] as const;

const PIPELINE_LABELS: Record<(typeof FORECAST_PIPELINE_ORDER)[number], string> = {
  "clustering-refresh": "1. Prepare Features & Clusters",
  "model-refresh": "2. Refresh Five-Model Roster",
  "forecast-publish": "3. Build Release Candidate",
  "forecast-snapshot-bundle": "4. Archive Forecast Snapshot",
};

type PipelineStatus =
  | "idle"
  | "submitting"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface PipelineRunState {
  status: PipelineStatus;
  step: number | null;
  totalSteps: number | null;
  message: string | null;
}

export interface PipelineLaunch {
  name: string;
  pipelineId?: string;
  submitting: boolean;
  submittedAt?: number;
}

interface Props {
  jobTypes: JobType[];
  pipelines: NamedPipelinePreset[];
  jobs?: Job[];
  launch?: PipelineLaunch;
  readiness?: PipelineReadiness;
  readinessLoading?: boolean;
  readinessError?: string | null;
  isLoading?: boolean;
  loadError?: string | null;
  onRun: (name: string) => void;
}

interface PipelineGuidance {
  blocked: boolean;
  message: string;
}

/** Translate cross-stage readiness checks into the next safe lifecycle action. */
export function derivePipelineGuidance(
  name: string,
  readiness?: PipelineReadiness
): PipelineGuidance | null {
  const staleStages = new Set(
    readiness?.checks.filter((check) => check.status === "stale").map((check) => check.stage) ?? []
  );
  if (staleStages.size === 0) return null;

  if (name === "clustering-refresh" && staleStages.has("clustering")) {
    return {
      blocked: false,
      message: "This workflow restores and promotes the cluster assignments required by step 2.",
    };
  }

  if (name === "model-refresh") {
    if (staleStages.has("clustering")) {
      return {
        blocked: true,
        message: "Run step 1: Prepare Features & Clusters before refreshing the model roster.",
      };
    }
    const resolvedStage = ["tuning", "champion", "forecast"].find((stage) =>
      staleStages.has(stage)
    );
    if (resolvedStage) {
      return {
        blocked: false,
        message: `This workflow resolves the current ${resolvedStage} readiness issue.`,
      };
    }
  }

  if (name === "forecast-publish") {
    return staleStages.has("clustering")
      ? {
          blocked: true,
          message:
            "Run step 1: Prepare Features & Clusters, then step 2: Refresh Five-Model Roster.",
        }
      : {
          blocked: true,
          message: "Run step 2: Refresh Five-Model Roster before building a release candidate.",
        };
  }

  return null;
}

export function selectForecastPipelines(pipelines: NamedPipelinePreset[]): NamedPipelinePreset[] {
  const byName = new Map(pipelines.map((pipeline) => [pipeline.name, pipeline]));
  return FORECAST_PIPELINE_ORDER.flatMap((name) => {
    const pipeline = byName.get(name);
    return pipeline ? [pipeline] : [];
  });
}

function numericParam(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() && Number.isFinite(Number(value))) {
    return Number(value);
  }
  return null;
}

function belongsToPipeline(job: Job, name: string): boolean {
  return (
    job.params?.__pipeline_label === name ||
    job.job_label === name ||
    job.job_label?.startsWith(`[${name} `)
  );
}

function jobTimestamp(job: Job): number {
  const value = job.completed_at ?? job.started_at ?? job.submitted_at;
  const timestamp = Date.parse(value);
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

export function derivePipelineRunState(
  name: string,
  jobs: Job[],
  launch?: PipelineLaunch
): PipelineRunState {
  const labelMatches = jobs
    .filter((job) => belongsToPipeline(job, name))
    .sort((left, right) => jobTimestamp(right) - jobTimestamp(left));
  const targetPipelineId =
    launch?.pipelineId ?? labelMatches.find((job) => job.pipeline_id)?.pipeline_id;
  const matchingJobs = targetPipelineId
    ? labelMatches.filter((job) => job.pipeline_id === targetPipelineId)
    : labelMatches;
  const activeJob = matchingJobs.find((job) => job.status === "queued" || job.status === "running");

  if (activeJob) {
    return {
      status: activeJob.status,
      step: numericParam(activeJob.pipeline_step ?? activeJob.params?.__pipeline_step),
      totalSteps: numericParam(activeJob.params?.__pipeline_total_steps),
      message: activeJob.progress_msg,
    };
  }

  const latestJob = matchingJobs[0];
  const launchIsNewer =
    launch?.name === name &&
    (latestJob === undefined ||
      (launch.submittedAt !== undefined && jobTimestamp(latestJob) < launch.submittedAt));
  if (launchIsNewer) {
    return {
      status: launch.submitting ? "submitting" : "queued",
      step: null,
      totalSteps: null,
      message: launch.pipelineId ? "Waiting for first step" : "Starting workflow",
    };
  }

  if (latestJob?.status === "failed" || latestJob?.status === "cancelled") {
    return {
      status: latestJob.status,
      step: numericParam(latestJob.pipeline_step ?? latestJob.params?.__pipeline_step),
      totalSteps: numericParam(latestJob.params?.__pipeline_total_steps),
      message: latestJob.error,
    };
  }

  if (latestJob?.status === "completed") {
    const step = numericParam(latestJob.pipeline_step ?? latestJob.params?.__pipeline_step);
    const totalSteps = numericParam(latestJob.params?.__pipeline_total_steps);
    const isFinalStep = step !== null && totalSteps !== null && step >= totalSteps;
    return {
      status: isFinalStep ? "completed" : "queued",
      step,
      totalSteps,
      message: isFinalStep ? "Workflow completed" : "Advancing to the next step",
    };
  }

  if (launch?.name === name && latestJob === undefined) {
    return {
      status: launch.submitting ? "submitting" : "queued",
      step: null,
      totalSteps: null,
      message: launch.pipelineId ? "Waiting for first step" : "Starting workflow",
    };
  }

  return { status: "idle", step: null, totalSteps: null, message: null };
}

const STATUS_STYLES: Record<PipelineStatus, string> = {
  idle: "bg-muted text-muted-foreground",
  submitting: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  queued: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  running: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  completed: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  failed: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
  cancelled: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
};

function statusLabel(status: PipelineStatus): string {
  if (status === "idle") return "Ready";
  return status.charAt(0).toUpperCase() + status.slice(1);
}

function actionLabel(status: PipelineStatus): string {
  if (status === "completed") return "Run again";
  if (status === "failed" || status === "cancelled") return "Restart from step 1";
  return "Run";
}

export function PipelineBuilderPanel({
  jobTypes,
  pipelines,
  jobs = [],
  launch,
  readiness,
  readinessLoading = false,
  readinessError = null,
  isLoading = false,
  loadError = null,
  onRun,
}: Props) {
  const [expanded, setExpanded] = useState(true);
  const jobTypeMap = useMemo(() => new Map(jobTypes.map((job) => [job.type_id, job])), [jobTypes]);
  const forecastPipelines = useMemo(() => selectForecastPipelines(pipelines), [pipelines]);

  return (
    <section className="rounded-xl border border-border bg-card">
      <button
        type="button"
        className="flex w-full items-center justify-between px-4 py-3 text-left"
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
      >
        <div>
          <div className="flex items-center gap-2">
            <Play className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold">Forecast Pipelines</span>
            <span className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
              {forecastPipelines.length} workflows
            </span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            Run the governed forecasting lifecycle in sequence; promotion remains a review step.
          </p>
        </div>
        {expanded ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>

      {expanded && (
        <div className="border-t border-border p-4">
          {isLoading && (
            <div className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading forecast workflows…
            </div>
          )}

          {!isLoading && loadError && (
            <div className="flex items-center gap-2 rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
              <AlertCircle className="h-4 w-4" /> {loadError}
            </div>
          )}

          {!isLoading && !loadError && readinessError && (
            <div className="mb-3 flex items-center gap-2 rounded-md border border-amber-300/50 bg-amber-50/50 p-3 text-sm text-amber-800 dark:border-amber-800 dark:bg-amber-950/20 dark:text-amber-300">
              <AlertCircle className="h-4 w-4" /> {readinessError}
            </div>
          )}

          {!isLoading && !loadError && forecastPipelines.length === 0 && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No forecasting workflows are configured on the server.
            </p>
          )}

          {!isLoading && !loadError && forecastPipelines.length > 0 && (
            <div className="grid gap-3 md:grid-cols-2">
              {forecastPipelines.map((pipeline) => {
                const state = derivePipelineRunState(pipeline.name, jobs, launch);
                const guidance = derivePipelineGuidance(pipeline.name, readiness);
                const requiresReadiness =
                  pipeline.name === "model-refresh" || pipeline.name === "forecast-publish";
                const checkingPrerequisites = readinessLoading && requiresReadiness;
                const readinessUnverified = Boolean(readinessError) && requiresReadiness;
                const idleReadinessLabel = guidance?.blocked
                  ? "Prerequisite"
                  : checkingPrerequisites
                    ? "Checking"
                    : readinessUnverified
                      ? "Unverified"
                      : null;
                const busy =
                  state.status === "submitting" ||
                  state.status === "queued" ||
                  state.status === "running";
                const pipelineKey = pipeline.name as (typeof FORECAST_PIPELINE_ORDER)[number];

                return (
                  <article
                    key={pipeline.name}
                    className="flex min-h-52 flex-col rounded-lg border border-border bg-muted/15 p-4"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <h3 className="text-sm font-semibold">{PIPELINE_LABELS[pipelineKey]}</h3>
                      <span
                        className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
                          idleReadinessLabel && state.status === "idle"
                            ? "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300"
                            : STATUS_STYLES[state.status]
                        }`}
                      >
                        {idleReadinessLabel && state.status === "idle"
                          ? idleReadinessLabel
                          : statusLabel(state.status)}
                      </span>
                    </div>
                    <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                      {pipeline.description ?? "Server-managed forecast workflow."}
                    </p>

                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {pipeline.steps.map((step) => {
                        const jobType = jobTypeMap.get(step);
                        const group = jobType?.group ?? "forecast";
                        const colors = GROUP_CONFIG[group] ?? GROUP_CONFIG.forecast;
                        return (
                          <span
                            key={step}
                            className={`rounded px-1.5 py-0.5 text-[11px] ${colors.bgColor} ${colors.color}`}
                          >
                            {jobType?.label ?? step.replace(/_/g, " ")}
                          </span>
                        );
                      })}
                    </div>

                    {guidance && (
                      <div
                        role={guidance.blocked ? "alert" : "status"}
                        className={
                          guidance.blocked
                            ? "mt-3 rounded-md border border-amber-300/50 bg-amber-50/50 p-2 text-[11px] text-amber-800 dark:border-amber-800 dark:bg-amber-950/20 dark:text-amber-300"
                            : "mt-3 rounded-md border border-blue-300/40 bg-blue-50/40 p-2 text-[11px] text-blue-800 dark:border-blue-800 dark:bg-blue-950/20 dark:text-blue-300"
                        }
                      >
                        {guidance.message}
                      </div>
                    )}

                    {(state.status === "failed" || state.status === "cancelled") && (
                      <p className="mt-2 text-[11px] text-muted-foreground">
                        Restart creates a new workflow from step 1; this run remains in history.
                      </p>
                    )}

                    <div className="mt-auto flex items-end justify-between gap-3 pt-4">
                      <div className="min-w-0 text-[11px] text-muted-foreground">
                        {state.step !== null && state.totalSteps !== null && (
                          <div className="font-medium text-foreground">
                            Step {state.step} of {state.totalSteps}
                          </div>
                        )}
                        {state.message && <div className="truncate">{state.message}</div>}
                        {launch?.name === pipeline.name && launch.pipelineId && (
                          <div className="font-mono">{launch.pipelineId}</div>
                        )}
                      </div>
                      <button
                        type="button"
                        onClick={() => onRun(pipeline.name)}
                        disabled={busy || guidance?.blocked || checkingPrerequisites}
                        className="inline-flex shrink-0 items-center gap-1 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {busy ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : state.status === "completed" ||
                          state.status === "failed" ||
                          state.status === "cancelled" ? (
                          <RotateCcw className="h-3.5 w-3.5" />
                        ) : (
                          <Play className="h-3.5 w-3.5" />
                        )}
                        {busy
                          ? statusLabel(state.status)
                          : checkingPrerequisites
                            ? "Checking prerequisites"
                          : guidance?.blocked
                            ? "Prerequisite required"
                            : actionLabel(state.status)}
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
