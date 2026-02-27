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
  STALE,
} from "@/api/queries";
import type { Job, JobType, JobSchedule } from "@/types/jobs";
import { GROUP_CONFIG } from "@/types/jobs";
import type { Theme } from "@/types";
import { useJobNotification } from "@/context/JobNotificationContext";
import { cn } from "@/lib/utils";
import {
  PlayCircle,
  Square,
  CheckCircle2,
  XCircle,
  Loader2,
  Trash2,
  ChevronDown,
  ChevronRight,
  Clock,
  AlertCircle,
  BarChart3,
  Zap,
  Trophy,
  Activity,
  Network,
  TrendingUp,
  Calendar,
  Timer,
  Repeat,
  X,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Group icon mapping
// ---------------------------------------------------------------------------
const GROUP_ICONS: Record<string, typeof Network> = {
  clustering: Network,
  backtest: TrendingUp,
  seasonality: Activity,
  champion: Trophy,
};

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------
const STATUS_CONFIG: Record<string, { icon: typeof CheckCircle2; color: string; bg: string; label: string }> = {
  queued: { icon: Clock, color: "text-yellow-600 dark:text-yellow-400", bg: "bg-yellow-100 dark:bg-yellow-900/30", label: "Queued" },
  running: { icon: Loader2, color: "text-blue-600 dark:text-blue-400", bg: "bg-blue-100 dark:bg-blue-900/30", label: "Running" },
  completed: { icon: CheckCircle2, color: "text-emerald-600 dark:text-emerald-400", bg: "bg-emerald-100 dark:bg-emerald-900/30", label: "Completed" },
  failed: { icon: XCircle, color: "text-red-600 dark:text-red-400", bg: "bg-red-100 dark:bg-red-900/30", label: "Failed" },
  cancelled: { icon: Square, color: "text-gray-500", bg: "bg-gray-100 dark:bg-gray-800", label: "Cancelled" },
};

function StatusBadge({ status }: { status: string }) {
  const config = STATUS_CONFIG[status] || STATUS_CONFIG.queued;
  const Icon = config.icon;
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium", config.bg, config.color)}>
      <Icon className={cn("h-3 w-3", status === "running" && "animate-spin")} />
      {config.label}
    </span>
  );
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m >= 60) {
    const h = Math.floor(m / 60);
    const rm = m % 60;
    return rm > 0 ? `${h}h ${rm}m` : `${h}h`;
  }
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

function formatTimestamp(iso: string | null): string {
  if (!iso) return "-";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function jobDuration(job: Job): string {
  if (!job.started_at) return "-";
  const start = new Date(job.started_at).getTime();
  const end = job.completed_at ? new Date(job.completed_at).getTime() : Date.now();
  return formatDuration((end - start) / 1000);
}

function getGroupKey(jobType: string): string {
  return Object.keys(GROUP_CONFIG).find((g) => jobType.startsWith(g.slice(0, 4))) || "clustering";
}

// ---------------------------------------------------------------------------
// KPI Card
// ---------------------------------------------------------------------------
function KpiCard({
  label,
  value,
  icon: Icon,
  color,
  subtitle,
}: {
  label: string;
  value: string | number;
  icon: typeof BarChart3;
  color: string;
  subtitle?: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-card p-4 transition-shadow hover:shadow-sm">
      <div className="flex items-center justify-between">
        <div className={cn("rounded-lg p-2", color)}>
          <Icon className="h-4 w-4" />
        </div>
      </div>
      <p className="mt-3 text-2xl font-bold tabular-nums text-foreground">{value}</p>
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      {subtitle && <p className="mt-0.5 text-[10px] text-muted-foreground/70">{subtitle}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SubmitJobPanel — grouped cards
// ---------------------------------------------------------------------------
function SubmitJobPanel({
  jobTypes,
  onSubmit,
  onSchedule,
  submitting,
}: {
  jobTypes: JobType[];
  onSubmit: (typeId: string, params: Record<string, unknown>, label: string) => void;
  onSchedule: (typeId: string) => void;
  submitting: boolean;
}) {
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const groups = useMemo(() => {
    const seen = new Set<string>();
    return jobTypes.reduce<string[]>((acc, t) => {
      if (!seen.has(t.group)) { seen.add(t.group); acc.push(t.group); }
      return acc;
    }, []);
  }, [jobTypes]);

  return (
    <div className="space-y-5">
      {groups.map((group) => {
        const cfg = GROUP_CONFIG[group] || GROUP_CONFIG.clustering;
        const GIcon = GROUP_ICONS[group] || Zap;
        const types = jobTypes.filter((t) => t.group === group);
        return (
          <div key={group}>
            <div className="flex items-center gap-2 mb-3">
              <div className={cn("rounded-md p-1.5", cfg.iconBg)}>
                <GIcon className={cn("h-3.5 w-3.5", cfg.color)} />
              </div>
              <h4 className={cn("text-xs font-semibold uppercase tracking-wider", cfg.color)}>
                {cfg.label}
              </h4>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {types.map((t) => {
                const isSelected = selectedType === t.type_id;
                return (
                  <button
                    key={t.type_id}
                    className={cn(
                      "group relative rounded-xl border p-4 text-left transition-all duration-200",
                      isSelected
                        ? cn("ring-2 ring-offset-1", cfg.borderColor, cfg.bgColor, "ring-current")
                        : "border-border bg-card hover:border-primary/30 hover:shadow-sm",
                    )}
                    onClick={() => setSelectedType(isSelected ? null : t.type_id)}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-foreground truncate">{t.label}</p>
                        <p className="text-xs text-muted-foreground mt-1 line-clamp-2">{t.description}</p>
                      </div>
                      <div className={cn("rounded-lg p-1.5 transition-colors", isSelected ? cfg.iconBg : "bg-muted/50")}>
                        <PlayCircle className={cn("h-4 w-4", isSelected ? cfg.color : "text-muted-foreground/50")} />
                      </div>
                    </div>
                    {isSelected && (
                      <div className="mt-3 pt-3 border-t border-border/50 flex gap-2">
                        <button
                          disabled={submitting}
                          onClick={(e) => {
                            e.stopPropagation();
                            onSubmit(t.type_id, t.params_schema || {}, t.label);
                          }}
                          className={cn(
                            "flex-1 rounded-lg px-3 py-2 text-xs font-semibold transition-all duration-200",
                            submitting
                              ? "bg-muted text-muted-foreground cursor-not-allowed"
                              : "bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm",
                          )}
                        >
                          {submitting ? (
                            <span className="inline-flex items-center gap-1.5">
                              <Loader2 className="h-3 w-3 animate-spin" /> Scheduling...
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1.5">
                              <Zap className="h-3 w-3" /> Run Now
                            </span>
                          )}
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSchedule(t.type_id);
                          }}
                          className="rounded-lg border border-border px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-muted/50 hover:text-foreground transition-colors"
                          title="Schedule recurring"
                        >
                          <Calendar className="h-3 w-3" />
                        </button>
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ActiveJobCard — live monitoring
// ---------------------------------------------------------------------------
function ActiveJobCard({ job, onCancel }: { job: Job; onCancel: (id: string) => void }) {
  const [elapsed, setElapsed] = useState("");
  const groupKey = getGroupKey(job.job_type);
  const cfg = GROUP_CONFIG[groupKey] || GROUP_CONFIG.clustering;

  useEffect(() => {
    if (!job.started_at || job.status !== "running") return;
    const update = () => {
      const start = new Date(job.started_at!).getTime();
      setElapsed(formatDuration((Date.now() - start) / 1000));
    };
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, [job.started_at, job.status]);

  return (
    <div className={cn("rounded-xl border-2 p-4 transition-all", cfg.borderColor, cfg.bgColor)}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <div className={cn("h-2 w-2 rounded-full animate-pulse", job.status === "running" ? "bg-blue-500" : "bg-yellow-500")} />
          <div className="min-w-0">
            <p className="text-sm font-semibold text-foreground truncate">{job.job_label}</p>
            <p className="text-[10px] text-muted-foreground">{job.job_type}</p>
          </div>
        </div>
        <StatusBadge status={job.status} />
      </div>

      {/* Progress bar */}
      <div className="w-full bg-background/50 rounded-full h-2 mb-2 overflow-hidden">
        <div
          className={cn(
            "h-2 rounded-full transition-all duration-700 ease-out",
            job.status === "running"
              ? "bg-gradient-to-r from-blue-500 to-blue-400"
              : "bg-yellow-500",
          )}
          style={{ width: `${Math.max(job.progress_pct, 3)}%` }}
        />
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="font-medium">{job.progress_pct}%</span>
          <span>{job.progress_msg || "Waiting..."}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs tabular-nums font-medium text-muted-foreground">
            <Timer className="h-3 w-3 inline mr-0.5" />
            {elapsed || jobDuration(job)}
          </span>
          {(job.status === "running" || job.status === "queued") && (
            <button
              onClick={() => onCancel(job.job_id)}
              className="rounded-md border border-destructive/30 px-2 py-0.5 text-[10px] font-medium text-destructive hover:bg-destructive/10 transition-colors"
            >
              Cancel
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ScheduleDialog — cron/interval builder
// ---------------------------------------------------------------------------
function ScheduleDialog({
  jobType,
  jobTypes,
  onClose,
  onSubmit,
}: {
  jobType: string;
  jobTypes: JobType[];
  onClose: () => void;
  onSubmit: (typeId: string, cron?: string, intervalMin?: number, label?: string) => void;
}) {
  const [mode, setMode] = useState<"cron" | "interval">("interval");
  const [cron, setCron] = useState("0 2 * * *");
  const [intervalMin, setIntervalMin] = useState(60);
  const typeDef = jobTypes.find((t) => t.type_id === jobType);

  const presets = [
    { label: "Every hour", cron: "0 * * * *", intervalMin: 60 },
    { label: "Every 6 hours", cron: "0 */6 * * *", intervalMin: 360 },
    { label: "Daily at 2 AM", cron: "0 2 * * *", intervalMin: 1440 },
    { label: "Weekly (Mon 2 AM)", cron: "0 2 * * 1", intervalMin: 10080 },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onClose}>
      <div className="w-full max-w-md rounded-2xl border border-border bg-card p-6 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-semibold text-foreground">Schedule Recurring Job</h3>
            <p className="text-xs text-muted-foreground mt-0.5">{typeDef?.label || jobType}</p>
          </div>
          <button onClick={onClose} className="rounded-md p-1 hover:bg-muted">
            <X className="h-4 w-4 text-muted-foreground" />
          </button>
        </div>

        {/* Presets */}
        <div className="flex flex-wrap gap-2 mb-4">
          {presets.map((p) => (
            <button
              key={p.label}
              onClick={() => {
                setCron(p.cron);
                setIntervalMin(p.intervalMin);
              }}
              className="rounded-full border border-border px-3 py-1 text-xs hover:bg-muted transition-colors"
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* Mode toggle */}
        <div className="flex rounded-lg bg-muted p-0.5 mb-4">
          <button
            className={cn(
              "flex-1 rounded-md py-1.5 text-xs font-medium transition-all",
              mode === "interval" ? "bg-card shadow-sm text-foreground" : "text-muted-foreground",
            )}
            onClick={() => setMode("interval")}
          >
            Interval
          </button>
          <button
            className={cn(
              "flex-1 rounded-md py-1.5 text-xs font-medium transition-all",
              mode === "cron" ? "bg-card shadow-sm text-foreground" : "text-muted-foreground",
            )}
            onClick={() => setMode("cron")}
          >
            Cron Expression
          </button>
        </div>

        {mode === "interval" ? (
          <div className="space-y-2">
            <label className="text-xs font-medium text-foreground">Run every</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={1}
                value={intervalMin}
                onChange={(e) => setIntervalMin(Number(e.target.value))}
                className="h-9 w-24 rounded-md border border-input bg-background px-3 text-sm"
              />
              <span className="text-sm text-muted-foreground">minutes</span>
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            <label className="text-xs font-medium text-foreground">Cron expression</label>
            <input
              type="text"
              value={cron}
              onChange={(e) => setCron(e.target.value)}
              placeholder="0 2 * * *"
              className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm font-mono"
            />
            <p className="text-[10px] text-muted-foreground">Format: minute hour day-of-month month day-of-week</p>
          </div>
        )}

        <div className="flex justify-end gap-2 mt-6">
          <button onClick={onClose} className="rounded-lg border border-border px-4 py-2 text-xs font-medium hover:bg-muted">
            Cancel
          </button>
          <button
            onClick={() => {
              if (mode === "cron") {
                onSubmit(jobType, cron, undefined, typeDef?.label);
              } else {
                onSubmit(jobType, undefined, intervalMin, typeDef?.label);
              }
              onClose();
            }}
            className="rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground hover:bg-primary/90 shadow-sm"
          >
            <span className="inline-flex items-center gap-1.5">
              <Repeat className="h-3 w-3" />
              Create Schedule
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SchedulesSection
// ---------------------------------------------------------------------------
function SchedulesSection({
  schedules,
  onDelete,
}: {
  schedules: JobSchedule[];
  onDelete: (id: string) => void;
}) {
  if (schedules.length === 0) return null;
  return (
    <section>
      <h3 className="text-sm font-semibold text-foreground/80 uppercase tracking-wider mb-3 flex items-center gap-2">
        <Repeat className="h-3.5 w-3.5" />
        Active Schedules ({schedules.length})
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {schedules.map((s) => {
          const groupKey = getGroupKey(s.job_type);
          const cfg = GROUP_CONFIG[groupKey] || GROUP_CONFIG.clustering;
          return (
            <div key={s.schedule_id} className={cn("rounded-xl border p-3", cfg.borderColor, cfg.bgColor)}>
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold text-foreground">{s.job_label}</p>
                <button onClick={() => onDelete(s.schedule_id)} className="text-muted-foreground hover:text-destructive">
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
              <div className="mt-2 flex items-center gap-3 text-[10px] text-muted-foreground">
                {s.cron_expr && <span className="font-mono bg-muted rounded px-1.5 py-0.5">{s.cron_expr}</span>}
                {s.interval_min && <span>Every {s.interval_min}min</span>}
                <span>Runs: {s.run_count}</span>
              </div>
              {s.next_run_at && (
                <p className="mt-1 text-[10px] text-muted-foreground">Next: {formatTimestamp(s.next_run_at)}</p>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// JobHistoryRow
// ---------------------------------------------------------------------------
function JobHistoryRow({ job, onDelete }: { job: Job; onDelete: (id: string) => void }) {
  const [expanded, setExpanded] = useState(false);
  const groupKey = getGroupKey(job.job_type);
  const cfg = GROUP_CONFIG[groupKey] || GROUP_CONFIG.clustering;

  return (
    <>
      <tr
        className="border-b border-border/50 hover:bg-muted/20 cursor-pointer transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <td className="px-3 py-2.5 text-xs">
          {expanded ? <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" /> : <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />}
        </td>
        <td className="px-3 py-2.5">
          <div className="flex items-center gap-2">
            <div className={cn("rounded-md p-1", cfg.iconBg)}>
              {(() => { const GI = GROUP_ICONS[groupKey] || Zap; return <GI className={cn("h-3 w-3", cfg.color)} />; })()}
            </div>
            <span className="text-sm font-medium text-foreground">{job.job_label}</span>
          </div>
        </td>
        <td className="px-3 py-2.5 text-xs text-muted-foreground">{job.job_type}</td>
        <td className="px-3 py-2.5"><StatusBadge status={job.status} /></td>
        <td className="px-3 py-2.5 text-xs tabular-nums text-muted-foreground">{jobDuration(job)}</td>
        <td className="px-3 py-2.5 text-xs text-muted-foreground">{formatTimestamp(job.submitted_at)}</td>
        <td className="px-3 py-2.5">
          <button
            onClick={(e) => { e.stopPropagation(); onDelete(job.job_id); }}
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
                  <code className="text-muted-foreground font-mono text-[10px] bg-muted rounded px-1.5 py-0.5">{job.job_id}</code>
                </div>
                <div>
                  <span className="font-semibold text-foreground/80 block mb-1">Progress</span>
                  <span className="text-muted-foreground">{job.progress_pct}% — {job.progress_msg || "-"}</span>
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
                <div>
                  <span className="font-semibold text-foreground/80 block mb-1">Result</span>
                  <pre className="text-muted-foreground bg-muted/50 rounded-lg p-2 overflow-x-auto text-[10px] font-mono">
                    {JSON.stringify(job.result, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
type JobsTabProps = { theme: Theme };

export default function JobsTab({ theme }: JobsTabProps) {
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
    queryKey: queryKeys.jobs({ status: historyFilter || undefined, job_type: historyTypeFilter || undefined, limit: 50, offset: 0 }),
    queryFn: () => fetchJobs({
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

  // ---- mutations ----
  const invalidateAll = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: queryKeys.activeJobs() });
    queryClient.invalidateQueries({ queryKey: ["jobs"] });
    queryClient.invalidateQueries({ queryKey: queryKeys.jobStats() });
  }, [queryClient]);

  const submitMutation = useMutation({
    mutationFn: ({ type, params, label }: { type: string; params: Record<string, unknown>; label: string }) =>
      submitJob(type, params, label),
    onSuccess: (data) => {
      jobNotification.startJob(data.job_id, "job", data.status);
      invalidateAll();
    },
  });

  const cancelMutation = useMutation({
    mutationFn: cancelJob,
    onSuccess: invalidateAll,
  });

  const deleteMutation = useMutation({
    mutationFn: deleteJob,
    onSuccess: invalidateAll,
  });

  const scheduleMutation = useMutation({
    mutationFn: ({ type, cron, intervalMin, label }: { type: string; cron?: string; intervalMin?: number; label?: string }) =>
      createSchedule(type, {}, label, cron, intervalMin),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.jobSchedules() });
    },
  });

  const deleteScheduleMutation = useMutation({
    mutationFn: deleteSchedule,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.jobSchedules() });
    },
  });

  const handleSubmit = useCallback(
    (typeId: string, params: Record<string, unknown>, label: string) => {
      submitMutation.mutate({ type: typeId, params, label });
    },
    [submitMutation],
  );

  const handleSchedule = useCallback(
    (typeId: string, cron?: string, intervalMin?: number, label?: string) => {
      scheduleMutation.mutate({ type: typeId, cron, intervalMin, label });
    },
    [scheduleMutation],
  );

  const handleCancel = useCallback((jobId: string) => cancelMutation.mutate(jobId), [cancelMutation]);
  const handleDelete = useCallback((jobId: string) => deleteMutation.mutate(jobId), [deleteMutation]);

  const jobTypes = typesData?.types || [];
  const activeJobs = activeData?.jobs || [];
  const historyJobs = historyData?.jobs?.filter((j: Job) => j.status !== "running" && j.status !== "queued") || [];
  const schedules = schedulesData?.schedules || [];
  const stats = statsData;

  const successRate = stats && (stats.completed + stats.failed) > 0
    ? Math.round((stats.completed / (stats.completed + stats.failed)) * 100)
    : 100;

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* ---- Header ---- */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-foreground">Job Scheduler</h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            Automate, schedule, and monitor long-running operations
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="rounded-full bg-primary/10 px-3 py-1 text-[10px] font-semibold text-primary uppercase tracking-wider">
            APScheduler Engine
          </span>
        </div>
      </div>

      {/* ---- KPI Dashboard ---- */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <KpiCard
            label="Total Jobs"
            value={stats.total}
            icon={BarChart3}
            color="bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400"
            subtitle={stats.last_24h.submitted > 0 ? `${stats.last_24h.submitted} today` : undefined}
          />
          <KpiCard
            label="Active Now"
            value={stats.active}
            icon={Zap}
            color="bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400"
          />
          <KpiCard
            label="Success Rate"
            value={`${successRate}%`}
            icon={CheckCircle2}
            color="bg-emerald-100 dark:bg-emerald-900/50 text-emerald-600 dark:text-emerald-400"
            subtitle={stats.failed > 0 ? `${stats.failed} failed` : undefined}
          />
          <KpiCard
            label="Avg Duration"
            value={stats.avg_duration_seconds > 0 ? formatDuration(stats.avg_duration_seconds) : "-"}
            icon={Timer}
            color="bg-amber-100 dark:bg-amber-900/50 text-amber-600 dark:text-amber-400"
          />
        </div>
      )}

      {/* ---- Submit error ---- */}
      {submitMutation.isError && (
        <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive flex items-center gap-2">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {(submitMutation.error as Error)?.message || "Failed to submit job"}
        </div>
      )}

      {/* ---- Active Jobs ---- */}
      {activeJobs.length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-foreground/80 uppercase tracking-wider mb-3 flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
            Active Jobs ({activeJobs.length})
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {activeJobs.map((job) => (
              <ActiveJobCard key={job.job_id} job={job} onCancel={handleCancel} />
            ))}
          </div>
        </section>
      )}

      {/* ---- Schedules ---- */}
      <SchedulesSection
        schedules={schedules}
        onDelete={(id) => deleteScheduleMutation.mutate(id)}
      />

      {/* ---- Available Jobs ---- */}
      <section className="rounded-2xl border border-border bg-card/50 p-5">
        <h3 className="text-sm font-semibold text-foreground/80 uppercase tracking-wider mb-4">
          Schedule New Job
        </h3>
        <SubmitJobPanel
          jobTypes={jobTypes}
          onSubmit={handleSubmit}
          onSchedule={setScheduleDialogType}
          submitting={submitMutation.isPending}
        />
      </section>

      {/* ---- Job History ---- */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-foreground/80 uppercase tracking-wider">
            Job History {historyData?.total != null && <span className="text-muted-foreground font-normal">({historyData.total})</span>}
          </h3>
          <div className="flex items-center gap-2">
            <select
              value={historyFilter}
              onChange={(e) => setHistoryFilter(e.target.value)}
              className="text-xs rounded-lg border border-border bg-card px-2.5 py-1.5 text-foreground"
            >
              <option value="">All Statuses</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
              <option value="cancelled">Cancelled</option>
            </select>
            <select
              value={historyTypeFilter}
              onChange={(e) => setHistoryTypeFilter(e.target.value)}
              className="text-xs rounded-lg border border-border bg-card px-2.5 py-1.5 text-foreground"
            >
              <option value="">All Types</option>
              {jobTypes.map((t) => (
                <option key={t.type_id} value={t.type_id}>{t.label}</option>
              ))}
            </select>
          </div>
        </div>

        {historyJobs.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-border bg-card/30 p-12 text-center">
            <BarChart3 className="h-8 w-8 text-muted-foreground/30 mx-auto mb-3" />
            <p className="text-sm font-medium text-muted-foreground">No jobs in history yet</p>
            <p className="text-xs text-muted-foreground/60 mt-1">Submit a job above to get started</p>
          </div>
        ) : (
          <div className="rounded-xl border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-muted/20">
                <tr className="border-b border-border/50">
                  <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground w-8" />
                  <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground">Job</th>
                  <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground">Type</th>
                  <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground">Status</th>
                  <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground">Duration</th>
                  <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground">Submitted</th>
                  <th className="px-3 py-2.5 text-left text-xs font-medium text-muted-foreground w-10" />
                </tr>
              </thead>
              <tbody>
                {historyJobs.map((job: Job) => (
                  <JobHistoryRow key={job.job_id} job={job} onDelete={handleDelete} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* ---- Schedule Dialog ---- */}
      {scheduleDialogType && (
        <ScheduleDialog
          jobType={scheduleDialogType}
          jobTypes={jobTypes}
          onClose={() => setScheduleDialogType(null)}
          onSubmit={handleSchedule}
        />
      )}
    </div>
  );
}
