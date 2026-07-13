// ---------------------------------------------------------------------------
// Job Scheduler types (Feature 39)
// ---------------------------------------------------------------------------

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type JobGroup = "clustering" | "backtest" | "seasonality" | "champion" | "ai" | "forecast" | "replenishment" | "inventory" | "tuning";

export interface JobType {
  type_id: string;
  label: string;
  description: string;
  group: string;
  params_schema: Record<string, unknown>;
}

export interface Job {
  job_id: string;
  job_type: string;
  job_label: string;
  status: JobStatus;
  params: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string | null;
  submitted_at: string;
  started_at: string | null;
  completed_at: string | null;
  progress_pct: number;
  progress_msg: string | null;
  logs?: Array<{ ts: string; pct: number; msg: string }>;
  pid: number | null;
  pipeline_id?: string | null;
  pipeline_step?: number | null;
  recovery_quarantine_reason?: string | null;
}

export interface JobLogsPayload {
  job_id: string;
  log: string;
  total_length: number;
  offset: number;
}

export interface JobListPayload {
  jobs: Job[];
  total: number;
  limit: number;
  offset: number;
}

export interface JobTypesPayload {
  types: JobType[];
}

export interface ActiveJobsPayload {
  jobs: Job[];
}

export interface JobStats {
  total: number;
  active: number;
  completed: number;
  failed: number;
  cancelled: number;
  avg_duration_seconds: number;
  last_24h: {
    submitted: number;
    completed: number;
    failed: number;
  };
}

export interface JobSchedule {
  schedule_id: string;
  job_type: string;
  job_label: string;
  cron_expr: string | null;
  interval_min: number | null;
  params: Record<string, unknown>;
  enabled: boolean;
  created_at: string | null;
  last_run_at: string | null;
  next_run_at: string | null;
  run_count: number;
}

export interface JobSchedulesPayload {
  schedules: JobSchedule[];
}

// ---------------------------------------------------------------------------
// Group visual config for UI
// ---------------------------------------------------------------------------

export interface GroupConfig {
  label: string;
  color: string;
  bgColor: string;
  borderColor: string;
  iconBg: string;
}

export const GROUP_CONFIG: Record<string, GroupConfig> = {
  clustering: {
    label: "Clustering",
    color: "text-blue-600 dark:text-blue-400",
    bgColor: "bg-blue-50 dark:bg-blue-950/30",
    borderColor: "border-blue-200 dark:border-blue-800",
    iconBg: "bg-blue-100 dark:bg-blue-900/50",
  },
  backtest: {
    label: "Backtesting",
    color: "text-sky-600 dark:text-sky-400",
    bgColor: "bg-sky-50 dark:bg-sky-950/30",
    borderColor: "border-sky-200 dark:border-sky-800",
    iconBg: "bg-sky-100 dark:bg-sky-900/50",
  },
  seasonality: {
    label: "Seasonality",
    color: "text-emerald-600 dark:text-emerald-400",
    bgColor: "bg-emerald-50 dark:bg-emerald-950/30",
    borderColor: "border-emerald-200 dark:border-emerald-800",
    iconBg: "bg-emerald-100 dark:bg-emerald-900/50",
  },
  champion: {
    label: "Champion",
    color: "text-amber-600 dark:text-amber-400",
    bgColor: "bg-amber-50 dark:bg-amber-950/30",
    borderColor: "border-amber-200 dark:border-amber-800",
    iconBg: "bg-amber-100 dark:bg-amber-900/50",
  },
  ai: {
    label: "AI Planning",
    color: "text-teal-600 dark:text-teal-400",
    bgColor: "bg-teal-50 dark:bg-teal-950/30",
    borderColor: "border-teal-200 dark:border-teal-800",
    iconBg: "bg-teal-100 dark:bg-teal-900/50",
  },
  forecast: {
    label: "Forecasting",
    color: "text-purple-600 dark:text-purple-400",
    bgColor: "bg-purple-50 dark:bg-purple-950/30",
    borderColor: "border-purple-200 dark:border-purple-800",
    iconBg: "bg-purple-100 dark:bg-purple-900/50",
  },
  replenishment: {
    label: "Replenishment",
    color: "text-orange-600 dark:text-orange-400",
    bgColor: "bg-orange-50 dark:bg-orange-950/30",
    borderColor: "border-orange-200 dark:border-orange-800",
    iconBg: "bg-orange-100 dark:bg-orange-900/50",
  },
  inventory: {
    label: "Inventory Planning",
    color: "text-cyan-600 dark:text-cyan-400",
    bgColor: "bg-cyan-50 dark:bg-cyan-950/30",
    borderColor: "border-cyan-200 dark:border-cyan-800",
    iconBg: "bg-cyan-100 dark:bg-cyan-900/50",
  },
  tuning: {
    label: "AI Tuning",
    color: "text-violet-600 dark:text-violet-400",
    bgColor: "bg-violet-50 dark:bg-violet-950/30",
    borderColor: "border-violet-200 dark:border-violet-800",
    iconBg: "bg-violet-100 dark:bg-violet-900/50",
  },
  platform: {
    label: "Platform",
    color: "text-slate-600 dark:text-slate-400",
    bgColor: "bg-slate-50 dark:bg-slate-950/30",
    borderColor: "border-slate-200 dark:border-slate-800",
    iconBg: "bg-slate-100 dark:bg-slate-900/50",
  },
};
