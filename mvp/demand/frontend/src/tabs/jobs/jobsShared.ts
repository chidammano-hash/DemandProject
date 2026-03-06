/**
 * Shared utilities, constants, and types used across Jobs panel components.
 */
import type { Job } from "@/types/jobs";
import { CheckCircle2, XCircle, Square, Clock, Loader2 } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Network, TrendingUp, Activity, Trophy } from "lucide-react";

// ---------------------------------------------------------------------------
// Group icon mapping
// ---------------------------------------------------------------------------
export const GROUP_ICONS: Record<string, LucideIcon> = {
  clustering: Network,
  backtest: TrendingUp,
  seasonality: Activity,
  champion: Trophy,
};

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------
export const STATUS_CONFIG: Record<
  string,
  { icon: LucideIcon; color: string; bg: string; label: string }
> = {
  queued: {
    icon: Clock,
    color: "text-yellow-600 dark:text-yellow-400",
    bg: "bg-yellow-100 dark:bg-yellow-900/30",
    label: "Queued",
  },
  running: {
    icon: Loader2,
    color: "text-blue-600 dark:text-blue-400",
    bg: "bg-blue-100 dark:bg-blue-900/30",
    label: "Running",
  },
  completed: {
    icon: CheckCircle2,
    color: "text-emerald-600 dark:text-emerald-400",
    bg: "bg-emerald-100 dark:bg-emerald-900/30",
    label: "Completed",
  },
  failed: {
    icon: XCircle,
    color: "text-red-600 dark:text-red-400",
    bg: "bg-red-100 dark:bg-red-900/30",
    label: "Failed",
  },
  cancelled: {
    icon: Square,
    color: "text-gray-500",
    bg: "bg-gray-100 dark:bg-gray-800",
    label: "Cancelled",
  },
};

// ---------------------------------------------------------------------------
// Formatting utilities
// ---------------------------------------------------------------------------
export function formatDuration(seconds: number): string {
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

export function formatTimestamp(iso: string | null): string {
  if (!iso) return "-";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function jobDuration(job: Job): string {
  if (!job.started_at) return "-";
  const start = new Date(job.started_at).getTime();
  const end = job.completed_at ? new Date(job.completed_at).getTime() : Date.now();
  return formatDuration((end - start) / 1000);
}

export function getGroupKey(jobType: string): string {
  return (
    Object.keys({ clustering: 1, backtest: 1, seasonality: 1, champion: 1 }).find((g) =>
      jobType.startsWith(g.slice(0, 4)),
    ) || "clustering"
  );
}
