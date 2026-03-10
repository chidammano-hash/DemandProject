/**
 * Shared utilities, constants, and types used across Jobs panel components.
 */
import type { Job } from "@/types/jobs";
import { CheckCircle2, XCircle, Square, Clock, Loader2 } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Network, TrendingUp, Activity, Trophy, Sparkles, BarChart2, Package, Boxes } from "lucide-react";

// ---------------------------------------------------------------------------
// Group icon mapping
// ---------------------------------------------------------------------------
export const GROUP_ICONS: Record<string, LucideIcon> = {
  clustering: Network,
  backtest: TrendingUp,
  seasonality: Activity,
  champion: Trophy,
  ai: Sparkles,
  forecast: BarChart2,
  replenishment: Package,
  inventory: Boxes,
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

// Prefix-based fallback group resolver (used when jobType list isn't available)
const _GROUP_PREFIXES: [string, string][] = [
  ["cluster", "clustering"],
  ["backtest", "backtest"],
  ["seasonality", "seasonality"],
  ["champion", "champion"],
  ["generate_ai", "ai"],
  ["generate_storyboard", "ai"],
  ["generate_production", "forecast"],
  ["compute_replenishment", "replenishment"],
  ["compute_safety", "inventory"],
  ["compute_eoq", "inventory"],
  ["assign_policies", "inventory"],
  ["generate_exceptions", "inventory"],
  ["classify_abc", "inventory"],
  ["compute_variability", "inventory"],
  ["compute_demand_signals", "inventory"],
  ["compute_investment", "inventory"],
  ["refresh_health", "inventory"],
  ["refresh_intramonth", "inventory"],
  ["run_ss_simulation", "inventory"],
];

export function getGroupKey(jobType: string): string {
  for (const [prefix, group] of _GROUP_PREFIXES) {
    if (jobType.startsWith(prefix)) return group;
  }
  return "clustering";
}

