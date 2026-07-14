/**
 * Shared Tuning Utilities — reusable components and helpers for
 * ModelTuningTab and ClusterExperimentsPanel.
 *
 * Extracted from ModelTuningTab to avoid duplication across
 * experiment management UIs that share the same visual patterns.
 */

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

// ---------------------------------------------------------------------------
// StatusBadge — compact colored badge for experiment/run status
// ---------------------------------------------------------------------------

type ExperimentStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

const STATUS_STYLES: Record<ExperimentStatus, string> = {
  completed:
    "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
  running:
    "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  failed:
    "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  queued:
    "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  cancelled:
    "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <Badge
      className={cn(
        "text-[10px] font-medium px-2 py-0.5",
        STATUS_STYLES[status as ExperimentStatus] ?? "bg-gray-100 text-gray-700",
      )}
      aria-label={`Status: ${status}`}
    >
      {status}
    </Badge>
  );
}

// ---------------------------------------------------------------------------
// formatDuration — renders elapsed time from start/end ISO strings
// ---------------------------------------------------------------------------

/**
 * Format duration between two ISO timestamps at a human scale:
 * "42s", "5m 20s", "2h 15m", "2d 22h". Long-running or runaway jobs must
 * not render as raw minutes ("4217m 31s").
 * If `completedAt` is null, uses current time (for in-progress items).
 */
export function formatDuration(
  startedAt: string | null,
  completedAt: string | null,
): string {
  if (!startedAt) return "--";
  const start = new Date(startedAt).getTime();
  const end = completedAt ? new Date(completedAt).getTime() : Date.now();
  const elapsed = Math.max(0, end - start);
  const totalSec = Math.floor(elapsed / 1000);
  const day = Math.floor(totalSec / 86_400);
  const hr = Math.floor((totalSec % 86_400) / 3_600);
  const min = Math.floor((totalSec % 3_600) / 60);
  const sec = totalSec % 60;
  if (day > 0) return `${day}d ${hr}h`;
  if (hr > 0) return `${hr}h ${min}m`;
  if (min > 0) return `${min}m ${sec}s`;
  return `${sec}s`;
}

// ---------------------------------------------------------------------------
// timeAgo — relative time display from an ISO date string
// ---------------------------------------------------------------------------

/**
 * Convert an ISO date string to a human-readable relative time.
 * Returns "--" for null inputs.
 */
export function timeAgo(dateStr: string | null): string {
  if (!dateStr) return "--";
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}
