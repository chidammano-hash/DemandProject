/**
 * Shared StatusBadge — used by ActiveJobsPanel and JobHistoryPanel.
 */
import { cn } from "@/lib/utils";
import { STATUS_CONFIG } from "./jobsShared";

export function StatusBadge({ status }: { status: string }) {
  const config = STATUS_CONFIG[status] || STATUS_CONFIG.queued;
  const Icon = config.icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium",
        config.bg,
        config.color,
      )}
    >
      <Icon className={cn("h-3 w-3", status === "running" && "animate-spin")} />
      {config.label}
    </span>
  );
}
