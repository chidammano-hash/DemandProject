import type { Job } from "../../api/queries/integration";

type JobStatus = Job["status"];

// Token tints are mode-correct in light, soft, and dark — no dark: siblings.
const STATUS_STYLES: Record<JobStatus, string> = {
  queued: "bg-muted text-muted-foreground",
  running: "bg-info/10 text-info",
  success: "bg-success/10 text-success",
  failed: "bg-destructive/10 text-destructive",
  skipped: "bg-warning/10 text-warning",
};

const PILL_BASE =
  "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium uppercase tracking-wide";

export function JobStatusBadge({ status }: { status: JobStatus }): JSX.Element {
  const style = STATUS_STYLES[status] ?? STATUS_STYLES.queued;
  const isRunning = status === "running";

  return (
    <span className={`${PILL_BASE} ${style}`} aria-label={`status: ${status}`}>
      <span
        aria-hidden="true"
        className={isRunning ? "animate-pulse" : undefined}
      >
        {"●"}
      </span>
      <span>{status}</span>
    </span>
  );
}
