import type { Job } from "../../api/queries/integration";

type JobStatus = Job["status"];

const STATUS_STYLES: Record<JobStatus, string> = {
  queued:
    "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  running:
    "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  success:
    "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300",
  failed:
    "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
  skipped:
    "bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-300",
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
