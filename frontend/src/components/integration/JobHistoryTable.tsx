import { Fragment } from "react";
import type { Job } from "../../api/queries/integration";
import { JobStatusBadge } from "./JobStatusBadge";

interface JobHistoryTableProps {
  jobs: Job[];
  onSelect?: (job: Job) => void;
  emptyMessage?: string;
  showRowsLoaded?: boolean;
}

// Module-scoped formatters (avoid recreating Intl.* per render)
const NUMBER_FMT = new Intl.NumberFormat();
const DATETIME_FMT = new Intl.DateTimeFormat(undefined, {
  month: "numeric",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

const ROWS_LOADED_STATUSES: ReadonlyArray<Job["status"]> = ["success", "skipped"];

function formatStartedAt(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "—";
  return DATETIME_FMT.format(d);
}

function formatDuration(durationMs: number | null | undefined): string {
  if (durationMs === null || durationMs === undefined || Number.isNaN(durationMs)) {
    return "—";
  }
  const total = Math.max(0, Math.round(durationMs / 1000));
  if (total < 60) return `${total}s`;
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}m ${s}s`;
}

function formatRowsLoaded(job: Job): string {
  const rows = job.rows_loaded ?? 0;
  if (rows === 0 && !ROWS_LOADED_STATUSES.includes(job.status)) {
    return "—";
  }
  return NUMBER_FMT.format(rows);
}

/**
 * Returns a "+1 inserted · 2 updated · 0 deleted" breakdown when the dispatcher
 * reported per-operation counts (delta path). Returns null when only the legacy
 * single rows_loaded value is available.
 */
function formatDiffBreakdown(job: Job): string | null {
  const { rows_inserted: ins, rows_updated: upd, rows_deleted: del } = job;
  if (ins === null && upd === null && del === null) return null;
  const parts: string[] = [];
  if (ins !== null) parts.push(`+${NUMBER_FMT.format(ins)} new`);
  if (upd !== null) parts.push(`${NUMBER_FMT.format(upd)} updated`);
  if (del !== null && del > 0) parts.push(`${NUMBER_FMT.format(del)} deleted`);
  return parts.length ? parts.join(" · ") : null;
}

const HEADER_CELL =
  "px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground";
const BODY_CELL = "px-3 py-2 text-sm align-middle";

export function JobHistoryTable(props: JobHistoryTableProps): JSX.Element {
  const {
    jobs,
    onSelect,
    emptyMessage = "No jobs.",
    showRowsLoaded = true,
  } = props;

  if (jobs.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-4 text-center">
        {emptyMessage}
      </p>
    );
  }

  const clickable = Boolean(onSelect);
  const rowClass = `border-t border-border ${
    clickable ? "cursor-pointer hover:bg-muted/50" : ""
  }`;

  return (
    <div className="overflow-x-auto rounded border border-border">
      <table className="w-full border-collapse divide-y divide-border">
        <thead className="sticky top-0 bg-card z-10">
          <tr>
            <th className={HEADER_CELL}>Status</th>
            <th className={HEADER_CELL}>Domain</th>
            <th className={HEADER_CELL}>Mode</th>
            <th className={HEADER_CELL}>Slice</th>
            {showRowsLoaded && <th className={HEADER_CELL}>Rows Loaded</th>}
            <th className={HEADER_CELL}>Started At</th>
            <th className={HEADER_CELL}>Duration</th>
            <th className={HEADER_CELL}>Triggered By</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {jobs.map((job) => {
            const colSpan = showRowsLoaded ? 8 : 7;
            const handleClick = onSelect ? () => onSelect(job) : undefined;
            return (
              <Fragment key={job.id}>
                <tr className={rowClass} onClick={handleClick}>
                  <td className={BODY_CELL}>
                    <JobStatusBadge status={job.status} />
                  </td>
                  <td className={BODY_CELL}>{job.domain}</td>
                  <td className={BODY_CELL}>{job.mode}</td>
                  <td className={BODY_CELL}>{job.slice ?? "—"}</td>
                  {showRowsLoaded && (
                    <td className={`${BODY_CELL} tabular-nums`}>
                      <div>{formatRowsLoaded(job)}</div>
                      {formatDiffBreakdown(job) && (
                        <div className="text-xs text-muted-foreground">
                          {formatDiffBreakdown(job)}
                        </div>
                      )}
                    </td>
                  )}
                  <td className={`${BODY_CELL} tabular-nums`}>
                    {formatStartedAt(job.started_at)}
                  </td>
                  <td className={`${BODY_CELL} tabular-nums`}>
                    {formatDuration(job.duration_ms)}
                  </td>
                  <td className={BODY_CELL}>{job.triggered_by ?? "—"}</td>
                </tr>
                {job.status === "failed" && job.error_message && (
                  <tr className="border-t-0">
                    <td
                      colSpan={colSpan}
                      className="px-3 pb-2 pt-0 text-xs text-red-600/80 dark:text-red-400/80"
                    >
                      {job.error_message}
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
