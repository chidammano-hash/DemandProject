/**
 * Real-time progress view for a multi-step Data Integration chain.
 *
 * Polls `getChain(chainId)` at 1.5s while the chain is queued/running, and
 * stops polling once it reaches a terminal state (success/failed/halted).
 * Renders a header, progress bar, vertical timeline of jobs, and footer
 * with start time + total duration.
 */

import { useQuery } from "@tanstack/react-query";
import { JobStatusBadge } from "./JobStatusBadge";
import {
  chainKeys,
  getChain,
  type ChainDetail,
  type ChainJob,
  type ChainStatus,
} from "../../api/queries/integration_chain";
import type { JobStatus } from "../../api/queries/integration";
import { CollapsibleSection } from "@/components/CollapsibleSection";

interface ChainProgressProps {
  chainId: string;
  onClose?: () => void;
}

const POLL_INTERVAL_MS = 1500;
const ACTIVE_CHAIN_STATUSES: ReadonlyArray<ChainStatus> = ["queued", "running"];

const NUMBER_FMT = new Intl.NumberFormat();
const DATETIME_FMT = new Intl.DateTimeFormat(undefined, {
  month: "numeric",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

/** Map a chain-level status to a JobStatus the badge knows how to render. */
function chainStatusToBadge(status: ChainStatus): JobStatus {
  if (status === "halted") return "failed";
  return status;
}

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

function formatDiffBreakdown(job: ChainJob): string | null {
  const { rows_inserted: ins, rows_updated: upd, rows_deleted: del } = job;
  if (ins === null && upd === null && del === null) return null;
  const parts: string[] = [];
  if (ins !== null) parts.push(`${NUMBER_FMT.format(ins)} new`);
  if (upd !== null) parts.push(`${NUMBER_FMT.format(upd)} updated`);
  if (del !== null) parts.push(`${NUMBER_FMT.format(del)} deleted`);
  return parts.length ? parts.join(" · ") : null;
}

function jobSubtitle(job: ChainJob): string {
  const base = `${job.domain} · ${job.mode}`;
  return job.slice ? `${base} · ${job.slice}` : base;
}

interface JobTimelineEntryProps {
  job: ChainJob;
  chainStatus: ChainStatus;
  isLast: boolean;
}

function JobTimelineEntry({
  job,
  chainStatus,
  isLast,
}: JobTimelineEntryProps): JSX.Element {
  const diff =
    job.status === "success" ? formatDiffBreakdown(job) : null;
  const cancelled = job.status === "queued" && chainStatus === "halted";

  return (
    <li className="relative pl-10 pb-4" data-testid={`chain-step-${job.step}`}>
      {!isLast && (
        <span
          aria-hidden="true"
          className="absolute left-3 top-7 bottom-0 w-px bg-border"
        />
      )}
      <span
        className="absolute left-0 top-0 flex h-6 w-6 items-center justify-center rounded-full border border-border bg-card text-xs font-semibold tabular-nums"
        aria-label={`step ${job.step}`}
      >
        {job.step}
      </span>
      <div className="rounded border border-border bg-card px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-medium">{jobSubtitle(job)}</span>
          <JobStatusBadge status={job.status} />
        </div>
        {job.status === "success" && diff && (
          <div className="mt-1 text-xs text-muted-foreground">{diff}</div>
        )}
        {job.status === "failed" && job.error_message && (
          <div className="mt-1 text-xs text-red-600/80 dark:text-red-400/80">
            {job.error_message}
          </div>
        )}
        {job.status === "running" && (
          <div className="mt-1 text-xs text-blue-600 dark:text-blue-400">
            Running…
          </div>
        )}
        {cancelled && (
          <div className="mt-1 text-xs text-muted-foreground">(cancelled)</div>
        )}
      </div>
    </li>
  );
}

export function ChainProgress(props: ChainProgressProps): JSX.Element {
  const { chainId, onClose } = props;

  const query = useQuery<ChainDetail, Error>({
    queryKey: chainKeys.chain(chainId),
    queryFn: () => getChain(chainId),
    refetchInterval: (q) => {
      const data = q.state.data;
      if (!data) return POLL_INTERVAL_MS;
      return ACTIVE_CHAIN_STATUSES.includes(data.status)
        ? POLL_INTERVAL_MS
        : false;
    },
  });

  if (query.isError) {
    return (
      <div className="rounded border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
        Failed to load chain: {query.error?.message ?? "Unknown error"}
      </div>
    );
  }

  const chain = query.data;
  if (!chain) {
    return <p className="text-sm text-muted-foreground">Loading chain…</p>;
  }

  const total = Math.max(1, chain.total_steps);
  const pct = Math.min(100, Math.round((chain.completed_steps / total) * 100));
  const sortedJobs = [...chain.jobs].sort((a, b) => a.step - b.step);
  const shortId = chain.id.slice(0, 8);

  const headerRight = (
    <div className="flex items-center gap-2">
      <JobStatusBadge status={chainStatusToBadge(chain.status)} />
      {onClose && (
        <button
          type="button"
          onClick={onClose}
          aria-label="close chain progress"
          className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          ×
        </button>
      )}
    </div>
  );

  return (
    <CollapsibleSection
      title={`Chain ${shortId}…`}
      storageKey={`integration.chain.${chainId}`}
      headerRight={headerRight}
    >
      <div className="space-y-4">
        <div>
          <div
            className="h-2 w-full overflow-hidden rounded bg-muted"
            role="progressbar"
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${pct}%` }}
            />
          </div>
          <p className="mt-1 text-xs text-muted-foreground tabular-nums">
            {chain.completed_steps} of {chain.total_steps} steps complete
          </p>
        </div>

        <ol className="list-none">
          {sortedJobs.map((job, i) => (
            <JobTimelineEntry
              key={job.job_id}
              job={job}
              chainStatus={chain.status}
              isLast={i === sortedJobs.length - 1}
            />
          ))}
        </ol>

        <div className="flex items-center justify-between gap-2 border-t border-border pt-2 text-xs text-muted-foreground tabular-nums">
          <span>Started {formatStartedAt(chain.started_at)}</span>
          <span>Duration {formatDuration(chain.duration_ms)}</span>
        </div>
      </div>
    </CollapsibleSection>
  );
}
