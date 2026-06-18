/**
 * ActiveJobsPanel — live monitoring of running and queued jobs.
 * Displays animated progress bars, elapsed timers, kill buttons, and a
 * collapsible log panel that fetches persistent execution logs from the backend.
 */
import { useEffect, useRef, useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { Timer, ScrollText } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Job } from "@/types/jobs";
import { GROUP_CONFIG } from "@/types/jobs";
import { fetchJobLogs, queryKeys } from "@/api/queries";
import { formatDuration, jobDuration, getGroupKey } from "./jobsShared";
import { StatusBadge } from "./StatusBadge";

// ---------------------------------------------------------------------------
// ActiveJobCard — single live job card
// ---------------------------------------------------------------------------

function ActiveJobCard({ job, onCancel }: { job: Job; onCancel: (id: string) => void }) {
  const [elapsed, setElapsed] = useState("");
  const [showLogs, setShowLogs] = useState(false);
  const [confirmKill, setConfirmKill] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);
  const groupKey = getGroupKey(job.job_type);
  const cfg = GROUP_CONFIG[groupKey] || GROUP_CONFIG.clustering;

  // Fetch persistent logs from the backend (polls every 3s when panel is open)
  const { data: logsData } = useQuery({
    queryKey: queryKeys.jobLogs(job.job_id),
    queryFn: () => fetchJobLogs(job.job_id),
    refetchInterval: showLogs ? 3_000 : false,
    enabled: showLogs,
  });

  const logText = logsData?.log ?? "";
  const logLines = logText ? logText.split("\n").filter(Boolean) : [];

  // Elapsed timer
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

  // Auto-scroll log panel to bottom
  useEffect(() => {
    if (showLogs && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logLines.length, showLogs]);

  const handleKill = useCallback(() => {
    if (confirmKill) {
      onCancel(job.job_id);
      setConfirmKill(false);
    } else {
      setConfirmKill(true);
      // Auto-reset confirmation after 3s
      setTimeout(() => setConfirmKill(false), 3000);
    }
  }, [confirmKill, onCancel, job.job_id]);

  // Many long jobs (e.g. backtests) stream log messages but never report a
  // numeric percentage, so progress_pct stays 0 the whole run. Show an
  // animated "working" bar in that case instead of a frozen 0% bar.
  const indeterminate = job.status === "running" && (job.progress_pct ?? 0) <= 0;

  return (
    <div className={cn("rounded-xl border-2 p-4 transition-all", cfg.borderColor, cfg.bgColor)}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <div
            className={cn(
              "h-2 w-2 rounded-full animate-pulse",
              job.status === "running" ? "bg-blue-500" : "bg-yellow-500",
            )}
          />
          <div className="min-w-0">
            <p className="text-sm font-semibold text-foreground truncate">{job.job_label}</p>
            <p className="text-[10px] text-muted-foreground">{job.job_type}</p>
          </div>
        </div>
        <StatusBadge status={job.status} />
      </div>

      {/* Progress bar — determinate when the job reports a %, else an animated
          "working" indicator so a 0%-reporting job doesn't look frozen. */}
      <div className="w-full bg-background/50 rounded-full h-2 mb-2 overflow-hidden">
        <div
          className={cn(
            "h-2 rounded-full",
            indeterminate
              ? "w-2/5 bg-gradient-to-r from-blue-500 to-blue-400 animate-pulse"
              : cn(
                  "transition-all duration-700 ease-out",
                  job.status === "running"
                    ? "bg-gradient-to-r from-blue-500 to-blue-400"
                    : "bg-yellow-500",
                ),
          )}
          style={indeterminate ? undefined : { width: `${Math.max(job.progress_pct, 3)}%` }}
        />
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="font-medium">{indeterminate ? "running…" : `${job.progress_pct}%`}</span>
          <span>{job.progress_msg || "Waiting..."}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs tabular-nums font-medium text-muted-foreground">
            <Timer className="h-3 w-3 inline mr-0.5" />
            {elapsed || jobDuration(job)}
          </span>
          <button
            onClick={() => setShowLogs((v) => !v)}
            className={cn(
              "flex items-center gap-1 rounded-md border px-2 py-0.5 text-[10px] font-medium transition-colors",
              showLogs
                ? "border-primary/40 bg-primary/10 text-primary"
                : "border-border text-muted-foreground hover:bg-muted",
            )}
            title="Toggle logs"
          >
            <ScrollText className="h-3 w-3" />
            Logs{logLines.length > 0 && ` (${logLines.length})`}
          </button>
          {(job.status === "running" || job.status === "queued") && (
            <button
              onClick={handleKill}
              className={cn(
                "rounded-md border px-2 py-0.5 text-[10px] font-medium transition-colors",
                confirmKill
                  ? "border-destructive bg-destructive/20 text-destructive animate-pulse"
                  : "border-destructive/30 text-destructive hover:bg-destructive/10",
              )}
            >
              {confirmKill ? "Confirm Kill" : "Kill"}
            </button>
          )}
        </div>
      </div>

      {/* Log panel — persistent logs from backend */}
      {showLogs && (
        <div className="mt-3 rounded-md border border-border bg-black/80 dark:bg-black/60 px-3 py-2 max-h-48 overflow-y-auto font-mono text-[10px] leading-relaxed whitespace-pre-wrap">
          {logLines.length === 0 ? (
            <span className="text-muted-foreground/60">Waiting for first log entry...</span>
          ) : (
            logLines.map((line, i) => (
              <div key={i} className="text-green-300">{line}</div>
            ))
          )}
          <div ref={logEndRef} />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ActiveJobsPanel
// ---------------------------------------------------------------------------
export interface ActiveJobsPanelProps {
  activeJobs: Job[];
  onCancel: (jobId: string) => void;
}

export function ActiveJobsPanel({ activeJobs, onCancel }: ActiveJobsPanelProps) {
  if (activeJobs.length === 0) return null;

  return (
    <section>
      <h3 className="text-sm font-semibold text-foreground/80 uppercase tracking-wider mb-3 flex items-center gap-2">
        <div className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
        Active Jobs ({activeJobs.length})
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {activeJobs.map((job) => (
          <ActiveJobCard key={job.job_id} job={job} onCancel={onCancel} />
        ))}
      </div>
    </section>
  );
}
