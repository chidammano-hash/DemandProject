/**
 * ActiveJobsPanel — live monitoring of running and queued jobs.
 * Displays animated progress bars, elapsed timers, and cancel buttons.
 */
import { useEffect, useState } from "react";
import { Timer } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Job } from "@/types/jobs";
import { GROUP_CONFIG } from "@/types/jobs";
import { formatDuration, jobDuration, getGroupKey, STATUS_CONFIG } from "./jobsShared";

// ---------------------------------------------------------------------------
// StatusBadge
// ---------------------------------------------------------------------------
function StatusBadge({ status }: { status: string }) {
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

// ---------------------------------------------------------------------------
// ActiveJobCard — single live job card
// ---------------------------------------------------------------------------
function ActiveJobCard({ job, onCancel }: { job: Job; onCancel: (id: string) => void }) {
  const [elapsed, setElapsed] = useState("");
  const groupKey = getGroupKey(job.job_type);
  const cfg = GROUP_CONFIG[groupKey] || GROUP_CONFIG.clustering;

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

      {/* Progress bar */}
      <div className="w-full bg-background/50 rounded-full h-2 mb-2 overflow-hidden">
        <div
          className={cn(
            "h-2 rounded-full transition-all duration-700 ease-out",
            job.status === "running"
              ? "bg-gradient-to-r from-blue-500 to-blue-400"
              : "bg-yellow-500",
          )}
          style={{ width: `${Math.max(job.progress_pct, 3)}%` }}
        />
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="font-medium">{job.progress_pct}%</span>
          <span>{job.progress_msg || "Waiting..."}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs tabular-nums font-medium text-muted-foreground">
            <Timer className="h-3 w-3 inline mr-0.5" />
            {elapsed || jobDuration(job)}
          </span>
          {(job.status === "running" || job.status === "queued") && (
            <button
              onClick={() => onCancel(job.job_id)}
              className="rounded-md border border-destructive/30 px-2 py-0.5 text-[10px] font-medium text-destructive hover:bg-destructive/10 transition-colors"
            >
              Cancel
            </button>
          )}
        </div>
      </div>
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
