/**
 * ActiveJobsPanel — live monitoring of running and queued jobs.
 * Displays animated progress bars, elapsed timers, cancel buttons, and a
 * collapsible log panel that accumulates progress messages in real-time.
 */
import { useEffect, useRef, useState } from "react";
import { Timer, ScrollText } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Job } from "@/types/jobs";
import { GROUP_CONFIG } from "@/types/jobs";
import { formatDuration, jobDuration, getGroupKey } from "./jobsShared";
import { StatusBadge } from "./StatusBadge";

const MAX_LOG_ENTRIES = 500;

// ---------------------------------------------------------------------------
// ActiveJobCard — single live job card
// ---------------------------------------------------------------------------
interface LogEntry {
  ts: string; // HH:MM:SS
  pct: number;
  msg: string;
}

function nowTs() {
  return new Date().toLocaleTimeString("en-US", { hour12: false });
}

function ActiveJobCard({ job, onCancel }: { job: Job; onCancel: (id: string) => void }) {
  const [elapsed, setElapsed] = useState("");
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const lastMsgRef = useRef<string>("");
  const logEndRef = useRef<HTMLDivElement>(null);
  const groupKey = getGroupKey(job.job_type);
  const cfg = GROUP_CONFIG[groupKey] || GROUP_CONFIG.clustering;

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

  // Accumulate progress_msg changes into a log (capped to MAX_LOG_ENTRIES)
  useEffect(() => {
    const msg = job.progress_msg || "";
    if (msg && msg !== lastMsgRef.current) {
      lastMsgRef.current = msg;
      setLogs((prev) => {
        const next = [...prev, { ts: nowTs(), pct: job.progress_pct ?? 0, msg }];
        return next.length > MAX_LOG_ENTRIES ? next.slice(-MAX_LOG_ENTRIES) : next;
      });
    }
  }, [job.progress_msg, job.progress_pct]);

  // Auto-scroll log panel to bottom
  useEffect(() => {
    if (showLogs) {
      logEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, showLogs]);

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
            Logs{logs.length > 0 && ` (${logs.length})`}
          </button>
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

      {/* Log panel */}
      {showLogs && (
        <div className="mt-3 rounded-md border border-border bg-black/80 dark:bg-black/60 px-3 py-2 max-h-48 overflow-y-auto font-mono text-[10px] leading-relaxed">
          {logs.length === 0 ? (
            <span className="text-muted-foreground/60">Waiting for first log entry…</span>
          ) : (
            logs.map((entry, i) => (
              <div key={i} className="flex gap-2">
                <span className="text-muted-foreground/50 shrink-0">{entry.ts}</span>
                <span className="text-green-400/70 shrink-0 w-7 text-right">{entry.pct}%</span>
                <span className="text-green-300">{entry.msg}</span>
              </div>
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
