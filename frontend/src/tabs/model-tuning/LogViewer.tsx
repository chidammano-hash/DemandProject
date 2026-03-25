/**
 * LogViewer -- Slide-over panel for viewing experiment execution logs.
 *
 * Polls GET /model-tuning/{model}/experiments/{runId}/logs every 2s while
 * the experiment is running. Stops when status transitions to completed/failed.
 * Supports auto-scroll with scroll-lock toggle, copy, and download.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  X,
  Copy,
  Download,
  Lock,
  Unlock,
  Loader2,
  CheckCircle,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import type { ModelType } from "@/api/queries";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface LogViewerProps {
  model: ModelType;
  runId: number;
  open: boolean;
  onClose: () => void;
}

interface LogResponse {
  run_id: number;
  status: "running" | "completed" | "failed" | "queued";
  log: string;
  offset: number;
  started_at: string | null;
  completed_at: string | null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const MODEL_PREFIX: Record<ModelType, string> = {
  lgbm: "/model-tuning/lgbm",
  catboost: "/model-tuning/catboost",
  xgboost: "/model-tuning/xgboost",
};

function formatDuration(startedAt: string | null): string {
  if (!startedAt) return "--";
  const start = new Date(startedAt).getTime();
  const elapsed = Math.max(0, Date.now() - start);
  const totalSec = Math.floor(elapsed / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  if (min > 0) return `${min}m ${sec}s`;
  return `${sec}s`;
}

async function fetchLogs(
  model: ModelType,
  runId: number,
): Promise<LogResponse> {
  const res = await fetch(
    `${MODEL_PREFIX[model]}/experiments/${runId}/logs`,
  );
  if (!res.ok) {
    throw new Error(`Failed to fetch logs: ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function LogViewer({ model, runId, open, onClose }: LogViewerProps) {
  const logRef = useRef<HTMLPreElement>(null);
  const [scrollLocked, setScrollLocked] = useState(false);
  const [copied, setCopied] = useState(false);
  const [durationStr, setDurationStr] = useState("--");

  // Fetch logs with polling
  const { data, isLoading, isError, error } = useQuery<LogResponse>({
    queryKey: ["model-tuning-logs", model, runId],
    queryFn: () => fetchLogs(model, runId),
    enabled: open,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed") return false;
      return 2000;
    },
    staleTime: 0,
  });

  const isRunning = data?.status === "running" || data?.status === "queued";
  const logText = data?.log ?? "";

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (!scrollLocked && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logText, scrollLocked]);

  // Duration counter while running
  useEffect(() => {
    if (!isRunning || !data?.started_at) return;
    setDurationStr(formatDuration(data.started_at));
    const timer = setInterval(() => {
      setDurationStr(formatDuration(data.started_at));
    }, 1000);
    return () => clearInterval(timer);
  }, [isRunning, data?.started_at]);

  // If completed, compute final duration
  useEffect(() => {
    if (!isRunning && data?.started_at && data?.completed_at) {
      const start = new Date(data.started_at).getTime();
      const end = new Date(data.completed_at).getTime();
      const elapsed = Math.max(0, end - start);
      const totalSec = Math.floor(elapsed / 1000);
      const min = Math.floor(totalSec / 60);
      const sec = totalSec % 60;
      setDurationStr(min > 0 ? `${min}m ${sec}s` : `${sec}s`);
    }
  }, [isRunning, data?.started_at, data?.completed_at]);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(logText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard not available
    }
  }, [logText]);

  const handleDownload = useCallback(() => {
    const blob = new Blob([logText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `tuning_run_${runId}_${model}.log`;
    a.click();
    URL.revokeObjectURL(url);
  }, [logText, runId, model]);

  if (!open) return null;

  // Status indicator
  const StatusIcon =
    data?.status === "completed"
      ? CheckCircle
      : data?.status === "failed"
        ? XCircle
        : Loader2;

  const statusColor =
    data?.status === "completed"
      ? "text-emerald-600 dark:text-emerald-400"
      : data?.status === "failed"
        ? "text-red-600 dark:text-red-400"
        : "text-amber-600 dark:text-amber-400";

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/40 backdrop-blur-sm">
      {/* Backdrop click to close */}
      <div className="flex-1" onClick={onClose} aria-hidden="true" />

      {/* Slide-over panel */}
      <div className="w-full max-w-2xl bg-card border-l border-border shadow-2xl flex flex-col animate-in slide-in-from-right duration-200">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <div>
            <p className="text-sm font-semibold text-foreground">
              Experiment Logs -- Run #{runId}
            </p>
            <div className="flex items-center gap-3 text-xs text-muted-foreground mt-0.5">
              <span className="flex items-center gap-1">
                <StatusIcon
                  className={cn(
                    "h-3 w-3",
                    statusColor,
                    data?.status === "running" && "animate-spin",
                  )}
                />
                {data?.status ?? "loading"}
              </span>
              <span>Duration: {durationStr}</span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1.5 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Log content */}
        <div className="flex-1 overflow-hidden relative">
          {isLoading && (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              <span className="ml-2 text-sm text-muted-foreground">
                Loading logs...
              </span>
            </div>
          )}

          {isError && (
            <div className="p-5 text-sm text-destructive">
              Failed to load logs: {(error as Error).message}
            </div>
          )}

          {!isLoading && !isError && logText.length === 0 && (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <p className="text-sm text-muted-foreground">
                No logs available yet. Logs will appear once the experiment starts.
              </p>
            </div>
          )}

          {!isLoading && logText.length > 0 && (
            <pre
              ref={logRef}
              className="h-full overflow-y-auto p-4 text-xs font-mono leading-relaxed text-foreground/90 whitespace-pre-wrap break-words bg-muted/20"
            >
              {logText}
            </pre>
          )}
        </div>

        {/* Footer toolbar */}
        <div className="flex items-center justify-between border-t border-border px-5 py-2.5">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs gap-1"
              onClick={() => setScrollLocked(!scrollLocked)}
              title={scrollLocked ? "Unlock auto-scroll" : "Lock scroll position"}
            >
              {scrollLocked ? (
                <Lock className="h-3 w-3" />
              ) : (
                <Unlock className="h-3 w-3" />
              )}
              {scrollLocked ? "Scroll Locked" : "Auto-scroll"}
            </Button>
          </div>

          <div className="flex items-center gap-1.5">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs gap-1"
              onClick={handleCopy}
              disabled={logText.length === 0}
            >
              <Copy className="h-3 w-3" />
              {copied ? "Copied" : "Copy"}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs gap-1"
              onClick={handleDownload}
              disabled={logText.length === 0}
            >
              <Download className="h-3 w-3" />
              Download
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
