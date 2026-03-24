import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, CheckCircle, XCircle, Clock } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  tuningChatKeys,
  fetchRunStatus,
  lgbmTuningKeys,
  type RunStatusResult,
} from "@/api/queries";
import { cn } from "@/lib/utils";

interface RunStatusCardProps {
  sessionId: string;
  runId: number;
  /** Pre-fetched result data (from a run_completed message) */
  completedResult?: {
    accuracy_pct?: number;
    wape?: number;
    bias?: number;
  };
  messageType: "run_started" | "run_completed" | "run_failed";
  errorMessage?: string;
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function RunStatusCard({
  sessionId,
  runId,
  completedResult,
  messageType,
  errorMessage,
}: RunStatusCardProps) {
  const queryClient = useQueryClient();
  const isActive = messageType === "run_started";

  // Poll only for active runs
  const { data } = useQuery({
    queryKey: tuningChatKeys.runStatus(sessionId, runId),
    queryFn: () => fetchRunStatus(sessionId, runId),
    refetchInterval: isActive ? 10_000 : false,
    enabled: isActive,
  });

  // Local elapsed timer for smoother UX
  const [localElapsed, setLocalElapsed] = useState(0);
  useEffect(() => {
    if (!isActive) return;
    const iv = setInterval(() => setLocalElapsed((e) => e + 1), 1000);
    return () => clearInterval(iv);
  }, [isActive]);

  // When run completes, invalidate the runs list so the table refreshes
  useEffect(() => {
    if (data?.status === "completed" || data?.status === "failed") {
      queryClient.invalidateQueries({ queryKey: lgbmTuningKeys.runs() });
    }
  }, [data?.status, queryClient]);

  const fallback = messageType === "run_started" ? "running" : messageType.replace("run_", "");
  const status = data?.status ?? fallback;
  const elapsed = data?.elapsed_seconds ?? localElapsed;
  const results = data?.results ?? completedResult;

  return (
    <div
      className={cn(
        "border rounded-lg p-3 my-2",
        status === "completed"
          ? "border-emerald-500/30 bg-emerald-50 dark:bg-emerald-950/20"
          : status === "failed"
            ? "border-red-500/30 bg-red-50 dark:bg-red-950/20"
            : "border-amber-500/30 bg-amber-50 dark:bg-amber-950/20",
      )}
    >
      <div className="flex items-center gap-2 mb-2">
        {status === "running" && (
          <Loader2 className="h-4 w-4 text-amber-600 dark:text-amber-400 animate-spin" />
        )}
        {status === "completed" && (
          <CheckCircle className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
        )}
        {status === "failed" && (
          <XCircle className="h-4 w-4 text-red-600 dark:text-red-400" />
        )}
        <span className="text-sm font-medium text-foreground">
          Run #{runId}
        </span>
        <Badge
          variant="outline"
          className={cn(
            "text-xs ml-auto",
            status === "completed"
              ? "text-emerald-700 dark:text-emerald-400"
              : status === "failed"
                ? "text-red-700 dark:text-red-400"
                : "text-amber-700 dark:text-amber-400",
          )}
        >
          {status}
        </Badge>
      </div>

      {status === "running" && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Clock className="h-3 w-3" />
          <span>{formatElapsed(elapsed)}</span>
          <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
            <div className="h-full bg-amber-500/50 rounded-full animate-pulse w-2/3" />
          </div>
        </div>
      )}

      {status === "completed" && results && (
        <div className="grid grid-cols-3 gap-3 mt-2">
          <div className="text-center">
            <div className="text-xs text-muted-foreground">Accuracy</div>
            <div className="text-sm font-mono text-emerald-700 dark:text-emerald-300">
              {results.accuracy_pct != null ? Number(results.accuracy_pct).toFixed(2) : "--"}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-muted-foreground">WAPE</div>
            <div className="text-sm font-mono text-foreground">
              {results.wape != null ? Number(results.wape).toFixed(2) : "--"}
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-muted-foreground">Bias</div>
            <div className="text-sm font-mono text-foreground">
              {results.bias != null ? Number(results.bias).toFixed(4) : "--"}
            </div>
          </div>
        </div>
      )}

      {status === "failed" && errorMessage && (
        <p className="text-xs text-red-600 dark:text-red-400 mt-1 truncate" title={errorMessage}>
          {errorMessage}
        </p>
      )}
    </div>
  );
}
