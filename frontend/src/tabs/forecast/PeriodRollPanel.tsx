import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Archive,
  Calculator,
  CheckCircle2,
  DatabaseZap,
  Loader2,
  Play,
  Trash2,
} from "lucide-react";

import { fetchJobs, jobKeys, runNamedPipeline } from "@/api/queries/jobs";
import { toast } from "@/components/Toaster";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatApiError } from "@/lib/formatApiError";
import type { Job, JobStatus } from "@/types/jobs";

const OPERATIONS = [
  { label: "Calculate Snapshot KPIs", icon: Calculator },
  { label: "Prepare Forecast Snapshot Contenders", icon: DatabaseZap },
  { label: "Archive Forecast Snapshot", icon: Archive },
  { label: "Clean Forecast Staging", icon: Trash2 },
] as const;

function operationStatus(jobs: Job[], position: number): JobStatus | "pending" {
  return (
    jobs.find((job) => Number(job.pipeline_step ?? job.params.__pipeline_step) === position)
      ?.status ?? "pending"
  );
}

export function PeriodRollPanel() {
  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  const { data } = useQuery({
    queryKey: jobKeys.periodRoll(pipelineId ?? "idle"),
    queryFn: () => fetchJobs({ limit: 100 }),
    enabled: pipelineId !== null,
    staleTime: 0,
    refetchInterval: isPolling ? 5_000 : false,
  });
  const jobs = useMemo(
    () => (data?.jobs ?? []).filter((job) => job.pipeline_id === pipelineId),
    [data?.jobs, pipelineId]
  );
  const completed = jobs.length > 0 && operationStatus(jobs, OPERATIONS.length) === "completed";
  const failed = jobs.some((job) => job.status === "failed" || job.status === "cancelled");
  const isRunning = isSubmitting || isPolling;
  const statusLabel = failed
    ? "Needs attention"
    : completed
      ? "Completed"
      : isRunning
        ? "Running"
        : "Ready";

  useEffect(() => {
    if (completed || failed) setIsPolling(false);
  }, [completed, failed]);

  async function handleRun() {
    if (isRunning) return;
    setIsSubmitting(true);
    setIsPolling(false);
    try {
      const result = await runNamedPipeline("period-roll");
      setPipelineId(result.pipeline_id);
      setIsPolling(true);
      toast.info(
        result.status === "already_running"
          ? "Period Roll is already running; showing its current progress."
          : "Period Roll started. Each operation runs durably in order."
      );
    } catch (error) {
      toast.error(formatApiError(error));
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div>
              <CardTitle>Period Roll · Score Prior + Archive Current</CardTitle>
              <p className="mt-1 text-sm font-medium text-emerald-600 dark:text-emerald-400">
                {statusLabel}
              </p>
            </div>
            <Button onClick={handleRun} disabled={isRunning}>
              {isRunning ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : completed ? (
                <CheckCircle2 className="mr-2 h-4 w-4" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              {isRunning
                ? "Period Roll Running"
                : completed
                  ? "Run Period Roll Again"
                  : "Run Period Roll"}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <p className="max-w-4xl text-sm text-muted-foreground">
            Beginning-of-month control: score any newly closed lags from actuals, prepare and
            archive the current planning month&apos;s champion-plus-three snapshot, then remove only
            staging generations that reconcile to an immutable archive. Safe to retry for an already
            archived planning month.
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-3 md:grid-cols-2">
        {OPERATIONS.map(({ label, icon: Icon }, index) => {
          const state = operationStatus(jobs, index + 1);
          return (
            <Card key={label}>
              <CardContent className="flex items-center gap-3 p-4">
                <Icon className="h-5 w-5 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">{label}</p>
                  <p className="text-xs capitalize text-muted-foreground">{state}</p>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
