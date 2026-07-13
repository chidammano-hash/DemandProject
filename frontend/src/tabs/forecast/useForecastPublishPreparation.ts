import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { backtestMgmtKeys } from "@/api/queries/backtest-management";
import { fetchJobs, runNamedPipeline } from "@/api/queries/jobs";
import { toast } from "@/components/Toaster";
import { formatApiError } from "@/lib/formatApiError";
import type { Job } from "@/types/jobs";
import type { SnapshotRosterReadiness } from "@/api/queries/backtest-management";

import { forecastPanelKeys } from "./forecastPanelShared";

const EMPTY_JOBS: Job[] = [];
type ReadinessPipeline = NonNullable<SnapshotRosterReadiness["action_pipeline"]>;

export function resolvePublishPipelineOutcome(
  jobs: Job[],
  pipelineId: string
): "completed" | "failed" | "cancelled" | null {
  const matching = jobs.filter((job) => job.pipeline_id === pipelineId);
  const failed = matching.find((job) => job.status === "failed" || job.status === "cancelled");
  if (failed?.status === "failed" || failed?.status === "cancelled") return failed.status;

  const finalStep = matching.find((job) => {
    if (job.status !== "completed") return false;
    const step = Number(job.pipeline_step ?? job.params.__pipeline_step);
    const total = Number(job.params.__pipeline_total_steps);
    return Number.isFinite(step) && Number.isFinite(total) && step >= total;
  });
  return finalStep ? "completed" : null;
}

export function useForecastPublishPreparation() {
  const queryClient = useQueryClient();
  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [activePipeline, setActivePipeline] = useState<ReadinessPipeline | null>(null);
  const [isPreparing, setIsPreparing] = useState(false);

  const { data } = useQuery({
    queryKey: forecastPanelKeys.publishPipeline(pipelineId ?? "idle"),
    queryFn: () => fetchJobs({ limit: 100 }),
    enabled: pipelineId !== null,
    staleTime: 0,
    refetchInterval: isPreparing ? 5_000 : false,
  });

  useEffect(() => {
    if (!pipelineId) return;
    const outcome = resolvePublishPipelineOutcome(data?.jobs ?? EMPTY_JOBS, pipelineId);
    if (!outcome) return;

    setIsPreparing(false);
    setPipelineId(null);
    setActivePipeline(null);
    queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.trainingStatus });
    queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.stagingSummary });
    queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.snapshotRosterReadiness });
    queryClient.invalidateQueries({ queryKey: forecastPanelKeys.jobs(0) });

    if (outcome === "completed") {
      toast.success(
        activePipeline === "model-refresh"
          ? "Model Refresh completed; governed champion evidence is being revalidated."
          : "Release preparation completed; publish evidence is being revalidated."
      );
    } else {
      const label = activePipeline === "model-refresh" ? "Model Refresh" : "Forecast Publish";
      toast.error(`${label} ${outcome}. Open Jobs for the failed step and retry.`);
    }
  }, [activePipeline, data?.jobs, pipelineId, queryClient]);

  async function preparePublish(pipeline: ReadinessPipeline = "forecast-publish") {
    if (isPreparing) return;
    setIsPreparing(true);
    setActivePipeline(pipeline);
    try {
      const submitted = await runNamedPipeline(pipeline);
      setPipelineId(submitted.pipeline_id);
      toast.info(
        pipeline === "model-refresh"
          ? "Model Refresh started. Tuning, five governed backtests, and champion selection will run in sequence."
          : "Forecast Publish started. Model fitting, champion generation, and top-three evidence will run in sequence."
      );
    } catch (error) {
      setIsPreparing(false);
      setPipelineId(null);
      setActivePipeline(null);
      toast.error(formatApiError(error));
    }
  }

  return { isPreparingPublish: isPreparing, preparePublish };
}
