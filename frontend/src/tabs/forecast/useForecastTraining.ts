import { useCallback, useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { backtestMgmtKeys, submitTraining } from "@/api/queries/backtest-management";
import { queryKeys } from "@/api/queries/core";
import { fetchJobDetail } from "@/api/queries/jobs";
import { toast } from "@/components/Toaster";
import { extractStatus, formatApiError } from "@/lib/formatApiError";

import {
  loadPendingTrainingRun,
  persistPendingTrainingRun,
  resolveTrainingTerminalOutcome,
  type PendingTrainingRun,
} from "./forecastTrainingRun";

export function useForecastTraining() {
  const queryClient = useQueryClient();
  const [pendingRun, setPendingRun] = useState<PendingTrainingRun | null>(() =>
    loadPendingTrainingRun()
  );
  const [isTraining, setIsTraining] = useState(pendingRun !== null);
  const [trainingModelId, setTrainingModelId] = useState<string | null>(
    pendingRun?.modelId ?? null
  );
  const pendingJobId = pendingRun?.jobId ?? null;

  const { data: trainingJob, error: lookupError } = useQuery({
    queryKey: queryKeys.jobDetail(pendingJobId ?? "pending-training"),
    queryFn: () => fetchJobDetail(pendingJobId!),
    enabled: pendingJobId !== null,
    staleTime: 0,
    retry: 2,
    refetchInterval: (query) =>
      resolveTrainingTerminalOutcome(query.state.data?.status) ? false : 5_000,
  });

  const clearPendingRun = useCallback(() => {
    setPendingRun(null);
    persistPendingTrainingRun(null);
    setIsTraining(false);
    setTrainingModelId(null);
  }, []);

  async function train(modelId: string) {
    setIsTraining(true);
    setTrainingModelId(modelId);
    try {
      const submitted = await submitTraining(modelId);
      const run = { jobId: submitted.job_id, modelId };
      setPendingRun(run);
      persistPendingTrainingRun(run);
    } catch (error) {
      toast.error(formatApiError(error));
      clearPendingRun();
    }
  }

  async function trainAll() {
    setIsTraining(true);
    setTrainingModelId("__all__");
    try {
      const submitted = await submitTraining("all");
      const run = { jobId: submitted.job_id, modelId: "__all__" };
      setPendingRun(run);
      persistPendingTrainingRun(run);
      toast.success("Production model training queued.");
    } catch (error) {
      toast.error(formatApiError(error));
      clearPendingRun();
    }
  }

  useEffect(() => {
    if (!pendingRun || !trainingJob) return;
    const outcome = resolveTrainingTerminalOutcome(trainingJob.status);
    if (!outcome) return;

    clearPendingRun();
    queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.trainingStatus });
    if (outcome === "completed") {
      toast.success("Production model training completed; refreshing artifact status.");
    } else {
      toast.error(
        `Production model training ${outcome}. Open Jobs for details, then retry from Step 1.`
      );
    }
  }, [clearPendingRun, pendingRun, queryClient, trainingJob]);

  useEffect(() => {
    if (!pendingRun || extractStatus(lookupError) !== 404) return;
    clearPendingRun();
    toast.error(
      "The submitted training job is no longer in history. Verify artifacts in Step 1 before retrying."
    );
  }, [clearPendingRun, lookupError, pendingRun]);

  return { isTraining, trainingModelId, train, trainAll };
}
