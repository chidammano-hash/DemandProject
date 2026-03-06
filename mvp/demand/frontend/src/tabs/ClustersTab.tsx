import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  queryKeys,
  fetchClusteringDefaults,
  runClusteringScenario,
  promoteScenario,
  fetchScenarioEstimate,
  fetchScenarioStatus,
  fetchJobDetail,
  fetchScenarioHistory,
  STALE,
} from "@/api/queries";
import type { ClusteringDefaultsPayload, ClusteringScenarioResult } from "@/api/queries";
import type { Job } from "@/types/jobs";
import { getScenarioJobParam, setScenarioJobParam } from "@/hooks/useUrlState";
import { useScenarioNotification } from "@/context/ScenarioNotificationContext";
import { CardContent } from "@/components/ui/card";

import ClusterOverviewPanel from "./clusters/ClusterOverviewPanel";
import WhatIfPanel from "./clusters/WhatIfPanel";
import ScenarioResultsPanel from "./clusters/ScenarioResultsPanel";
import PastScenariosPanel from "./clusters/PastScenariosPanel";

type ClustersTabProps = {
  domain: string;
  onDomainChange: (d: string) => void;
};

export default function ClustersTab({ domain, onDomainChange }: ClustersTabProps) {
  // Ensure dfu domain
  if (domain !== "dfu") onDomainChange("dfu");

  // What-If state
  const [showWhatIf, setShowWhatIf] = useState(false);
  const [scenarioRunning, setScenarioRunning] = useState(false);
  const [pollingScenarioId, setPollingScenarioId] = useState<string | null>(null);
  const [scenarioResult, setScenarioResult] = useState<ClusteringScenarioResult | null>(null);
  const [scenarioError, setScenarioError] = useState<string | null>(null);
  const [scenarioLabel, setScenarioLabel] = useState("A");
  const [nextScenarioIdx, setNextScenarioIdx] = useState(1);
  const [showPromoteConfirm, setShowPromoteConfirm] = useState(false);
  const [scheduledJobId, setScheduledJobId] = useState<string | null>(null);
  const [scenarioQueued, setScenarioQueued] = useState(false);

  const scenarioNotification = useScenarioNotification();
  const currentLabelRef = useRef("A");
  const scenarioResultRef = useRef<HTMLDivElement>(null);

  // Param state
  const [featureParams, setFeatureParams] = useState<ClusteringDefaultsPayload["feature_params"]>({
    time_window_months: 24,
    min_months_history: 1,
  });
  const [modelParams, setModelParams] = useState<ClusteringDefaultsPayload["model_params"]>({
    k_range: [3, 12],
    min_cluster_size_pct: 2.0,
    use_pca: false,
    pca_components: null,
    skip_gap: true,
    all_features: false,
  });
  const [labelParams, setLabelParams] = useState<ClusteringDefaultsPayload["label_params"]>({
    volume_high: 0.75,
    volume_low: 0.25,
    cv_steady: 0.3,
    cv_volatile: 0.8,
    seasonality_threshold: 0.5,
    zero_demand_threshold: 0.2,
  });

  const { data: defaults } = useQuery({
    queryKey: queryKeys.clusteringDefaults(),
    queryFn: fetchClusteringDefaults,
    staleTime: STALE.TEN_MIN,
  });

  const { data: estimate } = useQuery({
    queryKey: queryKeys.scenarioEstimate({ k_min: modelParams.k_range[0], k_max: modelParams.k_range[1], skip_gap: modelParams.skip_gap }),
    queryFn: () => fetchScenarioEstimate({ k_min: modelParams.k_range[0], k_max: modelParams.k_range[1], skip_gap: modelParams.skip_gap }),
    staleTime: STALE.THIRTY_SEC,
    enabled: showWhatIf,
  });

  const { data: statusData, isError: statusPollError } = useQuery({
    queryKey: queryKeys.scenarioStatus(pollingScenarioId ?? ""),
    queryFn: () => fetchScenarioStatus(pollingScenarioId!),
    enabled: !!pollingScenarioId && scenarioRunning,
    refetchInterval: 3000,
    retry: 2,
  });

  const { data: pastScenarios } = useQuery({
    queryKey: queryKeys.scenarioHistory(),
    queryFn: () => fetchScenarioHistory(10),
    staleTime: STALE.THIRTY_SEC,
    enabled: showWhatIf,
  });

  // Sync params from loaded defaults
  useEffect(() => {
    if (defaults) {
      setFeatureParams(defaults.feature_params);
      setModelParams(defaults.model_params);
      setLabelParams(defaults.label_params);
    }
  }, [defaults]);

  // Handle polling completion / error
  useEffect(() => {
    if (!pollingScenarioId) return;
    if (statusPollError) {
      setScenarioRunning(false);
      setScenarioError("Scenario failed — lost connection to background task");
      setPollingScenarioId(null);
      scenarioNotification.failScenario();
      return;
    }
    if (!statusData) return;
    if (statusData.status === "running" && scenarioQueued) setScenarioQueued(false);
    if (statusData.status === "completed" && statusData.result) {
      setScenarioRunning(false);
      setScenarioResult(statusData.result);
      setScenarioLabel(currentLabelRef.current);
      setNextScenarioIdx((c) => c + 1);
      setPollingScenarioId(null);
      scenarioNotification.completeScenario({ id: pollingScenarioId, label: currentLabelRef.current, runtimeSeconds: statusData.runtime_seconds ?? 0, result: statusData.result });
    } else if (statusData.status === "failed") {
      setScenarioRunning(false);
      setScenarioError(statusData.error ?? "Scenario failed");
      setPollingScenarioId(null);
      scenarioNotification.failScenario();
    }
  }, [statusData, statusPollError, pollingScenarioId, scenarioNotification, scenarioQueued]);

  // Auto-load scenario result from URL param (navigation from JobsTab)
  useEffect(() => {
    const jobId = getScenarioJobParam();
    if (!jobId) return;
    setScenarioJobParam(null);
    fetchJobDetail(jobId)
      .then((job: Job) => {
        if (job.job_type !== "cluster_scenario" || job.status !== "completed" || !job.result) {
          setScenarioError("Job is not a completed cluster scenario"); return;
        }
        const jr = job.result as Record<string, unknown>;
        const innerResult = (jr.result ?? jr) as ClusteringScenarioResult["result"];
        if (!innerResult) { setScenarioError("Job result does not contain scenario data"); return; }
        setScenarioResult({ scenario_id: (jr.scenario_id as string) || jobId, status: "completed", runtime_seconds: (jr.runtime_seconds as number) || 0, params: (jr.params as Record<string, unknown>) || {}, result: innerResult });
        setScenarioLabel(job.job_label?.replace("What-If Scenario ", "") || "R");
        setShowWhatIf(true);
        setTimeout(() => scenarioResultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 300);
      })
      .catch((err: unknown) => setScenarioError(`Failed to load scenario: ${err instanceof Error ? err.message : "Unknown error"}`));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRunScenario = useCallback(async () => {
    setScenarioRunning(true); setScenarioError(null); setScenarioResult(null); setScheduledJobId(null); setScenarioQueued(false);
    const label = String.fromCharCode(64 + nextScenarioIdx);
    currentLabelRef.current = label;
    try {
      const response = await runClusteringScenario({ feature_params: featureParams, model_params: modelParams, label_params: labelParams });
      setPollingScenarioId(response.scenario_id);
      if (response.job_id) setScheduledJobId(response.job_id);
      if (response.status === "queued") setScenarioQueued(true);
      scenarioNotification.startScenario(response.scenario_id, label);
    } catch (err) {
      setScenarioRunning(false);
      setScenarioError(err instanceof Error ? err.message : "Unknown error");
    }
  }, [featureParams, modelParams, labelParams, nextScenarioIdx, scenarioNotification]);

  const handleReset = useCallback(() => {
    if (defaults) { setFeatureParams(defaults.feature_params); setModelParams(defaults.model_params); setLabelParams(defaults.label_params); }
  }, [defaults]);

  const handlePromote = useCallback(async () => {
    if (!scenarioResult) return;
    try { await promoteScenario(scenarioResult.scenario_id); setShowPromoteConfirm(false); }
    catch (err) { setScenarioError(err instanceof Error ? err.message : "Promote failed"); setShowPromoteConfirm(false); }
  }, [scenarioResult]);

  return (
    <>
      <ClusterOverviewPanel onDomainChange={onDomainChange} />

      <WhatIfPanel
        showWhatIf={showWhatIf}
        onToggle={() => setShowWhatIf((v) => !v)}
        featureParams={featureParams}
        modelParams={modelParams}
        labelParams={labelParams}
        setFeatureParams={setFeatureParams}
        setModelParams={setModelParams}
        setLabelParams={setLabelParams}
        scenarioRunning={scenarioRunning}
        scenarioQueued={scenarioQueued}
        scenarioError={scenarioError}
        scheduledJobId={scheduledJobId}
        estimate={estimate}
        statusData={statusData}
        onRunScenario={handleRunScenario}
        onReset={handleReset}
      >
        <CardContent className="space-y-4 pt-0">
          <ScenarioResultsPanel
            scenarioResult={scenarioResult}
            scenarioLabel={scenarioLabel}
            scenarioResultRef={scenarioResultRef}
            showPromoteConfirm={showPromoteConfirm}
            onShowPromoteConfirm={() => setShowPromoteConfirm(true)}
            onCancelPromote={() => setShowPromoteConfirm(false)}
            onConfirmPromote={handlePromote}
          />
          <PastScenariosPanel
            pastScenarios={pastScenarios}
            scenarioResultRef={scenarioResultRef}
            onLoadResult={(result, label) => { setScenarioResult(result); setScenarioLabel(label); }}
          />
        </CardContent>
      </WhatIfPanel>
    </>
  );
}
