import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  queryKeys,
  fetchDfuClusters,
  fetchClusterProfiles,
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
import type { ClusterInfo } from "@/types";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
import { cn } from "@/lib/utils";
import { useScenarioNotification } from "@/context/ScenarioNotificationContext";
import { ScenarioCharts } from "@/components/ScenarioCharts";

import { ChevronDown, ChevronRight } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

type ClustersTabProps = {
  domain: string;
  onDomainChange: (d: string) => void;
};

// ---------------------------------------------------------------------------
// Param input helper
// ---------------------------------------------------------------------------
function ParamInput({
  label,
  value,
  onChange,
  step,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="font-semibold text-muted-foreground">{label}</span>
      <input
        type="number"
        className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm tabular-nums"
        value={value}
        step={step ?? 1}
        min={min}
        max={max}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

// ---------------------------------------------------------------------------
// ClustersTab
// ---------------------------------------------------------------------------
export default function ClustersTab({ domain, onDomainChange }: ClustersTabProps) {
  const [clusterSource, setClusterSource] = useState<"ml" | "source">("ml");
  const [selectedCluster, setSelectedCluster] = useState("");
  const [showClusterViz, setShowClusterViz] = useState(false);

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
  const [expandedHistoryId, setExpandedHistoryId] = useState<string | null>(null);
  const [historyResult, setHistoryResult] = useState<ClusteringScenarioResult | null>(null);

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

  // Ensure we're on the dfu domain
  if (domain !== "dfu") {
    onDomainChange("dfu");
  }

  // Existing queries
  const { data: clustersPayload } = useQuery({
    queryKey: queryKeys.dfuClusters(clusterSource),
    queryFn: () => fetchDfuClusters(clusterSource),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: profilesPayload } = useQuery({
    queryKey: queryKeys.clusterProfiles(),
    queryFn: fetchClusterProfiles,
    staleTime: STALE.TEN_MIN,
    enabled: clusterSource === "ml",
  });

  // Load clustering defaults
  const { data: defaults } = useQuery({
    queryKey: queryKeys.clusteringDefaults(),
    queryFn: fetchClusteringDefaults,
    staleTime: STALE.TEN_MIN,
  });

  // Runtime estimate query
  const { data: estimate } = useQuery({
    queryKey: queryKeys.scenarioEstimate({
      k_min: modelParams.k_range[0],
      k_max: modelParams.k_range[1],
      skip_gap: modelParams.skip_gap,
    }),
    queryFn: () => fetchScenarioEstimate({
      k_min: modelParams.k_range[0],
      k_max: modelParams.k_range[1],
      skip_gap: modelParams.skip_gap,
    }),
    staleTime: STALE.THIRTY_SEC,
    enabled: showWhatIf,
  });

  // Status polling query
  const { data: statusData, isError: statusPollError } = useQuery({
    queryKey: queryKeys.scenarioStatus(pollingScenarioId ?? ""),
    queryFn: () => fetchScenarioStatus(pollingScenarioId!),
    enabled: !!pollingScenarioId && scenarioRunning,
    refetchInterval: 3000,
    retry: 2,
  });

  // Handle status polling completion or error
  useEffect(() => {
    if (!pollingScenarioId) return;

    // Handle fetch errors (e.g. 404 when result not found on disk)
    if (statusPollError) {
      setScenarioRunning(false);
      setScenarioError("Scenario failed — lost connection to background task");
      setPollingScenarioId(null);
      scenarioNotification.failScenario();
      return;
    }

    if (!statusData) return;
    // Clear queued flag once scenario actually starts running
    if (statusData.status === "running" && scenarioQueued) {
      setScenarioQueued(false);
    }
    if (statusData.status === "completed" && statusData.result) {
      setScenarioRunning(false);
      setScenarioResult(statusData.result);
      setScenarioLabel(currentLabelRef.current);
      setNextScenarioIdx((c) => c + 1);
      setPollingScenarioId(null);
      scenarioNotification.completeScenario({
        id: pollingScenarioId,
        label: currentLabelRef.current,
        runtimeSeconds: statusData.runtime_seconds ?? 0,
        result: statusData.result,
      });
    } else if (statusData.status === "failed") {
      setScenarioRunning(false);
      setScenarioError(statusData.error ?? "Scenario failed");
      setPollingScenarioId(null);
      scenarioNotification.failScenario();
    }
  }, [statusData, statusPollError, pollingScenarioId, scenarioNotification, scenarioQueued]);

  // Sync params from loaded defaults
  useEffect(() => {
    if (defaults) {
      setFeatureParams(defaults.feature_params);
      setModelParams(defaults.model_params);
      setLabelParams(defaults.label_params);
    }
  }, [defaults]);

  // Past scenarios query (last 10 completed cluster_scenario jobs)
  const { data: pastScenarios } = useQuery({
    queryKey: queryKeys.scenarioHistory(),
    queryFn: () => fetchScenarioHistory(10),
    staleTime: STALE.THIRTY_SEC,
    enabled: showWhatIf,
  });

  // Auto-load scenario result from URL param (navigation from JobsTab)
  useEffect(() => {
    const jobId = getScenarioJobParam();
    if (!jobId) return;
    setScenarioJobParam(null); // clear immediately to avoid re-fetch

    fetchJobDetail(jobId)
      .then((job: Job) => {
        if (job.job_type !== "cluster_scenario" || job.status !== "completed" || !job.result) {
          setScenarioError("Job is not a completed cluster scenario");
          return;
        }
        // job.result shape: { scenario_id, status, runtime_seconds, params, result: { optimal_k, ... } }
        const jr = job.result as Record<string, unknown>;
        const innerResult = (jr.result ?? jr) as ClusteringScenarioResult["result"];
        if (!innerResult) {
          setScenarioError("Job result does not contain scenario data");
          return;
        }
        const transformed: ClusteringScenarioResult = {
          scenario_id: (jr.scenario_id as string) || jobId,
          status: "completed",
          runtime_seconds: (jr.runtime_seconds as number) || 0,
          params: (jr.params as Record<string, unknown>) || {},
          result: innerResult,
        };
        setScenarioResult(transformed);
        setScenarioLabel(job.job_label?.replace("What-If Scenario ", "") || "R");
        setShowWhatIf(true);
        setTimeout(() => {
          scenarioResultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 300);
      })
      .catch((err: unknown) => {
        setScenarioError(`Failed to load scenario: ${err instanceof Error ? err.message : "Unknown error"}`);
      });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const clusterSummary: ClusterInfo[] = clustersPayload?.clusters ?? [];
  const clusterMeta = profilesPayload?.metadata ?? null;

  const handleRunScenario = useCallback(async () => {
    setScenarioRunning(true);
    setScenarioError(null);
    setScenarioResult(null);
    setScheduledJobId(null);
    setScenarioQueued(false);
    const label = String.fromCharCode(64 + nextScenarioIdx); // A, B, C ...
    currentLabelRef.current = label;
    try {
      const response = await runClusteringScenario({
        feature_params: featureParams,
        model_params: modelParams,
        label_params: labelParams,
      });
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
    if (defaults) {
      setFeatureParams(defaults.feature_params);
      setModelParams(defaults.model_params);
      setLabelParams(defaults.label_params);
    }
  }, [defaults]);

  const handlePromote = useCallback(async () => {
    if (!scenarioResult) return;
    try {
      await promoteScenario(scenarioResult.scenario_id);
      setShowPromoteConfirm(false);
    } catch (err) {
      setScenarioError(err instanceof Error ? err.message : "Promote failed");
      setShowPromoteConfirm(false);
    }
  }, [scenarioResult]);

  return (
    <>
      {/* ---- Existing cluster summary card ---- */}
      <Card className="mt-4 animate-fade-in">
        <CardHeader>
          <CardTitle className="text-base">DFU Clustering</CardTitle>
          <CardDescription>
            Filter by demand-pattern cluster.{" "}
            {clusterSource === "ml"
              ? "ML pipeline clusters from KMeans."
              : "Source clusters from dfu.txt."}
          </CardDescription>
          <div className="flex flex-wrap items-end gap-3">
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Source
              <select
                className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                value={clusterSource}
                onChange={(e) => {
                  setClusterSource(e.target.value as "ml" | "source");
                  setSelectedCluster("");
                }}
              >
                <option value="ml">ML Pipeline</option>
                <option value="source">Source (dfu.txt)</option>
              </select>
            </label>
            {clusterSummary.length > 0 ? (
              <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Cluster
                <select
                  className="h-9 w-full min-w-[200px] rounded-md border border-input bg-background px-3 text-sm"
                  value={selectedCluster}
                  onChange={(e) => setSelectedCluster(e.target.value)}
                >
                  <option value="">All Clusters</option>
                  {clusterSummary.map((c) => (
                    <option key={c.label} value={c.label}>
                      {c.label} ({formatCompactNumber(c.count)})
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {clusterSummary.length > 0 ? (
            <>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Cluster summary &mdash; {clusterSummary.length} clusters,{" "}
                {formatCompactNumber(clusterSummary.reduce((s, c) => s + c.count, 0))} DFUs assigned
              </p>
              <div className="max-h-[320px] overflow-y-auto rounded-md border border-input">
                <Table>
                  <TableHeader>
                    <TableRow className="border-muted bg-muted/30">
                      <TableHead className="text-xs">Cluster</TableHead>
                      <TableHead className="text-xs text-right">DFUs</TableHead>
                      <TableHead className="text-xs text-right">%</TableHead>
                      <TableHead className="text-xs text-right">Avg demand</TableHead>
                      <TableHead className="text-xs text-right">CV</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {clusterSummary.map((c) => (
                      <TableRow
                        key={c.label}
                        className={cn(
                          "cursor-pointer transition-colors hover:bg-muted/40",
                          selectedCluster === c.label && "bg-primary/10 font-semibold",
                        )}
                        onClick={() => setSelectedCluster(selectedCluster === c.label ? "" : c.label)}
                      >
                        <TableCell className="font-medium text-sm">{c.label}</TableCell>
                        <TableCell className="text-right text-sm tabular-nums">{formatNumber(c.count)}</TableCell>
                        <TableCell className="text-right text-sm tabular-nums">{formatNumber(c.pct_of_total)}%</TableCell>
                        <TableCell className="text-right text-sm tabular-nums">{formatNumber(c.avg_demand)}</TableCell>
                        <TableCell className="text-right text-sm tabular-nums">{formatNumber(c.cv_demand)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              <p className="text-xs text-muted-foreground">
                Click a row or use the dropdown above to filter the table below.
              </p>

              {/* Model metadata (ML source only) */}
              {clusterSource === "ml" && clusterMeta?.optimal_k ? (
                <div className="mt-3 flex flex-wrap gap-3 text-xs">
                  <span className="rounded bg-muted px-2 py-1">K = {clusterMeta.optimal_k}</span>
                  <span className="rounded bg-muted px-2 py-1">
                    Silhouette = {clusterMeta.silhouette_score?.toFixed(4)}
                  </span>
                  <span className="rounded bg-muted px-2 py-1">
                    Inertia = {formatCompactNumber(clusterMeta.inertia ?? 0)}
                  </span>
                </div>
              ) : null}

              {/* Visualization toggle (ML source only) */}
              {clusterSource === "ml" ? (
                <>
                  <button
                    className="mt-2 text-xs text-primary underline underline-offset-2 hover:text-primary/80"
                    onClick={() => setShowClusterViz(!showClusterViz)}
                  >
                    {showClusterViz ? "Hide visualizations" : "Show cluster visualizations"}
                  </button>
                  {showClusterViz ? (
                    <div className="mt-2 grid gap-4 md:grid-cols-2">
                      <div>
                        <p className="mb-1 text-xs font-semibold text-muted-foreground">
                          K Selection (Elbow / Silhouette / Gap)
                        </p>
                        <img
                          src="/domains/dfu/clusters/visualization/k_selection_plots.png"
                          alt="K Selection Plots"
                          className="w-full rounded-md border"
                          onError={(e) => {
                            (e.target as HTMLImageElement).style.display = "none";
                          }}
                        />
                      </div>
                      <div>
                        <p className="mb-1 text-xs font-semibold text-muted-foreground">
                          Cluster Visualization (2D PCA)
                        </p>
                        <img
                          src="/domains/dfu/clusters/visualization/cluster_visualization.png"
                          alt="Cluster PCA Visualization"
                          className="w-full rounded-md border"
                          onError={(e) => {
                            (e.target as HTMLImageElement).style.display = "none";
                          }}
                        />
                      </div>
                    </div>
                  ) : null}
                </>
              ) : null}
            </>
          ) : (
            <p className="text-sm text-muted-foreground">
              No cluster assignments yet. Run the clustering pipeline:{" "}
              <code className="rounded bg-muted px-1">make cluster-features</code>, then{" "}
              <code className="rounded bg-muted px-1">make cluster-train</code>,{" "}
              <code className="rounded bg-muted px-1">make cluster-label</code>, and{" "}
              <code className="rounded bg-muted px-1">make cluster-update</code> to see clusters here.
            </p>
          )}
        </CardContent>
      </Card>

      {/* ---- What-If Scenarios ---- */}
      <Card className="mt-4 animate-fade-in">
        <CardHeader className="cursor-pointer" onClick={() => setShowWhatIf(!showWhatIf)}>
          <CardTitle className="flex items-center gap-2 text-base">
            <span className="text-xs">{showWhatIf ? "\u25BC" : "\u25B6"}</span>
            What-If Scenarios
          </CardTitle>
        </CardHeader>

        {showWhatIf && (
          <CardContent className="space-y-4">
            {/* Parameter sections */}
            <div className="grid gap-4 md:grid-cols-3">
              {/* Data Scope */}
              <div className="space-y-2 rounded-lg border border-input p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Data Scope</p>
                <ParamInput
                  label="Time Window (months)"
                  value={featureParams.time_window_months}
                  onChange={(v) => setFeatureParams((p) => ({ ...p, time_window_months: v }))}
                  min={1}
                  max={120}
                />
                <ParamInput
                  label="Min History (months)"
                  value={featureParams.min_months_history}
                  onChange={(v) => setFeatureParams((p) => ({ ...p, min_months_history: v }))}
                  min={1}
                  max={60}
                />
              </div>

              {/* Model */}
              <div className="space-y-2 rounded-lg border border-input p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Model</p>
                <div>
                  <span className="text-xs font-semibold text-muted-foreground">K Range</span>
                  <div className="mt-1 flex items-center gap-2">
                    <input
                      type="number"
                      className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm tabular-nums"
                      value={modelParams.k_range[0]}
                      min={2}
                      onChange={(e) =>
                        setModelParams((p) => ({ ...p, k_range: [Number(e.target.value), p.k_range[1]] }))
                      }
                    />
                    <span className="text-xs text-muted-foreground">to</span>
                    <input
                      type="number"
                      className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm tabular-nums"
                      value={modelParams.k_range[1]}
                      min={2}
                      onChange={(e) =>
                        setModelParams((p) => ({ ...p, k_range: [p.k_range[0], Number(e.target.value)] }))
                      }
                    />
                  </div>
                </div>
                <ParamInput
                  label="Min Cluster Size (%)"
                  value={modelParams.min_cluster_size_pct}
                  onChange={(v) => setModelParams((p) => ({ ...p, min_cluster_size_pct: v }))}
                  step={0.5}
                  min={0}
                  max={50}
                />
              </div>

              {/* Labeling Thresholds */}
              <div className="space-y-2 rounded-lg border border-input p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Labeling Thresholds</p>
                <ParamInput
                  label="Volume High (pctl)"
                  value={labelParams.volume_high}
                  onChange={(v) => setLabelParams((p) => ({ ...p, volume_high: v }))}
                  step={0.05}
                  min={0}
                  max={1}
                />
                <ParamInput
                  label="Volume Low (pctl)"
                  value={labelParams.volume_low}
                  onChange={(v) => setLabelParams((p) => ({ ...p, volume_low: v }))}
                  step={0.05}
                  min={0}
                  max={1}
                />
                <ParamInput
                  label="CV Steady (<)"
                  value={labelParams.cv_steady}
                  onChange={(v) => setLabelParams((p) => ({ ...p, cv_steady: v }))}
                  step={0.05}
                  min={0}
                />
                <ParamInput
                  label="CV Volatile (>)"
                  value={labelParams.cv_volatile}
                  onChange={(v) => setLabelParams((p) => ({ ...p, cv_volatile: v }))}
                  step={0.05}
                  min={0}
                />
                <ParamInput
                  label="Seasonality Threshold"
                  value={labelParams.seasonality_threshold}
                  onChange={(v) => setLabelParams((p) => ({ ...p, seasonality_threshold: v }))}
                  step={0.05}
                  min={0}
                  max={1}
                />
                <ParamInput
                  label="Zero Demand Threshold"
                  value={labelParams.zero_demand_threshold}
                  onChange={(v) => setLabelParams((p) => ({ ...p, zero_demand_threshold: v }))}
                  step={0.05}
                  min={0}
                  max={1}
                />
              </div>
            </div>

            {/* Action buttons + estimate */}
            <div className="flex items-center gap-3">
              <button
                className={cn(
                  "rounded-md px-4 py-2 text-sm font-medium transition-colors",
                  scenarioRunning
                    ? "cursor-wait bg-primary/50 text-primary-foreground"
                    : "bg-primary text-primary-foreground hover:bg-primary/90",
                )}
                disabled={scenarioRunning}
                onClick={handleRunScenario}
              >
                {scenarioRunning ? (
                  <span className="flex items-center gap-2">
                    <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                    {scenarioQueued ? "Queued" : "Scheduled"}{statusData?.elapsed_seconds != null
                      ? ` (${statusData.elapsed_seconds >= 60
                          ? `${Math.floor(statusData.elapsed_seconds / 60)}m ${Math.round(statusData.elapsed_seconds % 60)}s`
                          : `${Math.round(statusData.elapsed_seconds)}s`})`
                      : "..."}
                  </span>
                ) : (
                  "Schedule Scenario Job"
                )}
              </button>
              <button
                className="rounded-md border border-input bg-background px-4 py-2 text-sm font-medium hover:bg-muted/50"
                onClick={handleReset}
              >
                Reset to Defaults
              </button>
              {estimate && !scenarioRunning && (
                <span className="rounded-full bg-muted px-3 py-1 text-xs font-medium tabular-nums text-muted-foreground">
                  Est. ~{estimate.estimated_seconds >= 120
                    ? `${Math.round(estimate.estimated_seconds / 60)}m`
                    : estimate.estimated_seconds >= 60
                    ? `${Math.floor(estimate.estimated_seconds / 60)}m ${Math.round(estimate.estimated_seconds % 60)}s`
                    : `${Math.round(estimate.estimated_seconds)}s`}
                  {estimate.dfu_count > 0 && (
                    <span className="ml-1 text-[10px]">
                      ({formatCompactNumber(estimate.dfu_count)} DFUs{estimate.sampled ? `, ${formatCompactNumber(estimate.training_sample)} sample` : ""})
                    </span>
                  )}
                </span>
              )}
            </div>

            {/* Job link - shows after scheduling */}
            {scheduledJobId && scenarioRunning && (
              <div className="rounded-md border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/30 p-3 text-sm flex items-center justify-between">
                <span className="text-blue-700 dark:text-blue-300 text-xs">
                  {scenarioQueued
                    ? "Your scenario is queued. A clustering job is currently running \u2014 yours will start automatically when it finishes."
                    : "Job scheduled successfully. Track progress in the Jobs tab."}
                </span>
                <span className="text-[10px] font-mono text-blue-500 bg-blue-100 dark:bg-blue-900/50 rounded px-1.5 py-0.5">
                  {scheduledJobId}
                </span>
              </div>
            )}

            {/* Error display */}
            {scenarioError && (
              <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
                {scenarioError}
              </div>
            )}

            {/* Scenario results */}
            {scenarioResult?.status === "completed" && scenarioResult.result && (
              <div ref={scenarioResultRef} className="space-y-3 rounded-lg border border-border p-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-semibold">
                    Scenario {scenarioLabel} &mdash; K={scenarioResult.result.optimal_k},{" "}
                    {scenarioResult.result.total_dfus} DFUs, {scenarioResult.runtime_seconds.toFixed(1)}s
                  </p>
                  <button
                    className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
                    onClick={() => setShowPromoteConfirm(true)}
                  >
                    Promote Scenario {scenarioLabel}
                  </button>
                </div>

                {/* Result profile table */}
                <div className="max-h-[240px] overflow-y-auto rounded-md border border-input">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-muted bg-muted/30">
                        <TableHead className="text-xs">Cluster</TableHead>
                        <TableHead className="text-xs text-right">DFUs</TableHead>
                        <TableHead className="text-xs text-right">%</TableHead>
                        <TableHead className="text-xs text-right">Avg demand</TableHead>
                        <TableHead className="text-xs text-right">CV</TableHead>
                        <TableHead className="text-xs text-right">Seasonality</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {scenarioResult.result.profiles.map((p) => (
                        <TableRow key={p.label}>
                          <TableCell className="text-sm font-medium">{p.label}</TableCell>
                          <TableCell className="text-right text-sm tabular-nums">{formatNumber(p.count)}</TableCell>
                          <TableCell className="text-right text-sm tabular-nums">{formatNumber(p.pct_of_total)}%</TableCell>
                          <TableCell className="text-right text-sm tabular-nums">{formatNumber(p.mean_demand)}</TableCell>
                          <TableCell className="text-right text-sm tabular-nums">{formatNumber(p.cv_demand)}</TableCell>
                          <TableCell className="text-right text-sm tabular-nums">{formatNumber(p.seasonality_strength)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>

                {/* Charts */}
                <ScenarioCharts result={scenarioResult.result} />
              </div>
            )}

            {/* Promote confirmation dialog */}
            {showPromoteConfirm && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
                <div className="w-full max-w-md rounded-lg border border-border bg-card p-6 shadow-xl">
                  <p className="text-sm font-semibold">Promote Scenario {scenarioLabel} to Production?</p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    This will update <code>dim_dfu.ml_cluster</code> with the new cluster assignments.
                  </p>
                  <div className="mt-4 flex justify-end gap-3">
                    <button
                      className="rounded-md border border-input bg-background px-4 py-2 text-sm hover:bg-muted/50"
                      onClick={() => setShowPromoteConfirm(false)}
                    >
                      Cancel
                    </button>
                    <button
                      className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
                      onClick={handlePromote}
                    >
                      Confirm Promote
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Past Scenarios History (last 10 runs) */}
            {pastScenarios && pastScenarios.length > 0 && (
              <div className="space-y-2 mt-4">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Past Scenarios ({pastScenarios.length})
                </p>
                <div className="space-y-1.5">
                  {pastScenarios.map((pj) => {
                    const jr = pj.result as Record<string, unknown> | null;
                    const inner = jr ? ((jr.result ?? jr) as Record<string, unknown>) : null;
                    const optK = inner?.optimal_k as number | undefined;
                    const totalDfus = inner?.total_dfus as number | undefined;
                    const runtimeSec = (jr?.runtime_seconds as number) ?? 0;
                    const scenId = (jr?.scenario_id as string) || pj.job_id;
                    const isExpanded = expandedHistoryId === pj.job_id;

                    return (
                      <div key={pj.job_id} className="rounded-lg border border-border/60">
                        <button
                          className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-muted/20 transition-colors rounded-lg"
                          onClick={() => {
                            if (isExpanded) {
                              setExpandedHistoryId(null);
                              setHistoryResult(null);
                            } else {
                              setExpandedHistoryId(pj.job_id);
                              if (inner) {
                                setHistoryResult({
                                  scenario_id: scenId,
                                  status: "completed",
                                  runtime_seconds: runtimeSec,
                                  params: (jr?.params as Record<string, unknown>) || {},
                                  result: inner as ClusteringScenarioResult["result"],
                                });
                              }
                            }
                          }}
                        >
                          <div className="flex items-center gap-2">
                            {isExpanded ? (
                              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                            ) : (
                              <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                            )}
                            <span className="text-sm font-medium">
                              {pj.job_label || "Scenario"}
                              {optK != null && <span className="text-muted-foreground font-normal"> &mdash; K={optK}</span>}
                              {totalDfus != null && <span className="text-muted-foreground font-normal">, {formatCompactNumber(totalDfus)} DFUs</span>}
                              {runtimeSec > 0 && <span className="text-muted-foreground font-normal">, {runtimeSec.toFixed(1)}s</span>}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-muted-foreground">
                              {new Date(pj.submitted_at).toLocaleString()}
                            </span>
                            {inner && (
                              <button
                                className="rounded-md px-2 py-0.5 text-[10px] font-medium bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setScenarioResult({
                                    scenario_id: scenId,
                                    status: "completed",
                                    runtime_seconds: runtimeSec,
                                    params: (jr?.params as Record<string, unknown>) || {},
                                    result: inner as ClusteringScenarioResult["result"],
                                  });
                                  setScenarioLabel(pj.job_label?.replace("What-If Scenario ", "") || "H");
                                  setTimeout(() => {
                                    scenarioResultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
                                  }, 100);
                                }}
                              >
                                Promote
                              </button>
                            )}
                          </div>
                        </button>
                        {isExpanded && historyResult?.result && (
                          <div className="px-3 pb-3 space-y-3">
                            <div className="max-h-[180px] overflow-y-auto rounded-md border border-input">
                              <Table>
                                <TableHeader>
                                  <TableRow className="border-muted bg-muted/30">
                                    <TableHead className="text-xs">Cluster</TableHead>
                                    <TableHead className="text-xs text-right">DFUs</TableHead>
                                    <TableHead className="text-xs text-right">%</TableHead>
                                    <TableHead className="text-xs text-right">Avg demand</TableHead>
                                    <TableHead className="text-xs text-right">CV</TableHead>
                                    <TableHead className="text-xs text-right">Seasonality</TableHead>
                                  </TableRow>
                                </TableHeader>
                                <TableBody>
                                  {historyResult.result.profiles.map((p) => (
                                    <TableRow key={p.label}>
                                      <TableCell className="text-xs font-medium">{p.label}</TableCell>
                                      <TableCell className="text-right text-xs tabular-nums">{formatNumber(p.count)}</TableCell>
                                      <TableCell className="text-right text-xs tabular-nums">{formatNumber(p.pct_of_total)}%</TableCell>
                                      <TableCell className="text-right text-xs tabular-nums">{formatNumber(p.mean_demand)}</TableCell>
                                      <TableCell className="text-right text-xs tabular-nums">{formatNumber(p.cv_demand)}</TableCell>
                                      <TableCell className="text-right text-xs tabular-nums">{formatNumber(p.seasonality_strength)}</TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </div>
                            <ScenarioCharts result={historyResult.result} />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </CardContent>
        )}
      </Card>
    </>
  );
}
