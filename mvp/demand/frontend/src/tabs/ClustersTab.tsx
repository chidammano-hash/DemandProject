import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ReferenceLine,
  Cell,
  PieChart,
  Pie,
} from "recharts";
import {
  queryKeys,
  fetchDfuClusters,
  fetchClusterProfiles,
  fetchClusteringDefaults,
  runClusteringScenario,
  promoteScenario,
  fetchScenarioEstimate,
  fetchScenarioStatus,
  STALE,
} from "@/api/queries";
import type { ClusteringDefaultsPayload, ClusteringScenarioResult } from "@/api/queries";
import type { Theme, ClusterInfo } from "@/types";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
import { cn } from "@/lib/utils";
import { useScenarioNotification } from "@/context/ScenarioNotificationContext";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

type ClustersTabProps = {
  domain: string;
  onDomainChange: (d: string) => void;
  theme: Theme;
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
// Silhouette quality zone helper
// ---------------------------------------------------------------------------
function silhouetteColor(score: number): string {
  if (score >= 0.71) return "#22c55e"; // green — strong
  if (score >= 0.51) return "#3b82f6"; // blue — reasonable
  if (score >= 0.26) return "#eab308"; // yellow — weak
  return "#ef4444"; // red — no structure
}

function silhouetteQuality(score: number): string {
  if (score >= 0.71) return "Strong";
  if (score >= 0.51) return "Reasonable";
  if (score >= 0.26) return "Weak";
  return "No structure";
}

// ---------------------------------------------------------------------------
// Enhanced What-If scenario charts
// ---------------------------------------------------------------------------
const PIE_COLORS = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6", "#f97316", "#84cc16"];

function ScenarioCharts({ result }: { result: NonNullable<ClusteringScenarioResult["result"]> }) {
  const kData = result.k_selection_results.k_values.map((k, i) => ({
    k,
    inertia: result.k_selection_results.inertias[i],
    silhouette: result.k_selection_results.silhouette_scores[i],
    ...(result.k_selection_results.gap_stats?.[i] != null ? { gap: result.k_selection_results.gap_stats[i] } : {}),
  }));

  const sizeData = result.profiles.map((p) => ({
    label: p.label,
    count: p.count,
    pct: p.pct_of_total,
  }));

  // Normalise radar features to 0-1 range
  const radarKeys = ["cv_demand", "seasonality_strength", "trend_slope", "growth_rate", "zero_demand_pct"] as const;
  const radarData = radarKeys.map((key) => {
    const entry: Record<string, string | number> = { feature: key };
    for (const p of result.profiles) {
      entry[p.label] = Math.abs(p[key]);
    }
    return entry;
  });
  const radarColors = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"];

  // Feature importance (top 10)
  const featureImportance = (result.feature_importance ?? [])
    .slice(0, 10)
    .map((f) => ({ ...f, pct: Math.round(f.variance_ratio * 100) }));

  // Gap statistic data
  const hasGapStats = result.k_selection_results.gap_stats && result.k_selection_results.gap_stats.length > 0;

  return (
    <div className="mt-4 grid gap-4 md:grid-cols-2">
      {/* Elbow chart with optimal K marker */}
      <div>
        <p className="mb-1 text-xs font-semibold text-muted-foreground">Elbow (WCSS/Inertia)</p>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={kData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="k" />
            <YAxis />
            <Tooltip
              formatter={(value: number, name: string) => [
                name === "Inertia" ? formatCompactNumber(value) : value.toFixed(4),
                name,
              ]}
              labelFormatter={(k) => `K = ${k}${Number(k) === result.optimal_k ? " (Optimal)" : ""}`}
            />
            <Legend />
            <ReferenceLine
              x={result.optimal_k}
              stroke="#ef4444"
              strokeDasharray="5 5"
              label={{ value: `Optimal K=${result.optimal_k}`, position: "top", fill: "#ef4444", fontSize: 11 }}
            />
            <Line type="monotone" dataKey="inertia" stroke="#6366f1" name="Inertia" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Silhouette chart with quality zones */}
      <div>
        <p className="mb-1 text-xs font-semibold text-muted-foreground">
          Silhouette Score
          <span className="ml-2 text-[10px] font-normal text-muted-foreground">
            ({silhouetteQuality(result.silhouette_score)} — {result.silhouette_score.toFixed(3)})
          </span>
        </p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={kData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="k" />
            <YAxis domain={[0, 1]} />
            <Tooltip
              formatter={(value: number) => [value.toFixed(4), "Silhouette"]}
              labelFormatter={(k) => `K = ${k}${Number(k) === result.optimal_k ? " (Optimal)" : ""}`}
            />
            <ReferenceLine y={0.71} stroke="#22c55e" strokeDasharray="3 3" label={{ value: "Strong", position: "right", fontSize: 9 }} />
            <ReferenceLine y={0.51} stroke="#3b82f6" strokeDasharray="3 3" label={{ value: "Reasonable", position: "right", fontSize: 9 }} />
            <ReferenceLine y={0.26} stroke="#eab308" strokeDasharray="3 3" label={{ value: "Weak", position: "right", fontSize: 9 }} />
            <Bar dataKey="silhouette" name="Silhouette">
              {kData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.k === result.optimal_k ? "#6366f1" : silhouetteColor(entry.silhouette)}
                  stroke={entry.k === result.optimal_k ? "#312e81" : undefined}
                  strokeWidth={entry.k === result.optimal_k ? 2 : 0}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Cluster size distribution — Pie chart */}
      <div>
        <p className="mb-1 text-xs font-semibold text-muted-foreground">Cluster Size Distribution</p>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie
              data={sizeData}
              dataKey="count"
              nameKey="label"
              cx="50%"
              cy="50%"
              outerRadius={80}
              label={({ label, pct }) => `${label} (${pct.toFixed(0)}%)`}
              labelLine={{ strokeWidth: 1 }}
            >
              {sizeData.map((_, i) => (
                <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value: number, name: string) => [formatNumber(value), name]} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Radar profile */}
      <div>
        <p className="mb-1 text-xs font-semibold text-muted-foreground">Cluster Profile Radar</p>
        <ResponsiveContainer width="100%" height={220}>
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="feature" />
            <PolarRadiusAxis />
            {result.profiles.map((p, i) => (
              <Radar
                key={p.label}
                name={p.label}
                dataKey={p.label}
                stroke={radarColors[i % radarColors.length]}
                fill={radarColors[i % radarColors.length]}
                fillOpacity={0.15}
              />
            ))}
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Feature importance (horizontal bar) */}
      {featureImportance.length > 0 && (
        <div>
          <p className="mb-1 text-xs font-semibold text-muted-foreground">Feature Importance (Variance Ratio)</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={featureImportance} layout="vertical" margin={{ left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, "auto"]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <YAxis type="category" dataKey="feature" width={75} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, "Importance"]} />
              <Bar dataKey="variance_ratio" fill="#8b5cf6" name="Importance">
                {featureImportance.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? "#6366f1" : "#a5b4fc"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Gap statistic chart (conditional) */}
      {hasGapStats && (
        <div>
          <p className="mb-1 text-xs font-semibold text-muted-foreground">Gap Statistic</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={kData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="k" />
              <YAxis />
              <Tooltip
                formatter={(value: number) => [value.toFixed(4), "Gap"]}
                labelFormatter={(k) => `K = ${k}${Number(k) === result.optimal_k ? " (Optimal)" : ""}`}
              />
              <ReferenceLine
                x={result.optimal_k}
                stroke="#ef4444"
                strokeDasharray="5 5"
                label={{ value: `K=${result.optimal_k}`, position: "top", fill: "#ef4444", fontSize: 11 }}
              />
              <Line type="monotone" dataKey="gap" stroke="#f59e0b" name="Gap" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ClustersTab
// ---------------------------------------------------------------------------
export default function ClustersTab({ domain, onDomainChange, theme }: ClustersTabProps) {
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

  const scenarioNotification = useScenarioNotification();
  const currentLabelRef = useRef("A");

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
  }, [statusData, statusPollError, pollingScenarioId, scenarioNotification]);

  // Sync params from loaded defaults
  useEffect(() => {
    if (defaults) {
      setFeatureParams(defaults.feature_params);
      setModelParams(defaults.model_params);
      setLabelParams(defaults.label_params);
    }
  }, [defaults]);

  const clusterSummary: ClusterInfo[] = clustersPayload?.clusters ?? [];
  const clusterMeta = profilesPayload?.metadata ?? null;

  const handleRunScenario = useCallback(async () => {
    setScenarioRunning(true);
    setScenarioError(null);
    setScenarioResult(null);
    setScheduledJobId(null);
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
                    Scheduled{statusData?.elapsed_seconds != null
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
                  Job scheduled successfully. Track progress in the Jobs tab.
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
              <div className="space-y-3 rounded-lg border border-border p-4">
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
          </CardContent>
        )}
      </Card>
    </>
  );
}
