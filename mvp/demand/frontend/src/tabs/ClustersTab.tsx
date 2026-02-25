import { useCallback, useEffect, useState } from "react";
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
} from "recharts";
import {
  queryKeys,
  fetchDfuClusters,
  fetchClusterProfiles,
  fetchClusteringDefaults,
  runClusteringScenario,
  promoteScenario,
  STALE,
} from "@/api/queries";
import type { ClusteringDefaultsPayload, ClusteringScenarioResult, ScenarioProfile } from "@/api/queries";
import type { Theme, ClusterInfo } from "@/types";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
import { cn } from "@/lib/utils";

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
// What-If scenario charts
// ---------------------------------------------------------------------------
function ScenarioCharts({ result }: { result: NonNullable<ClusteringScenarioResult["result"]> }) {
  const kData = result.k_selection_results.k_values.map((k, i) => ({
    k,
    inertia: result.k_selection_results.inertias[i],
    silhouette: result.k_selection_results.silhouette_scores[i],
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

  return (
    <div className="mt-4 grid gap-4 md:grid-cols-2">
      {/* Elbow chart */}
      <div>
        <p className="mb-1 text-xs font-semibold text-muted-foreground">Elbow (WCSS/Inertia)</p>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={kData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="k" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="inertia" stroke="#6366f1" name="Inertia" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Silhouette chart */}
      <div>
        <p className="mb-1 text-xs font-semibold text-muted-foreground">Silhouette Score</p>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={kData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="k" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="silhouette" stroke="#10b981" name="Silhouette" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Cluster size distribution */}
      <div>
        <p className="mb-1 text-xs font-semibold text-muted-foreground">Cluster Size Distribution</p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={sizeData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#6366f1" name="DFUs" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Radar profile */}
      <div>
        <p className="mb-1 text-xs font-semibold text-muted-foreground">Cluster Profile Radar</p>
        <ResponsiveContainer width="100%" height={200}>
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
  const [scenarioResult, setScenarioResult] = useState<ClusteringScenarioResult | null>(null);
  const [scenarioError, setScenarioError] = useState<string | null>(null);
  const [scenarioLabel, setScenarioLabel] = useState("A");
  const [nextScenarioIdx, setNextScenarioIdx] = useState(1);
  const [showPromoteConfirm, setShowPromoteConfirm] = useState(false);

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
    const currentLabel = String.fromCharCode(64 + nextScenarioIdx); // A, B, C ...
    try {
      const result = await runClusteringScenario({
        feature_params: featureParams,
        model_params: modelParams,
        label_params: labelParams,
      });
      setScenarioResult(result);
      if (result.status === "failed") {
        setScenarioError(result.error);
      } else {
        setScenarioLabel(currentLabel);
        setNextScenarioIdx((c) => c + 1);
      }
    } catch (err) {
      setScenarioError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setScenarioRunning(false);
    }
  }, [featureParams, modelParams, labelParams, nextScenarioIdx]);

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

            {/* Action buttons */}
            <div className="flex gap-3">
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
                {scenarioRunning ? "Running..." : "Run Scenario"}
              </button>
              <button
                className="rounded-md border border-input bg-background px-4 py-2 text-sm font-medium hover:bg-muted/50"
                onClick={handleReset}
              >
                Reset to Defaults
              </button>
            </div>

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
