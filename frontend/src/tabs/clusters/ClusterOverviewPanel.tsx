import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { queryKeys, fetchSkuClusters, fetchClusterProfiles, submitJob, STALE } from "@/api/queries";
import type { ClusterInfo } from "@/types";
import { formatNumber, formatCompactNumber, formatClusterLabel } from "@/lib/formatters";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

type ClusterOverviewPanelProps = {
  onDomainChange: (d: string) => void;
};

export default function ClusterOverviewPanel({ onDomainChange: _onDomainChange }: ClusterOverviewPanelProps) {
  const [clusterSource, setClusterSource] = useState<"ml" | "source">("ml");
  const [selectedCluster, setSelectedCluster] = useState("");
  const [showClusterViz, setShowClusterViz] = useState(false);
  const [showRunConfirm, setShowRunConfirm] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineJobId, setPipelineJobId] = useState<string | null>(null);
  const [pipelineError, setPipelineError] = useState<string | null>(null);

  const { data: clustersPayload } = useQuery({
    queryKey: queryKeys.skuClusters(clusterSource),
    queryFn: () => fetchSkuClusters(clusterSource),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: profilesPayload } = useQuery({
    queryKey: queryKeys.clusterProfiles(),
    queryFn: fetchClusterProfiles,
    staleTime: STALE.TEN_MIN,
    enabled: clusterSource === "ml",
  });

  const clusterSummary: ClusterInfo[] = clustersPayload?.clusters ?? [];
  const clusterMeta = profilesPayload?.metadata ?? null;

  const handleRunPipeline = async () => {
    setShowRunConfirm(false);
    setPipelineRunning(true);
    setPipelineError(null);
    setPipelineJobId(null);
    try {
      const resp = await submitJob("cluster_pipeline", {}, "Production Clustering Pipeline");
      setPipelineJobId(resp.job_id);
    } catch (err) {
      setPipelineRunning(false);
      setPipelineError(err instanceof Error ? err.message : "Failed to start pipeline");
    }
  };

  return (
    <Card className="mt-4 animate-fade-in">
      <CardHeader>
        <CardTitle className="text-base">DFU Clustering</CardTitle>
        <CardDescription>
          Filter by demand-pattern cluster.{" "}
          {clusterSource === "ml"
            ? "ML pipeline clusters from KMeans."
            : "Source clusters from sku.txt."}
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
              <option value="source">Source (sku.txt)</option>
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
                    {formatClusterLabel(c.label)} ({formatCompactNumber(c.count)})
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
              {formatCompactNumber(clusterSummary.reduce((s, c) => s + c.count, 0))} SKUs assigned
            </p>
            <div className="max-h-[320px] overflow-y-auto rounded-md border border-input">
              <Table>
                <TableHeader>
                  <TableRow className="border-muted bg-muted/30">
                    <TableHead className="text-xs">Cluster</TableHead>
                    <TableHead className="text-xs text-right">SKUs</TableHead>
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
                      <TableCell className="font-medium text-sm" title={c.label}>{formatClusterLabel(c.label)}</TableCell>
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
                        src="/domains/sku/clusters/visualization/k_selection_plots.png"
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
                        src="/domains/sku/clusters/visualization/cluster_visualization.png"
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

            {/* Run Production Pipeline button */}
            {clusterSource === "ml" ? (
              <div className="mt-3 flex items-center gap-3">
                <button
                  className={cn(
                    "rounded-md px-4 py-2 text-sm font-medium transition-colors",
                    pipelineRunning
                      ? "cursor-wait bg-primary/50 text-primary-foreground"
                      : "bg-primary text-primary-foreground hover:bg-primary/90",
                  )}
                  disabled={pipelineRunning}
                  onClick={() => setShowRunConfirm(true)}
                >
                  {pipelineRunning ? (
                    <span className="flex items-center gap-2">
                      <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                      Pipeline Running...
                    </span>
                  ) : (
                    "Re-run Clustering Pipeline"
                  )}
                </button>
                {pipelineJobId && (
                  <span className="text-[10px] font-mono text-muted-foreground bg-muted rounded px-1.5 py-0.5">
                    {pipelineJobId}
                  </span>
                )}
              </div>
            ) : null}
            {pipelineError && (
              <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
                {pipelineError}
              </div>
            )}
          </>
        ) : (
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">
              No cluster assignments yet. Run the clustering pipeline to group SKUs by demand patterns.
            </p>
            <button
              className={cn(
                "rounded-md px-4 py-2 text-sm font-medium transition-colors",
                pipelineRunning
                  ? "cursor-wait bg-primary/50 text-primary-foreground"
                  : "bg-primary text-primary-foreground hover:bg-primary/90",
              )}
              disabled={pipelineRunning}
              onClick={() => setShowRunConfirm(true)}
            >
              {pipelineRunning ? (
                <span className="flex items-center gap-2">
                  <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                  Pipeline Running...
                </span>
              ) : (
                "Run Clustering Pipeline"
              )}
            </button>
            {pipelineJobId && (
              <div className="rounded-md border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/30 p-3 text-sm flex items-center justify-between">
                <span className="text-blue-700 dark:text-blue-300 text-xs">
                  Pipeline job scheduled. Track progress in the Jobs tab.
                </span>
                <span className="text-[10px] font-mono text-blue-500 bg-blue-100 dark:bg-blue-900/50 rounded px-1.5 py-0.5">
                  {pipelineJobId}
                </span>
              </div>
            )}
            {pipelineError && (
              <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
                {pipelineError}
              </div>
            )}
          </div>
        )}
      </CardContent>

      {/* Run Pipeline Confirmation Dialog */}
      {showRunConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-lg border border-border bg-card p-6 shadow-xl">
            <p className="text-base font-semibold">Run Production Clustering Pipeline?</p>

            <div className="mt-3 rounded-md border border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/30 p-3">
              <p className="text-sm font-medium text-amber-800 dark:text-amber-300">
                This will overwrite existing cluster assignments
              </p>
              <ul className="mt-1.5 space-y-1 text-xs text-amber-700 dark:text-amber-400">
                <li>1. <strong>Generate features</strong> — extract demand patterns from sales history</li>
                <li>2. <strong>Train model</strong> — find optimal K and fit KMeans</li>
                <li>3. <strong>Label clusters</strong> — assign business labels (e.g., high_volume_steady)</li>
                <li>4. <strong>Update assignments</strong> — write new labels to <code className="text-[10px] bg-amber-100 dark:bg-amber-900/50 rounded px-0.5">dim_sku.ml_cluster</code></li>
              </ul>
              <p className="mt-2 text-[11px] text-amber-600 dark:text-amber-500">
                After completion, re-run backtests to validate model accuracy with the new clusters.
              </p>
            </div>

            <p className="mt-3 text-xs text-muted-foreground">
              Uses parameters from the promoted cluster experiment (or system defaults).
              Use What-If Scenarios below to experiment with custom parameters first.
            </p>

            <div className="mt-4 flex justify-end gap-3">
              <button
                className="rounded-md border border-input bg-background px-4 py-2 text-sm hover:bg-muted/50"
                onClick={() => setShowRunConfirm(false)}
              >
                Cancel
              </button>
              <button
                className="rounded-md bg-primary hover:bg-primary/90 px-4 py-2 text-sm font-medium text-primary-foreground"
                onClick={handleRunPipeline}
              >
                Run Pipeline
              </button>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
