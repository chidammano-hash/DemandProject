import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { queryKeys, fetchDfuClusters, fetchClusterProfiles, STALE } from "@/api/queries";
import type { ClusterInfo } from "@/types";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
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

  const clusterSummary: ClusterInfo[] = clustersPayload?.clusters ?? [];
  const clusterMeta = profilesPayload?.metadata ?? null;

  return (
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
  );
}
