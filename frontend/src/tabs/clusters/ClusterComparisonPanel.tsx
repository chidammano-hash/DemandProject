/**
 * ClusterComparisonPanel -- Comparison panel when two cluster experiments
 * are selected. Shows quality metrics, cluster profile comparison,
 * DFU migration summary, and promote button.
 */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowUp, ArrowDown, Minus, Crown } from "lucide-react";
import {
  clusterExperimentKeys,
  CLUSTER_EXP_STALE,
  fetchClusterComparison,
  type ClusterExperiment,
  type ClusterExperimentComparison,
} from "@/api/queries";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { cn } from "@/lib/utils";
import { formatClusterLabel } from "@/lib/formatters";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { LoadingElement } from "@/components/LoadingElement";
import type { PCAScatterData } from "@/api/queries";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ClusterComparisonPanelProps {
  baselineId: number;
  candidateId: number;
  onPromote: (experiment: ClusterExperiment) => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function DeltaIndicator({
  value,
  higherIsBetter = true,
  suffix = "",
}: {
  value: number;
  higherIsBetter?: boolean;
  suffix?: string;
}) {
  if (Math.abs(value) < 0.0001) {
    return (
      <span className="inline-flex items-center gap-0.5 text-xs text-muted-foreground">
        <Minus className="h-3 w-3" />
        <span className="tabular-nums">0{suffix}</span>
      </span>
    );
  }
  const isPositive = value > 0;
  const improved = higherIsBetter ? isPositive : !isPositive;
  const Icon = isPositive ? ArrowUp : ArrowDown;
  const color = improved
    ? "text-emerald-600 dark:text-emerald-400"
    : "text-red-600 dark:text-red-400";
  return (
    <span className={cn("inline-flex items-center gap-0.5 text-xs", color)}>
      <Icon className="h-3 w-3" />
      <span className="tabular-nums">
        {isPositive ? "+" : ""}
        {typeof value === "number" && Math.abs(value) < 100
          ? value.toFixed(4)
          : value.toLocaleString()}
        {suffix}
      </span>
    </span>
  );
}

function MetricRow({
  label,
  valueA,
  valueB,
  delta,
  higherIsBetter = true,
  formatter = (v: number) => v.toFixed(4),
}: {
  label: string;
  valueA: number | null;
  valueB: number | null;
  delta: number;
  higherIsBetter?: boolean;
  formatter?: (v: number) => string;
}) {
  return (
    <div className="grid grid-cols-4 gap-2 items-center py-1.5 border-b border-border/30 last:border-0">
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      <p className="text-sm tabular-nums text-center">
        {valueA != null ? formatter(valueA) : "--"}
      </p>
      <p className="text-sm tabular-nums text-center">
        {valueB != null ? formatter(valueB) : "--"}
      </p>
      <div className="text-center">
        <DeltaIndicator value={delta} higherIsBetter={higherIsBetter} />
      </div>
    </div>
  );
}

const PCA_COLORS = [
  "#2563EB", "#f59e0b", "#10b981", "#ef4444", "#0891B2",
  "#06b6d4", "#ec4899", "#14b8a6", "#f97316", "#84cc16",
];

function PCAScatterChart({ pca }: { pca: PCAScatterData }) {
  const byCluster = new Map<number, { pc1: number; pc2: number }[]>();
  for (const pt of pca.points) {
    const arr = byCluster.get(pt.cluster) ?? [];
    arr.push({ pc1: pt.pc1, pc2: pt.pc2 });
    byCluster.set(pt.cluster, arr);
  }

  return (
    <div className="rounded-md border border-input bg-background p-2">
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 10, right: 30, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="pc1"
            name="PC1"
            tick={{ fontSize: 10 }}
            label={{ value: `PC1 (${pca.pc1_variance}% variance)`, position: "insideBottom", offset: -10, fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="pc2"
            name="PC2"
            tick={{ fontSize: 10 }}
            label={{ value: `PC2 (${pca.pc2_variance}% variance)`, angle: -90, position: "insideLeft", offset: -5, fontSize: 11 }}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            formatter={(value: number, name: string) => [value.toFixed(2), name]}
          />
          {Array.from(byCluster.entries())
            .sort(([a], [b]) => a - b)
            .map(([cluster, points]) => (
              <Scatter
                key={cluster}
                name={`Cluster ${cluster}`}
                data={points}
                fill={PCA_COLORS[cluster % PCA_COLORS.length]}
                fillOpacity={0.6}
                r={3}
              />
            ))}
          <Legend verticalAlign="top" align="right" layout="vertical" wrapperStyle={{ fontSize: 10, paddingLeft: 12 }} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function VerdictBadge({ verdict }: { verdict: string }) {
  const v = verdict.toUpperCase();
  const variant =
    v === "IMPROVED"
      ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
      : v === "DEGRADED"
        ? "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300"
        : "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300";
  return (
    <Badge className={cn("text-xs font-semibold px-3 py-1", variant)}>
      {v}
    </Badge>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ClusterComparisonPanel({
  baselineId,
  candidateId,
  onPromote,
}: ClusterComparisonPanelProps) {
  const [showMigrationTable, setShowMigrationTable] = useState(false);

  const { data, isLoading, isError, error } = useQuery<ClusterExperimentComparison>({
    queryKey: clusterExperimentKeys.compare(baselineId, candidateId),
    queryFn: () => fetchClusterComparison(baselineId, candidateId),
    staleTime: CLUSTER_EXP_STALE.COMPARE,
    enabled: baselineId > 0 && candidateId > 0,
  });

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-12">
          <LoadingElement message="Loading comparison..." />
        </CardContent>
      </Card>
    );
  }

  if (isError) {
    return (
      <Card>
        <CardContent className="py-12 text-center text-sm text-destructive">
          Failed to load comparison: {(error as Error).message}
        </CardContent>
      </Card>
    );
  }

  if (!data) return null;

  const { experiment_a, experiment_b, quality_comparison, profile_comparison } = data;

  // Migration data
  const migrationLabels = Object.keys(data.migration_matrix);
  const allTargetLabels = new Set<string>();
  for (const row of Object.values(data.migration_matrix)) {
    for (const col of Object.keys(row)) {
      allTargetLabels.add(col);
    }
  }
  const targetLabels = Array.from(allTargetLabels).sort();

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">
            #{experiment_a.experiment_id} vs #{experiment_b.experiment_id}
          </CardTitle>
          <VerdictBadge verdict={quality_comparison.verdict} />
        </div>
        <p className="text-xs text-muted-foreground">
          {experiment_a.label} (baseline) vs {experiment_b.label} (candidate)
        </p>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* 1. Quality Metrics Header */}
        <div className="space-y-1">
          <div className="grid grid-cols-4 gap-2 text-[10px] text-muted-foreground uppercase tracking-wider px-1">
            <span>Metric</span>
            <span className="text-center">Baseline</span>
            <span className="text-center">Candidate</span>
            <span className="text-center">Delta</span>
          </div>
          <div className="rounded-lg border border-border/60 bg-card px-3 py-1">
            <MetricRow
              label="Silhouette"
              valueA={experiment_a.silhouette_score}
              valueB={experiment_b.silhouette_score}
              delta={quality_comparison.silhouette_delta}
              higherIsBetter
            />
            <MetricRow
              label="Inertia"
              valueA={experiment_a.inertia}
              valueB={experiment_b.inertia}
              delta={quality_comparison.inertia_delta}
              higherIsBetter={false}
              formatter={(v) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            />
            <MetricRow
              label="K (clusters)"
              valueA={experiment_a.optimal_k}
              valueB={experiment_b.optimal_k}
              delta={quality_comparison.k_delta}
              higherIsBetter
              formatter={(v) => String(Math.round(v))}
            />
            <MetricRow
              label="DFUs"
              valueA={experiment_a.total_dfus}
              valueB={experiment_b.total_dfus}
              delta={(experiment_b.total_dfus ?? 0) - (experiment_a.total_dfus ?? 0)}
              higherIsBetter
              formatter={(v) => v.toLocaleString()}
            />
          </div>
        </div>

        {/* 2. Cluster Profile Comparison */}
        {(profile_comparison.common_clusters.length > 0 ||
          profile_comparison.clusters_only_in_a.length > 0 ||
          profile_comparison.clusters_only_in_b.length > 0) && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-foreground">Cluster Profiles</p>

            {profile_comparison.clusters_only_in_a.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-1">
                <span className="text-[10px] text-muted-foreground mr-1">Only in baseline:</span>
                {profile_comparison.clusters_only_in_a.map((c) => (
                  <span
                    key={c}
                    className="inline-block px-1.5 py-0.5 text-[10px] rounded bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300"
                    title={c}
                  >
                    {formatClusterLabel(c)}
                  </span>
                ))}
              </div>
            )}

            {profile_comparison.clusters_only_in_b.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-1">
                <span className="text-[10px] text-muted-foreground mr-1">Only in candidate:</span>
                {profile_comparison.clusters_only_in_b.map((c) => (
                  <span
                    key={c}
                    className="inline-block px-1.5 py-0.5 text-[10px] rounded bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300"
                    title={c}
                  >
                    {formatClusterLabel(c)}
                  </span>
                ))}
              </div>
            )}

            {profile_comparison.common_clusters.length > 0 && (
              <div className="overflow-x-auto max-h-[200px] overflow-y-auto border rounded-md">
                <table className="w-full text-xs">
                  <thead className="bg-muted/50 sticky top-0">
                    <tr>
                      <th className="text-left px-2 py-1.5 font-medium">Cluster</th>
                      <th className="text-right px-2 py-1.5 font-medium">Baseline</th>
                      <th className="text-right px-2 py-1.5 font-medium">Candidate</th>
                      <th className="text-right px-2 py-1.5 font-medium">Delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {profile_comparison.common_clusters.map((c) => {
                      const delta = c.count_b - c.count_a;
                      return (
                        <tr
                          key={c.label}
                          className="border-t border-border/40 hover:bg-muted/30"
                        >
                          <td className="px-2 py-1 truncate max-w-[140px]" title={c.label}>
                            {formatClusterLabel(c.label)}
                          </td>
                          <td className="text-right px-2 py-1 tabular-nums">
                            {c.count_a.toLocaleString()}
                          </td>
                          <td className="text-right px-2 py-1 tabular-nums">
                            {c.count_b.toLocaleString()}
                          </td>
                          <td
                            className={cn(
                              "text-right px-2 py-1 tabular-nums font-medium",
                              delta > 0
                                ? "text-emerald-600 dark:text-emerald-400"
                                : delta < 0
                                  ? "text-red-600 dark:text-red-400"
                                  : "text-muted-foreground",
                            )}
                          >
                            {delta > 0 ? "+" : ""}
                            {delta.toLocaleString()}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* 3. DFU Migration Summary */}
        <div className="space-y-2">
          <p className="text-xs font-medium text-foreground">DFU Migration</p>
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-md border border-border/60 px-3 py-2 text-center">
              <p className="text-[10px] text-muted-foreground">Migrated</p>
              <p className="text-lg font-bold tabular-nums text-amber-600 dark:text-amber-400">
                {(data.total_dfus_migrated ?? 0).toLocaleString()}
              </p>
            </div>
            <div className="rounded-md border border-border/60 px-3 py-2 text-center">
              <p className="text-[10px] text-muted-foreground">Unchanged</p>
              <p className="text-lg font-bold tabular-nums text-emerald-600 dark:text-emerald-400">
                {(data.total_dfus_unchanged ?? 0).toLocaleString()}
              </p>
            </div>
          </div>

          {/* Migration matrix table view */}
          {migrationLabels.length > 0 && (
            <div>
              <button
                onClick={() => setShowMigrationTable((v) => !v)}
                className="text-[10px] text-primary hover:underline"
              >
                {showMigrationTable ? "Hide migration matrix" : "Show migration matrix"}
              </button>
              {showMigrationTable && (
                <div className="overflow-x-auto mt-2 border rounded-md" style={{ minWidth: 500 }}>
                  <table className="w-full text-[10px]" role="table" aria-label="DFU migration matrix">
                    <thead className="bg-muted/50">
                      <tr>
                        <th className="text-left px-2 py-1 font-medium">From / To</th>
                        {targetLabels.map((col) => (
                          <th key={col} className="text-right px-2 py-1 font-medium truncate max-w-[80px]" title={col}>
                            {formatClusterLabel(col)}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {migrationLabels.map((row) => (
                        <tr key={row} className="border-t border-border/40">
                          <td className="px-2 py-1 font-medium truncate max-w-[100px]" title={row}>
                            {formatClusterLabel(row)}
                          </td>
                          {targetLabels.map((col) => {
                            const count = data.migration_matrix[row]?.[col] ?? 0;
                            return (
                              <td
                                key={col}
                                className={cn(
                                  "text-right px-2 py-1 tabular-nums",
                                  count > 0 && row !== col
                                    ? "text-amber-600 dark:text-amber-400 font-medium"
                                    : count > 0
                                      ? "text-foreground"
                                      : "text-muted-foreground/40",
                                )}
                                title={
                                  count > 0
                                    ? `${count.toLocaleString()} DFUs moved from ${row} to ${col}`
                                    : undefined
                                }
                              >
                                {count > 0 ? count.toLocaleString() : "-"}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>

        {/* 4. PCA Cluster Visualizations (side by side) */}
        {(() => {
          const pcaA: PCAScatterData | undefined = experiment_a.k_selection_results?.pca_scatter ?? undefined;
          const pcaB: PCAScatterData | undefined = experiment_b.k_selection_results?.pca_scatter ?? undefined;
          if (!pcaA && !pcaB) return null;

          return (
            <div className="space-y-2">
              <p className="text-xs font-medium text-foreground">Cluster Visualization (2D PCA)</p>
              <div className="grid grid-cols-1 gap-4">
                {pcaA && pcaA.points.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-[10px] text-muted-foreground font-medium">
                      Baseline — {experiment_a.label} (K={experiment_a.optimal_k})
                    </p>
                    <PCAScatterChart pca={pcaA} />
                  </div>
                )}
                {pcaB && pcaB.points.length > 0 && (
                  <div className="space-y-1">
                    <p className="text-[10px] text-muted-foreground font-medium">
                      Candidate — {experiment_b.label} (K={experiment_b.optimal_k})
                    </p>
                    <PCAScatterChart pca={pcaB} />
                  </div>
                )}
              </div>
            </div>
          );
        })()}

        {/* Promote button */}
        <div className="flex justify-end gap-2 pt-2 border-t border-border/40">
          <Button
            size="sm"
            className="text-xs gap-1 bg-amber-600 hover:bg-amber-700 text-white"
            onClick={() => onPromote(experiment_b)}
            disabled={experiment_b.status !== "completed"}
          >
            <Crown className="h-3 w-3" />
            Promote Candidate
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
