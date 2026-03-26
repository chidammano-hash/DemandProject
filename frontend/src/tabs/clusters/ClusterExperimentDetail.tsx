/**
 * ClusterExperimentDetail — Shows charts and profile table for a selected
 * cluster experiment. Reuses the existing ScenarioCharts component by
 * mapping ClusterExperiment fields to the expected result shape.
 */
import { Crown } from "lucide-react";
import type { ClusterExperiment } from "@/api/queries";
import { formatNumber, formatFixed, formatClusterLabel } from "@/lib/formatters";
import { ScenarioCharts, silhouetteQuality } from "@/components/ScenarioCharts";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface ClusterExperimentDetailProps {
  experiment: ClusterExperiment;
  onPromote?: () => void;
}

export function ClusterExperimentDetail({
  experiment,
  onPromote,
}: ClusterExperimentDetailProps) {
  const hasResults =
    experiment.status === "completed" &&
    experiment.k_selection_results != null &&
    experiment.profiles != null &&
    experiment.optimal_k != null &&
    experiment.silhouette_score != null;

  if (!hasResults) {
    return (
      <Card className="h-full">
        <CardContent className="flex flex-col items-center justify-center py-20 text-center">
          <p className="text-sm text-muted-foreground">
            {experiment.status === "running" || experiment.status === "queued"
              ? "Experiment is still running. Charts will appear when it completes."
              : experiment.status === "failed"
                ? "Experiment failed. No results to display."
                : "No visualization data available for this experiment."}
          </p>
        </CardContent>
      </Card>
    );
  }

  // Map ClusterExperiment → ScenarioCharts expected shape
  const kSelRaw = experiment.k_selection_results!;
  const pcaScatter = kSelRaw.pca_scatter ?? undefined;
  const chartResult = {
    optimal_k: experiment.optimal_k!,
    silhouette_score: experiment.silhouette_score!,
    inertia: experiment.inertia ?? 0,
    total_dfus: experiment.total_dfus ?? 0,
    k_selection_results: kSelRaw,
    profiles: experiment.profiles!,
  };

  const quality = silhouetteQuality(experiment.silhouette_score!);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base flex items-center gap-2">
              {experiment.label}
              {experiment.is_promoted && (
                <span title="Promoted to production">
                  <Crown className="h-4 w-4 text-amber-500" />
                </span>
              )}
            </CardTitle>
            <div className="flex items-center gap-2 mt-1">
              <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                K={experiment.optimal_k}
              </Badge>
              <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                Sil={formatFixed(experiment.silhouette_score, 4)} ({quality})
              </Badge>
              <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                {experiment.total_dfus?.toLocaleString()} DFUs
              </Badge>
              {experiment.runtime_seconds != null && (
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                  {experiment.runtime_seconds < 60
                    ? `${Math.round(experiment.runtime_seconds)}s`
                    : `${Math.floor(experiment.runtime_seconds / 60)}m ${Math.round(experiment.runtime_seconds % 60)}s`}
                </Badge>
              )}
            </div>
          </div>
          {onPromote && experiment.status === "completed" && !experiment.is_promoted && (
            <button
              className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
              onClick={onPromote}
            >
              Promote
            </button>
          )}
        </div>
        {experiment.notes && (
          <p className="text-xs text-muted-foreground mt-1">
            {experiment.notes}
          </p>
        )}
      </CardHeader>

      <CardContent className="space-y-4 pt-0">
        {/* Profile table */}
        <div className="max-h-[240px] overflow-y-auto rounded-md border border-input">
          <Table>
            <TableHeader>
              <TableRow className="border-muted bg-muted/30">
                <TableHead className="text-xs">Cluster</TableHead>
                <TableHead className="text-xs text-right">DFUs</TableHead>
                <TableHead className="text-xs text-right">%</TableHead>
                <TableHead className="text-xs text-right">Avg Demand</TableHead>
                <TableHead className="text-xs text-right">CV</TableHead>
                <TableHead className="text-xs text-right">Seasonality</TableHead>
                <TableHead className="text-xs text-right">Trend</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {experiment.profiles!.map((p) => (
                <TableRow key={p.label}>
                  <TableCell className="text-sm font-medium" title={p.label}>{formatClusterLabel(p.label)}</TableCell>
                  <TableCell className="text-right text-sm tabular-nums">
                    {formatNumber(p.count)}
                  </TableCell>
                  <TableCell className="text-right text-sm tabular-nums">
                    {formatNumber(p.pct_of_total)}%
                  </TableCell>
                  <TableCell className="text-right text-sm tabular-nums">
                    {formatNumber(p.mean_demand)}
                  </TableCell>
                  <TableCell className="text-right text-sm tabular-nums">
                    {formatNumber(p.cv_demand)}
                  </TableCell>
                  <TableCell className="text-right text-sm tabular-nums">
                    {formatNumber(p.seasonality_strength)}
                  </TableCell>
                  <TableCell className="text-right text-sm tabular-nums">
                    {formatFixed(p.trend_slope, 3)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        {/* Charts */}
        <ScenarioCharts result={chartResult} pcaScatter={pcaScatter} />
      </CardContent>
    </Card>
  );
}
