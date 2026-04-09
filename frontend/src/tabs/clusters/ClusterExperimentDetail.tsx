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

// ---------------------------------------------------------------------------
// Config summary helper
// ---------------------------------------------------------------------------

function ExperimentConfigSummary({
  featureParams,
  modelParams,
  labelParams,
}: {
  featureParams: ClusterExperiment["feature_params"];
  modelParams: ClusterExperiment["model_params"];
  labelParams: ClusterExperiment["label_params"];
}) {
  const items: { label: string; value: string }[] = [];

  if (modelParams) {
    if (modelParams.k_range)
      items.push({ label: "K Range", value: `${modelParams.k_range[0]}--${modelParams.k_range[1]}` });
    if (modelParams.min_cluster_size_pct != null)
      items.push({ label: "Min Cluster %", value: `${modelParams.min_cluster_size_pct}%` });
    items.push({ label: "Use PCA", value: modelParams.use_pca ? "Yes" : "No" });
    if (modelParams.use_pca && modelParams.pca_components != null)
      items.push({ label: "PCA Components", value: String(modelParams.pca_components) });
    if (modelParams.all_features != null)
      items.push({ label: "All Features", value: modelParams.all_features ? "Yes" : "No" });
  }

  if (featureParams) {
    if (featureParams.time_window_months != null)
      items.push({ label: "Time Window", value: `${featureParams.time_window_months}mo` });
    if (featureParams.min_months_history != null)
      items.push({ label: "Min History", value: `${featureParams.min_months_history}mo` });
  }

  if (labelParams) {
    if (labelParams.volume_high != null)
      items.push({ label: "Vol High", value: String(labelParams.volume_high) });
    if (labelParams.volume_low != null)
      items.push({ label: "Vol Low", value: String(labelParams.volume_low) });
    if (labelParams.cv_steady != null)
      items.push({ label: "CV Steady", value: String(labelParams.cv_steady) });
    if (labelParams.cv_volatile != null)
      items.push({ label: "CV Volatile", value: String(labelParams.cv_volatile) });
    if (labelParams.seasonality_threshold != null)
      items.push({ label: "Seasonality", value: String(labelParams.seasonality_threshold) });
    if (labelParams.zero_demand_threshold != null)
      items.push({ label: "Zero Demand", value: String(labelParams.zero_demand_threshold) });
  }

  if (items.length === 0) return null;

  return (
    <div className="rounded-md border border-border/60 bg-muted/20 px-3 py-2">
      <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1.5">
        Configuration
      </p>
      <div className="grid grid-cols-3 gap-x-4 gap-y-1">
        {items.map((item) => (
          <div key={item.label} className="flex items-baseline justify-between gap-1">
            <span className="text-[10px] text-muted-foreground truncate">{item.label}</span>
            <span className="text-[11px] font-medium tabular-nums">{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

interface ClusterExperimentDetailProps {
  experiment: ClusterExperiment;
}

export function ClusterExperimentDetail({
  experiment,
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
        </div>
        {experiment.notes && (
          <p className="text-xs text-muted-foreground mt-1">
            {experiment.notes}
          </p>
        )}
      </CardHeader>

      <CardContent className="space-y-4 pt-0">
        {/* Experiment Config Summary */}
        {(experiment.feature_params || experiment.model_params || experiment.label_params) && (
          <ExperimentConfigSummary
            featureParams={experiment.feature_params}
            modelParams={experiment.model_params}
            labelParams={experiment.label_params}
          />
        )}

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
