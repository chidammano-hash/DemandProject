/**
 * ClusterScenarioConfigPanel — Inline clustering parameter controls
 * for the cluster_scenario job card in JobGroupsPanel.
 * Shows key params (time window, K range, min cluster size) and a Run button.
 */
import { useCallback, useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2, Zap } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  queryKeys,
  STALE,
  fetchClusteringDefaults,
  submitJob,
  type ClusteringDefaultsPayload,
} from "@/api/queries";

export interface ClusterScenarioConfigPanelProps {
  onJobSubmitted?: () => void;
}

export function ClusterScenarioConfigPanel({ onJobSubmitted }: ClusterScenarioConfigPanelProps) {
  const [featureParams, setFeatureParams] = useState<ClusteringDefaultsPayload["feature_params"] | null>(null);
  const [modelParams, setModelParams] = useState<ClusteringDefaultsPayload["model_params"] | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const { data: defaults } = useQuery({
    queryKey: queryKeys.clusteringDefaults(),
    queryFn: fetchClusteringDefaults,
    staleTime: STALE.TEN_MIN,
  });

  // Init local state from defaults
  useEffect(() => {
    if (defaults && featureParams === null) {
      setFeatureParams(defaults.feature_params);
      setModelParams(defaults.model_params);
    }
  }, [defaults, featureParams]);

  const handleSubmit = useCallback(async () => {
    if (!featureParams || !modelParams) return;
    setSubmitting(true);
    try {
      await submitJob(
        "cluster_scenario",
        { feature_params: featureParams, model_params: modelParams },
        "Clustering What-If",
      );
      onJobSubmitted?.();
    } finally {
      setSubmitting(false);
    }
  }, [featureParams, modelParams, onJobSubmitted]);

  if (!featureParams || !modelParams) {
    return <p className="text-xs text-muted-foreground">Loading defaults...</p>;
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-4">
        {/* Time Window */}
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Time Window
          <input
            type="number"
            className="h-8 w-20 rounded-md border border-input bg-background px-2 text-sm tabular-nums"
            value={featureParams.time_window_months}
            min={1}
            max={120}
            disabled={submitting}
            onChange={(e) => setFeatureParams((p) => p && ({ ...p, time_window_months: Number(e.target.value) }))}
          />
          <span className="text-[10px] font-normal normal-case text-muted-foreground/70 block">months</span>
        </label>

        {/* K Range */}
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          K Range
          <div className="flex items-center gap-1.5">
            <input
              type="number"
              className="h-8 w-16 rounded-md border border-input bg-background px-2 text-sm tabular-nums"
              value={modelParams.k_range[0]}
              min={2}
              disabled={submitting}
              onChange={(e) => setModelParams((p) => p && ({ ...p, k_range: [Number(e.target.value), p.k_range[1]] }))}
            />
            <span className="text-xs text-muted-foreground">–</span>
            <input
              type="number"
              className="h-8 w-16 rounded-md border border-input bg-background px-2 text-sm tabular-nums"
              value={modelParams.k_range[1]}
              min={2}
              disabled={submitting}
              onChange={(e) => setModelParams((p) => p && ({ ...p, k_range: [p.k_range[0], Number(e.target.value)] }))}
            />
          </div>
        </label>

        {/* Min Cluster Size */}
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Min Size
          <div className="flex items-center gap-1">
            <input
              type="number"
              className="h-8 w-16 rounded-md border border-input bg-background px-2 text-sm tabular-nums"
              value={modelParams.min_cluster_size_pct}
              step={0.5}
              min={0}
              max={50}
              disabled={submitting}
              onChange={(e) => setModelParams((p) => p && ({ ...p, min_cluster_size_pct: Number(e.target.value) }))}
            />
            <span className="text-xs text-muted-foreground">%</span>
          </div>
        </label>

        {/* Run button */}
        <button
          disabled={submitting}
          onClick={handleSubmit}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
            submitting
              ? "bg-muted text-muted-foreground cursor-not-allowed"
              : "bg-primary text-primary-foreground hover:bg-primary/90",
          )}
        >
          {submitting ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <Zap className="h-3 w-3" />
          )}
          Run Scenario
        </button>
      </div>
    </div>
  );
}
