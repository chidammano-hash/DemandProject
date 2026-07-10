import type { RefObject } from "react";
import type { ClusteringDefaultsPayload, ClusteringScenarioResult } from "@/api/queries";
import { cn } from "@/lib/utils";
import { formatCompactNumber } from "@/lib/formatters";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// ---------------------------------------------------------------------------
// Param input helper
// ---------------------------------------------------------------------------
function ParamInput({
  label,
  description,
  value,
  onChange,
  step,
  min,
  max,
}: {
  label: string;
  description?: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="font-semibold text-muted-foreground">{label}</span>
      {description && (
        <span className="text-[10px] leading-tight text-muted-foreground/70">{description}</span>
      )}
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
// Types
// ---------------------------------------------------------------------------
type ScenarioEstimate = {
  estimated_seconds: number;
  sku_count: number;
  sampled: boolean;
  training_sample: number;
};

type StatusData = {
  status: string;
  elapsed_seconds?: number | null;
};

type WhatIfPanelProps = {
  showWhatIf: boolean;
  onToggle: () => void;

  featureParams: ClusteringDefaultsPayload["feature_params"];
  modelParams: ClusteringDefaultsPayload["model_params"];
  labelParams: ClusteringDefaultsPayload["label_params"];
  setFeatureParams: React.Dispatch<React.SetStateAction<ClusteringDefaultsPayload["feature_params"]>>;
  setModelParams: React.Dispatch<React.SetStateAction<ClusteringDefaultsPayload["model_params"]>>;
  setLabelParams: React.Dispatch<React.SetStateAction<ClusteringDefaultsPayload["label_params"]>>;

  scenarioRunning: boolean;
  scenarioQueued: boolean;
  scenarioError: string | null;
  scheduledJobId: string | null;
  estimate: ScenarioEstimate | undefined;
  statusData: StatusData | undefined;

  onRunScenario: () => void;
  onReset: () => void;
  children?: React.ReactNode;
};

// ---------------------------------------------------------------------------
// WhatIfPanel
// ---------------------------------------------------------------------------
export default function WhatIfPanel({
  showWhatIf,
  onToggle,
  featureParams,
  modelParams,
  labelParams,
  setFeatureParams,
  setModelParams,
  setLabelParams,
  scenarioRunning,
  scenarioQueued,
  scenarioError,
  scheduledJobId,
  estimate,
  statusData,
  onRunScenario,
  onReset,
  children,
}: WhatIfPanelProps) {
  return (
    <Card className="mt-4 animate-fade-in">
      <CardHeader className="cursor-pointer" onClick={onToggle}>
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
                description="How many months of sales history to analyze. Longer windows capture more patterns but may include outdated trends. Recommended: 24-48 months."
                value={featureParams.time_window_months}
                onChange={(v) => setFeatureParams((p) => ({ ...p, time_window_months: v }))}
                min={1}
                max={120}
              />
              <ParamInput
                label="Min History (months)"
                description="SKUs with fewer months of data are excluded from clustering. Lower values include newer products but may produce noisy features."
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
                <span className="block text-[10px] leading-tight text-muted-foreground/70 mt-0.5">Range of cluster counts to evaluate. The algorithm tests each K value and selects the optimal one using combined Silhouette + Calinski-Harabasz scoring.</span>
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
                description="Minimum percentage of SKUs required in each cluster. Prevents tiny, unstable clusters. Higher values produce more balanced groups for model training."
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
                description="Percentile threshold for 'high volume' classification. SKUs above this percentile are labeled high/very_high volume."
                value={labelParams.volume_high}
                onChange={(v) => setLabelParams((p) => ({ ...p, volume_high: v }))}
                step={0.05}
                min={0}
                max={1}
              />
              <ParamInput
                label="Volume Low (pctl)"
                description="Percentile threshold for 'low volume' classification. SKUs below this percentile are labeled low/very_low volume."
                value={labelParams.volume_low}
                onChange={(v) => setLabelParams((p) => ({ ...p, volume_low: v }))}
                step={0.05}
                min={0}
                max={1}
              />
              <ParamInput
                label="CV Steady (<)"
                description="Coefficient of Variation threshold below which demand is considered 'steady'. Lower CV = more predictable demand."
                value={labelParams.cv_steady}
                onChange={(v) => setLabelParams((p) => ({ ...p, cv_steady: v }))}
                step={0.05}
                min={0}
              />
              <ParamInput
                label="CV Volatile (>)"
                description="Coefficient of Variation threshold above which demand is considered 'volatile'. Higher CV = more erratic demand patterns."
                value={labelParams.cv_volatile}
                onChange={(v) => setLabelParams((p) => ({ ...p, cv_volatile: v }))}
                step={0.05}
                min={0}
              />
              <ParamInput
                label="Seasonality Threshold"
                description="Seasonal amplitude ratio above which a DFU is classified as 'seasonal'. Measures peak-to-trough swing relative to average demand."
                value={labelParams.seasonality_threshold}
                onChange={(v) => setLabelParams((p) => ({ ...p, seasonality_threshold: v }))}
                step={0.05}
                min={0}
                max={1}
              />
              <ParamInput
                label="Zero Demand Threshold"
                description="Fraction of months with zero sales above which a DFU is classified as 'intermittent'. E.g., 0.15 means >15% zero months."
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
              onClick={onRunScenario}
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
              onClick={onReset}
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
                {estimate.sku_count > 0 && (
                  <span className="ml-1 text-[10px]">
                    ({formatCompactNumber(estimate.sku_count)} SKUs{estimate.sampled ? `, ${formatCompactNumber(estimate.training_sample)} sample` : ""})
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
                  : "Job scheduled successfully. Track progress in the Workflow Library."}
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
        </CardContent>
      )}
      {showWhatIf && children}
    </Card>
  );
}

// Re-export the ref type so ScenarioResultsPanel can use the same shape
export type { ScenarioEstimate, StatusData };
