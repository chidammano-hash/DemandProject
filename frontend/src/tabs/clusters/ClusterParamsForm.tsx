/**
 * ClusterParamsForm -- Reusable 3-column parameter form for cluster experiment
 * configuration. Groups: Data Scope, Model, Labeling Thresholds.
 *
 * Used by ClusterExperimentBuilder and potentially the WhatIfPanel.
 */
import type { FeatureParams, ModelParams, LabelParams } from "@/api/queries";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

// Core features used when all_features is off (matches train_clustering_model.py CORE_FEATURES)
const CORE_FEATURES = [
  "mean_demand", "cv_demand", "iqr_demand", "trend_slope_norm", "trend_r2",
  "cagr", "seasonal_amplitude", "seasonal_r2", "yoy_correlation",
  "periodicity_strength", "zero_demand_pct", "adi", "months_available", "recency_ratio",
];

export interface ClusterParamsFormProps {
  featureParams: FeatureParams;
  modelParams: ModelParams;
  labelParams: LabelParams;
  onChange: (
    section: "feature" | "model" | "label",
    key: string,
    value: number | boolean | null,
  ) => void;
  defaults?: {
    featureParams?: FeatureParams;
    modelParams?: ModelParams;
    labelParams?: LabelParams;
  };
  disabled?: boolean;
  /** Optional feature list fetched from API; falls back to hardcoded CORE_FEATURES */
  features?: string[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function NumberInput({
  label,
  value,
  onChange,
  defaultValue,
  step = 1,
  min,
  max,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  defaultValue?: number;
  step?: number;
  min?: number;
  max?: number;
  disabled?: boolean;
}) {
  const isDelta = defaultValue !== undefined && value !== defaultValue;
  return (
    <div className="space-y-1">
      <label className="text-xs text-muted-foreground font-medium">{label}</label>
      <div className="flex items-center gap-2">
        <input
          type="number"
          value={value}
          onChange={(e) => {
            const v = step < 1 ? parseFloat(e.target.value) : parseInt(e.target.value, 10);
            if (Number.isFinite(v)) onChange(v);
          }}
          step={step}
          min={min}
          max={max}
          disabled={disabled}
          className={cn(
            "w-full rounded-md border border-border bg-background px-2.5 py-1.5 text-sm tabular-nums",
            "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-1",
            "disabled:cursor-not-allowed disabled:opacity-50",
            isDelta && "border-amber-400 dark:border-amber-600",
          )}
        />
        {isDelta && defaultValue !== undefined && (
          <span className="text-[10px] text-amber-600 dark:text-amber-400 whitespace-nowrap tabular-nums">
            {value > defaultValue ? "+" : ""}
            {step < 1
              ? (value - defaultValue).toFixed(2)
              : String(value - defaultValue)}
          </span>
        )}
      </div>
    </div>
  );
}

function CheckboxInput({
  label,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className="flex items-center gap-2 cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        className="rounded border-border"
      />
      <span className="text-xs text-muted-foreground font-medium">{label}</span>
    </label>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ClusterParamsForm({
  featureParams,
  modelParams,
  labelParams,
  onChange,
  defaults,
  disabled = false,
  features,
}: ClusterParamsFormProps) {
  const resolvedFeatures = features ?? CORE_FEATURES;
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {/* Data Scope */}
      <div className="space-y-3">
        <h4 className="text-xs font-semibold text-foreground uppercase tracking-wider">
          Data Scope
        </h4>
        <NumberInput
          label="Time Window (months)"
          value={featureParams.time_window_months}
          onChange={(v) => onChange("feature", "time_window_months", v)}
          defaultValue={defaults?.featureParams?.time_window_months}
          min={6}
          max={120}
          disabled={disabled}
        />
        <NumberInput
          label="Min Months History"
          value={featureParams.min_months_history}
          onChange={(v) => onChange("feature", "min_months_history", v)}
          defaultValue={defaults?.featureParams?.min_months_history}
          min={1}
          max={24}
          disabled={disabled}
        />
      </div>

      {/* Model */}
      <div className="space-y-3">
        <h4 className="text-xs font-semibold text-foreground uppercase tracking-wider">
          Model
        </h4>
        <div className="grid grid-cols-2 gap-2">
          <NumberInput
            label="K Range Min"
            value={modelParams.k_range[0]}
            onChange={(v) => onChange("model", "k_range_min", v)}
            defaultValue={defaults?.modelParams?.k_range[0]}
            min={2}
            max={modelParams.k_range[1]}
            disabled={disabled}
          />
          <NumberInput
            label="K Range Max"
            value={modelParams.k_range[1]}
            onChange={(v) => onChange("model", "k_range_max", v)}
            defaultValue={defaults?.modelParams?.k_range[1]}
            min={modelParams.k_range[0]}
            max={50}
            disabled={disabled}
          />
        </div>
        <NumberInput
          label="Min Cluster Size %"
          value={modelParams.min_cluster_size_pct}
          onChange={(v) => onChange("model", "min_cluster_size_pct", v)}
          defaultValue={defaults?.modelParams?.min_cluster_size_pct}
          step={0.5}
          min={0.5}
          max={20}
          disabled={disabled}
        />
        <CheckboxInput
          label="All Features"
          checked={modelParams.all_features ?? false}
          onChange={(v) => onChange("model", "all_features", v)}
          disabled={disabled}
        />
        {!(modelParams.all_features ?? false) && (
          <p className="text-[10px] text-muted-foreground -mt-1">
            Core: volume, trend, seasonality, intermittency ({resolvedFeatures.length} features)
          </p>
        )}
        {(modelParams.all_features ?? false) && (
          <p className="text-[10px] text-muted-foreground -mt-1">
            All numeric features from clustering feature set
          </p>
        )}
        <CheckboxInput
          label="Use PCA"
          checked={modelParams.use_pca}
          onChange={(v) => onChange("model", "use_pca", v)}
          disabled={disabled}
        />
        {modelParams.use_pca && (
          <NumberInput
            label="PCA Components"
            value={modelParams.pca_components ?? 10}
            onChange={(v) => onChange("model", "pca_components", v)}
            defaultValue={defaults?.modelParams?.pca_components ?? 10}
            min={2}
            max={50}
            disabled={disabled}
          />
        )}
      </div>

      {/* Labeling Thresholds */}
      <div className="space-y-3">
        <h4 className="text-xs font-semibold text-foreground uppercase tracking-wider">
          Labeling Thresholds
        </h4>
        <NumberInput
          label="Volume High"
          value={labelParams.volume_high ?? 0.75}
          onChange={(v) => onChange("label", "volume_high", v)}
          defaultValue={defaults?.labelParams?.volume_high}
          step={0.05}
          min={0}
          max={1}
          disabled={disabled}
        />
        <NumberInput
          label="Volume Low"
          value={labelParams.volume_low ?? 0.25}
          onChange={(v) => onChange("label", "volume_low", v)}
          defaultValue={defaults?.labelParams?.volume_low}
          step={0.05}
          min={0}
          max={1}
          disabled={disabled}
        />
        <NumberInput
          label="CV Steady"
          value={labelParams.cv_steady ?? 0.3}
          onChange={(v) => onChange("label", "cv_steady", v)}
          defaultValue={defaults?.labelParams?.cv_steady}
          step={0.05}
          min={0}
          max={2}
          disabled={disabled}
        />
        <NumberInput
          label="CV Volatile"
          value={labelParams.cv_volatile ?? 0.8}
          onChange={(v) => onChange("label", "cv_volatile", v)}
          defaultValue={defaults?.labelParams?.cv_volatile}
          step={0.05}
          min={0}
          max={3}
          disabled={disabled}
        />
        <NumberInput
          label="Seasonality Threshold"
          value={labelParams.seasonality_threshold ?? 0.5}
          onChange={(v) => onChange("label", "seasonality_threshold", v)}
          defaultValue={defaults?.labelParams?.seasonality_threshold}
          step={0.05}
          min={0}
          max={1}
          disabled={disabled}
        />
        <NumberInput
          label="Zero Demand Threshold"
          value={labelParams.zero_demand_threshold ?? 0.2}
          onChange={(v) => onChange("label", "zero_demand_threshold", v)}
          defaultValue={defaults?.labelParams?.zero_demand_threshold}
          step={0.05}
          min={0}
          max={1}
          disabled={disabled}
        />
      </div>
    </div>
  );
}
