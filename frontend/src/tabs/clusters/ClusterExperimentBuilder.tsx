/**
 * ClusterExperimentBuilder -- Full-screen modal for creating cluster experiments.
 *
 * Sections:
 * 1. Label + Notes inputs
 * 2. Template radio buttons (7 templates)
 * 3. ClusterParamsForm with current values
 * 4. Estimate bar (runtime + DFU count from /clustering/scenario/estimate)
 * 5. Cancel + Launch footer
 */
import { useCallback, useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { X, Loader2, FlaskConical, AlertTriangle, Clock, Users } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  clusterExperimentKeys,
  CLUSTER_EXP_STALE,
  createClusterExperiment,
  fetchClusterTemplates,
  type ClusterExperimentTemplate,
  type FeatureParams,
  type ModelParams,
  type LabelParams,
} from "@/api/queries";
import { fetchScenarioEstimate, STALE } from "@/api/queries";
import { ClusterParamsForm } from "./ClusterParamsForm";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ClusterExperimentBuilderProps {
  open: boolean;
  onClose: () => void;
  onSubmitted: () => void;
  /** Pre-populate from a cloned experiment */
  cloneFrom?: {
    featureParams: FeatureParams;
    modelParams: ModelParams;
    labelParams: LabelParams;
    label?: string;
    notes?: string;
  };
}

// ---------------------------------------------------------------------------
// Default params
// ---------------------------------------------------------------------------

const DEFAULT_FEATURE_PARAMS: FeatureParams = {
  time_window_months: 24,
  min_months_history: 1,
};

const DEFAULT_MODEL_PARAMS: ModelParams = {
  k_range: [3, 12],
  min_cluster_size_pct: 2.0,
  use_pca: false,
  pca_components: null,
  all_features: false,
};

const DEFAULT_LABEL_PARAMS: LabelParams = {
  volume_high: 0.75,
  volume_low: 0.25,
  cv_steady: 0.3,
  cv_volatile: 0.8,
  seasonality_threshold: 0.5,
  zero_demand_threshold: 0.2,
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ClusterExperimentBuilder({
  open,
  onClose,
  onSubmitted,
  cloneFrom,
}: ClusterExperimentBuilderProps) {
  const queryClient = useQueryClient();

  // ---- State ---------------------------------------------------------------
  const [label, setLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [selectedTemplate, setSelectedTemplate] = useState<string>("custom");

  const [featureParams, setFeatureParams] = useState<FeatureParams>(
    cloneFrom?.featureParams ?? { ...DEFAULT_FEATURE_PARAMS },
  );
  const [modelParams, setModelParams] = useState<ModelParams>(
    cloneFrom?.modelParams ?? { ...DEFAULT_MODEL_PARAMS, k_range: [...DEFAULT_MODEL_PARAMS.k_range] },
  );
  const [labelParams, setLabelParams] = useState<LabelParams>(
    cloneFrom?.labelParams ?? { ...DEFAULT_LABEL_PARAMS },
  );

  // ---- Clone pre-fill -------------------------------------------------------
  useEffect(() => {
    if (cloneFrom) {
      setFeatureParams(cloneFrom.featureParams);
      setModelParams({ ...cloneFrom.modelParams, k_range: [...cloneFrom.modelParams.k_range] });
      setLabelParams(cloneFrom.labelParams);
      if (cloneFrom.label) setLabel(`${cloneFrom.label} (clone)`);
      if (cloneFrom.notes) setNotes(cloneFrom.notes);
    }
  }, [cloneFrom]);

  // ---- Fetch templates -------------------------------------------------------
  const { data: templatesData } = useQuery({
    queryKey: clusterExperimentKeys.templates(),
    queryFn: fetchClusterTemplates,
    staleTime: CLUSTER_EXP_STALE.TEMPLATES,
    enabled: open,
  });

  const templates: ClusterExperimentTemplate[] = templatesData?.templates ?? [];

  // ---- Fetch estimate -------------------------------------------------------
  const { data: estimate } = useQuery({
    queryKey: ["scenario-estimate", modelParams.k_range[0], modelParams.k_range[1]],
    queryFn: () =>
      fetchScenarioEstimate({
        k_min: modelParams.k_range[0],
        k_max: modelParams.k_range[1],
      }),
    staleTime: STALE.THIRTY_SEC,
    enabled: open,
  });

  // ---- Template selection handler -------------------------------------------
  const handleTemplateSelect = useCallback(
    (templateId: string) => {
      setSelectedTemplate(templateId);
      const tmpl = templates.find((t) => t.id === templateId);
      if (!tmpl || templateId === "custom") {
        // Reset to defaults for custom
        setFeatureParams({ ...DEFAULT_FEATURE_PARAMS });
        setModelParams({ ...DEFAULT_MODEL_PARAMS, k_range: [...DEFAULT_MODEL_PARAMS.k_range] });
        setLabelParams({ ...DEFAULT_LABEL_PARAMS });
        return;
      }
      // Apply template overrides on top of defaults
      setFeatureParams({
        ...DEFAULT_FEATURE_PARAMS,
        ...(tmpl.feature_params ?? {}),
      });
      const tmplModelOverrides = tmpl.model_params ?? {};
      const resolvedKRange = tmpl.model_params?.k_range
        ? [...tmpl.model_params.k_range] as [number, number]
        : [...DEFAULT_MODEL_PARAMS.k_range] as [number, number];
      setModelParams({
        ...DEFAULT_MODEL_PARAMS,
        ...tmplModelOverrides,
        k_range: resolvedKRange,
      });
      setLabelParams({
        ...DEFAULT_LABEL_PARAMS,
        ...(tmpl.label_params ?? {}),
      });
    },
    [templates],
  );

  // ---- Param change handler -------------------------------------------------
  const handleParamChange = useCallback(
    (section: "feature" | "model" | "label", key: string, value: number | boolean | null) => {
      if (section === "feature") {
        setFeatureParams((prev) => ({ ...prev, [key]: value }));
      } else if (section === "model") {
        if (key === "k_range_min") {
          setModelParams((prev) => ({
            ...prev,
            k_range: [value as number, prev.k_range[1]],
          }));
        } else if (key === "k_range_max") {
          setModelParams((prev) => ({
            ...prev,
            k_range: [prev.k_range[0], value as number],
          }));
        } else {
          setModelParams((prev) => ({ ...prev, [key]: value }));
        }
      } else {
        setLabelParams((prev) => ({ ...prev, [key]: value }));
      }
    },
    [],
  );

  // ---- Create mutation -------------------------------------------------------
  const createMut = useMutation({
    mutationFn: () =>
      createClusterExperiment({
        label: label.trim(),
        notes: notes.trim() || undefined,
        template: selectedTemplate !== "custom" ? selectedTemplate : undefined,
        feature_params: featureParams,
        model_params: modelParams,
        label_params: labelParams,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: clusterExperimentKeys.all });
      onSubmitted();
      // Reset
      setLabel("");
      setNotes("");
      setSelectedTemplate("custom");
      setFeatureParams({ ...DEFAULT_FEATURE_PARAMS });
      setModelParams({ ...DEFAULT_MODEL_PARAMS, k_range: [...DEFAULT_MODEL_PARAMS.k_range] });
      setLabelParams({ ...DEFAULT_LABEL_PARAMS });
    },
  });

  // ---- Validation -----------------------------------------------------------
  const canSubmit = label.trim().length > 0 && !createMut.isPending;

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="w-full max-w-4xl rounded-xl border bg-card shadow-xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-5 py-4 shrink-0">
          <div className="flex items-center gap-2">
            <FlaskConical className="h-5 w-5 text-muted-foreground" />
            <div>
              <p className="text-sm font-semibold text-foreground">
                New Cluster Experiment
              </p>
              <p className="text-xs text-muted-foreground">
                Configure and launch a clustering scenario with tracked results
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-6">
          {/* 1. Label + Notes */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="text-xs font-medium text-foreground">
                Experiment Label <span className="text-destructive">*</span>
              </label>
              <Input
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="e.g. High-K Seasonal Focus"
                className="text-sm"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-foreground">Notes</label>
              <Input
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Optional description or hypothesis"
                className="text-sm"
              />
            </div>
          </div>

          {/* 2. Template selection */}
          <div className="space-y-2">
            <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider">
              Template
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {/* Built-in templates */}
              {[
                { id: "production_baseline", label: "Production Baseline", desc: "Current production config" },
                { id: "high_k_granular", label: "High-K Granular", desc: "K=12-25, finer segments" },
                { id: "low_k_broad", label: "Low-K Broad", desc: "K=3-8, robust clusters" },
                { id: "seasonal_focus", label: "Seasonal Focus", desc: "48mo window, low threshold" },
                { id: "intermittent_specialist", label: "Intermittent Specialist", desc: "Low zero-demand threshold" },
                { id: "pca_compressed", label: "PCA Compressed", desc: "All features + PCA" },
                { id: "custom", label: "Custom", desc: "Start from defaults" },
              ].map((t) => (
                <button
                  key={t.id}
                  onClick={() => handleTemplateSelect(t.id)}
                  className={cn(
                    "rounded-lg border p-3 text-left transition-colors",
                    selectedTemplate === t.id
                      ? "border-primary bg-primary/5 ring-1 ring-primary"
                      : "border-border hover:border-muted-foreground/30 hover:bg-muted/30",
                  )}
                >
                  <p className="text-xs font-medium text-foreground">{t.label}</p>
                  <p className="text-[10px] text-muted-foreground mt-0.5">{t.desc}</p>
                </button>
              ))}
            </div>
          </div>

          {/* 3. Parameter form */}
          <div className="space-y-2">
            <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider">
              Parameters
            </h3>
            <div className="rounded-lg border border-border p-4">
              <ClusterParamsForm
                featureParams={featureParams}
                modelParams={modelParams}
                labelParams={labelParams}
                onChange={handleParamChange}
                defaults={{
                  featureParams: DEFAULT_FEATURE_PARAMS,
                  modelParams: DEFAULT_MODEL_PARAMS,
                  labelParams: DEFAULT_LABEL_PARAMS,
                }}
              />
            </div>
          </div>

          {/* 4. Estimate bar */}
          {estimate && (
            <div className="flex items-center gap-4 rounded-lg border border-border bg-muted/20 px-4 py-2.5">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Clock className="h-3.5 w-3.5" />
                <span>Est. runtime: </span>
                <span className="font-medium text-foreground">
                  {estimate.estimated_runtime_seconds != null
                    ? `~${Math.ceil(estimate.estimated_runtime_seconds / 60)}min`
                    : "--"}
                </span>
              </div>
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Users className="h-3.5 w-3.5" />
                <span>DFUs: </span>
                <span className="font-medium text-foreground">
                  {estimate.total_dfus?.toLocaleString() ?? "--"}
                </span>
              </div>
            </div>
          )}

          {/* Error banner */}
          {createMut.isError && (
            <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-400 flex items-center gap-1.5">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
              {(createMut.error as Error).message}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 border-t px-5 py-3 shrink-0">
          <Button variant="ghost" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button
            size="sm"
            onClick={() => createMut.mutate()}
            disabled={!canSubmit}
            className="gap-1.5"
          >
            {createMut.isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <FlaskConical className="h-3.5 w-3.5" />
            )}
            Launch Experiment
          </Button>
        </div>
      </div>
    </div>
  );
}
