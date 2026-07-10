/**
 * ExperimentBuilder -- Full-screen dialog for configuring and launching a new
 * tuning experiment. Features template radio buttons (production baseline,
 * 4 expert templates, custom), a hyperparameter form with Value/Default/Delta
 * columns, training config section, validation with inline errors, and launch
 * with loading state.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { X, Loader2, FlaskConical, AlertTriangle, ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import type { ModelType } from "@/api/queries";
import {
  clusterExperimentKeys,
  fetchCompletedClusterExperiments,
  submitModelExperiment,
  type ClusterExperiment,
} from "@/api/queries";

import {
  getParamSpecs,
  getDefaults,
  getTemplates,
  validate,
  DEFAULT_CONFIG,
  type TrainingConfig,
  type ValidationError,
} from "@/lib/model-params";
import { TemplateSelector } from "./TemplateSelector";
import { ParameterTable } from "./ParameterTable";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface ExperimentBuilderProps {
  model: ModelType;
  open: boolean;
  onClose: () => void;
  onSubmitted: () => void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const MODEL_LABELS: Record<ModelType, string> = {
  lgbm: "LightGBM",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function ExperimentBuilder({ model, open, onClose, onSubmitted }: ExperimentBuilderProps) {
  const queryClient = useQueryClient();
  const templates = useMemo(() => getTemplates(model), [model]);
  const paramSpecs = useMemo(() => getParamSpecs(model), [model]);
  const defaults = useMemo(() => getDefaults(model), [model]);

  const [selectedTemplate, setSelectedTemplate] = useState<string>("production");
  const [runLabel, setRunLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [params, setParams] = useState<Record<string, unknown>>({ ...defaults });
  const [config, setConfig] = useState<TrainingConfig>({ ...DEFAULT_CONFIG });
  const [errors, setErrors] = useState<ValidationError[]>([]);
  const [paramsExpanded, setParamsExpanded] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [clusterSource, setClusterSource] = useState<"production" | "experimental">("production");
  const [clusterExperimentId, setClusterExperimentId] = useState<number | null>(null);

  // Fetch completed cluster experiments for the dropdown
  const { data: completedExperimentsData } = useQuery({
    queryKey: clusterExperimentKeys.completed(),
    queryFn: fetchCompletedClusterExperiments,
    staleTime: 300_000, // 5 min
  });
  const completedExperiments = completedExperimentsData?.experiments ?? [];

  const warnings: string[] = [];

  // Reset form when model changes
  useEffect(() => {
    const d = getDefaults(model);
    setParams({ ...d });
    setSelectedTemplate("production");
    setRunLabel("");
    setNotes("");
    setErrors([]);
    setConfig({ ...DEFAULT_CONFIG });
    setParamsExpanded(false);
    setAdvancedOpen(false);
    setClusterSource("production");
    setClusterExperimentId(null);
  }, [model]);

  // Apply template
  const applyTemplate = useCallback(
    (templateId: string) => {
      setSelectedTemplate(templateId);
      const tmpl = templates.find((t) => t.id === templateId);
      if (tmpl) {
        setParams({ ...tmpl.params });
        setConfig({ ...tmpl.config });
        if (templateId !== "custom" && templateId !== "production") {
          setRunLabel(tmpl.label);
        }
      }
      // Auto-expand parameter table for Custom; collapse for named templates
      setParamsExpanded(templateId === "custom");
      setErrors([]);
    },
    [templates]
  );

  const updateParam = useCallback((key: string, value: unknown) => {
    setParams((prev) => ({ ...prev, [key]: value }));
    setErrors((prev) => prev.filter((e) => e.field !== key));
  }, []);

  // Submit mutation
  const submitMut = useMutation({
    mutationFn: () =>
      submitModelExperiment(model, {
        run_label: runLabel.trim(),
        notes: notes.trim(),
        template: selectedTemplate,
        params,
        config: {
          ...config,
          cluster_source: clusterSource,
          cluster_experiment_id: clusterSource === "experimental" ? clusterExperimentId : undefined,
        },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["model-tuning"] });
      queryClient.invalidateQueries({ queryKey: [`${model}-tuning`] });
      onSubmitted();
    },
  });

  const handleSubmit = useCallback(() => {
    const validationErrors = validate(model, params, runLabel);
    if (validationErrors.length > 0) {
      setErrors(validationErrors);
      return;
    }
    setErrors([]);
    submitMut.mutate();
  }, [model, params, runLabel, submitMut]);

  const getError = useCallback(
    (field: string) => errors.find((e) => e.field === field)?.message,
    [errors]
  );

  // Count changed params
  const changedCount = useMemo(() => {
    let count = 0;
    for (const key of Object.keys(params)) {
      if (defaults[key] !== undefined && params[key] !== defaults[key]) {
        count++;
      }
    }
    return count;
  }, [params, defaults]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm">
      <div className="w-full max-w-3xl rounded-xl border bg-card shadow-xl max-h-[92vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-5 py-4 shrink-0">
          <div className="flex items-center gap-2">
            <FlaskConical className="h-4 w-4 text-primary" />
            <div>
              <p className="text-sm font-semibold text-foreground">
                New Experiment -- {MODEL_LABELS[model]}
              </p>
              <p className="text-xs text-muted-foreground">
                Configure hyperparameters and launch a tuning backtest
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
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
          {/* Label + Notes (UX-8 a11y: explicit htmlFor + onBlur validation + consistent required marker). */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label
                htmlFor="experiment-run-label"
                className="text-xs font-medium text-foreground mb-1 block"
              >
                Experiment Label
                <span aria-hidden="true" className="ml-0.5 text-destructive">
                  *
                </span>
                <span className="sr-only"> (required)</span>
              </label>
              <Input
                id="experiment-run-label"
                name="run_label"
                value={runLabel}
                aria-required="true"
                aria-invalid={getError("run_label") ? true : undefined}
                aria-describedby={getError("run_label") ? "experiment-run-label-err" : undefined}
                onChange={(e) => {
                  setRunLabel(e.target.value);
                  setErrors((prev) => prev.filter((er) => er.field !== "run_label"));
                }}
                onBlur={(e) => {
                  if (!e.target.value.trim()) {
                    setErrors((prev) => [
                      ...prev.filter((er) => er.field !== "run_label"),
                      { field: "run_label", message: "Label is required" },
                    ]);
                  }
                }}
                placeholder="e.g., Aggressive Depth + Heavy Reg"
                className={cn("text-sm", getError("run_label") && "border-red-500")}
              />
              {getError("run_label") && (
                <p
                  id="experiment-run-label-err"
                  role="alert"
                  className="text-[10px] text-red-600 mt-0.5"
                >
                  {getError("run_label")}
                </p>
              )}
            </div>
            <div>
              <label
                htmlFor="experiment-notes"
                className="text-xs font-medium text-foreground mb-1 block"
              >
                Notes
                <span className="ml-1 text-muted-foreground font-normal">(optional)</span>
              </label>
              <Input
                id="experiment-notes"
                name="notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Hypothesis or rationale..."
                className="text-sm"
              />
            </div>
          </div>

          {/* Template selection */}
          <TemplateSelector
            templates={templates}
            selectedTemplate={selectedTemplate}
            onSelect={applyTemplate}
          />

          {/* Hyperparameters */}
          <ParameterTable
            paramSpecs={paramSpecs}
            params={params}
            defaults={defaults}
            errors={errors}
            expanded={paramsExpanded}
            onToggleExpand={() => setParamsExpanded((p) => !p)}
            alwaysExpanded={selectedTemplate === "custom"}
            changedCount={changedCount}
            onUpdateParam={updateParam}
          />

          {/* Advanced Training Options (collapsed by default) */}
          <div className="border rounded-md overflow-hidden">
            <button
              onClick={() => setAdvancedOpen((prev) => !prev)}
              className="w-full flex items-center justify-between px-3 py-2 bg-muted/30 hover:bg-muted/50 transition-colors text-left"
            >
              <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Advanced Training Options
              </span>
              {advancedOpen ? (
                <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
              )}
            </button>
            {advancedOpen && (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 px-3 py-3">
                {/* Cluster Source selector */}
                <div>
                  <label className="text-[10px] text-muted-foreground mb-1 block">
                    Cluster Source
                  </label>
                  <select
                    value={
                      clusterSource === "experimental" && clusterExperimentId
                        ? String(clusterExperimentId)
                        : "production"
                    }
                    onChange={(e) => {
                      const val = e.target.value;
                      if (val === "production") {
                        setClusterSource("production");
                        setClusterExperimentId(null);
                      } else {
                        setClusterSource("experimental");
                        setClusterExperimentId(Number(val));
                      }
                    }}
                    className="w-full h-8 rounded-md border border-input bg-background px-2 text-xs focus:outline-none focus:ring-1 focus:ring-ring"
                    aria-label="Cluster Source"
                  >
                    <option value="production">Production Clusters</option>
                    {completedExperiments.length > 0 && <option disabled>---</option>}
                    {completedExperiments.length === 0 && (
                      <option disabled>No cluster experiments yet</option>
                    )}
                    {completedExperiments.map((exp: ClusterExperiment) => (
                      <option key={exp.experiment_id} value={String(exp.experiment_id)}>
                        {exp.label} — K={exp.optimal_k ?? "?"}, Sil=
                        {exp.silhouette_score != null ? exp.silhouette_score.toFixed(3) : "?"}
                      </option>
                    ))}
                  </select>
                  {clusterSource === "experimental" && clusterExperimentId && (
                    <div className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                      Using clusters from experiment #{clusterExperimentId}
                    </div>
                  )}
                  {completedExperiments.length === 0 && clusterSource === "production" && (
                    <p className="text-[10px] text-muted-foreground mt-1">
                      <a
                        href="#"
                        className="text-blue-600 dark:text-blue-400 hover:underline"
                        onClick={(e) => e.preventDefault()}
                      >
                        Create one in the Clusters tab &rarr;
                      </a>
                    </p>
                  )}
                </div>

                <div>
                  <label className="text-[10px] text-muted-foreground mb-1 block">
                    Cluster Strategy
                  </label>
                  <Select
                    value={config.cluster_strategy}
                    onValueChange={(v) => setConfig((prev) => ({ ...prev, cluster_strategy: v }))}
                  >
                    <SelectTrigger className="h-8 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="per_cluster">per_cluster</SelectItem>
                      <SelectItem value="global">global</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <label className="flex items-center gap-2 pt-4">
                  <input
                    type="checkbox"
                    checked={config.recursive}
                    onChange={(e) =>
                      setConfig((prev) => ({
                        ...prev,
                        recursive: e.target.checked,
                      }))
                    }
                    className="rounded"
                  />
                  <span className="text-xs text-foreground">Recursive</span>
                </label>

                <label className="flex items-center gap-2 pt-4">
                  <input
                    type="checkbox"
                    checked={config.shap_select}
                    onChange={(e) =>
                      setConfig((prev) => ({
                        ...prev,
                        shap_select: e.target.checked,
                      }))
                    }
                    className="rounded"
                  />
                  <span className="text-xs text-foreground">SHAP Selection</span>
                </label>

                {config.shap_select && (
                  <>
                    <div>
                      <label className="text-[10px] text-muted-foreground mb-1 block">
                        SHAP Threshold
                      </label>
                      <input
                        type="number"
                        value={config.shap_threshold}
                        onChange={(e) =>
                          setConfig((prev) => ({
                            ...prev,
                            shap_threshold: parseFloat(e.target.value) || 0.95,
                          }))
                        }
                        step="0.01"
                        min="0.5"
                        max="1.0"
                        className="w-full h-8 rounded-md border border-input bg-background px-2 text-xs tabular-nums focus:outline-none focus:ring-1 focus:ring-ring"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] text-muted-foreground mb-1 block">
                        SHAP Sample Size
                      </label>
                      <input
                        type="number"
                        value={config.shap_sample_size}
                        onChange={(e) =>
                          setConfig((prev) => ({
                            ...prev,
                            shap_sample_size: parseInt(e.target.value, 10) || 500,
                          }))
                        }
                        step="100"
                        min="100"
                        max="5000"
                        className="w-full h-8 rounded-md border border-input bg-background px-2 text-xs tabular-nums focus:outline-none focus:ring-1 focus:ring-ring"
                      />
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          {/* Warnings */}
          {warnings.length > 0 && (
            <div className="space-y-1">
              {warnings.map((w, i) => (
                <div
                  key={i}
                  className="rounded-md border border-amber-300 bg-amber-50 dark:border-amber-700 dark:bg-amber-950/30 px-3 py-2 text-xs text-amber-800 dark:text-amber-300 flex items-center gap-1.5"
                >
                  <AlertTriangle className="h-3 w-3 shrink-0" />
                  {w}
                </div>
              ))}
            </div>
          )}

          {/* Summary */}
          <div className="rounded-md bg-muted/30 border border-border/60 px-4 py-3">
            <p className="text-xs text-muted-foreground">
              Model: <span className="font-medium text-foreground">{MODEL_LABELS[model]}</span> |
              Strategy:{" "}
              <span className="font-medium text-foreground">{config.cluster_strategy}</span> |
              Recursive:{" "}
              <span className="font-medium text-foreground">{config.recursive ? "Yes" : "No"}</span>
            </p>
            <p className="text-xs text-muted-foreground mt-0.5">
              {paramSpecs.length} hyperparameters configured |{" "}
              <span
                className={cn(
                  changedCount > 0 ? "text-amber-700 dark:text-amber-400 font-medium" : ""
                )}
              >
                {changedCount} changed from production
              </span>
            </p>
          </div>
        </div>

        {/* Submit error */}
        {submitMut.isError && (
          <div className="mx-5 mb-0 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-400 flex items-center gap-1.5">
            <AlertTriangle className="h-3 w-3 shrink-0" />
            {(submitMut.error as Error).message}
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-between border-t px-5 py-3 shrink-0">
          <div className="flex items-center gap-2">
            {changedCount > 0 && (
              <Badge variant="outline" className="text-[10px]">
                {changedCount} param{changedCount !== 1 ? "s" : ""} changed
              </Badge>
            )}
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" size="sm" onClick={onClose}>
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={handleSubmit}
              disabled={submitMut.isPending}
              className="gap-1.5"
            >
              {submitMut.isPending && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              Launch Experiment
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
