/**
 * ChampionExperimentBuilder — Full-screen modal for creating champion selection experiments.
 *
 * Template selector, strategy picker, dynamic params form, model checkboxes,
 * metric/lag selectors, and launch button.
 */
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FlaskConical, X } from "lucide-react";

import {
  championExperimentKeys,
  CHAMPION_EXP_STALE,
  createChampionExperiment,
  fetchChampionTemplates,
  type ChampionExperimentTemplate,
} from "@/api/queries";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Strategy param defaults
// ---------------------------------------------------------------------------

const STRATEGY_DEFAULTS: Record<string, Record<string, unknown>> = {
  expanding: { min_prior_months: 3 },
  rolling: { window_months: 6, min_prior_months: 3 },
  decay: { decay_factor: 0.9, min_prior_months: 3 },
  ensemble: { top_k: 3, weight_method: "inverse_wape", min_prior_months: 3 },
  meta_learner: { min_prior_months: 3 },
};

const META_DEFAULTS = {
  model_type: "random_forest",
  n_estimators: 200,
  max_depth: 15,
  test_months: 3,
  performance_window: 6,
};

const ALL_MODELS = ["lgbm_cluster", "catboost_cluster", "xgboost_cluster"];

const STRATEGY_LABELS: Record<string, string> = {
  expanding: "Expanding Window",
  rolling: "Rolling Window",
  decay: "Exponential Decay",
  ensemble: "Ensemble (Blended)",
  meta_learner: "Meta-Learner (ML)",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface Props {
  open: boolean;
  onClose: () => void;
  onSubmitted?: () => void;
}

export function ChampionExperimentBuilder({ open, onClose, onSubmitted }: Props) {
  const queryClient = useQueryClient();

  // Form state
  const [label, setLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [selectedTemplate, setSelectedTemplate] = useState<string>("custom");
  const [strategy, setStrategy] = useState("expanding");
  const [strategyParams, setStrategyParams] = useState<Record<string, unknown>>(
    STRATEGY_DEFAULTS.expanding,
  );
  const [metaLearnerParams, setMetaLearnerParams] = useState(META_DEFAULTS);
  const [models, setModels] = useState<string[]>([...ALL_MODELS]);
  const [metric, setMetric] = useState("accuracy_pct");
  const [lagMode, setLagMode] = useState("execution");
  const [minSkuRows, setMinSkuRows] = useState(3);

  // Templates query
  const { data: templatesData } = useQuery({
    queryKey: championExperimentKeys.templates(),
    queryFn: fetchChampionTemplates,
    staleTime: CHAMPION_EXP_STALE.TEMPLATES,
  });
  const templates = templatesData?.templates ?? [];

  // Submit mutation
  const submitMutation = useMutation({
    mutationFn: () =>
      createChampionExperiment({
        label,
        notes: notes || undefined,
        template: selectedTemplate !== "custom" ? selectedTemplate : undefined,
        strategy,
        strategy_params: strategyParams,
        meta_learner_params: strategy === "meta_learner" ? metaLearnerParams : undefined,
        models,
        metric,
        lag_mode: lagMode,
        min_sku_rows: minSkuRows,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: championExperimentKeys.all });
      resetForm();
      onSubmitted?.();
      onClose();
    },
  });

  function resetForm() {
    setLabel("");
    setNotes("");
    setSelectedTemplate("custom");
    setStrategy("expanding");
    setStrategyParams(STRATEGY_DEFAULTS.expanding);
    setMetaLearnerParams(META_DEFAULTS);
    setModels([...ALL_MODELS]);
    setMetric("accuracy_pct");
    setLagMode("execution");
    setMinSkuRows(3);
  }

  function handleTemplateSelect(tmpl: ChampionExperimentTemplate) {
    setSelectedTemplate(tmpl.id);
    if (tmpl.strategy) setStrategy(tmpl.strategy);
    if (tmpl.strategy_params) {
      setStrategyParams({
        ...STRATEGY_DEFAULTS[tmpl.strategy ?? "expanding"],
        ...tmpl.strategy_params,
      });
    }
    if (tmpl.meta_learner_params) {
      setMetaLearnerParams({ ...META_DEFAULTS, ...(tmpl.meta_learner_params as typeof META_DEFAULTS) });
    }
    if (tmpl.models) setModels([...tmpl.models]);
    if (tmpl.metric) setMetric(tmpl.metric);
    if (tmpl.lag_mode) setLagMode(tmpl.lag_mode);
    if (tmpl.min_sku_rows) setMinSkuRows(tmpl.min_sku_rows);
  }

  // Reset strategy params when strategy changes
  useEffect(() => {
    if (selectedTemplate === "custom") {
      setStrategyParams(STRATEGY_DEFAULTS[strategy] ?? {});
    }
  }, [strategy, selectedTemplate]);

  function toggleModel(m: string) {
    setModels((prev) =>
      prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m],
    );
  }

  const canSubmit = label.trim().length > 0 && models.length >= 2 && !submitMutation.isPending;

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="relative flex w-full max-w-3xl max-h-[90vh] flex-col rounded-lg border bg-background shadow-lg">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-6 py-4">
          <div className="flex items-center gap-2">
            <FlaskConical className="h-5 w-5" />
            <h2 className="text-lg font-semibold">New Champion Experiment</h2>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Body — scrollable */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
          {/* Label + Notes */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium">Label *</label>
              <input
                className="mt-1 w-full rounded border px-3 py-2 text-sm"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="e.g. Rolling 6m vs Expanding"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Notes</label>
              <input
                className="mt-1 w-full rounded border px-3 py-2 text-sm"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Optional description"
              />
            </div>
          </div>

          {/* Templates */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="text-sm">Template</CardTitle>
            </CardHeader>
            <CardContent className="pb-3">
              <div className="grid grid-cols-2 gap-2">
                {templates.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => handleTemplateSelect(t)}
                    className={cn(
                      "rounded border p-2 text-left text-xs transition-colors",
                      selectedTemplate === t.id
                        ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                        : "hover:bg-muted",
                    )}
                  >
                    <div className="font-medium">{t.label}</div>
                    <div className="text-muted-foreground mt-0.5 line-clamp-1">
                      {t.description}
                    </div>
                  </button>
                ))}
                <button
                  onClick={() => {
                    setSelectedTemplate("custom");
                    setStrategyParams(STRATEGY_DEFAULTS[strategy] ?? {});
                  }}
                  className={cn(
                    "rounded border p-2 text-left text-xs transition-colors",
                    selectedTemplate === "custom"
                      ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                      : "hover:bg-muted",
                  )}
                >
                  <div className="font-medium">Custom</div>
                  <div className="text-muted-foreground mt-0.5">Configure manually</div>
                </button>
              </div>
            </CardContent>
          </Card>

          {/* Strategy selector */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="text-sm">Strategy</CardTitle>
            </CardHeader>
            <CardContent className="pb-3 space-y-4">
              <select
                className="w-full rounded border px-3 py-2 text-sm"
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
              >
                {Object.entries(STRATEGY_LABELS).map(([k, v]) => (
                  <option key={k} value={k}>{v}</option>
                ))}
              </select>

              {/* Dynamic strategy params */}
              <div className="grid grid-cols-2 gap-3">
                {strategy === "rolling" && (
                  <ParamInput
                    label="Window Months"
                    type="number"
                    value={strategyParams.window_months as number ?? 6}
                    onChange={(v) =>
                      setStrategyParams((p) => ({ ...p, window_months: Number(v) }))
                    }
                    min={2}
                    max={24}
                  />
                )}
                {strategy === "decay" && (
                  <ParamInput
                    label="Decay Factor"
                    type="number"
                    value={strategyParams.decay_factor as number ?? 0.9}
                    onChange={(v) =>
                      setStrategyParams((p) => ({ ...p, decay_factor: Number(v) }))
                    }
                    min={0.5}
                    max={0.99}
                    step={0.01}
                  />
                )}
                {strategy === "ensemble" && (
                  <>
                    <ParamInput
                      label="Top K"
                      type="number"
                      value={strategyParams.top_k as number ?? 3}
                      onChange={(v) =>
                        setStrategyParams((p) => ({ ...p, top_k: Number(v) }))
                      }
                      min={2}
                      max={5}
                    />
                    <div>
                      <label className="text-xs font-medium">Weight Method</label>
                      <select
                        className="mt-1 w-full rounded border px-2 py-1.5 text-xs"
                        value={(strategyParams.weight_method as string) ?? "inverse_wape"}
                        onChange={(e) =>
                          setStrategyParams((p) => ({
                            ...p,
                            weight_method: e.target.value,
                          }))
                        }
                      >
                        <option value="inverse_wape">Inverse WAPE</option>
                        <option value="equal">Equal</option>
                      </select>
                    </div>
                  </>
                )}
                <ParamInput
                  label="Min Prior Months"
                  type="number"
                  value={strategyParams.min_prior_months as number ?? 3}
                  onChange={(v) =>
                    setStrategyParams((p) => ({ ...p, min_prior_months: Number(v) }))
                  }
                  min={1}
                  max={12}
                />
              </div>

              {/* Meta-learner params */}
              {strategy === "meta_learner" && (
                <div className="border-t pt-3 mt-3">
                  <h4 className="text-xs font-medium mb-2">Meta-Learner Config</h4>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs font-medium">Model Type</label>
                      <select
                        className="mt-1 w-full rounded border px-2 py-1.5 text-xs"
                        value={metaLearnerParams.model_type}
                        onChange={(e) =>
                          setMetaLearnerParams((p) => ({ ...p, model_type: e.target.value }))
                        }
                      >
                        <option value="random_forest">Random Forest</option>
                        <option value="xgboost">XGBoost</option>
                      </select>
                    </div>
                    <ParamInput
                      label="N Estimators"
                      type="number"
                      value={metaLearnerParams.n_estimators}
                      onChange={(v) =>
                        setMetaLearnerParams((p) => ({ ...p, n_estimators: Number(v) }))
                      }
                      min={50}
                      max={1000}
                    />
                    <ParamInput
                      label="Max Depth"
                      type="number"
                      value={metaLearnerParams.max_depth}
                      onChange={(v) =>
                        setMetaLearnerParams((p) => ({ ...p, max_depth: Number(v) }))
                      }
                      min={3}
                      max={30}
                    />
                    <ParamInput
                      label="Performance Window"
                      type="number"
                      value={metaLearnerParams.performance_window}
                      onChange={(v) =>
                        setMetaLearnerParams((p) => ({ ...p, performance_window: Number(v) }))
                      }
                      min={3}
                      max={24}
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Models + Metric + Lag */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="text-sm">Competition Config</CardTitle>
            </CardHeader>
            <CardContent className="pb-3 space-y-3">
              <div>
                <label className="text-xs font-medium">Models (min 2)</label>
                <div className="flex gap-3 mt-1">
                  {ALL_MODELS.map((m) => (
                    <label key={m} className="flex items-center gap-1.5 text-xs">
                      <input
                        type="checkbox"
                        checked={models.includes(m)}
                        onChange={() => toggleModel(m)}
                      />
                      {m.replace("_cluster", "")}
                    </label>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="text-xs font-medium">Metric</label>
                  <select
                    className="mt-1 w-full rounded border px-2 py-1.5 text-xs"
                    value={metric}
                    onChange={(e) => setMetric(e.target.value)}
                  >
                    <option value="accuracy_pct">Accuracy %</option>
                    <option value="wape">WAPE</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs font-medium">Lag Mode</label>
                  <select
                    className="mt-1 w-full rounded border px-2 py-1.5 text-xs"
                    value={lagMode}
                    onChange={(e) => setLagMode(e.target.value)}
                  >
                    <option value="execution">Execution</option>
                    {[0, 1, 2, 3, 4].map((l) => (
                      <option key={l} value={String(l)}>Lag {l}</option>
                    ))}
                  </select>
                </div>
                <ParamInput
                  label="Min SKU Rows"
                  type="number"
                  value={minSkuRows}
                  onChange={(v) => setMinSkuRows(Number(v))}
                  min={1}
                  max={24}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 border-t px-6 py-3">
          {submitMutation.isError && (
            <span className="text-xs text-red-500 mr-auto">
              {(submitMutation.error as Error).message}
            </span>
          )}
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button disabled={!canSubmit} onClick={() => submitMutation.mutate()}>
            {submitMutation.isPending ? "Launching..." : "Launch Experiment"}
          </Button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ParamInput helper
// ---------------------------------------------------------------------------

function ParamInput({
  label,
  type,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  type: string;
  value: number | string;
  onChange: (v: string) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <div>
      <label className="text-xs font-medium">{label}</label>
      <input
        type={type}
        className="mt-1 w-full rounded border px-2 py-1.5 text-xs"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        min={min}
        max={max}
        step={step}
      />
    </div>
  );
}
