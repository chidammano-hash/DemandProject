/**
 * ChampionExperimentBuilder — Full-screen modal for creating champion selection experiments.
 *
 * Template selector, strategy picker, dynamic params form, model checkboxes,
 * metric/lag selectors, and launch button.
 */
import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FlaskConical, X, Star, ChevronDown, ChevronRight, Pencil } from "lucide-react";

import {
  championExperimentKeys,
  CHAMPION_EXP_STALE,
  createChampionExperiment,
  fetchChampionExperiments,
  fetchChampionTemplates,
  type ChampionExperimentTemplate,
} from "@/api/queries";
import {
  fetchPipelineConfig, pipelineConfigKeys,
} from "@/api/queries/unified-model-tuning";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LoadingElement } from "@/components/LoadingElement";
import { modelLabel, MODEL_TYPE_COLORS } from "@/lib/model-labels";
import { cn } from "@/lib/utils";

const TYPE_BADGE_COLORS: Record<string, string> = {
  tree: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
  foundation: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
  statistical: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
  deep_learning: "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300",
};

// ---------------------------------------------------------------------------
// Strategy param defaults
// ---------------------------------------------------------------------------

const STRATEGY_DEFAULTS: Record<string, Record<string, unknown>> = {
  expanding: { min_prior_months: 3 },
  rolling: { window_months: 6, min_prior_months: 3 },
  decay: { decay_factor: 0.9, min_prior_months: 3 },
  ensemble: { top_k: 3, weight_method: "inverse_wape", min_prior_months: 3 },
  meta_learner: { min_prior_months: 3 },
  hybrid_warmup: { min_prior_months: 3, warmup_strategy: "rolling", warmup_window: 2, warmup_min_prior: 1, primary_strategy: "adaptive_ensemble", primary_top_k: 3 },
  adaptive_ensemble: { min_k: 2, max_k: 5, spread_threshold: 0.15, min_prior_months: 3, weight_method: "inverse_wape" },
  ensemble_rolling: { top_k: 3, window_months: 6, weight_method: "inverse_wape", min_prior_months: 3 },
  optimized_decay: { decay_candidates: [0.75, 0.80, 0.85, 0.90, 0.95], min_prior_months: 3, validation_months: 3 },
  learned_blend: { min_prior_months: 6, alpha: 100.0 },
  ridge_blend: { min_prior_months: 3, ridge_alpha: 100.0, min_train_months: 6 },
  shrinkage_blend: { min_prior_months: 3, shrinkage_intensity: 0.5 },
  bayesian_model_avg: { min_prior_months: 3 },
  per_segment: { min_prior_months: 3, adi_threshold: 1.32, cv2_threshold: 0.49 },
  per_cluster: { min_prior_months: 3 },
  seasonal: { min_prior_months: 2, fallback_strategy: "expanding" },
  hybrid_meta_router: { min_prior_months: 3, confidence_threshold: 0.6, blend_top_k: 3 },
  diverse_ensemble: { min_prior_months: 3, top_k: 3, correlation_penalty: 0.5 },
  uncertainty_aware: { min_prior_months: 3, uncertainty_weight: 0.3 },
  cascade_ensemble: { min_prior_months: 3, solo_threshold: 0.10, mid_threshold: 0.25, mid_k: 2, wide_k: 5 },
  adversarial_filter: { min_prior_months: 3, outlier_z_threshold: 1.5, top_k: 3 },
  dynamic_window: { min_prior_months: 3, window_candidates: [2, 3, 4, 6, 9, 12], cv_months: 3 },
  regime_adaptive: { min_prior_months: 3, variance_window: 4, variance_threshold: 2.0 },
  error_correcting: { min_prior_months: 3, correction_window: 3, correction_strength: 0.5 },
  thompson_sampling: { min_prior_months: 2, discount: 0.95 },
  thompson_ensemble: { min_prior_months: 2, discount: 0.95, top_k: 3 },
  linucb: { min_prior_months: 3, alpha_ucb: 1.0 },
  exp3: { min_prior_months: 2, gamma: 0.10 },
  dfu_strategy_router: { min_prior_months: 3, eval_months: 3 },
  stacked_strategies: { min_prior_months: 3, eval_months: 3 },
  cluster_regime_hybrid: { min_prior_months: 3, variance_window: 4, variance_threshold: 2.0 },
};

const META_DEFAULTS = {
  model_type: "random_forest",
  n_estimators: 200,
  max_depth: 15,
  test_months: 3,
  performance_window: 6,
};

/** Fallback model IDs used before pipeline config loads */
const DEFAULT_MODELS = [
  "lgbm_cluster", "catboost_cluster", "xgboost_cluster",
  "chronos2_enriched",
  "mstl", "nbeats", "nhits", "seasonal_naive", "rolling_mean", "rolling_median",
];

const STRATEGY_LABELS: Record<string, string> = {
  // Core
  expanding: "Expanding Window",
  rolling: "Rolling Window",
  decay: "Exponential Decay",
  ensemble: "Ensemble (Blended)",
  meta_learner: "Meta-Learner (ML)",
  // Hybrid / Adaptive
  hybrid_warmup: "Hybrid Warmup",
  adaptive_ensemble: "Adaptive Ensemble",
  ensemble_rolling: "Ensemble Rolling",
  optimized_decay: "Walk-Forward Decay",
  // Learning-based
  learned_blend: "Learned Blend (Ridge)",
  ridge_blend: "Ridge Blend",
  shrinkage_blend: "Shrinkage Blend",
  bayesian_model_avg: "Bayesian Model Avg",
  // Segment / Cluster
  per_segment: "Per-Segment (SBA)",
  per_cluster: "Per-Cluster Champion",
  seasonal: "Seasonal (Same-Quarter)",
  // Advanced
  hybrid_meta_router: "Hybrid Meta-Router",
  diverse_ensemble: "Diverse Ensemble",
  uncertainty_aware: "Uncertainty-Aware",
  cascade_ensemble: "Cascade Ensemble",
  adversarial_filter: "Adversarial Filter",
  dynamic_window: "Dynamic Window",
  regime_adaptive: "Regime Adaptive",
  error_correcting: "Error Correcting",
  // Bandit / RL
  thompson_sampling: "Thompson Sampling",
  thompson_ensemble: "Thompson Ensemble",
  linucb: "LinUCB",
  exp3: "EXP3",
  // Meta / Routing
  dfu_strategy_router: "DFU Strategy Router",
  stacked_strategies: "Stacked Strategies",
  cluster_regime_hybrid: "Cluster-Regime Hybrid",
};

// ---------------------------------------------------------------------------
// Strategy tiers & descriptions
// ---------------------------------------------------------------------------

const RECOMMENDED_STRATEGIES = ["expanding", "rolling", "adaptive_ensemble", "per_cluster", "hybrid_warmup"];
const ADVANCED_STRATEGIES = [
  "decay", "ensemble", "meta_learner", "optimized_decay", "ridge_blend",
  "per_segment", "diverse_ensemble", "seasonal", "ensemble_rolling",
];
// Everything else is "Expert"
const EXPERT_STRATEGIES = Object.keys(STRATEGY_LABELS).filter(
  (k) => !RECOMMENDED_STRATEGIES.includes(k) && !ADVANCED_STRATEGIES.includes(k),
);

const STRATEGY_DESCRIPTIONS: Record<string, string> = {
  expanding: "Uses all available history. Best for stable demand.",
  rolling: "Uses last N months only. Better for changing trends.",
  adaptive_ensemble: "Blends top models. Best general-purpose choice.",
  per_cluster: "Picks best model per demand cluster.",
  hybrid_warmup: "Expanding for mature DFUs, rolling for new ones.",
};

// ---------------------------------------------------------------------------
// Strategy tier button — extracted to reduce duplication in the 3 tiers
// ---------------------------------------------------------------------------

function StrategyButton({
  strategyKey,
  selected,
  onSelect,
  showDescription = false,
}: {
  strategyKey: string;
  selected: boolean;
  onSelect: (key: string) => void;
  showDescription?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(strategyKey)}
      className={cn(
        "w-full rounded border px-3 py-2 text-left text-sm transition-colors",
        selected
          ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
          : "hover:bg-muted",
      )}
    >
      <div className="font-medium">{STRATEGY_LABELS[strategyKey]}</div>
      {showDescription && STRATEGY_DESCRIPTIONS[strategyKey] && (
        <div className="text-[11px] text-muted-foreground mt-0.5">
          {STRATEGY_DESCRIPTIONS[strategyKey]}
        </div>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Collapsible strategy tier — shared toggle + list pattern
// ---------------------------------------------------------------------------

function StrategyTier({
  label,
  strategies,
  expanded,
  onToggle,
  selected,
  onSelect,
}: {
  label: string;
  strategies: string[];
  expanded: boolean;
  onToggle: () => void;
  selected: string;
  onSelect: (key: string) => void;
}) {
  return (
    <div>
      <button
        type="button"
        onClick={onToggle}
        className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors py-1"
      >
        {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        {label} ({strategies.length})
      </button>
      {expanded && (
        <div className="space-y-1 mt-1">
          {strategies.map((k) => (
            <StrategyButton
              key={k}
              strategyKey={k}
              selected={selected === k}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

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

  // Fetch pipeline config to get dynamic model list
  const { data: pipelineConfig } = useQuery({
    queryKey: pipelineConfigKeys.config,
    queryFn: fetchPipelineConfig,
    staleTime: 60_000,
  });

  // All algorithms from pipeline config (show all, pre-select competing ones)
  const ALL_MODELS = useMemo(() => {
    if (!pipelineConfig?.algorithms) return DEFAULT_MODELS;
    const all = Object.keys(pipelineConfig.algorithms);
    return all.length > 0 ? all : DEFAULT_MODELS;
  }, [pipelineConfig]);

  // Models that have compete: true (pre-selected by default)
  const COMPETING_MODELS = useMemo(() => {
    if (!pipelineConfig?.algorithms) return DEFAULT_MODELS;
    return Object.entries(pipelineConfig.algorithms)
      .filter(([, algo]) => algo.enabled && algo.compete)
      .map(([id]) => id);
  }, [pipelineConfig]);

  // Form state
  const [label, setLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [selectedTemplate, setSelectedTemplate] = useState<string>("custom");
  const [strategy, setStrategy] = useState("expanding");
  const [strategyParams, setStrategyParams] = useState<Record<string, unknown>>(
    STRATEGY_DEFAULTS.expanding,
  );
  const [metaLearnerParams, setMetaLearnerParams] = useState(META_DEFAULTS);
  const [models, setModels] = useState<string[]>([...DEFAULT_MODELS]);
  const [metric, setMetric] = useState("accuracy_pct");
  const [lagMode, setLagMode] = useState("execution");
  const [minSkuRows, setMinSkuRows] = useState(3);

  // Strategy tier collapse state
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showExpert, setShowExpert] = useState(false);

  // Model editing state — collapsed by default to show pipeline config summary
  const [modelsExpanded, setModelsExpanded] = useState(false);

  // Sync model selection when dynamic model list loads — pre-select competing models
  useEffect(() => {
    if (COMPETING_MODELS !== DEFAULT_MODELS && COMPETING_MODELS.length > 0) {
      setModels([...COMPETING_MODELS]);
    }
  }, [COMPETING_MODELS]);

  // Templates query
  const { data: templatesData, isLoading: templatesLoading } = useQuery({
    queryKey: championExperimentKeys.templates(),
    queryFn: fetchChampionTemplates,
    staleTime: CHAMPION_EXP_STALE.TEMPLATES,
  });
  const templates = templatesData?.templates ?? [];

  // Past experiments — compute best accuracy per strategy for auto-star
  const { data: pastData } = useQuery({
    queryKey: championExperimentKeys.experiments({ status: "completed" }),
    queryFn: () => fetchChampionExperiments({ status: "completed", limit: 200 }),
    staleTime: CHAMPION_EXP_STALE.TEMPLATES,
    enabled: open,
  });

  // Auto-star: rank strategies by best accuracy, top 5 get stars
  const strategyPerf = useMemo(() => {
    const exps = pastData?.experiments ?? [];
    const bestByStrategy = new Map<string, number>();
    for (const e of exps) {
      if (e.champion_accuracy == null) continue;
      const prev = bestByStrategy.get(e.strategy) ?? 0;
      if (e.champion_accuracy > prev) bestByStrategy.set(e.strategy, e.champion_accuracy);
    }
    // Sort by best accuracy descending
    const sorted = [...bestByStrategy.entries()].sort((a, b) => b[1] - a[1]);
    const result = new Map<string, { rank: number; accuracy: number; stars: number }>();
    sorted.forEach(([strat, acc], i) => {
      // Top 5 get 5→1 stars, rest get 0
      const stars = i < 5 ? 5 - i : 0;
      result.set(strat, { rank: i + 1, accuracy: acc, stars });
    });
    return result;
  }, [pastData]);

  // User favorites — persisted in localStorage, can override auto-stars
  const [favorites, setFavorites] = useState<Set<string>>(() => {
    try {
      const stored = localStorage.getItem("champion_template_favorites");
      return stored ? new Set(JSON.parse(stored)) : new Set();
    } catch { return new Set(); }
  });

  function toggleFavorite(templateId: string) {
    setFavorites(prev => {
      const next = new Set(prev);
      if (next.has(templateId)) next.delete(templateId);
      else next.add(templateId);
      localStorage.setItem("champion_template_favorites", JSON.stringify([...next]));
      return next;
    });
  }

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
    setShowAdvanced(false);
    setShowExpert(false);
    setModelsExpanded(false);
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
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">Template</CardTitle>
                {strategyPerf.size > 0 && (
                  <span className="text-[10px] text-muted-foreground">
                    Stars based on {pastData?.experiments?.length ?? 0} past runs
                  </span>
                )}
              </div>
            </CardHeader>
            <CardContent className="pb-3">
              {templatesLoading ? (
                <LoadingElement message="Loading templates..." />
              ) : templates.length === 0 ? (
                <p className="text-xs text-muted-foreground text-center py-4">
                  No templates available. Use the Custom option below.
                </p>
              ) : null}
              <div className="grid grid-cols-2 gap-2">
                {templates.map((t) => {
                  const perf = t.strategy ? strategyPerf.get(t.strategy) : undefined;
                  const isFav = favorites.has(t.id);
                  return (
                  <button
                    key={t.id}
                    onClick={() => handleTemplateSelect(t)}
                    className={cn(
                      "relative rounded border p-2 text-left text-xs transition-colors",
                      selectedTemplate === t.id
                        ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                        : "hover:bg-muted",
                      isFav && selectedTemplate !== t.id && "border-amber-300 dark:border-amber-700",
                    )}
                  >
                    {/* Favorite toggle */}
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); toggleFavorite(t.id); }}
                      className="absolute top-1 right-1 p-0.5 rounded hover:bg-muted"
                      title={isFav ? "Remove favorite" : "Mark as favorite"}
                    >
                      <Star className={cn("h-3 w-3", isFav ? "fill-amber-400 text-amber-400" : "text-muted-foreground/40")} />
                    </button>
                    <div className="font-medium pr-5">{t.label}</div>
                    <div className="text-muted-foreground mt-0.5 line-clamp-1">
                      {t.description}
                    </div>
                    {/* Auto-star performance row */}
                    {perf && (
                      <div className="flex items-center gap-1.5 mt-1">
                        <span className="text-[10px] text-amber-500 tracking-tight">
                          {"★".repeat(perf.stars)}{"☆".repeat(5 - perf.stars)}
                        </span>
                        <span className="text-[10px] text-muted-foreground">
                          {perf.accuracy.toFixed(1)}% best
                        </span>
                      </div>
                    )}
                  </button>
                  );
                })}
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

          {/* Strategy selector — tiered */}
          <Card>
            <CardHeader className="py-3">
              <CardTitle className="text-sm">Strategy</CardTitle>
            </CardHeader>
            <CardContent className="pb-3 space-y-4">
              <div className="space-y-2">
                {/* Recommended tier — always visible */}
                <div className="space-y-1">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Recommended</span>
                  {RECOMMENDED_STRATEGIES.map((k) => (
                    <StrategyButton
                      key={k}
                      strategyKey={k}
                      selected={strategy === k}
                      onSelect={setStrategy}
                      showDescription
                    />
                  ))}
                </div>

                {/* Advanced tier — collapsed */}
                <StrategyTier
                  label="Show more strategies"
                  strategies={ADVANCED_STRATEGIES}
                  expanded={showAdvanced}
                  onToggle={() => setShowAdvanced((p) => !p)}
                  selected={strategy}
                  onSelect={setStrategy}
                />

                {/* Expert tier — collapsed */}
                <StrategyTier
                  label="Show all strategies"
                  strategies={EXPERT_STRATEGIES}
                  expanded={showExpert}
                  onToggle={() => setShowExpert((p) => !p)}
                  selected={strategy}
                  onSelect={setStrategy}
                />

                {/* Show current selection if it's in a collapsed tier */}
                {!RECOMMENDED_STRATEGIES.includes(strategy) && (
                  <div className="text-xs text-muted-foreground mt-1">
                    Selected: <span className="font-medium text-foreground">{STRATEGY_LABELS[strategy] ?? strategy}</span>
                  </div>
                )}
              </div>

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
                <div className="flex items-center gap-2">
                  <label className="text-xs font-medium">Models (min 2)</label>
                  {!modelsExpanded && (
                    <span className="text-[11px] text-muted-foreground">
                      Using {models.length} models from pipeline config
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={() => setModelsExpanded((p) => !p)}
                    className="flex items-center gap-0.5 text-[11px] text-primary hover:underline"
                  >
                    {modelsExpanded ? (
                      <>
                        <ChevronDown className="h-3 w-3" />
                        Collapse
                      </>
                    ) : (
                      <>
                        <Pencil className="h-3 w-3" />
                        Edit
                      </>
                    )}
                  </button>
                </div>
                {modelsExpanded && (
                  <div className="mt-2 space-y-2">
                    <div className="flex gap-2 text-[10px]">
                      <button type="button" onClick={() => setModels([...ALL_MODELS])} className="text-primary hover:underline">Select All</button>
                      <button type="button" onClick={() => setModels([...COMPETING_MODELS])} className="text-primary hover:underline">Competing Only</button>
                      <button type="button" onClick={() => setModels([])} className="text-muted-foreground hover:underline">Clear</button>
                      <span className="ml-auto text-muted-foreground">{models.length}/{ALL_MODELS.length} selected</span>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                      {ALL_MODELS.map((m) => {
                        const algo = pipelineConfig?.algorithms?.[m];
                        const selected = models.includes(m);
                        const isCompeting = algo?.enabled && algo?.compete;
                        const algoType = algo?.type ?? "unknown";
                        return (
                          <button
                            key={m}
                            type="button"
                            onClick={() => toggleModel(m)}
                            className={cn(
                              "rounded-lg border p-2.5 text-left transition-all",
                              selected
                                ? "border-primary bg-primary/5 ring-1 ring-primary"
                                : "opacity-50 hover:opacity-80 hover:bg-muted/50",
                            )}
                          >
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs font-semibold truncate">{modelLabel(m)}</span>
                              {isCompeting && (
                                <span className="h-2 w-2 rounded-full bg-primary shrink-0" title="Competing" />
                              )}
                            </div>
                            <div className="flex items-center gap-1.5">
                              <span className={`text-[9px] px-1.5 py-0.5 rounded-full font-medium ${TYPE_BADGE_COLORS[algoType] ?? "bg-gray-100 text-gray-600"}`}>
                                {algoType.replace("_", " ")}
                              </span>
                              {!algo?.enabled && (
                                <span className="text-[9px] text-muted-foreground">disabled</span>
                              )}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
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
