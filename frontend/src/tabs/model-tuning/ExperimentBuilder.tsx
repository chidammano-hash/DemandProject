/**
 * ExperimentBuilder -- Full-screen dialog for configuring and launching a new
 * tuning experiment. Features template radio buttons (production baseline,
 * 4 expert templates, custom), a hyperparameter form with Value/Default/Delta
 * columns, training config section, validation with inline errors, and launch
 * with loading state.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  X,
  Loader2,
  FlaskConical,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
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
  type ClusterExperiment,
} from "@/api/queries";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface ExperimentBuilderProps {
  model: ModelType;
  open: boolean;
  onClose: () => void;
  onSubmitted: () => void;
}

interface ParamSpec {
  key: string;
  label: string;
  type: "int" | "float" | "bool" | "select";
  min?: number;
  max?: number;
  options?: string[];
  group: "tree" | "regularization" | "sampling" | "advanced";
  tooltip?: string;
  /** Only show when a condition is met (e.g., DART booster for XGBoost) */
  visibleWhen?: (params: Record<string, unknown>) => boolean;
  /** Disable when a condition is met */
  disabledWhen?: (params: Record<string, unknown>) => boolean;
  disabledTooltip?: string;
}

interface TemplateOption {
  id: string;
  label: string;
  description: string;
  params: Record<string, unknown>;
  config: TrainingConfig;
}

interface TrainingConfig {
  cluster_strategy: string;
  recursive: boolean;
  shap_select: boolean;
  shap_threshold: number;
  shap_sample_size: number;
}

interface ValidationError {
  field: string;
  message: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const MODEL_LABELS: Record<ModelType, string> = {
  lgbm: "LightGBM",
  catboost: "CatBoost",
  xgboost: "XGBoost",
};

const MODEL_PREFIX: Record<ModelType, string> = {
  lgbm: "/model-tuning/lgbm",
  catboost: "/model-tuning/catboost",
  xgboost: "/model-tuning/xgboost",
};

const GROUP_LABELS: Record<string, string> = {
  tree: "Tree Structure",
  regularization: "Regularization",
  sampling: "Sampling",
  advanced: "Advanced",
};

const DEFAULT_CONFIG: TrainingConfig = {
  cluster_strategy: "per_cluster",
  recursive: true,
  shap_select: true,
  shap_threshold: 0.95,
  shap_sample_size: 500,
};

// ---------------------------------------------------------------------------
// LGBM Parameter Specs
// ---------------------------------------------------------------------------
const LGBM_PARAMS: ParamSpec[] = [
  { key: "n_estimators", label: "n_estimators", type: "int", min: 100, max: 10000, group: "tree", tooltip: "Number of boosting rounds" },
  { key: "learning_rate", label: "learning_rate", type: "float", min: 0.001, max: 0.5, group: "tree", tooltip: "Shrinkage rate per tree" },
  { key: "num_leaves", label: "num_leaves", type: "int", min: 2, max: 512, group: "tree", tooltip: "Maximum number of leaves per tree" },
  { key: "max_depth", label: "max_depth", type: "int", min: -1, max: 20, group: "tree", tooltip: "Max tree depth (-1 = unlimited)" },
  { key: "min_child_samples", label: "min_child_samples", type: "int", min: 1, max: 500, group: "tree", tooltip: "Min samples per leaf node" },
  { key: "reg_lambda", label: "reg_lambda", type: "float", min: 0, max: 100, group: "regularization", tooltip: "L2 regularization weight" },
  { key: "reg_alpha", label: "reg_alpha", type: "float", min: 0, max: 100, group: "regularization", tooltip: "L1 regularization weight" },
  { key: "path_smooth", label: "path_smooth", type: "float", min: 0, max: 50, group: "regularization", tooltip: "Leaf output smoothing" },
  { key: "min_gain_to_split", label: "min_gain_to_split", type: "float", min: 0, max: 10, group: "regularization", tooltip: "Min gain required to split" },
  { key: "subsample", label: "subsample", type: "float", min: 0.1, max: 1.0, group: "sampling", tooltip: "Row subsampling ratio per iteration" },
  { key: "bagging_freq", label: "bagging_freq", type: "int", min: 0, max: 10, group: "sampling", tooltip: "Bagging frequency (0 = disabled)" },
  { key: "colsample_bytree", label: "colsample_bytree", type: "float", min: 0.1, max: 1.0, group: "sampling", tooltip: "Feature fraction per tree" },
  { key: "feature_fraction_bynode", label: "feature_fraction_bynode", type: "float", min: 0.1, max: 1.0, group: "sampling", tooltip: "Feature fraction per split node" },
  { key: "max_bin", label: "max_bin", type: "int", min: 15, max: 512, group: "advanced", tooltip: "Max histogram bin count" },
];

const LGBM_DEFAULTS: Record<string, unknown> = {
  n_estimators: 1500, learning_rate: 0.02, num_leaves: 127, max_depth: -1,
  min_child_samples: 40, reg_lambda: 1.0, reg_alpha: 0.1, path_smooth: 4.0,
  min_gain_to_split: 0.01, subsample: 0.8, bagging_freq: 1, colsample_bytree: 0.8,
  feature_fraction_bynode: 0.7, max_bin: 127,
};

// ---------------------------------------------------------------------------
// CatBoost Parameter Specs
// ---------------------------------------------------------------------------
const CATBOOST_PARAMS: ParamSpec[] = [
  { key: "iterations", label: "iterations", type: "int", min: 100, max: 10000, group: "tree", tooltip: "Number of boosting iterations" },
  { key: "learning_rate", label: "learning_rate", type: "float", min: 0.001, max: 0.5, group: "tree", tooltip: "Shrinkage rate per iteration" },
  { key: "depth", label: "depth", type: "int", min: 1, max: 16, group: "tree", tooltip: "Max tree depth" },
  { key: "max_leaves", label: "max_leaves", type: "int", min: 2, max: 512, group: "tree", tooltip: "Max leaves (Lossguide only)", disabledWhen: (p) => p.grow_policy === "SymmetricTree", disabledTooltip: "SymmetricTree uses depth to control tree size" },
  { key: "min_data_in_leaf", label: "min_data_in_leaf", type: "int", min: 1, max: 500, group: "tree", tooltip: "Min samples per leaf" },
  { key: "l2_leaf_reg", label: "l2_leaf_reg", type: "float", min: 0, max: 100, group: "regularization", tooltip: "L2 regularization on leaf values" },
  { key: "reg_lambda", label: "reg_lambda", type: "float", min: 0, max: 100, group: "regularization", tooltip: "Additional L2 regularization" },
  { key: "random_strength", label: "random_strength", type: "float", min: 0, max: 10, group: "regularization", tooltip: "Score randomization strength" },
  { key: "model_size_reg", label: "model_size_reg", type: "float", min: 0, max: 1, group: "regularization", tooltip: "Model size regularization coefficient" },
  { key: "subsample", label: "subsample", type: "float", min: 0.1, max: 1.0, group: "sampling", tooltip: "Row sampling ratio (ignored with Ordered bootstrap)" },
  { key: "colsample_bylevel", label: "colsample_bylevel", type: "float", min: 0.1, max: 1.0, group: "sampling", tooltip: "Feature fraction per depth level" },
  { key: "bagging_temperature", label: "bagging_temperature", type: "float", min: 0, max: 10, group: "sampling", tooltip: "Controls Bayesian bootstrap intensity" },
  { key: "border_count", label: "border_count", type: "int", min: 1, max: 255, group: "advanced", tooltip: "Number of split borders for numeric features" },
  { key: "grow_policy", label: "grow_policy", type: "select", options: ["SymmetricTree", "Lossguide", "Depthwise"], group: "advanced", tooltip: "Tree growth policy" },
  { key: "bootstrap_type", label: "bootstrap_type", type: "select", options: ["MVS", "Bayesian", "Ordered", "No"], group: "advanced", tooltip: "Bootstrap sampling method" },
  { key: "score_function", label: "score_function", type: "select", options: ["L2", "Cosine", "NewtonL2"], group: "advanced", tooltip: "Score function for leaf splits" },
  { key: "leaf_estimation_method", label: "leaf_estimation_method", type: "select", options: ["Newton", "Gradient", "Exact"], group: "advanced", tooltip: "Leaf value estimation method" },
  { key: "leaf_estimation_iterations", label: "leaf_estimation_iterations", type: "int", min: 1, max: 100, group: "advanced", tooltip: "Leaf estimation iteration count" },
  { key: "boost_from_average", label: "boost_from_average", type: "bool", group: "advanced", tooltip: "Initialize with target mean" },
  { key: "max_ctr_complexity", label: "max_ctr_complexity", type: "int", min: 0, max: 10, group: "advanced", tooltip: "Max categorical feature combinations" },
  { key: "langevin", label: "langevin", type: "bool", group: "advanced", tooltip: "Enable Langevin gradient boosting (requires Bayesian bootstrap)" },
  { key: "diffusion_temperature", label: "diffusion_temperature", type: "float", min: 1, max: 100000, group: "advanced", tooltip: "Langevin diffusion temperature", visibleWhen: (p) => p.langevin === true },
];

const CATBOOST_DEFAULTS: Record<string, unknown> = {
  iterations: 3000, learning_rate: 0.008, depth: 10, max_leaves: 127,
  min_data_in_leaf: 28, l2_leaf_reg: 7.5, reg_lambda: 3.5, random_strength: 0.5,
  model_size_reg: 0.08, subsample: 0.85, colsample_bylevel: 0.85,
  bagging_temperature: 0.4, border_count: 64, grow_policy: "Lossguide",
  bootstrap_type: "MVS", score_function: "L2", leaf_estimation_method: "Newton",
  leaf_estimation_iterations: 10, boost_from_average: true, max_ctr_complexity: 1,
  langevin: false, diffusion_temperature: 10000,
};

// ---------------------------------------------------------------------------
// XGBoost Parameter Specs
// ---------------------------------------------------------------------------
const XGBOOST_PARAMS: ParamSpec[] = [
  { key: "n_estimators", label: "n_estimators", type: "int", min: 100, max: 10000, group: "tree", tooltip: "Number of boosting rounds" },
  { key: "learning_rate", label: "learning_rate", type: "float", min: 0.001, max: 0.5, group: "tree", tooltip: "Shrinkage rate per tree" },
  { key: "max_depth", label: "max_depth", type: "int", min: 1, max: 20, group: "tree", tooltip: "Max tree depth" },
  { key: "max_leaves", label: "max_leaves", type: "int", min: 0, max: 512, group: "tree", tooltip: "Max leaves (lossguide only)", disabledWhen: (p) => p.grow_policy !== "lossguide", disabledTooltip: "Only active with lossguide grow policy" },
  { key: "min_child_weight", label: "min_child_weight", type: "float", min: 0, max: 500, group: "tree", tooltip: "Min sum of instance weight in a child" },
  { key: "reg_lambda", label: "reg_lambda", type: "float", min: 0, max: 100, group: "regularization", tooltip: "L2 regularization weight" },
  { key: "reg_alpha", label: "reg_alpha", type: "float", min: 0, max: 100, group: "regularization", tooltip: "L1 regularization weight" },
  { key: "gamma", label: "gamma", type: "float", min: 0, max: 10, group: "regularization", tooltip: "Min loss reduction for a split" },
  { key: "subsample", label: "subsample", type: "float", min: 0.1, max: 1.0, group: "sampling", tooltip: "Row subsampling ratio" },
  { key: "colsample_bytree", label: "colsample_bytree", type: "float", min: 0.1, max: 1.0, group: "sampling", tooltip: "Feature fraction per tree" },
  { key: "colsample_bylevel", label: "colsample_bylevel", type: "float", min: 0.1, max: 1.0, group: "sampling", tooltip: "Feature fraction per depth level" },
  { key: "max_bin", label: "max_bin", type: "int", min: 2, max: 1024, group: "advanced", tooltip: "Max histogram bin count" },
  { key: "grow_policy", label: "grow_policy", type: "select", options: ["depthwise", "lossguide"], group: "advanced", tooltip: "Tree growth policy" },
  { key: "booster", label: "booster", type: "select", options: ["gbtree", "dart"], group: "advanced", tooltip: "Booster type (DART enables dropout)" },
  { key: "rate_drop", label: "rate_drop", type: "float", min: 0, max: 1, group: "advanced", tooltip: "DART dropout rate", visibleWhen: (p) => p.booster === "dart" },
  { key: "skip_drop", label: "skip_drop", type: "float", min: 0, max: 1, group: "advanced", tooltip: "DART skip-drop probability", visibleWhen: (p) => p.booster === "dart" },
];

const XGBOOST_DEFAULTS: Record<string, unknown> = {
  n_estimators: 500, learning_rate: 0.05, max_depth: 6, max_leaves: 0,
  min_child_weight: 5, reg_lambda: 1.0, reg_alpha: 0, gamma: 0,
  subsample: 0.8, colsample_bytree: 0.8, colsample_bylevel: 1.0,
  max_bin: 256, grow_policy: "depthwise", booster: "gbtree",
  rate_drop: 0.08, skip_drop: 0.5,
};

// ---------------------------------------------------------------------------
// Param/template maps per model
// ---------------------------------------------------------------------------
function getParamSpecs(model: ModelType): ParamSpec[] {
  if (model === "lgbm") return LGBM_PARAMS;
  if (model === "catboost") return CATBOOST_PARAMS;
  return XGBOOST_PARAMS;
}

function getDefaults(model: ModelType): Record<string, unknown> {
  if (model === "lgbm") return { ...LGBM_DEFAULTS };
  if (model === "catboost") return { ...CATBOOST_DEFAULTS };
  return { ...XGBOOST_DEFAULTS };
}

// ---------------------------------------------------------------------------
// Expert Templates
// ---------------------------------------------------------------------------
function getTemplates(model: ModelType): TemplateOption[] {
  const defaults = getDefaults(model);
  const baseTemplate: TemplateOption = {
    id: "production",
    label: "Current Production Settings",
    description: "The parameters currently running in production",
    params: { ...defaults },
    config: { ...DEFAULT_CONFIG },
  };

  if (model === "lgbm") {
    return [
      baseTemplate,
      {
        id: "aggressive_depth", label: "Conservative (Stable Demand)",
        description: "Best for stable, low-variability items with strong regularization",
        params: { ...defaults, max_depth: 10, num_leaves: 63, reg_lambda: 3.5, reg_alpha: 0.5, path_smooth: 8.0, min_child_samples: 60 },
        config: { ...DEFAULT_CONFIG },
      },
      {
        id: "ultra_slow_lr", label: "High Precision (Long Training)",
        description: "Maximizes accuracy with extended training for subtle patterns",
        params: { ...defaults, learning_rate: 0.008, n_estimators: 3000, subsample: 0.85, colsample_bytree: 0.85 },
        config: { ...DEFAULT_CONFIG },
      },
      {
        id: "feature_boost", label: "Intermittent Demand",
        description: "Optimized for sparse or intermittent demand patterns",
        params: { ...defaults, feature_fraction_bynode: 0.9, colsample_bytree: 0.9, min_child_samples: 100, min_gain_to_split: 0.05, reg_alpha: 1.0, path_smooth: 12.0 },
        config: { ...DEFAULT_CONFIG },
      },
      {
        id: "balanced_champion", label: "Balanced (Best All-Around)",
        description: "Well-rounded settings combining top findings from prior experiments",
        params: { ...defaults, learning_rate: 0.015, n_estimators: 2000, max_depth: 12, num_leaves: 95, reg_lambda: 2.5, reg_alpha: 0.3, feature_fraction_bynode: 0.8, path_smooth: 6.0, min_child_samples: 50 },
        config: { ...DEFAULT_CONFIG },
      },
      {
        id: "custom", label: "Custom",
        description: "Start from production baseline and modify freely",
        params: { ...defaults },
        config: { ...DEFAULT_CONFIG },
      },
    ];
  }

  if (model === "catboost") {
    return [
      baseTemplate,
      {
        id: "symmetric_ordered", label: "Temporal Optimized",
        description: "Best for time-series data with strong seasonal patterns",
        params: { ...defaults, grow_policy: "SymmetricTree", depth: 8, bootstrap_type: "Ordered", iterations: 4000, learning_rate: 0.006, random_strength: 1.0 },
        config: { ...DEFAULT_CONFIG },
      },
      {
        id: "high_border", label: "Fine-Grained Splits",
        description: "Higher resolution splits for nuanced demand signals",
        params: { ...defaults, border_count: 128, l2_leaf_reg: 3.0, min_data_in_leaf: 40, bagging_temperature: 0.6, model_size_reg: 0.02 },
        config: { ...DEFAULT_CONFIG },
      },
      {
        id: "langevin", label: "Noise-Resistant (Exploratory)",
        description: "Uses diffusion noise to handle noisy demand data",
        params: { ...defaults, langevin: true, diffusion_temperature: 10000, learning_rate: 0.005, iterations: 5000, l2_leaf_reg: 5.0, bootstrap_type: "Bayesian", bagging_temperature: 1.0 },
        config: { ...DEFAULT_CONFIG },
      },
      {
        id: "ensemble_optimized", label: "Ensemble Complement",
        description: "Designed to work alongside other models in blended forecasts",
        params: { ...defaults, iterations: 3500, learning_rate: 0.01, depth: 12, l2_leaf_reg: 4.0, max_leaves: 191, subsample: 0.9, colsample_bylevel: 0.9, reg_lambda: 2.0, model_size_reg: 0.04 },
        config: { ...DEFAULT_CONFIG },
      },
      {
        id: "custom", label: "Custom",
        description: "Start from production baseline and modify freely",
        params: { ...defaults },
        config: { ...DEFAULT_CONFIG },
      },
    ];
  }

  // XGBoost
  return [
    baseTemplate,
    {
      id: "lossguide_heavy_reg", label: "Conservative (Heavy Guardrails)",
      description: "Strong regularization for stable, predictable forecasts",
      params: { ...defaults, grow_policy: "lossguide", max_leaves: 127, max_depth: 10, n_estimators: 2000, learning_rate: 0.015, reg_lambda: 5.0, reg_alpha: 0.5, gamma: 0.2, min_child_weight: 15, max_bin: 256, colsample_bylevel: 0.8 },
      config: { ...DEFAULT_CONFIG },
    },
    {
      id: "dart_conservative", label: "Balanced Diversity",
      description: "Uses tree dropout for robust, well-generalized forecasts",
      params: { ...defaults, booster: "dart", rate_drop: 0.08, skip_drop: 0.5, n_estimators: 2500, learning_rate: 0.012, max_depth: 8, subsample: 0.85, colsample_bytree: 0.85, reg_lambda: 3.0 },
      config: { ...DEFAULT_CONFIG },
    },
    {
      id: "ultra_high_trees", label: "High Precision (Long Training)",
      description: "Extended training to capture fine-grained demand patterns",
      params: { ...defaults, n_estimators: 3000, learning_rate: 0.008, max_depth: 10, max_leaves: 95, grow_policy: "lossguide", max_bin: 256, subsample: 0.82, colsample_bylevel: 0.85, reg_lambda: 4.0, gamma: 0.15 },
      config: { ...DEFAULT_CONFIG },
    },
    {
      id: "champion_blend", label: "Balanced (Best All-Around)",
      description: "Top findings combined for a well-rounded candidate",
      params: { ...defaults, booster: "dart", rate_drop: 0.05, skip_drop: 0.6, n_estimators: 2800, learning_rate: 0.01, grow_policy: "lossguide", max_leaves: 127, max_depth: 10, max_bin: 256, min_child_weight: 12, subsample: 0.85, colsample_bylevel: 0.85, reg_lambda: 5.0, reg_alpha: 0.3, gamma: 0.12 },
      config: { ...DEFAULT_CONFIG },
    },
    {
      id: "custom", label: "Custom",
      description: "Start from production baseline and modify freely",
      params: { ...defaults },
      config: { ...DEFAULT_CONFIG },
    },
  ];
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
function validate(
  model: ModelType,
  params: Record<string, unknown>,
  label: string,
): ValidationError[] {
  const errors: ValidationError[] = [];
  const specs = getParamSpecs(model);

  if (!label.trim()) {
    errors.push({ field: "run_label", message: "Experiment label is required" });
  }

  for (const spec of specs) {
    const val = params[spec.key];
    if (val === undefined || val === null || val === "") continue;

    // Skip hidden fields
    if (spec.visibleWhen && !spec.visibleWhen(params)) continue;

    if (spec.type === "int") {
      const n = Number(val);
      if (!Number.isInteger(n)) {
        errors.push({ field: spec.key, message: `${spec.label} must be an integer` });
      } else if (spec.min !== undefined && n < spec.min) {
        errors.push({ field: spec.key, message: `${spec.label} min is ${spec.min}` });
      } else if (spec.max !== undefined && n > spec.max) {
        errors.push({ field: spec.key, message: `${spec.label} max is ${spec.max}` });
      }
    }

    if (spec.type === "float") {
      const n = Number(val);
      if (isNaN(n)) {
        errors.push({ field: spec.key, message: `${spec.label} must be a number` });
      } else if (spec.min !== undefined && n < spec.min) {
        errors.push({ field: spec.key, message: `${spec.label} min is ${spec.min}` });
      } else if (spec.max !== undefined && n > spec.max) {
        errors.push({ field: spec.key, message: `${spec.label} max is ${spec.max}` });
      }
    }
  }

  // Cross-param validation
  if (model === "catboost") {
    if (params.langevin === true && params.bootstrap_type !== "Bayesian") {
      errors.push({
        field: "langevin",
        message: "Langevin requires bootstrap_type=Bayesian",
      });
    }
  }

  return errors;
}

// ---------------------------------------------------------------------------
// Fetcher
// ---------------------------------------------------------------------------
async function submitExperiment(
  model: ModelType,
  payload: {
    run_label: string;
    notes: string;
    template: string;
    params: Record<string, unknown>;
    config: TrainingConfig & {
      cluster_source?: "production" | "experimental";
      cluster_experiment_id?: number | null;
    };
  },
): Promise<{ run_id: number; status: string }> {
  const res = await fetch(`${MODEL_PREFIX[model]}/experiments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(body.detail ?? `Submit failed: ${res.status}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Delta formatter
// ---------------------------------------------------------------------------
function formatDelta(
  current: unknown,
  defaultVal: unknown,
  type: "int" | "float" | "bool" | "select",
): string {
  if (type === "bool" || type === "select") {
    if (current === defaultVal) return "--";
    return `${String(defaultVal)} -> ${String(current)}`;
  }
  const c = Number(current);
  const d = Number(defaultVal);
  if (isNaN(c) || isNaN(d) || d === 0) {
    if (c === d) return "--";
    const diff = c - d;
    return diff > 0 ? `+${diff}` : String(diff);
  }
  if (Math.abs(c - d) < 0.0001) return "--";
  const pct = ((c - d) / Math.abs(d)) * 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(0)}%`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function ExperimentBuilder({
  model,
  open,
  onClose,
  onSubmitted,
}: ExperimentBuilderProps) {
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
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({});
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

  // Cross-param warnings (non-blocking)
  const warnings = useMemo(() => {
    const w: string[] = [];
    if (model === "catboost" && params.bootstrap_type === "Ordered") {
      w.push("subsample has no effect with Ordered bootstrap");
    }
    return w;
  }, [model, params]);

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
    [templates],
  );

  const updateParam = useCallback((key: string, value: unknown) => {
    setParams((prev) => ({ ...prev, [key]: value }));
    setErrors((prev) => prev.filter((e) => e.field !== key));
  }, []);

  const toggleGroup = useCallback((group: string) => {
    setCollapsedGroups((prev) => ({ ...prev, [group]: !prev[group] }));
  }, []);

  // Submit mutation
  const submitMut = useMutation({
    mutationFn: () =>
      submitExperiment(model, {
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
    [errors],
  );

  // Group param specs
  const groupedParams = useMemo(() => {
    const groups: Record<string, ParamSpec[]> = {};
    for (const spec of paramSpecs) {
      if (spec.visibleWhen && !spec.visibleWhen(params)) continue;
      if (!groups[spec.group]) groups[spec.group] = [];
      groups[spec.group].push(spec);
    }
    return groups;
  }, [paramSpecs, params]);

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
          {/* Label + Notes */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="text-xs font-medium text-foreground mb-1 block">
                Experiment Label *
              </label>
              <Input
                value={runLabel}
                onChange={(e) => {
                  setRunLabel(e.target.value);
                  setErrors((prev) =>
                    prev.filter((er) => er.field !== "run_label"),
                  );
                }}
                placeholder="e.g., Aggressive Depth + Heavy Reg"
                className={cn(
                  "text-sm",
                  getError("run_label") && "border-red-500",
                )}
              />
              {getError("run_label") && (
                <p className="text-[10px] text-red-600 mt-0.5">
                  {getError("run_label")}
                </p>
              )}
            </div>
            <div>
              <label className="text-xs font-medium text-foreground mb-1 block">
                Notes (optional)
              </label>
              <Input
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Hypothesis or rationale..."
                className="text-sm"
              />
            </div>
          </div>

          {/* Template selection */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
              Template
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {templates.map((tmpl) => (
                <label
                  key={tmpl.id}
                  className={cn(
                    "flex items-start gap-2 rounded-md border px-3 py-2 cursor-pointer transition-colors",
                    selectedTemplate === tmpl.id
                      ? "border-primary bg-primary/5 ring-1 ring-primary/30"
                      : "border-border hover:bg-muted/30",
                  )}
                >
                  <input
                    type="radio"
                    name="template"
                    value={tmpl.id}
                    checked={selectedTemplate === tmpl.id}
                    onChange={() => applyTemplate(tmpl.id)}
                    className="mt-0.5"
                  />
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-foreground">
                      {tmpl.label}
                    </p>
                    <p className="text-[10px] text-muted-foreground truncate">
                      {tmpl.description}
                    </p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Hyperparameters — collapsed disclosure for named templates */}
          <div>
            {selectedTemplate !== "custom" && !paramsExpanded ? (
              <button
                onClick={() => setParamsExpanded(true)}
                className="w-full flex items-center justify-between rounded-md border border-border px-3 py-2 hover:bg-muted/30 transition-colors text-left"
              >
                <span className="text-xs font-medium text-foreground">
                  View Parameters ({paramSpecs.length} configured, {changedCount} differ from production)
                </span>
                <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
              </button>
            ) : (
              <>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Hyperparameters
                  </p>
                  {selectedTemplate !== "custom" && (
                    <button
                      onClick={() => setParamsExpanded(false)}
                      className="text-[10px] text-muted-foreground hover:text-foreground underline"
                    >
                      Collapse
                    </button>
                  )}
                </div>
              </>
            )}

            {(paramsExpanded || selectedTemplate === "custom") && Object.entries(groupedParams).map(([group, specs]) => {
              const isCollapsed = collapsedGroups[group] ?? false;
              return (
                <div
                  key={group}
                  className="border rounded-md mb-2 overflow-hidden"
                >
                  <button
                    onClick={() => toggleGroup(group)}
                    className="w-full flex items-center justify-between px-3 py-2 bg-muted/30 hover:bg-muted/50 transition-colors text-left"
                  >
                    <span className="text-xs font-medium text-foreground">
                      {GROUP_LABELS[group] ?? group}
                    </span>
                    {isCollapsed ? (
                      <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                    )}
                  </button>

                  {!isCollapsed && (
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead className="bg-muted/20">
                          <tr>
                            <th className="text-left px-3 py-1.5 font-medium w-1/4">
                              Parameter
                            </th>
                            <th className="text-left px-3 py-1.5 font-medium w-1/4">
                              Value
                            </th>
                            <th className="text-right px-3 py-1.5 font-medium w-1/6">
                              Default
                            </th>
                            <th className="text-right px-3 py-1.5 font-medium w-1/6">
                              Delta
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {specs.map((spec) => {
                            const val = params[spec.key];
                            const defVal = defaults[spec.key];
                            const changed = val !== defVal;
                            const disabled =
                              spec.disabledWhen?.(params) ?? false;
                            const fieldError = getError(spec.key);

                            return (
                              <tr
                                key={spec.key}
                                className={cn(
                                  "border-t border-border/40",
                                  changed &&
                                    "bg-amber-50/50 dark:bg-amber-900/10",
                                  disabled && "opacity-50",
                                )}
                              >
                                <td className="px-3 py-1.5">
                                  <span
                                    className="font-mono text-xs"
                                    title={spec.tooltip}
                                  >
                                    {spec.label}
                                  </span>
                                  {disabled && spec.disabledTooltip && (
                                    <p className="text-[9px] text-amber-600 dark:text-amber-400">
                                      {spec.disabledTooltip}
                                    </p>
                                  )}
                                </td>
                                <td className="px-3 py-1.5">
                                  {spec.type === "bool" ? (
                                    <input
                                      type="checkbox"
                                      checked={val === true}
                                      onChange={(e) =>
                                        updateParam(
                                          spec.key,
                                          e.target.checked,
                                        )
                                      }
                                      disabled={disabled}
                                      className="rounded"
                                    />
                                  ) : spec.type === "select" ? (
                                    <Select
                                      value={String(val ?? "")}
                                      onValueChange={(v) =>
                                        updateParam(spec.key, v)
                                      }
                                    >
                                      <SelectTrigger className="h-7 text-xs w-full min-w-[100px]">
                                        <SelectValue placeholder="Select..." />
                                      </SelectTrigger>
                                      <SelectContent>
                                        {(spec.options ?? []).map((opt) => (
                                          <SelectItem key={opt} value={opt}>
                                            {opt}
                                          </SelectItem>
                                        ))}
                                      </SelectContent>
                                    </Select>
                                  ) : (
                                    <div>
                                      <input
                                        type="number"
                                        value={
                                          val !== undefined ? String(val) : ""
                                        }
                                        onChange={(e) => {
                                          const raw = e.target.value;
                                          if (raw === "") {
                                            updateParam(spec.key, "");
                                            return;
                                          }
                                          updateParam(
                                            spec.key,
                                            spec.type === "int"
                                              ? parseInt(raw, 10)
                                              : parseFloat(raw),
                                          );
                                        }}
                                        step={
                                          spec.type === "float"
                                            ? "0.001"
                                            : "1"
                                        }
                                        disabled={disabled}
                                        className={cn(
                                          "w-full h-7 rounded-md border border-input bg-background px-2 text-xs tabular-nums",
                                          "focus:outline-none focus:ring-1 focus:ring-ring",
                                          changed && "font-semibold",
                                          fieldError && "border-red-500",
                                          disabled &&
                                            "cursor-not-allowed opacity-50",
                                        )}
                                      />
                                      {fieldError && (
                                        <p className="text-[9px] text-red-600 mt-0.5">
                                          {fieldError}
                                        </p>
                                      )}
                                    </div>
                                  )}
                                </td>
                                <td className="text-right px-3 py-1.5 tabular-nums text-muted-foreground font-mono">
                                  {defVal !== undefined ? String(defVal) : "--"}
                                </td>
                                <td
                                  className={cn(
                                    "text-right px-3 py-1.5 tabular-nums font-mono",
                                    changed
                                      ? "text-amber-700 dark:text-amber-400 font-medium"
                                      : "text-muted-foreground/50",
                                  )}
                                >
                                  {formatDelta(val, defVal, spec.type)}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

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
            {advancedOpen && <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 px-3 py-3">
              {/* Cluster Source selector */}
              <div>
                <label className="text-[10px] text-muted-foreground mb-1 block">
                  Cluster Source
                </label>
                <select
                  value={clusterSource === "experimental" && clusterExperimentId ? String(clusterExperimentId) : "production"}
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
                  {completedExperiments.length > 0 && (
                    <option disabled>---</option>
                  )}
                  {completedExperiments.length === 0 && (
                    <option disabled>No cluster experiments yet</option>
                  )}
                  {completedExperiments.map((exp: ClusterExperiment) => (
                    <option key={exp.experiment_id} value={String(exp.experiment_id)}>
                      {exp.label} — K={exp.optimal_k ?? "?"}, Sil=
                      {exp.silhouette_score != null
                        ? exp.silhouette_score.toFixed(3)
                        : "?"}
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
                    <a href="#" className="text-blue-600 dark:text-blue-400 hover:underline" onClick={(e) => e.preventDefault()}>
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
                  onValueChange={(v) =>
                    setConfig((prev) => ({ ...prev, cluster_strategy: v }))
                  }
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
                          shap_sample_size:
                            parseInt(e.target.value, 10) || 500,
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
            </div>}
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
              Model:{" "}
              <span className="font-medium text-foreground">
                {MODEL_LABELS[model]}
              </span>{" "}
              | Strategy:{" "}
              <span className="font-medium text-foreground">
                {config.cluster_strategy}
              </span>{" "}
              | Recursive:{" "}
              <span className="font-medium text-foreground">
                {config.recursive ? "Yes" : "No"}
              </span>
            </p>
            <p className="text-xs text-muted-foreground mt-0.5">
              {paramSpecs.length} hyperparameters configured |{" "}
              <span
                className={cn(
                  changedCount > 0
                    ? "text-amber-700 dark:text-amber-400 font-medium"
                    : "",
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
              {submitMut.isPending && (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              )}
              Launch Experiment
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
