/**
 * model-params.ts -- Hyperparameter specs, defaults, and template definitions
 * for LightGBM, CatBoost, and XGBoost tuning experiments.
 *
 * Extracted from ExperimentBuilder to keep data separate from UI.
 */
import type { ModelType } from "@/api/queries";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface ParamSpec {
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

export interface TrainingConfig {
  cluster_strategy: string;
  recursive: boolean;
  shap_select: boolean;
  shap_threshold: number;
  shap_sample_size: number;
}

export interface TemplateOption {
  id: string;
  label: string;
  description: string;
  params: Record<string, unknown>;
  config: TrainingConfig;
}

export interface ValidationError {
  field: string;
  message: string;
}

// ---------------------------------------------------------------------------
// Default training config
// ---------------------------------------------------------------------------
export const DEFAULT_CONFIG: TrainingConfig = {
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
export function getParamSpecs(model: ModelType): ParamSpec[] {
  if (model === "lgbm") return LGBM_PARAMS;
  if (model === "catboost") return CATBOOST_PARAMS;
  return XGBOOST_PARAMS;
}

export function getDefaults(model: ModelType): Record<string, unknown> {
  if (model === "lgbm") return { ...LGBM_DEFAULTS };
  if (model === "catboost") return { ...CATBOOST_DEFAULTS };
  return { ...XGBOOST_DEFAULTS };
}

export const GROUP_LABELS: Record<string, string> = {
  tree: "Tree Structure",
  regularization: "Regularization",
  sampling: "Sampling",
  advanced: "Advanced",
};

// ---------------------------------------------------------------------------
// Expert Templates
// ---------------------------------------------------------------------------
export function getTemplates(model: ModelType): TemplateOption[] {
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
export function validate(
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
// Delta formatter
// ---------------------------------------------------------------------------
export function formatDelta(
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
