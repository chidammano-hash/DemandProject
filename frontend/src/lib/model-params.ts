/** LightGBM hyperparameter specs and experiment templates. */
import type { ModelType } from "@/api/queries";

export interface ParamSpec {
  key: string;
  label: string;
  type: "int" | "float" | "bool" | "select";
  min?: number;
  max?: number;
  options?: string[];
  group: "tree" | "regularization" | "sampling" | "advanced";
  tooltip?: string;
  visibleWhen?: (params: Record<string, unknown>) => boolean;
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

export const DEFAULT_CONFIG: TrainingConfig = {
  cluster_strategy: "per_cluster",
  recursive: true,
  shap_select: true,
  shap_threshold: 0.95,
  shap_sample_size: 500,
};

const LGBM_PARAMS: ParamSpec[] = [
  {
    key: "n_estimators",
    label: "n_estimators",
    type: "int",
    min: 100,
    max: 10000,
    group: "tree",
    tooltip: "Number of boosting rounds",
  },
  {
    key: "learning_rate",
    label: "learning_rate",
    type: "float",
    min: 0.001,
    max: 0.5,
    group: "tree",
    tooltip: "Shrinkage rate per tree",
  },
  {
    key: "num_leaves",
    label: "num_leaves",
    type: "int",
    min: 2,
    max: 512,
    group: "tree",
    tooltip: "Maximum leaves per tree",
  },
  {
    key: "max_depth",
    label: "max_depth",
    type: "int",
    min: -1,
    max: 20,
    group: "tree",
    tooltip: "Maximum tree depth (-1 is unlimited)",
  },
  {
    key: "min_child_samples",
    label: "min_child_samples",
    type: "int",
    min: 1,
    max: 500,
    group: "tree",
    tooltip: "Minimum samples per leaf",
  },
  {
    key: "reg_lambda",
    label: "reg_lambda",
    type: "float",
    min: 0,
    max: 100,
    group: "regularization",
    tooltip: "L2 regularization",
  },
  {
    key: "reg_alpha",
    label: "reg_alpha",
    type: "float",
    min: 0,
    max: 100,
    group: "regularization",
    tooltip: "L1 regularization",
  },
  {
    key: "path_smooth",
    label: "path_smooth",
    type: "float",
    min: 0,
    max: 50,
    group: "regularization",
    tooltip: "Leaf-output smoothing",
  },
  {
    key: "min_gain_to_split",
    label: "min_gain_to_split",
    type: "float",
    min: 0,
    max: 10,
    group: "regularization",
    tooltip: "Minimum gain required to split",
  },
  {
    key: "subsample",
    label: "subsample",
    type: "float",
    min: 0.1,
    max: 1,
    group: "sampling",
    tooltip: "Row sampling ratio",
  },
  {
    key: "bagging_freq",
    label: "bagging_freq",
    type: "int",
    min: 0,
    max: 10,
    group: "sampling",
    tooltip: "Bagging frequency",
  },
  {
    key: "colsample_bytree",
    label: "colsample_bytree",
    type: "float",
    min: 0.1,
    max: 1,
    group: "sampling",
    tooltip: "Feature fraction per tree",
  },
  {
    key: "feature_fraction_bynode",
    label: "feature_fraction_bynode",
    type: "float",
    min: 0.1,
    max: 1,
    group: "sampling",
    tooltip: "Feature fraction per split",
  },
  {
    key: "max_bin",
    label: "max_bin",
    type: "int",
    min: 15,
    max: 512,
    group: "advanced",
    tooltip: "Maximum histogram bins",
  },
];

const LGBM_DEFAULTS: Record<string, unknown> = {
  n_estimators: 1500,
  learning_rate: 0.02,
  num_leaves: 127,
  max_depth: -1,
  min_child_samples: 40,
  reg_lambda: 1,
  reg_alpha: 0.1,
  path_smooth: 4,
  min_gain_to_split: 0.01,
  subsample: 0.8,
  bagging_freq: 1,
  colsample_bytree: 0.8,
  feature_fraction_bynode: 0.7,
  max_bin: 127,
};

export function getParamSpecs(_model: ModelType): ParamSpec[] {
  return LGBM_PARAMS;
}

export function getDefaults(_model: ModelType): Record<string, unknown> {
  return { ...LGBM_DEFAULTS };
}

export const GROUP_LABELS: Record<string, string> = {
  tree: "Tree Structure",
  regularization: "Regularization",
  sampling: "Sampling",
  advanced: "Advanced",
};

export function getTemplates(model: ModelType): TemplateOption[] {
  const defaults = getDefaults(model);
  const template = (
    id: string,
    label: string,
    description: string,
    overrides: Record<string, unknown> = {}
  ): TemplateOption => ({
    id,
    label,
    description,
    params: { ...defaults, ...overrides },
    config: { ...DEFAULT_CONFIG },
  });

  return [
    template(
      "production_baseline",
      "Current Production Settings",
      "The parameters currently running in production"
    ),
    template(
      "curated_conservative",
      "Conservative (Stable Demand)",
      "Strong regularization for stable, low-variability items",
      {
        max_depth: 10,
        num_leaves: 63,
        reg_lambda: 3.5,
        reg_alpha: 0.5,
        path_smooth: 8,
        min_child_samples: 60,
      }
    ),
    template(
      "curated_high_precision",
      "High Precision (Long Training)",
      "Extended training for subtle patterns",
      { learning_rate: 0.008, n_estimators: 3000, subsample: 0.85, colsample_bytree: 0.85 }
    ),
    template("curated_intermittent", "Intermittent Demand", "Regularized settings for sparse demand", {
      feature_fraction_bynode: 0.9,
      colsample_bytree: 0.9,
      min_child_samples: 100,
      min_gain_to_split: 0.05,
      reg_alpha: 1,
      path_smooth: 12,
    }),
    template(
      "curated_balanced",
      "Balanced (Best All-Around)",
      "Balanced accuracy and training cost",
      {
        learning_rate: 0.015,
        n_estimators: 2000,
        max_depth: 12,
        num_leaves: 95,
        reg_lambda: 2.5,
        reg_alpha: 0.3,
        feature_fraction_bynode: 0.8,
        path_smooth: 6,
        min_child_samples: 50,
      }
    ),
    template("custom", "Custom", "Start from production settings and modify freely"),
  ];
}

export function validate(
  model: ModelType,
  params: Record<string, unknown>,
  label: string
): ValidationError[] {
  const errors: ValidationError[] = [];
  if (!label.trim()) errors.push({ field: "run_label", message: "Experiment label is required" });

  for (const spec of getParamSpecs(model)) {
    const value = params[spec.key];
    if (value === undefined || value === null || value === "") continue;
    if (spec.visibleWhen && !spec.visibleWhen(params)) continue;
    if (spec.type !== "int" && spec.type !== "float") continue;

    const numeric = Number(value);
    if (Number.isNaN(numeric)) {
      errors.push({ field: spec.key, message: `${spec.label} must be a number` });
    } else if (spec.type === "int" && !Number.isInteger(numeric)) {
      errors.push({ field: spec.key, message: `${spec.label} must be an integer` });
    } else if (spec.min !== undefined && numeric < spec.min) {
      errors.push({ field: spec.key, message: `${spec.label} min is ${spec.min}` });
    } else if (spec.max !== undefined && numeric > spec.max) {
      errors.push({ field: spec.key, message: `${spec.label} max is ${spec.max}` });
    }
  }
  return errors;
}

export function formatDelta(
  current: unknown,
  defaultVal: unknown,
  type: "int" | "float" | "bool" | "select"
): string {
  if (type === "bool" || type === "select")
    return current === defaultVal ? "--" : `${String(defaultVal)} -> ${String(current)}`;
  const currentNumber = Number(current);
  const defaultNumber = Number(defaultVal);
  if (Number.isNaN(currentNumber) || Number.isNaN(defaultNumber) || defaultNumber === 0) {
    if (currentNumber === defaultNumber) return "--";
    const difference = currentNumber - defaultNumber;
    return difference > 0 ? `+${difference}` : String(difference);
  }
  if (Math.abs(currentNumber - defaultNumber) < 0.0001) return "--";
  const percent = ((currentNumber - defaultNumber) / Math.abs(defaultNumber)) * 100;
  return `${percent > 0 ? "+" : ""}${percent.toFixed(0)}%`;
}
