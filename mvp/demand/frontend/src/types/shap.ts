// TypeScript types for SHAP feature importance API (Feature 42)

export interface ShapFeatureSummary {
  feature: string;
  mean_abs_shap_across_timeframes: number;
  mean_rank: number;
  selected_count: number;
  n_timeframes: number;
}

export interface ShapFeatureDetail {
  feature: string;
  mean_abs_shap: number;
  rank: number;
  selected: boolean;
  timeframe: number;
  cutoff_date: string;
}

export interface ShapTimeframeEntry {
  index: number;
  label: string;
  cutoff_date: string;
}

export interface ShapModelsPayload {
  models: string[];
}

export interface ShapSummaryPayload {
  model_id: string;
  total_features: number;
  features: ShapFeatureSummary[];
}

export interface ShapTimeframesPayload {
  model_id: string;
  timeframes: ShapTimeframeEntry[];
}

export interface ShapTimeframeDetailPayload {
  model_id: string;
  timeframe_idx: number;
  label: string;
  cutoff_date: string;
  total_features: number;
  features: ShapFeatureDetail[];
}
