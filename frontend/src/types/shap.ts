// TypeScript types for SHAP feature importance API (Feature 42)

/** Global filter params for SHAP endpoints — resolves to clusters server-side */
export interface ShapFilterParams {
  item?: string;
  location?: string;
  brand?: string;
  category?: string;
  market?: string;
}

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
  cluster?: string;
  available_clusters?: string[];
  features: ShapFeatureDetail[];
}

// Per-DFU on-demand SHAP types (new endpoint)
export interface DfuShapFeatureContribution {
  name: string;
  value: number; // signed: positive = feature pushes forecast up
}

export interface DfuShapPoint {
  month: string; // "2024-03-01"
  is_future: boolean;
  base_value: number; // model's expected value (base prediction)
  other_shap: number; // sum of contributions from non-top-N features
  features: DfuShapFeatureContribution[];
}

export interface DfuShapPayload {
  item_no: string;
  loc: string;
  model_id: string;
  cluster_id: string;
  top_n: number;
  computed_at: string;
  /** Model whose stored forecasts were used as lag source for future months.
   *  Differs from model_id when the requested model is not the production champion.
   *  Future-month SHAP is approximate in that case. */
  future_lag_model_id: string | null;
  points: DfuShapPoint[];
}
