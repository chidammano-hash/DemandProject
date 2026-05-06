/**
 * SKU Features tab — shared constants.
 */

export const PAGE_SIZE = 50;

export const SEASONALITY_OPTIONS = ["", "none", "low", "moderate", "strong"] as const;
export const VARIABILITY_OPTIONS = ["", "smooth", "erratic", "intermittent", "lumpy"] as const;
export const TREND_OPTIONS = ["", "declining", "flat", "growing"] as const;

export interface SortableColumn {
  key: string;
  label: string;
  align?: "right";
}

export const SORTABLE_COLUMNS: SortableColumn[] = [
  { key: "sku_ck", label: "SKU" },
  { key: "item_id", label: "Item" },
  { key: "loc", label: "Location" },
  { key: "ml_cluster", label: "Cluster", align: "right" },
  { key: "seasonality_profile", label: "Seasonality" },
  { key: "variability_class", label: "Variability" },
  { key: "trend_direction", label: "Trend" },
  { key: "cv_demand", label: "CV Demand", align: "right" },
  { key: "seasonal_amplitude", label: "Seasonal Amp", align: "right" },
  { key: "zero_demand_pct", label: "Zero %", align: "right" },
  { key: "cagr", label: "CAGR", align: "right" },
  { key: "recency_ratio", label: "Recency", align: "right" },
];

export const CHART_MARGIN = { top: 4, right: 16, left: 0, bottom: 4 };

export const DISTRIBUTION_TITLES: Record<string, string> = {
  seasonality_profile: "Seasonality Profile",
  variability_class: "Variability Class",
  trend_direction: "Trend Direction",
};

export const HISTOGRAM_LABELS: Record<string, string> = {
  cv_demand: "CV Demand",
  seasonal_amplitude: "Seasonal Amplitude",
  trend_r2: "Trend R²",
  zero_demand_pct: "Zero Demand %",
  adi: "ADI",
  cagr: "CAGR",
};

export const HISTOGRAM_FEATURES = [
  "cv_demand",
  "seasonal_amplitude",
  "trend_r2",
  "zero_demand_pct",
  "adi",
  "cagr",
] as const;

// Trend direction number -> label mapping
export const TREND_LABELS: Record<string, string> = {
  "-1": "declining",
  "0": "flat",
  "1": "growing",
};
