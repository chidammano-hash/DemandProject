import type { ShapTimeframeDetailPayload } from "@/types/shap";

export interface ShapChartFeatureRow {
  feature: string;
  value: number;
  selected: boolean;
}

/** Keep the chart label and its response on the same selected cluster. */
export function detailShapFeatureRows(
  detail: ShapTimeframeDetailPayload | undefined,
  selectedCluster: string
): ShapChartFeatureRow[] {
  if (!detail || detail.cluster !== selectedCluster) return [];
  return detail.features.map((feature) => ({
    feature: feature.feature,
    value: feature.mean_abs_shap,
    selected: feature.selected,
  }));
}
