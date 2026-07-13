import { describe, expect, it } from "vitest";

import { detailShapFeatureRows } from "../shapFeatureRows";

describe("detailShapFeatureRows", () => {
  const detail = {
    model_id: "lgbm_cluster",
    timeframe_idx: 3,
    label: "D",
    cutoff_date: "2025-11-01",
    total_features: 1,
    cluster: "all",
    features: [
      {
        feature: "mean_demand",
        mean_abs_shap: 638.9566,
        rank: 1,
        selected: true,
        timeframe: 3,
        cutoff_date: "2025-11-01",
      },
    ],
  };

  it("does not render an all-clusters response under a selected cluster", () => {
    expect(detailShapFeatureRows(detail, "high_volume_periodic")).toEqual([]);
  });

  it("maps feature rows when the response matches the selected cluster", () => {
    const matching = { ...detail, cluster: "high_volume_periodic" };

    expect(detailShapFeatureRows(matching, "high_volume_periodic")).toEqual([
      { feature: "mean_demand", value: 638.9566, selected: true },
    ]);
  });
});
