import { describe, it, expect } from "vitest";
import { FORECAST_MODEL_IDS, formatChampionLabel, isForecastModelId } from "../model-labels";

describe("formatChampionLabel", () => {
  it("renders the blend mix sorted by weight, as percentages", () => {
    const mix = [
      { model: "lgbm_cluster", weight: 0.35 },
      { model: "nbeats", weight: 0.4 },
      { model: "chronos2_enriched", weight: 0.25 },
    ];
    expect(formatChampionLabel(mix)).toBe("champion (40% N-BEATS, 35% LightGBM, 25% Chronos 2E)");
  });

  it("falls back to the single source model when no mix", () => {
    expect(formatChampionLabel(null, "nbeats")).toBe("champion (N-BEATS)");
    expect(formatChampionLabel([], "mstl")).toBe("champion (MSTL)");
  });

  it("falls back to a bare 'champion' when neither is known", () => {
    expect(formatChampionLabel()).toBe("champion");
    expect(formatChampionLabel(null, null)).toBe("champion");
  });

  it("prefers the mix over the single source when both are present", () => {
    const mix = [{ model: "nbeats", weight: 1.0 }];
    expect(formatChampionLabel(mix, "lgbm_cluster")).toBe("champion (100% N-BEATS)");
  });

  it("keeps the selectable base-model roster to the canonical five", () => {
    expect(FORECAST_MODEL_IDS).toEqual([
      "lgbm_cluster",
      "nhits",
      "nbeats",
      "mstl",
      "chronos2_enriched",
    ]);
    expect(isForecastModelId("catboost_cluster")).toBe(false);
    expect(isForecastModelId("external")).toBe(false);
  });
});
