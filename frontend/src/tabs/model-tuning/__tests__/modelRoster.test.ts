import { describe, expect, it } from "vitest";

import type { PipelineAlgorithm } from "@/api/queries/unified-model-tuning";
import { deriveModelsFromConfig } from "../_helpers";

function algorithm(type: string): PipelineAlgorithm {
  return {
    type,
    enabled: true,
    tune: type === "tree",
    backtest: true,
    compete: true,
    forecast: true,
    output_dir: "data/backtest",
  };
}

describe("deriveModelsFromConfig", () => {
  it("ignores retired algorithms even if stale config marks them enabled", () => {
    const models = deriveModelsFromConfig({
      lgbm_cluster: algorithm("tree"),
      nhits: algorithm("deep_learning"),
      nbeats: algorithm("deep_learning"),
      mstl: algorithm("statistical"),
      chronos2_enriched: algorithm("foundation"),
      catboost_cluster: algorithm("tree"),
    });

    expect(models.map((model) => model.id)).toEqual([
      "lgbm_cluster",
      "nhits",
      "nbeats",
      "mstl",
      "chronos2_enriched",
    ]);
  });
});
