import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchForecastModels } from "@/api/queries/domains";

describe("fetchForecastModels", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("filters stale database model IDs to the retained five-model roster", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            models: [
              "lgbm_cluster",
              "nhits",
              "nbeats",
              "mstl",
              "chronos2_enriched",
              "catboost_cluster",
              "xgboost_cluster",
            ],
          }),
      })
    );

    await expect(fetchForecastModels()).resolves.toEqual([
      "lgbm_cluster",
      "nhits",
      "nbeats",
      "mstl",
      "chronos2_enriched",
    ]);
  });
});
