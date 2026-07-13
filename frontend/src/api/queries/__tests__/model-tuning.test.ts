import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchModelSummary } from "../model-tuning";

describe("fetchModelSummary", () => {
  afterEach(() => vi.restoreAllMocks());

  it("counts queued and running experiments as active", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          experiments: [
            { run_id: 1, status: "queued", accuracy_pct: null },
            { run_id: 2, status: "running", accuracy_pct: null },
            { run_id: 3, status: "cancelled", accuracy_pct: null },
          ],
          total: 3,
        }),
    } as Response);

    await expect(fetchModelSummary("lgbm")).resolves.toMatchObject({
      runs: 3,
      active: 2,
    });
  });
});
