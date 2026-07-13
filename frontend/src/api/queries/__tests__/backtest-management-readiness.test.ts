import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchSnapshotRosterReadiness } from "@/api/queries/backtest-management";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchSnapshotRosterReadiness", () => {
  it("loads the current champion-plus-top-three publish prerequisite", async () => {
    const payload = {
      planning_month: "2026-07-01",
      ready: false,
      champion_ready: true,
      roster_model_count: 4,
      ready_contender_count: 2,
      required_contender_count: 3,
      contenders: [],
      stale_reason: "One contender is stale.",
      action_pipeline: "forecast-publish",
    };
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchSnapshotRosterReadiness()).resolves.toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith(
      "/backtest-management/snapshot-roster-readiness",
      undefined
    );
  });
});
