import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchBacktestCurrent } from "@/api/queries/backtest-management";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchBacktestCurrent", () => {
  it("returns current metadata when the model has artifacts", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ accuracy_pct: 74.0, wape: 25.99 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    const current = await fetchBacktestCurrent("lgbm_cluster");

    expect(fetchMock.mock.calls[0][0]).toBe("/backtest-management/lgbm_cluster/current");
    expect(current).toEqual({ accuracy_pct: 74.0, wape: 25.99 });
  });

  it("resolves to null on 404 — no artifacts is a normal state, not an error", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "No current backtest for lgbm_cluster" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchBacktestCurrent("lgbm_cluster")).resolves.toBeNull();
  });

  it("still throws on non-404 errors", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "boom" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchBacktestCurrent("lgbm_cluster")).rejects.toThrow();
  });
});
