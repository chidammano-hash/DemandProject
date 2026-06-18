/**
 * Tests for fetchCandidateForecasts — the per-model backtest (past) fetcher
 * that powers the Item Analysis "Backtest" overlay.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fetchCandidateForecasts } from "@/api/queries/production-forecast";

describe("fetchCandidateForecasts URL + payload", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      text: () =>
        Promise.resolve(
          JSON.stringify({ item_id: "100320", loc: "1401-BULK", models: {} }),
        ),
      json: () =>
        Promise.resolve({ item_id: "100320", loc: "1401-BULK", models: {} }),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("hits /forecast/candidate with item_id + loc", async () => {
    await fetchCandidateForecasts({ item_id: "100320", loc: "1401-BULK" });
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("/forecast/candidate");
    expect(url).toContain("item_id=100320");
    expect(url).toContain("loc=1401-BULK");
    expect(url).not.toContain("model_id=");
  });

  it("appends model_id when provided", async () => {
    await fetchCandidateForecasts({ item_id: "X", loc: "Y", model_id: "lgbm_cluster" });
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("model_id=lgbm_cluster");
  });
});
