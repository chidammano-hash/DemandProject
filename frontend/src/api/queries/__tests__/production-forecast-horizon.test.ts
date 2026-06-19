/**
 * Regression: fetchProductionForecast must request the full 24-month horizon
 * by default. The old default of 18 truncated the promoted/champion line in
 * Item Analysis while the staging lines (no horizon filter) ran 24 months, so
 * the production forecast appeared to "stop" 6 months early.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fetchProductionForecast } from "@/api/queries/production-forecast";

describe("fetchProductionForecast default horizon", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify({ forecasts: [] })),
      json: () => Promise.resolve({ forecasts: [] }),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("defaults to horizon=24 when none is passed", async () => {
    await fetchProductionForecast({ item_id: "100320", loc: "1401-BULK" });
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("horizon=24");
  });

  it("honors an explicit horizon override", async () => {
    await fetchProductionForecast({ item_id: "X", loc: "Y", horizon: 12 });
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("horizon=12");
  });
});
