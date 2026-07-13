import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { fetchSkuShap } from "@/api/queries/shap";

describe("fetchSkuShap", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ points: [] }),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("calls the backend DFU route with the supported item and location grain", async () => {
    await fetchSkuShap("lgbm_cluster", "100/200", "1401 BULK", 15);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toBe(
      "/forecast/shap/lgbm_cluster/dfu?item_id=100%2F200&loc=1401+BULK&top_n=15",
    );
    expect(url).not.toContain("/sku?");
    expect(url).not.toContain("customer_group=");
  });

  it("includes customer_group when Item Analysis has an unambiguous DFU", async () => {
    await fetchSkuShap("lgbm_cluster", "100320", "1401-BULK", 10, "RETAIL / EAST");

    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("/forecast/shap/lgbm_cluster/dfu?");
    expect(url).toContain("customer_group=RETAIL+%2F+EAST");
  });
});
