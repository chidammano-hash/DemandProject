import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { fetchCustomerBlendTrend } from "@/api/queries/customerForecast";

describe("fetchCustomerBlendTrend", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve(JSON.stringify({ months: [] })),
      json: () => Promise.resolve({ months: [] }),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("binds an exact blend run and serializes Portfolio filters", async () => {
    await fetchCustomerBlendTrend({
      run_id: "blend-1",
      window: 18,
      brand: ["Brand A"],
      item_id: ["ITEM-1", "ITEM-2"],
      location_id: ["LOC-1"],
      cluster: ["seasonal"],
    });

    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("/customer-forecast/blend/trend?");
    expect(url).toContain("run_id=blend-1");
    expect(url).toContain("window=18");
    expect(url).toContain("brand=Brand+A");
    expect(url).toContain("item_id=ITEM-1%2CITEM-2");
    expect(url).toContain("location_id=LOC-1");
    expect(url).toContain("cluster_assignment=seasonal");
  });
});
