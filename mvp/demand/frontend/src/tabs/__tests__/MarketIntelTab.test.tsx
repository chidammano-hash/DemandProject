import { describe, it, expect, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", () => ({
  queryKeys: {
    samplePair: (d: string) => ["sample-pair", d],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchSamplePair: vi.fn().mockResolvedValue({ item: "100320", location: "1401-BULK" }),
  fetchMarketIntel: vi.fn().mockResolvedValue({
    item_no: "100320",
    location_id: "1401-BULK",
    item_desc: "Test",
    brand_name: null,
    category: null,
    state_id: null,
    site_desc: null,
    search_results: [],
    narrative: "Test narrative",
    generated_at: "2024-01-01",
  }),
}));

const MarketIntelTab = (await import("@/tabs/MarketIntelTab")).default;

describe("MarketIntelTab", () => {
  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <MarketIntelTab theme="light" />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });
});
