import { describe, it, expect, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";
import { GlobalFilterProvider } from "@/context/GlobalFilterContext";
import type { GlobalFilterContextValue } from "@/context/GlobalFilterContext";
import type { GlobalFilters } from "@/types/theme";

vi.mock("@/api/queries", () => ({
  queryKeys: {
    domains: () => ["domains"],
    domainMeta: (d: string) => ["domain-meta", d],
    domainPage: (d: string, p: unknown) => ["domain-page", d, p],
    domainSuggest: (...args: unknown[]) => ["domain-suggest", ...args],
    samplePair: (d: string) => ["sample-pair", d],
    forecastModels: () => ["forecast-models"],
    skuClusters: (s: string) => ["sku-clusters", s],
    clusterProfiles: () => ["cluster-profiles"],
    accuracySlice: (p: unknown) => ["accuracy-slice", p],
    lagCurve: (p: unknown) => ["lag-curve", p],
    competitionConfig: () => ["competition-config"],
    competitionSummary: () => ["competition-summary"],
    skuAnalysis: (p: unknown) => ["sku-analysis", p],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchDomains: vi.fn().mockResolvedValue({ domains: ["item", "location", "customer", "time", "sku", "sales", "forecast"] }),
  fetchDomainMeta: vi.fn().mockResolvedValue({
    name: "item",
    plural: "items",
    default_sort: "item_id",
    columns: ["item_id", "item_desc", "brand_name"],
    numeric_fields: [],
    date_fields: [],
    category_fields: [],
  }),
  fetchDomainPage: vi.fn().mockResolvedValue({
    total: 2,
    limit: 50,
    offset: 0,
    rows: [
      { item_ck: 1, item_id: "100320", item_desc: "Test Item", brand_name: "TestBrand" },
      { item_ck: 2, item_id: "100321", item_desc: "Another Item", brand_name: "OtherBrand" },
    ],
  }),
  fetchDomainSuggest: vi.fn().mockResolvedValue([]),
  fetchSamplePair: vi.fn().mockResolvedValue({ item: "100320", location: "1401-BULK" }),
  fetchForecastModels: vi.fn().mockResolvedValue(["external"]),
  fetchSkuClusters: vi.fn().mockResolvedValue({ domain: "sku", total_assigned: 0, clusters: [] }),
  fetchClusterProfiles: vi.fn().mockResolvedValue({ profiles: [], metadata: {} }),
  fetchAccuracySlice: vi.fn().mockResolvedValue({ group_by: "cluster_assignment", rows: [] }),
  fetchLagCurve: vi.fn().mockResolvedValue({ by_lag: [] }),
  fetchCompetitionConfig: vi.fn().mockResolvedValue(null),
  fetchCompetitionSummary: vi.fn().mockResolvedValue(null),
  fetchSkuAnalysis: vi.fn().mockResolvedValue({ mode: "item_location", item: "", location: "", points: 0, models: [], series: [], model_monthly: {}, dfu_attributes: [] }),
  saveCompetitionConfig: vi.fn(),
  runCompetition: vi.fn(),
  fetchMarketIntel: vi.fn().mockResolvedValue({}),
  sendChatMessage: vi.fn().mockResolvedValue({}),
}));

vi.mock("@/components/DataTable", () => ({
  DataTable: ({ data }: { data: unknown[] }) => (
    <div data-testid="data-table">{data?.length ?? 0} rows</div>
  ),
}));

const { ExplorerTab } = await import("@/tabs/ExplorerTab");

function makeFilterContext(): GlobalFilterContextValue {
  const filters: GlobalFilters = {
    brand: [],
    category: [],
    market: [],
    channel: [],
    item: [],
    location: [],
      cluster: [],
    timeGrain: "month",
  };
  return {
    filters,
    setFilters: vi.fn(),
    resetFilters: vi.fn(),
    hasActiveFilters: false,
    planningDate: null,
  };
}

describe("ExplorerTab", () => {
  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <ExplorerTab domain="item" onDomainChange={vi.fn()} />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });

  it("renders domain selector", async () => {
    render(
      <TestQueryWrapper>
        <GlobalFilterProvider value={makeFilterContext()}>
          <ExplorerTab domain="item" onDomainChange={vi.fn()} />
        </GlobalFilterProvider>
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(document.body.textContent!.length).toBeGreaterThan(0);
    });
  });
});
