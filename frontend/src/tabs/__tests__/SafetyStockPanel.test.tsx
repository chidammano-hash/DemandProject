import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    safetyStockKeys: {
      summary: () => ["ss-summary"],
      detail: (p?: unknown) => ["ss-detail", p],
      explain: (item: string, loc: string) => ["ss-explain", item, loc],
    },
    STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
    fetchSafetyStockSummary: vi.fn(),
    fetchSafetyStockDetail: vi.fn(),
    fetchSafetyStockExplain: vi.fn(),
  };
});

import {
  fetchSafetyStockSummary,
  fetchSafetyStockDetail,
} from "@/api/queries";
import { SafetyStockPanel } from "@/tabs/inv-planning/SafetyStockPanel";

beforeEach(() => {
  vi.clearAllMocks();
  (fetchSafetyStockSummary as any).mockResolvedValue({
    below_ss_count: 25,
    avg_ss_coverage: 0.85,
    total_skus: 500,
    avg_ss_days: 12.3,
    by_abc: [
      { abc_vol: "A", count: 100, below_ss_count: 10, avg_coverage: 0.9 },
    ],
  });
  (fetchSafetyStockDetail as any).mockResolvedValue({
    total: 1,
    rows: [
      {
        item_id: "100320",
        loc: "1401-BULK",
        ss_combined: 200,
        ss_demand_only: 150,
        ss_coverage: 0.75,
        z_score: 1.65,
        is_below_ss: true,
        reorder_point: 350,
        abc_vol: "A",
      },
    ],
  });
});

describe("SafetyStockPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <SafetyStockPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("At Stockout Risk")).toBeInTheDocument();
    });
    expect(screen.getByText("Buffer Health")).toBeInTheDocument();
    expect(screen.getByText("Total SKUs")).toBeInTheDocument();
    expect(screen.getByText("Buffer Days")).toBeInTheDocument();
  });

  it("renders detail table with item data", async () => {
    render(
      <TestQueryWrapper>
        <SafetyStockPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("100320")).toBeInTheDocument();
  });

  it("renders Safety Stock by ABC Class section", async () => {
    render(
      <TestQueryWrapper>
        <SafetyStockPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Safety Buffer by ABC Class")).toBeInTheDocument();
  });

  it("renders Below SS Only toggle button", async () => {
    render(
      <TestQueryWrapper>
        <SafetyStockPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("At Risk Only")).toBeInTheDocument();
  });
});
