import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    fillRateKeys: {
      summary: (p?: unknown) => ["fr-summary", p],
      trend: (p?: unknown) => ["fr-trend", p],
      gapAnalysis: (p?: unknown) => ["fr-gap", p],
    },
    STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
    fetchFillRateSummary: vi.fn().mockResolvedValue({
      portfolio_fill_rate: 0.965,
      total_ordered: 120000,
      total_shortage_qty: 4200,
      partial_fulfillment_events: 85,
    }),
    fetchFillRateTrend: vi.fn().mockResolvedValue({
      months: [
        { month_start: "2026-01-01", fill_rate: 0.95 },
        { month_start: "2026-02-01", fill_rate: 0.97 },
      ],
    }),
    fetchFillRateGapAnalysis: vi.fn().mockResolvedValue({
      target_fill_rate: 0.97,
      actual_fill_rate: 0.93,
      gap_pct: -4.0,
      decomposition: [
        { cause: "Safety Stock Shortfall", impact_pct: -2.1, sku_count: 45, shortage_qty: 1200 },
        { cause: "Demand Spike (>20% above forecast)", impact_pct: -1.2, sku_count: 23, shortage_qty: 680 },
        { cause: "Lead Time Delay", impact_pct: -0.5, sku_count: 12, shortage_qty: 250 },
        { cause: "Other / Data Gap", impact_pct: -0.2, sku_count: 8, shortage_qty: 95 },
      ],
      month: "2026-03",
    }),
  };
});

import { FillRatePanel } from "@/tabs/inv-planning/FillRatePanel";

describe("FillRatePanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <FillRatePanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Portfolio Fill Rate")).toBeDefined();
    expect(await screen.findByText("Total Ordered")).toBeDefined();
    expect(await screen.findByText("Total Shortage")).toBeDefined();
    expect(await screen.findByText("Partial Fulfillment Events")).toBeDefined();
  });

  it("renders monthly trend chart", async () => {
    render(
      <TestQueryWrapper>
        <FillRatePanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Monthly Fill Rate Trend (Target: 98%)")).toBeDefined();
  });

  it("renders gap decomposition waterfall", async () => {
    render(
      <TestQueryWrapper>
        <FillRatePanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText(/Fill Rate Gap Decomposition/)).toBeDefined();
    expect(await screen.findByText("Safety Stock Shortfall")).toBeDefined();
    expect(await screen.findByText("Lead Time Delay")).toBeDefined();
    expect(await screen.findByText("Other / Data Gap")).toBeDefined();
  });
});
