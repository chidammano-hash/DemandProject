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
});
