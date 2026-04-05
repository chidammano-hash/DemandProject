import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    investmentKeys: {
      summary: (a?: unknown, b?: unknown) => ["inv-summary", a, b],
      detail: (p?: unknown) => ["inv-detail", p],
      frontier: (a?: unknown, b?: unknown) => ["inv-frontier", a, b],
    },
    STALE: { FIVE_MIN: 300000, TEN_MIN: 600000, ONE_MIN: 60000, TWO_MIN: 120000 },
    fetchInvestmentSummary: vi.fn(),
    fetchInvestmentDetail: vi.fn(),
    fetchInvestmentFrontier: vi.fn(),
    runInvestmentPlan: vi.fn(),
  };
});

import {
  fetchInvestmentSummary,
  fetchInvestmentDetail,
  fetchInvestmentFrontier,
} from "@/api/queries";
import { InvestmentPanel } from "@/tabs/inv-planning/InvestmentPanel";

beforeEach(() => {
  vi.clearAllMocks();
  (fetchInvestmentSummary as any).mockResolvedValue({
    total_investment_gap: 250000,
    avg_current_csl: 0.92,
    avg_recommended_csl: 0.97,
    total_items: 300,
  });
  (fetchInvestmentDetail as any).mockResolvedValue({
    total: 1,
    rows: [
      {
        investment_rank: 1,
        item_id: "100320",
        loc: "1401-BULK",
        abc_vol: "A",
        current_csl: 0.9,
        recommended_csl: 0.98,
        investment_increment: 5000,
        marginal_roi: 2.5,
      },
    ],
  });
  (fetchInvestmentFrontier as any).mockResolvedValue([
    { cumulative_investment: 0, achievable_csl: 0.85 },
    { cumulative_investment: 100000, achievable_csl: 0.95 },
  ]);
});

describe("InvestmentPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <InvestmentPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Investment Gap")).toBeInTheDocument();
    });
    expect(screen.getByText("Current Portfolio CSL")).toBeInTheDocument();
    expect(screen.getByText("Target Portfolio CSL")).toBeInTheDocument();
    expect(screen.getByText("SKUs Analyzed")).toBeInTheDocument();
  });

  it("renders Run Plan button", async () => {
    render(
      <TestQueryWrapper>
        <InvestmentPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Run Plan")).toBeInTheDocument();
  });

  it("renders efficient frontier chart", async () => {
    render(
      <TestQueryWrapper>
        <InvestmentPanel />
      </TestQueryWrapper>,
    );
    // "Efficient Frontier" appears as heading and info banner
    expect((await screen.findAllByText("Efficient Frontier")).length).toBeGreaterThanOrEqual(1);
  });

  it("renders detail table row", async () => {
    render(
      <TestQueryWrapper>
        <InvestmentPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("100320")).toBeInTheDocument();
  });
});
