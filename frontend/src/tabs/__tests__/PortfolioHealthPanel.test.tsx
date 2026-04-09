import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    healthKeys: {
      summary: (p?: unknown) => ["health-summary", p],
      detail: (p?: unknown) => ["health-detail", p],
      heatmap: (a?: unknown, b?: unknown) => ["health-heatmap", a, b],
    },
    STALE: { FIVE_MIN: 300000, TEN_MIN: 600000, ONE_MIN: 60000, TWO_MIN: 120000 },
    fetchHealthSummary: vi.fn(),
    fetchHealthDetail: vi.fn(),
    fetchHealthHeatmap: vi.fn(),
  };
});

import {
  fetchHealthSummary,
  fetchHealthDetail,
  fetchHealthHeatmap,
} from "@/api/queries";
import { PortfolioHealthPanel } from "@/tabs/inv-planning/PortfolioHealthPanel";

beforeEach(() => {
  vi.clearAllMocks();
  (fetchHealthSummary as any).mockResolvedValue({
    total_skus: 1000,
    avg_health_score: 72.5,
    by_tier: { healthy: 400, monitor: 300, at_risk: 200, critical: 100 },
    component_avgs: {
      ss_coverage: 18.5,
      dos_target: 20.0,
      stockout_risk: 15.0,
      forecast_accuracy: 19.0,
    },
  });
  (fetchHealthDetail as any).mockResolvedValue({
    total: 1,
    rows: [
      {
        item_id: "100320",
        loc: "1401-BULK",
        health_score: 45,
        health_tier: "at_risk",
        score_ss_coverage: 10,
        score_dos_target: 12,
        score_stockout_risk: 8,
        score_forecast_accuracy: 15,
      },
    ],
  });
  (fetchHealthHeatmap as any).mockResolvedValue({
    x_labels: ["A", "B", "C"],
    y_labels: ["low", "medium", "high"],
    cells: [{ x: "A", y: "low", avg_health_score: 85 }],
  });
});

describe("PortfolioHealthPanel", () => {
  it("renders heading", async () => {
    render(
      <TestQueryWrapper>
        <PortfolioHealthPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Portfolio Risk Overview")).toBeInTheDocument();
    });
  });

  it("renders tier KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <PortfolioHealthPanel />
      </TestQueryWrapper>,
    );
    // Tier names appear in KPI cards and pie chart legend
    expect((await screen.findAllByText("Healthy")).length).toBeGreaterThanOrEqual(1);
    expect((await screen.findAllByText("Monitor")).length).toBeGreaterThanOrEqual(1);
    expect((await screen.findAllByText("At Risk")).length).toBeGreaterThanOrEqual(1);
    expect((await screen.findAllByText("Critical")).length).toBeGreaterThanOrEqual(1);
  });

  it("renders Health Detail table", async () => {
    render(
      <TestQueryWrapper>
        <PortfolioHealthPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Health Detail")).toBeInTheDocument();
    expect(await screen.findByText("100320")).toBeInTheDocument();
  });

  it("renders score component bars", async () => {
    render(
      <TestQueryWrapper>
        <PortfolioHealthPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Buffer Adequacy")).toBeInTheDocument();
    expect(await screen.findByText("Supply Coverage")).toBeInTheDocument();
  });
});
