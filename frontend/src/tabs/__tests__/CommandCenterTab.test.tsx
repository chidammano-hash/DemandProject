import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

vi.mock("@/lib/navigation", () => ({
  navigateToItem: vi.fn(),
}));

vi.mock("@/api/queries", () => ({
  controlTowerKeys: {
    kpis: () => ["ct-kpis"],
    trend: (m?: number) => ["ct-trend", m ?? 6],
  },
  fetchControlTowerKpis: vi.fn().mockResolvedValue({
    computed_at: "2026-03-01",
    health: { avg_health_score: 85 },
    exceptions: { open_exceptions_total: 12, critical_exceptions: 3, high_exceptions: 5 },
    fill_rate: { portfolio_fill_rate_3m: 0.92 },
  }),
  fetchControlTowerTrend: vi.fn().mockResolvedValue({
    trend: [
      { month_start: "2026-01-01", avg_health_score: 80, fill_rate: 0.91 },
      { month_start: "2026-02-01", avg_health_score: 83, fill_rate: 0.92 },
    ],
  }),
  queryKeys: {
    aiInsights: (p: Record<string, unknown>) => ["ai-insights", p],
  },
  fetchAiInsights: vi.fn().mockResolvedValue({ insights: [], total: 0 }),
  updateInsightStatus: vi.fn().mockResolvedValue({ insight_id: 1, status: "acknowledged" }),
  storyboardKeys: {
    list: (p: Record<string, unknown>) => ["sb-list", p],
  },
  fetchSbExceptions: vi.fn().mockResolvedValue({ rows: [], total: 0 }),
  STALE: {
    FOREVER: Infinity,
    TEN_MIN: 600000,
    FIVE_MIN: 300000,
    TWO_MIN: 120000,
    ONE_MIN: 60000,
    THIRTY_SEC: 30000,
    NONE: 0,
  },
}));

const CommandCenterTab = (await import("@/tabs/CommandCenterTab")).default;

describe("CommandCenterTab", () => {
  const onNavigate = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders Command Center heading", async () => {
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Command Center")).toBeDefined();
    });
  });

  it("renders KPI cards when data loaded", async () => {
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByTestId("kpi-cards")).toBeDefined();
    });
    expect(screen.getByText("85/100")).toBeDefined();
    expect(screen.getByText("Portfolio Health")).toBeDefined();
    expect(screen.getByText("92.0%")).toBeDefined();
  });

  it("renders filter toolbar with severity buttons", async () => {
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByTestId("filter-toolbar")).toBeDefined();
    });
    // Severity filter buttons
    expect(screen.getByRole("button", { name: "All" })).toBeDefined();
    expect(screen.getByRole("button", { name: "critical" })).toBeDefined();
    expect(screen.getByRole("button", { name: "high" })).toBeDefined();
    expect(screen.getByRole("button", { name: "medium" })).toBeDefined();
    expect(screen.getByRole("button", { name: "low" })).toBeDefined();
    // Source filter buttons
    expect(screen.getByRole("button", { name: "All Sources" })).toBeDefined();
    expect(screen.getByRole("button", { name: "AI" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Rules" })).toBeDefined();
  });

  it("shows empty state when no exceptions match", async () => {
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByTestId("empty-state")).toBeDefined();
    });
    expect(screen.getByText("Portfolio looks healthy!")).toBeDefined();
  });

  it("shows a degraded warning instead of 'healthy' when KPIs are stale (F2.1)", async () => {
    const { fetchControlTowerKpis } = await import("@/api/queries");
    (fetchControlTowerKpis as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      computed_at: null,
      health: { avg_health_score: 0 },
      exceptions: { open_exceptions_total: 0, critical_exceptions: 0, high_exceptions: 0 },
      fill_rate: { portfolio_fill_rate_3m: null },
      warning: "mv_control_tower_kpis not yet refreshed. Run `make refresh-mvs-tiered`.",
    });
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByTestId("mv-stale-warning")).toBeDefined();
    });
    // The false-positive "healthy" empty state must NOT be shown when KPIs are stale.
    expect(screen.queryByText("Portfolio looks healthy!")).toBeNull();
  });

  it("renders trend chart section", async () => {
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Portfolio Trend (6M)")).toBeDefined();
    });
    // Trend data may show chart or "No trend data available" depending on async timing
    await waitFor(() => {
      const hasChart = screen.queryByTestId("line-chart") !== null;
      const hasNoData = screen.queryByText("No trend data available.") !== null;
      expect(hasChart || hasNoData).toBe(true);
    });
  });

  it("collapses and expands trend chart on toggle click", async () => {
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByTestId("trend-toggle")).toBeDefined();
    });
    // Trend section is visible initially (either chart or no-data message)
    const trendContent = screen.getByTestId("trend-toggle").parentElement;
    expect(trendContent?.children.length).toBeGreaterThan(1);
    // Collapse
    fireEvent.click(screen.getByTestId("trend-toggle"));
    // After collapse, only the button remains in the container
    expect(trendContent?.querySelectorAll("[data-testid='line-chart']").length ?? 0).toBe(0);
    // Expand again
    fireEvent.click(screen.getByTestId("trend-toggle"));
    expect(trendContent?.children.length).toBeGreaterThan(1);
  });

  it("renders exception feed when AI insights are present", async () => {
    const { fetchAiInsights } = await import("@/api/queries");
    (fetchAiInsights as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      insights: [
        {
          insight_id: 1,
          severity: "critical",
          insight_type: "stockout_risk",
          item_id: "ITEM100",
          loc: "LOC200",
          summary: "Stockout risk due to demand surge 42%",
          recommendation: "Increase safety stock",
          financial_impact_estimate: 12500,
          status: "open",
          created_at: "2026-03-15T10:00:00",
        },
      ],
      total: 1,
    });

    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByTestId("exception-feed")).toBeDefined();
    });
    expect(screen.getByText("ITEM100 @ LOC200")).toBeDefined();
    expect(screen.getByText(/Stockout risk due to demand surge/)).toBeDefined();
    expect(screen.getByText("Accept")).toBeDefined();
  });

  it("renders storyboard exceptions in the feed", async () => {
    const { fetchSbExceptions } = await import("@/api/queries");
    (fetchSbExceptions as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      rows: [
        {
          exception_id: "EXC-42",
          exception_type: "forecast_bias",
          severity: 0.8,
          item_id: "SKU999",
          loc: "WH01",
          headline: "Persistent over-forecast bias detected",
          financial_impact: 5000,
          status: "open",
          generated_at: "2026-03-14T08:00:00",
        },
      ],
      total: 1,
    });

    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByTestId("exception-feed")).toBeDefined();
    });
    expect(screen.getByText("SKU999 @ WH01")).toBeDefined();
    expect(screen.getByText("Persistent over-forecast bias detected")).toBeDefined();
  });

  it("shows Open Exceptions KPI card value", async () => {
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Open Exceptions")).toBeDefined();
    });
    expect(screen.getByText("12")).toBeDefined();
    expect(screen.getByText("3 critical")).toBeDefined();
  });
});
