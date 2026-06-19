import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent, within } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

vi.mock("@/lib/navigation", () => ({
  navigateToItem: vi.fn(),
}));

// The dependency-readiness banner self-fetches; stub it out — its own behaviour
// is covered by PipelineReadinessBanner.test.tsx.
vi.mock("@/components/PipelineReadinessBanner", () => ({
  PipelineReadinessBanner: () => null,
}));

vi.mock("@/api/queries", () => ({
  controlTowerKeys: {
    kpis: () => ["ct-kpis"],
    trend: (m?: number) => ["ct-trend", m ?? 6],
  },
  fetchControlTowerKpis: vi.fn().mockResolvedValue({
    computed_at: "2026-03-01",
    health: { avg_health_score: 85 },
    exceptions: { open_exceptions_total: 12, critical_exceptions: 3, high_exceptions: 5, recommended_order_value: 1077000 },
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

  // U6.7 — the feed fetches at most 50 rows but the tile reads the full
  // open-exception total. Without a caption a planner can't tell the feed is a
  // slice. The caption must reference the full total (open_exceptions_total),
  // not just the page length, mirroring the Inv Planning feed.
  it("captions the feed with 'showing N of {total}' referencing the full open-exception total", async () => {
    const { fetchSbExceptions } = await import("@/api/queries");
    (fetchSbExceptions as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      rows: [
        {
          exception_id: "EXC-1",
          exception_type: "forecast_bias",
          severity: 0.8,
          item_id: "SKU1",
          loc: "WH01",
          headline: "Over-forecast bias",
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
    // open_exceptions_total is 12 in the default kpi mock; the feed shows 1 row.
    const caption = await screen.findByTestId("feed-count-caption");
    expect(caption.textContent).toMatch(/showing\s+top\s+1\s+of\s+12/i);
  });

  // U7.10 — the storyboard feed is sorted critical-first and capped at 50, so
  // the lower-severity chips (High/Medium/Low) were structurally empty: filtering
  // the all-critical page client-side yielded nothing. Selecting a severity chip
  // must push that severity BAND down to the server so the feed reloads with the
  // matching rows (e.g. "High" => severity_min 0.5, severity_max 0.75).
  it("pushes the selected severity band to the storyboard query (High chip => [0.5, 0.75))", async () => {
    const { fetchSbExceptions } = await import("@/api/queries");
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByTestId("filter-toolbar")).toBeDefined();
    });
    (fetchSbExceptions as ReturnType<typeof vi.fn>).mockClear();
    fireEvent.click(screen.getByRole("button", { name: "high" }));
    await waitFor(() => {
      expect(fetchSbExceptions).toHaveBeenCalledWith(
        expect.objectContaining({ severity_min: 0.5, severity_max: 0.75 }),
      );
    });
  });

  it("does not constrain severity when the All chip is selected", async () => {
    const { fetchSbExceptions } = await import("@/api/queries");
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByTestId("filter-toolbar")).toBeDefined();
    });
    // The default "all" selection must not push a severity band (no min/max).
    const call = (fetchSbExceptions as ReturnType<typeof vi.fn>).mock.calls[0]?.[0];
    expect(call?.severity_min).toBeUndefined();
    expect(call?.severity_max).toBeUndefined();
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

  it("comma-formats the Open Exceptions tile + critical badge matching the feed footer (U2.1)", async () => {
    const { fetchControlTowerKpis } = await import("@/api/queries");
    (fetchControlTowerKpis as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      computed_at: "2026-03-01",
      health: { avg_health_score: 85 },
      exceptions: { open_exceptions_total: 6141, critical_exceptions: 2464, high_exceptions: 5, recommended_order_value: 1077000 },
      fill_rate: { portfolio_fill_rate_3m: 0.92 },
    });
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Open Exceptions")).toBeDefined();
    });
    // Prominent tile + badge must carry thousands separators, matching formatInt
    // used by the feed footer ("Showing top N of 6,141"). The bare "6141"/"2464"
    // must NOT appear.
    expect(screen.getByText("6,141")).toBeDefined();
    expect(screen.getByText("2,464 critical")).toBeDefined();
    expect(screen.queryByText("6141")).toBeNull();
    expect(screen.queryByText("2464 critical")).toBeNull();
  });

  it("labels the exception KPIs with their replenishment scope (U5.14)", async () => {
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Open Exceptions")).toBeDefined();
    });
    // A scope caption explains why this differs from the Inventory Action Feed
    // (which also folds in PO risks / demand signals).
    expect(screen.getAllByText(/replenishment exceptions only/i).length).toBeGreaterThanOrEqual(1);
  });

  it("exposes aria-pressed on filter chips and labeled role=group (U2.4)", async () => {
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Open Exceptions")).toBeDefined();
    });
    // Severity defaults to "all" -> the "All" chip must report pressed=true.
    const severityGroup = screen.getByRole("group", { name: /severity filter/i });
    expect(severityGroup).toBeDefined();
    const activeAll = within(severityGroup).getByRole("button", { pressed: true });
    expect(activeAll.textContent).toBe("All");
    // An inactive chip must report pressed=false (not absent), so a screen
    // reader can announce the toggle state.
    const criticalChip = within(severityGroup).getByRole("button", { name: /^critical$/i });
    expect(criticalChip.getAttribute("aria-pressed")).toBe("false");
    // Source + status groups are also labeled groups.
    expect(screen.getByRole("group", { name: /source filter/i })).toBeDefined();
    expect(screen.getByRole("group", { name: /status filter/i })).toBeDefined();
  });

  it("does not duplicate the critical count in a 4th KPI tile (U2.3)", async () => {
    render(
      <TestQueryWrapper>
        <CommandCenterTab onNavigate={onNavigate} />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Open Exceptions")).toBeDefined();
    });
    // The old duplicate "Critical Items" tile (which restated the "3 critical"
    // badge already on Open Exceptions) is replaced by a distinct $ metric.
    expect(screen.queryByText("Critical Items")).toBeNull();
    expect(screen.getByText("Order Value at Risk")).toBeDefined();
    // recommended_order_value 1,077,000 rendered via formatCurrency -> "$1.1M".
    expect(screen.getByText("$1.1M")).toBeDefined();
  });
});
