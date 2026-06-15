import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "../../__tests__/test-utils";
import { TodaysPlanBanner } from "../TodaysPlanBanner";

// Live-like payloads: financial_at_risk 3598.89 (banner must read "$3.6K" to
// match the Action Feed KPI — U8.1); briefing leaves total_skus/excess at 0
// while below_ss_count is real 3152 (must not render "0 SKUs" — U8.2).
vi.mock("@/api/queries", () => ({
  insightKeys: {
    actionFeed: () => ["inv-planning", "action-feed"],
    dailyBriefing: () => ["inv-planning", "daily-briefing"],
  },
  STALE_INSIGHTS: { ONE_MIN: 60000, FIVE_MIN: 300000 },
  fetchActionFeed: vi.fn().mockResolvedValue({
    summary: {
      total: 4252,
      critical: 2537,
      high: 1715,
      financial_at_risk: 3598.89,
      financial_at_risk_basis:
        "7-day lost gross margin (open exceptions) + proposed order value",
    },
    actions: [],
  }),
  fetchDailyBriefing: vi.fn().mockResolvedValue({
    date: "2026-04-02",
    stats: {
      total_skus: 0,
      below_ss_count: 3152,
      excess_count: 0,
      total_excess_value: 0,
      total_stockout_risk_value: 9766.81,
      avg_health_score: null,
    },
  }),
}));

describe("TodaysPlanBanner trust in headline numbers", () => {
  it("comma-formats the Urgent/High priority counts matching the Action-Feed KPIs (U2.2)", async () => {
    render(<TodaysPlanBanner onCollapse={vi.fn()} />, { wrapper: TestQueryWrapper });
    // critical=2537, high=1715 must render with thousands separators, matching the
    // comma-formatted Action-Feed "Critical 2,537" / "High Priority 1,715" below.
    await waitFor(() => expect(screen.getByText("2,537")).toBeInTheDocument());
    expect(screen.getByText("1,715")).toBeInTheDocument();
    // The bare unformatted integers must NOT appear.
    expect(screen.queryByText("2537")).not.toBeInTheDocument();
    expect(screen.queryByText("1715")).not.toBeInTheDocument();
  });

  it("renders the At Risk tile as $3.6K matching the Action Feed (U8.1)", async () => {
    render(<TodaysPlanBanner onCollapse={vi.fn()} />, { wrapper: TestQueryWrapper });
    await waitFor(() => expect(screen.getByText("$3.6K")).toBeInTheDocument());
    expect(screen.queryByText("$4K")).not.toBeInTheDocument();
  });

  it("names the at-risk basis on the banner chip so it is self-explaining (F2.1)", async () => {
    // The banner "At Risk" chip must name its basis (matching the Action Feed
    // panel sublabel), so a planner doesn't have to reconcile it against the
    // Command Center "Order Value at Risk" tile (a different metric).
    render(<TodaysPlanBanner onCollapse={vi.fn()} />, { wrapper: TestQueryWrapper });
    await waitFor(() => expect(screen.getByText("$3.6K")).toBeInTheDocument());
    // The chip's tooltip carries the explicit basis from the action-feed summary.
    const chip = screen.getByText("$3.6K").closest("[title]");
    expect(chip).not.toBeNull();
    expect(chip?.getAttribute("title")).toMatch(/7-day lost gross margin/i);
  });

  it("stamps the banner with the planning/data as-of date, not wall-clock (U1.1)", async () => {
    // The briefing's as-of date is Apr 2, 2026 (the planning date the action
    // rows + KPIs are computed against). The banner header date must match it,
    // not the browser's `new Date()`, so "Today's Plan · <date>" can't imply
    // same-day data when the figures are anchored to a frozen planning date.
    render(<TodaysPlanBanner onCollapse={vi.fn()} />, { wrapper: TestQueryWrapper });
    await waitFor(() =>
      expect(screen.getByText(/Apr 2, 2026/)).toBeInTheDocument(),
    );
    // Wall-clock today (test runs 2026-06-14) must NOT appear.
    const wallClock = new Date().toLocaleDateString("en-US", {
      weekday: "long",
      month: "short",
      day: "numeric",
    });
    expect(screen.queryByText(wallClock)).not.toBeInTheDocument();
  });

  it("does not show a literal '0 SKUs' next to a non-zero at-risk count (U8.2)", async () => {
    render(<TodaysPlanBanner onCollapse={vi.fn()} />, { wrapper: TestQueryWrapper });
    await waitFor(() => expect(screen.getByText(/3,152 at risk/)).toBeInTheDocument());
    expect(screen.queryByText(/^0 SKUs$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/0 SKUs/)).not.toBeInTheDocument();
    // The unpopulated excess chip is dropped entirely, not rendered as "0 excess ($0K)".
    expect(screen.queryByText(/0 excess/)).not.toBeInTheDocument();
  });
});
