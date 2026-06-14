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
    summary: { total: 20, critical: 20, high: 0, financial_at_risk: 3598.89 },
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
  it("renders the At Risk tile as $3.6K matching the Action Feed (U8.1)", async () => {
    render(<TodaysPlanBanner onCollapse={vi.fn()} />, { wrapper: TestQueryWrapper });
    await waitFor(() => expect(screen.getByText("$3.6K")).toBeInTheDocument());
    expect(screen.queryByText("$4K")).not.toBeInTheDocument();
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
