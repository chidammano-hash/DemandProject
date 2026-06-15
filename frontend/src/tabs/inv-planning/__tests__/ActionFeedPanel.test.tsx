import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "../../__tests__/test-utils";
import { ActionFeedPanel } from "../ActionFeedPanel";

// U9.1: the headline KPIs reflect the FULL candidate population (4,252 critical /
// $12.1K), while only 20 rows are shown. The panel must (a) surface the full
// counts and (b) caption the list as a truncated top-N so a planner does not
// read "20 critical" and conclude the day is light.
vi.mock("@/api/queries", () => ({
  insightKeys: {
    actionFeed: () => ["inv-planning", "action-feed"],
  },
  STALE_INSIGHTS: { ONE_MIN: 60000, FIVE_MIN: 300000 },
  fetchActionFeed: vi.fn().mockResolvedValue({
    summary: {
      total: 6214,
      critical: 4252,
      high: 0,
      financial_at_risk: 12099.96,
      financial_at_risk_basis:
        "7-day lost gross margin (open exceptions) + proposed order value",
      displayed: 20,
    },
    actions: [
      {
        id: "exception:627099:1401-BULK:0",
        source: "exception",
        item_id: "627099",
        loc: "1401-BULK",
        action_type: "stockout",
        urgency_score: 0.95,
        financial_impact: 571.98,
        severity: "critical",
        title: "Resolve Stockout",
        detail: "Stockout — TITOS HANDMADE VODKA 80 (627099 @ 1401-BULK)",
        item_desc: "TITOS HANDMADE VODKA 80",
        created_at: "2026-04-01",
      },
    ],
  }),
}));

describe("ActionFeedPanel full-population headline (U9.1)", () => {
  it("shows the FULL critical count, not the displayed page size", async () => {
    render(<ActionFeedPanel />, { wrapper: TestQueryWrapper });
    await waitFor(() =>
      expect(screen.getByText("4,252")).toBeInTheDocument(),
    );
    // The display-page size (20) must NOT be presented as the critical count.
    expect(screen.queryByText(/^20$/)).not.toBeInTheDocument();
  });

  it("captions the list as a truncated top-N of the full population", async () => {
    render(<ActionFeedPanel />, { wrapper: TestQueryWrapper });
    await waitFor(() =>
      expect(screen.getByText(/showing top 20 of 6,214/i)).toBeInTheDocument(),
    );
  });

  it("renders the human-readable item description on each action row (U1.8)", async () => {
    render(<ActionFeedPanel />, { wrapper: TestQueryWrapper });
    await waitFor(() =>
      expect(screen.getByText("TITOS HANDMADE VODKA 80")).toBeInTheDocument(),
    );
  });

  it("labels the $ at risk tile with its 7-day lost-margin basis (F1.2)", async () => {
    render(<ActionFeedPanel />, { wrapper: TestQueryWrapper });
    await waitFor(() =>
      expect(
        screen.getByText(/7-day lost gross margin/i),
      ).toBeInTheDocument(),
    );
  });
});
