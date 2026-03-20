import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries/evolution", () => ({
  eventKeys: {
    calendar: (p?: unknown) => ["event-calendar", p],
    event: (id?: string) => ["event", id],
    impactPreview: (p?: unknown) => ["event-impact", p],
    performance: (p?: unknown) => ["event-performance", p],
  },
  STALE_EVO: { FIVE_MIN: 300000, ONE_MIN: 60000 },
  fetchEventCalendar: vi.fn().mockResolvedValue({
    total: 2,
    events: [
      {
        event_id: "evt-1",
        event_name: "Spring Promo",
        event_type: "promotion",
        start_date: "2026-03-15",
        end_date: "2026-03-31",
        item_no: null,
        loc: null,
        uplift_multiplier: 1.3,
        additive_qty: 0,
        is_hard_override: false,
        override_qty: null,
        status: "approved",
        created_by: "planner1",
        created_at: "2026-02-28T10:00:00Z",
      },
      {
        event_id: "evt-2",
        event_name: "Easter Uplift",
        event_type: "holiday",
        start_date: "2026-04-05",
        end_date: "2026-04-06",
        item_no: "100320",
        loc: "1401-BULK",
        uplift_multiplier: 1.5,
        additive_qty: 50,
        is_hard_override: false,
        override_qty: null,
        status: "pending",
        created_by: "planner2",
        created_at: "2026-03-01T09:00:00Z",
      },
    ],
  }),
}));

import { EventCalendarPanel } from "@/tabs/inv-planning/EventCalendarPanel";

describe("EventCalendarPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <EventCalendarPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Total Events")).toBeDefined();
    const approvedEls = await screen.findAllByText("Approved");
    expect(approvedEls.length).toBeGreaterThan(0);
    expect(await screen.findByText("Pending Approval")).toBeDefined();
  });

  it("renders event rows in table", async () => {
    render(
      <TestQueryWrapper>
        <EventCalendarPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Spring Promo")).toBeDefined();
    expect(await screen.findByText("Easter Uplift")).toBeDefined();
  });

  it("renders status badges", async () => {
    render(
      <TestQueryWrapper>
        <EventCalendarPanel />
      </TestQueryWrapper>
    );
    // In KPI summary + table
    const approvedBadges = await screen.findAllByText("approved");
    expect(approvedBadges.length).toBeGreaterThan(0);
  });

  it("renders New Event button", async () => {
    render(
      <TestQueryWrapper>
        <EventCalendarPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText(/New Event/)).toBeDefined();
  });
});
