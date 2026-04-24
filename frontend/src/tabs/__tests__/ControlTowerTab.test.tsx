import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

vi.mock("@/api/queries", () => ({
  controlTowerKeys: {
    kpis:       () => ["ct-kpis"],
    alerts:     (p?: Record<string, unknown>) => ["ct-alerts", p ?? {}],
    topCritical:(n?: number) => ["ct-top-critical", n ?? 10],
    trend:      (m?: number) => ["ct-trend", m ?? 6],
  },
  fetchControlTowerKpis: vi.fn().mockResolvedValue({
    computed_at: "2026-03-01",
    health: {
      total_skus: 500, healthy_count: 310, monitor_count: 133,
      at_risk_count: 45, critical_count: 12,
      avg_health_score: 68.5, avg_ss_coverage: 1.4,
      below_ss_count: 85, below_ss_pct: 0.17, avg_portfolio_dos: 22.5,
    },
    exceptions: {
      open_exceptions_total: 28, critical_exceptions: 8,
      high_exceptions: 12, recommended_order_value: 45000,
    },
    fill_rate: { portfolio_fill_rate_3m: 0.934, total_shortage_qty_3m: 8500 },
    demand_signals: { urgent_demand_signals: 18, projected_stockouts_today: 6 },
    intramonth: { items_with_stockout_this_month: 35, extended_stockouts_this_month: 9 },
  }),
  fetchControlTowerAlerts: vi.fn().mockResolvedValue({
    total: 2,
    alerts: [
      { alert_id: "EXC-1", source: "exception", severity: "critical", item_id: "ITEM001",
        loc: "LOC1", alert_type: "stockout", description: "Stockout risk detected",
        action: "Place emergency order", alert_ts: "2026-03-04T00:00:00", abc_vol: "A" },
      { alert_id: "DS-ITEM002-LOC2", source: "demand_signal", severity: "high", item_id: "ITEM002",
        loc: "LOC2", alert_type: "above_plan", description: "Demand 25% above plan",
        action: "Monitor demand pace", alert_ts: "2026-03-04T10:00:00", abc_vol: "B" },
    ],
  }),
  fetchControlTowerTopCritical: vi.fn().mockResolvedValue({
    items: [
      { item_id: "ITEM001", loc: "LOC1", abc_vol: "A", abc_xyz_segment: "AX",
        health_score: 18, health_tier: "critical",
        ss_coverage: 0.3, is_below_ss: true, current_dos: 4.5,
        target_dos_min: 15.0, target_dos_max: 30.0, open_exceptions: 3, fill_rate_3m: 0.72 },
      { item_id: "ITEM002", loc: "LOC2", abc_vol: "B", abc_xyz_segment: "BY",
        health_score: 32, health_tier: "critical",
        ss_coverage: 0.6, is_below_ss: false, current_dos: 7.0,
        target_dos_min: 15.0, target_dos_max: 30.0, open_exceptions: 1, fill_rate_3m: 0.81 },
    ],
  }),
  fetchControlTowerTrend: vi.fn().mockResolvedValue({
    trend: [
      { month_start: "2026-01-01", avg_health_score: 65.0, portfolio_fill_rate: 0.91,
        open_exceptions: 35, intramonth_stockout_rate: 0.09, active_signals: 22 },
      { month_start: "2026-02-01", avg_health_score: 67.2, portfolio_fill_rate: 0.93,
        open_exceptions: 30, intramonth_stockout_rate: 0.08, active_signals: 19 },
      { month_start: "2026-03-01", avg_health_score: 68.5, portfolio_fill_rate: 0.934,
        open_exceptions: 28, intramonth_stockout_rate: 0.07, active_signals: 18 },
    ],
  }),
  STALE: {
    FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000,
    TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0,
  },
}));

const ControlTowerTab = (await import("@/tabs/ControlTowerTab")).default;

describe("ControlTowerTab", () => {
  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <ControlTowerTab />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });

  it("renders Control Tower heading", async () => {
    render(
      <TestQueryWrapper>
        <ControlTowerTab />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText(/control tower/i)).toBeDefined();
    });
  });

  it("renders KPI cards with health score", async () => {
    render(
      <TestQueryWrapper>
        <ControlTowerTab />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText(/68\.5/)).toBeDefined();
    });
  });

  it("renders exception queue section", async () => {
    render(
      <TestQueryWrapper>
        <ControlTowerTab />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getAllByText(/Open Exceptions/i).length).toBeGreaterThan(0);
    });
  });

  it("renders trend chart section", async () => {
    render(
      <TestQueryWrapper>
        <ControlTowerTab />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText(/Portfolio Trend/i)).toBeDefined();
    });
  });
});

// Regression: when upstream MVs are not yet refreshed, the backend returns
// a zero-filled payload + `warning` field. Control Tower must render without
// throwing `Cannot read properties of undefined (reading 'toFixed')`.
describe("ControlTowerTab — unpopulated MV state", () => {
  it("renders without crashing when the backend returns zeros + warning", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchControlTowerKpis as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      computed_at: null,
      health: {
        total_dfus: 0, healthy_count: 0, monitor_count: 0,
        at_risk_count: 0, critical_count: 0,
        avg_health_score: null, avg_ss_coverage: null,
        below_ss_count: 0, below_ss_pct: null, avg_portfolio_dos: null,
      },
      exceptions: {
        open_exceptions_total: 0, critical_exceptions: 0,
        high_exceptions: 0, recommended_order_value: null,
      },
      fill_rate: { portfolio_fill_rate_3m: null, total_shortage_qty_3m: 0 },
      demand_signals: { urgent_demand_signals: 0, projected_stockouts_today: 0 },
      intramonth: { items_with_stockout_this_month: 0, extended_stockouts_this_month: 0 },
      warning: "mv_control_tower_kpis not yet refreshed. Run `make refresh-mvs-tiered`.",
    });
    (queries.fetchControlTowerTrend as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      trend: [],
      warning: "Upstream materialized view not yet refreshed.",
    });

    render(
      <TestQueryWrapper>
        <ControlTowerTab />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      // The surface heading must still render (proving no ErrorBoundary fired).
      expect(screen.getByText(/Inventory Control Tower/i)).toBeDefined();
      // The warning banner should appear.
      expect(screen.getByText(/refresh-mvs-tiered/)).toBeDefined();
    });
  });
});
