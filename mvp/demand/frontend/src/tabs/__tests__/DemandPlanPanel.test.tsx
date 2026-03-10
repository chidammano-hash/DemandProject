/**
 * F2.2 — DemandPlanPanel smoke tests
 */

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";
import { DemandPlanPanel } from "../inv-planning/DemandPlanPanel";
import { TestQueryWrapper } from "./test-utils";

vi.mock("../../api/queries", () => ({
  fetchDemandPlanVersions: vi.fn().mockResolvedValue({
    versions: [
      {
        plan_version: "2026-04-01_production",
        plan_date: "2026-04-01",
        plan_label: "production",
        model_id: "lgbm_quantile_cluster",
        horizon_months: 12,
        dfu_count: 4823,
        status: "active",
        generated_at: "2026-04-01T06:00:00Z",
      },
    ],
  }),
  fetchDemandPlan: vi.fn().mockResolvedValue({
    item_no: "100320",
    loc: "1401-BULK",
    plan_version: "2026-04-01_production",
    generated_at: "2026-04-01T06:00:00Z",
    horizon_months: 12,
    rows: [
      {
        plan_month: "2026-04-01",
        horizon_months: 1,
        p10: 320.0,
        p50: 450.0,
        p90: 580.0,
        sigma_forecast: 101.6,
        sigma_demand: 80.0,
        sigma_combined: 129.3,
      },
      {
        plan_month: "2026-05-01",
        horizon_months: 2,
        p10: 290.0,
        p50: 420.0,
        p90: 560.0,
        sigma_forecast: 105.4,
        sigma_demand: 80.0,
        sigma_combined: 132.0,
      },
    ],
  }),
  fetchDemandPlanWeekly: vi.fn().mockResolvedValue({
    item_no: "100320",
    loc: "1401-BULK",
    plan_version: "2026-04-01_production",
    weeks: [
      {
        plan_week: "2026-03-30",
        iso_week: 14,
        iso_year: 2026,
        parent_month: "2026-04-01",
        weekly_weight: 0.067,
        p10_weekly: 21.0,
        p50_weekly: 30.0,
        p90_weekly: 39.0,
      },
      {
        plan_week: "2026-04-06",
        iso_week: 15,
        iso_year: 2026,
        parent_month: "2026-04-01",
        weekly_weight: 0.233,
        p10_weekly: 75.0,
        p50_weekly: 105.0,
        p90_weekly: 135.0,
      },
    ],
  }),
  STALE: { TWO_MIN: 120000, FIVE_MIN: 300000, ONE_MIN: 60000 },
}));

describe("DemandPlanPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders without crashing", () => {
    render(
      <TestQueryWrapper>
        <DemandPlanPanel />
      </TestQueryWrapper>
    );
    expect(screen.getByRole("heading", { name: "Demand Plan" })).toBeDefined();
  });

  it("shows item and location inputs", () => {
    render(
      <TestQueryWrapper>
        <DemandPlanPanel />
      </TestQueryWrapper>
    );
    expect(screen.getByPlaceholderText("e.g. 100320")).toBeDefined();
    expect(screen.getByPlaceholderText("e.g. 1401-BULK")).toBeDefined();
  });

  it("renders horizon buttons", () => {
    render(
      <TestQueryWrapper>
        <DemandPlanPanel />
      </TestQueryWrapper>
    );
    expect(screen.getByRole("button", { name: "3M" })).toBeDefined();
    expect(screen.getByRole("button", { name: "12M" })).toBeDefined();
    expect(screen.getByRole("button", { name: "18M" })).toBeDefined();
  });

  it("shows empty state before search", () => {
    render(
      <TestQueryWrapper>
        <DemandPlanPanel />
      </TestQueryWrapper>
    );
    expect(screen.getByText(/Enter an item and location/i)).toBeDefined();
  });

  it("renders demand plan table after submitting", async () => {
    render(
      <TestQueryWrapper>
        <DemandPlanPanel />
      </TestQueryWrapper>
    );

    fireEvent.change(screen.getByPlaceholderText("e.g. 100320"), {
      target: { value: "100320" },
    });
    fireEvent.change(screen.getByPlaceholderText("e.g. 1401-BULK"), {
      target: { value: "1401-BULK" },
    });
    fireEvent.click(screen.getByRole("button", { name: "View Plan" }));

    await waitFor(() => {
      expect(screen.getByText("P10")).toBeDefined();
      expect(screen.getByText("P50 (Median)")).toBeDefined();
      expect(screen.getByText("P90")).toBeDefined();
    });
  });

  it("renders sigma chips after data loads", async () => {
    render(
      <TestQueryWrapper>
        <DemandPlanPanel />
      </TestQueryWrapper>
    );

    fireEvent.change(screen.getByPlaceholderText("e.g. 100320"), {
      target: { value: "100320" },
    });
    fireEvent.change(screen.getByPlaceholderText("e.g. 1401-BULK"), {
      target: { value: "1401-BULK" },
    });
    fireEvent.click(screen.getByRole("button", { name: "View Plan" }));

    await waitFor(() => {
      expect(screen.getByText("Forecast σ")).toBeDefined();
      expect(screen.getByText("Combined σ")).toBeDefined();
    });
  });

  it("renders weekly view after data loads", async () => {
    render(
      <TestQueryWrapper>
        <DemandPlanPanel />
      </TestQueryWrapper>
    );

    fireEvent.change(screen.getByPlaceholderText("e.g. 100320"), {
      target: { value: "100320" },
    });
    fireEvent.change(screen.getByPlaceholderText("e.g. 1401-BULK"), {
      target: { value: "1401-BULK" },
    });
    fireEvent.click(screen.getByRole("button", { name: "View Plan" }));

    await waitFor(() => {
      expect(screen.getByText(/Weekly View/i)).toBeDefined();
    });
  });

  it("version selector is populated from API", async () => {
    render(
      <TestQueryWrapper>
        <DemandPlanPanel />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/2026-04-01_production/)).toBeDefined();
    });
  });
});
