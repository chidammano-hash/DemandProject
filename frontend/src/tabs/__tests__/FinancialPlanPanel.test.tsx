import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
}));

vi.mock("@/api/queries/evolution", () => ({
  financialPlanKeys: {
    plan: (p?: unknown) => ["financial-plan", p],
    budget: (p?: unknown) => ["financial-budget", p],
    workingCapital: (p?: unknown) => ["working-capital", p],
    excess: (p?: unknown) => ["excess-value", p],
  },
  STALE_EVO: { FIVE_MIN: 300000, ONE_MIN: 60000 },
  fetchBudgetStatus: vi.fn().mockResolvedValue({
    total: 2,
    budgets: [
      {
        budget_id: "budget-1",
        category: "Safety Stock",
        budget_cap: 500000,
        committed_spend: 420000,
        utilization_pct: 84.0,
        is_breached: false,
        effective_from: "2026-01-01",
      },
      {
        budget_id: "budget-2",
        category: "Excess Clearance",
        budget_cap: 100000,
        committed_spend: 115000,
        utilization_pct: 115.0,
        is_breached: true,
        effective_from: "2026-01-01",
      },
    ],
  }),
  fetchWorkingCapitalTrend: vi.fn().mockResolvedValue({
    months: [
      { month: "2026-03", inventory_value: 2500000, carrying_cost: 52083, excess_value: 80000 },
      { month: "2026-04", inventory_value: 2400000, carrying_cost: 50000, excess_value: 70000 },
    ],
  }),
}));

import { FinancialPlanPanel } from "@/tabs/inv-planning/FinancialPlanPanel";

describe("FinancialPlanPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <FinancialPlanPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Inventory Value")).toBeDefined();
    expect(await screen.findByText("Carrying Cost (6-mo)")).toBeDefined();
    expect(await screen.findByText("Excess Inventory")).toBeDefined();
    expect(await screen.findByText("Budget Breaches")).toBeDefined();
  });

  it("renders budget status table with BREACHED badge", async () => {
    render(
      <TestQueryWrapper>
        <FinancialPlanPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Budget Status")).toBeDefined();
    expect(await screen.findByText("BREACHED")).toBeDefined();
  });

  it("renders working capital chart", async () => {
    render(
      <TestQueryWrapper>
        <FinancialPlanPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Working Capital Timeline")).toBeDefined();
    expect(await screen.findByTestId("line-chart")).toBeDefined();
  });
});
