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

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    queryKeys: {
      ...(actual as any).queryKeys,
      eoqSummary: (p?: unknown) => ["eoq-summary", p],
      eoqDetail: (p?: unknown) => ["eoq-detail", p],
      eoqSensitivity: (p?: unknown) => ["eoq-sens", p],
    },
    STALE: { FIVE_MIN: 300000, TEN_MIN: 600000, ONE_MIN: 60000, TWO_MIN: 120000 },
    fetchEoqSummary: vi.fn().mockResolvedValue({
      total_cycle_stock: 50000,
      avg_effective_eoq: 1200,
      avg_order_frequency: 4.5,
      total_annual_cost: 320000,
      by_abc: [
        { abc_vol: "A", count: 100, avg_eoq: 1500, total_cycle_stock: 30000, total_annual_cost: 200000 },
      ],
    }),
    fetchEoqDetail: vi.fn().mockResolvedValue({
      total: 1,
      rows: [
        {
          item_no: "100320",
          loc: "1401-BULK",
          abc_vol: "A",
          eoq: 1500.5,
          effective_eoq: 1501.0,
          eoq_cycle_stock: 750.5,
          order_frequency: 6.2,
          total_annual_cost: 5000.0,
        },
      ],
    }),
    fetchEoqSensitivity: vi.fn().mockResolvedValue({
      curve: [
        { ordering_cost: 10, effective_eoq: 500, total_annual_cost: 3000 },
        { ordering_cost: 50, effective_eoq: 1200, total_annual_cost: 5500 },
      ],
    }),
  };
});

import { EoqPanel } from "@/tabs/inv-planning/EoqPanel";

describe("EoqPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <EoqPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Total Cycle Stock")).toBeDefined();
    expect(await screen.findByText("Avg EOQ Size")).toBeDefined();
    expect(await screen.findByText("Avg Order Frequency")).toBeDefined();
    expect(await screen.findByText("Total Annual Cost")).toBeDefined();
  });

  it("renders EOQ Detail table", async () => {
    render(
      <TestQueryWrapper>
        <EoqPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("EOQ Detail")).toBeDefined();
    expect(await screen.findByText("100320")).toBeDefined();
  });

  it("renders sensitivity chart section", async () => {
    render(
      <TestQueryWrapper>
        <EoqPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("EOQ Sensitivity")).toBeDefined();
  });

  it("renders By ABC Class section", async () => {
    render(
      <TestQueryWrapper>
        <EoqPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("By ABC Class")).toBeDefined();
  });
});
