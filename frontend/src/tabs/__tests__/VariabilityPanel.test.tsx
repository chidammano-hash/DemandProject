import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  PieChart: ({ children }: { children: React.ReactNode }) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => null,
  Cell: () => null,
  Tooltip: () => null,
}));

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    queryKeys: {
      ...(actual as any).queryKeys,
      variabilitySummary: (p?: unknown) => ["var-summary", p],
      variabilityDetail: (p?: unknown) => ["var-detail", p],
    },
    STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
    fetchVariabilitySummary: vi.fn().mockResolvedValue({
      total_skus: 500,
      avg_cv: 0.45,
      by_class: { low: 200, medium: 180, high: 100, lumpy: 20 },
    }),
    fetchVariabilityDetail: vi.fn().mockResolvedValue({
      rows: [
        {
          item_id: "100320",
          loc: "1401-BULK",
          demand_cv: 0.92,
          variability_class: "high",
        },
      ],
    }),
  };
});

import { VariabilityPanel } from "@/tabs/inv-planning/VariabilityPanel";

describe("VariabilityPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <VariabilityPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Stable Items")).toBeDefined();
    expect(await screen.findByText("Volatile Items")).toBeDefined();
    expect(await screen.findByText("Avg CV")).toBeDefined();
  });

  it("renders volatile items table", async () => {
    render(
      <TestQueryWrapper>
        <VariabilityPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Top Volatile Items")).toBeDefined();
    expect(await screen.findByText("100320")).toBeDefined();
  });
});
