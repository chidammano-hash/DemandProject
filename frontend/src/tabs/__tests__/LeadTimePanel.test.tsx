import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    queryKeys: {
      ...(actual as any).queryKeys,
      ltSummary: (p?: unknown) => ["lt-summary", p],
      ltProfile: (p?: unknown) => ["lt-profile", p],
    },
    STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
    fetchLtSummary: vi.fn().mockResolvedValue({
      avg_lt_mean_days: 14.5,
      avg_lt_cv: 0.28,
      by_class: { stable: 200, moderate: 80, volatile: 20 },
    }),
    fetchLtProfile: vi.fn().mockResolvedValue({
      rows: [
        {
          item_id: "100320",
          loc: "1401-BULK",
          lt_mean_days: 21.3,
          lt_std_days: 9.5,
          lt_cv: 0.45,
          lt_variability_class: "volatile",
        },
      ],
    }),
  };
});

import { LeadTimePanel } from "@/tabs/inv-planning/LeadTimePanel";

describe("LeadTimePanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <LeadTimePanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Avg Lead Time")).toBeDefined();
    expect(await screen.findByText("Volatile Suppliers")).toBeDefined();
    expect(await screen.findByText("Avg LT CV")).toBeDefined();
  });

  it("renders class breakdown chips", async () => {
    render(
      <TestQueryWrapper>
        <LeadTimePanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Stable")).toBeDefined();
    expect(await screen.findByText("Moderate")).toBeDefined();
    expect(await screen.findByText("Volatile")).toBeDefined();
  });

  it("renders volatile items table", async () => {
    render(
      <TestQueryWrapper>
        <LeadTimePanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Top Volatile Lead Time Items")).toBeDefined();
    expect(await screen.findByText("100320")).toBeDefined();
  });
});
