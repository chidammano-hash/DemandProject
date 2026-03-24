import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries/evolution", () => ({
  biasKeys: {
    summary: (p?: unknown) => ["bias-summary", p],
    flagged: (p?: unknown) => ["bias-flagged", p],
    list: (p?: unknown) => ["bias-list", p],
    history: (s?: string, v?: string) => ["bias-history", s, v],
  },
  STALE_EVO: { FIVE_MIN: 300000, ONE_MIN: 60000 },
  fetchBiasCorrectionSummary: vi.fn().mockResolvedValue({
    total_corrections: 120,
    sku_count: 95,
    flagged_count: 8,
    clipped_count: 3,
    avg_rolling_bias: 0.12,
    avg_correction_factor: 0.895,
    last_computed_at: "2026-03-01T03:00:00Z",
    plan_month: "2026-04-01",
  }),
  fetchFlaggedBiasCorrections: vi.fn().mockResolvedValue({
    total: 1,
    page: 1,
    flagged: [
      {
        item_id: "100320",
        loc: "1401-BULK",
        plan_month: "2026-04-01",
        segment_type: "cluster",
        rolling_bias_3m: 0.25,
        correction_factor_raw: 0.8,
        correction_factor: 0.80,
        correction_was_clipped: false,
        months_of_data: 3,
      },
    ],
  }),
}));

import { BiasCorrectionsPanel } from "@/tabs/accuracy/BiasCorrectionsPanel";

describe("BiasCorrectionsPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <BiasCorrectionsPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("SKUs Corrected")).toBeDefined();
    expect(await screen.findByText("Avg Correction Factor")).toBeDefined();
    expect(await screen.findByText("Flagged for Review")).toBeDefined();
    expect(await screen.findByText("Clipped (Guard Rail)")).toBeDefined();
  });

  it("renders flagged items table row", async () => {
    render(
      <TestQueryWrapper>
        <BiasCorrectionsPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("100320")).toBeDefined();
    expect(await screen.findByText("Items Flagged for Review")).toBeDefined();
  });

  it("renders avg rolling bias section", async () => {
    render(
      <TestQueryWrapper>
        <BiasCorrectionsPanel />
      </TestQueryWrapper>
    );
    expect(await screen.findByText(/Avg Rolling 3-Month Bias/)).toBeDefined();
  });
});
