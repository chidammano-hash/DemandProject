import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    abcXyzKeys: {
      matrix: (p?: unknown) => ["abc-xyz-matrix", p],
      summary: (p?: unknown) => ["abc-xyz-summary", p],
    },
    STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
    fetchAbcXyzMatrix: vi.fn().mockResolvedValue({
      total_classified: 500,
      cells: [
        { segment: "AX", sku_count: 120, avg_service_level: 0.98 },
        { segment: "BZ", sku_count: 30, avg_service_level: 0.85 },
      ],
    }),
    fetchAbcXyzSummary: vi.fn().mockResolvedValue({
      total_skus: 600,
      z_count: 45,
    }),
  };
});

import { AbcXyzPanel } from "@/tabs/inv-planning/AbcXyzPanel";

describe("AbcXyzPanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <AbcXyzPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Total SKUs")).toBeDefined();
    expect(await screen.findByText("Classified")).toBeDefined();
    expect(await screen.findByText("Z-Class (High Variability)")).toBeDefined();
  });

  it("renders the 3x3 matrix table", async () => {
    render(
      <TestQueryWrapper>
        <AbcXyzPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("120")).toBeDefined();
    expect(await screen.findByText("30")).toBeDefined();
  });
});
