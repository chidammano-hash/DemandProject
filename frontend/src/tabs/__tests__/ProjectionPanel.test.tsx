import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ComposedChart: ({ children }: { children: React.ReactNode }) => <div data-testid="composed-chart">{children}</div>,
  Line: () => null,
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
  ReferenceLine: () => null,
}));

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    projectionKeys: {
      sku: (p?: unknown) => ["projection-sku", p],
      atRisk: (days?: unknown) => ["projection-at-risk", days],
    },
    queryKeys: {
      ...(actual as any).queryKeys,
      planningDate: () => ["planning-date"],
    },
    STALE: { FIVE_MIN: 300000, TEN_MIN: 600000, ONE_MIN: 60000, TWO_MIN: 120000 },
    fetchProjection: vi.fn(),
    fetchProjectionAtRisk: vi.fn(),
    refreshProjection: vi.fn(),
    fetchPlanningDate: vi.fn(),
  };
});

import { fetchProjectionAtRisk, fetchPlanningDate } from "@/api/queries";
import { ProjectionPanel } from "@/tabs/inv-planning/ProjectionPanel";

beforeEach(() => {
  vi.clearAllMocks();
  (fetchProjectionAtRisk as any).mockResolvedValue({ total: 0, items: [] });
  (fetchPlanningDate as any).mockResolvedValue({
    planning_date: "2026-02-24",
    is_frozen: true,
    source: "config",
  });
});

describe("ProjectionPanel", () => {
  it("renders item selector inputs", async () => {
    render(
      <TestQueryWrapper>
        <ProjectionPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByPlaceholderText("e.g. 100320")).toBeInTheDocument();
    });
    expect(screen.getByPlaceholderText("e.g. 1401-BULK")).toBeInTheDocument();
  });

  it("renders Project button", async () => {
    render(
      <TestQueryWrapper>
        <ProjectionPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Project")).toBeInTheDocument();
  });

  it("renders empty state when no DFU selected", async () => {
    render(
      <TestQueryWrapper>
        <ProjectionPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Select a DFU to view inventory projection")).toBeInTheDocument();
  });

  it("renders at-risk banner when items at risk", async () => {
    (fetchProjectionAtRisk as any).mockResolvedValue({
      total: 2,
      items: [
        { item_id: "100320", loc: "1401-BULK", days_until_stockout: 5 },
        { item_id: "200450", loc: "2001-BULK", days_until_stockout: 12 },
      ],
    });
    render(
      <TestQueryWrapper>
        <ProjectionPanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("2 SKUs at stockout risk within 30 days")).toBeInTheDocument();
  });
});
