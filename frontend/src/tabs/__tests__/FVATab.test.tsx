import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// Mock recharts
vi.mock("recharts", () => ({
  BarChart: ({ children }: any) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
  Cell: () => null,
}));

vi.mock("@/api/queries", () => ({
  fetchFVAWaterfall: vi.fn().mockResolvedValue({ waterfall: { models: [] } }),
  fetchFVAROI: vi.fn().mockResolvedValue({ total_interventions: 0, measured: 0, total_estimated_impact: 0, total_actual_impact: 0 }),
  fetchFVAInterventions: vi.fn().mockResolvedValue({ interventions: [] }),
  fvaKeys: { waterfall: (m: number) => ["fva", "waterfall", m], roi: (m: number) => ["fva", "roi", m], interventions: ["fva", "interventions"] },
  STALE_PLATFORM: 300000,
}));

vi.mock("@/components/KpiCard", () => ({
  KpiCard: ({ label, value }: any) => <div data-testid="kpi-card">{label}: {value}</div>,
}));

vi.mock("@/context/ThemeContext", () => ({
  useThemeContext: () => ({ theme: { mode: "light" } }),
  ThemeProvider: ({ children }: any) => children,
}));

describe("FVATab", () => {
  it("renders without crashing", async () => {
    const { default: FVATab } = await import("../FVATab");
    render(
      <TestQueryWrapper>
        <FVATab />
      </TestQueryWrapper>
    );
    expect(screen.getByText("Forecast Value Added")).toBeInTheDocument();
  });

  it("renders month selector", async () => {
    const { default: FVATab } = await import("../FVATab");
    render(
      <TestQueryWrapper>
        <FVATab />
      </TestQueryWrapper>
    );
    expect(screen.getByText("12 months")).toBeInTheDocument();
  });

  it("shows KPI cards", async () => {
    const { default: FVATab } = await import("../FVATab");
    render(
      <TestQueryWrapper>
        <FVATab />
      </TestQueryWrapper>
    );
    const cards = screen.getAllByTestId("kpi-card");
    expect(cards.length).toBe(4);
  });

  it("shows empty interventions message", async () => {
    const { default: FVATab } = await import("../FVATab");
    render(
      <TestQueryWrapper>
        <FVATab />
      </TestQueryWrapper>
    );
    expect(screen.getByText("No interventions recorded yet")).toBeInTheDocument();
  });
});
