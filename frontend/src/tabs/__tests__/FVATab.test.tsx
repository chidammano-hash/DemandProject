import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// Mock recharts — uses shared mock at frontend/__mocks__/recharts.tsx
vi.mock("recharts");

vi.mock("@/api/queries", () => ({
  fetchFVAWaterfall: vi.fn().mockResolvedValue({
    waterfall: {
      stages: [
        { stage_id: "seasonal_naive", label: "Naive Seasonal", accuracy_pct: 52.1, delta_vs_prev: null, state: "actual" },
        { stage_id: "external", label: "External", accuracy_pct: 60.4, delta_vs_prev: 8.3, state: "actual" },
        { stage_id: "champion", label: "Champion", accuracy_pct: 67.8, delta_vs_prev: 7.4, state: "actual" },
        { stage_id: "ai_adjusted", label: "AI Adjusted", accuracy_pct: null, delta_vs_prev: null, state: "planned" },
        { stage_id: "planner_adjusted", label: "Planner Adjusted", accuracy_pct: null, delta_vs_prev: null, state: "planned" },
      ],
      benchmark: { stage_id: "ceiling", label: "Ceiling Benchmark", accuracy_pct: 73.9, state: "actual" },
      models: [],
    },
  }),
  fetchFVAROI: vi.fn().mockResolvedValue({ total_interventions: 0, measured: 0, total_estimated_impact: 0, total_actual_impact: 0 }),
  fetchFVAInterventions: vi.fn().mockResolvedValue({ interventions: [] }),
  fetchFVASnapshotMonths: vi.fn().mockResolvedValue({ months: [] }),
  fetchFVASnapshotAccuracy: vi.fn().mockResolvedValue({ rows: [] }),
  fetchFVAHistoricalBacktestMonths: vi.fn().mockResolvedValue({ months: [] }),
  fetchFVAHistoricalBacktestAccuracy: vi.fn().mockResolvedValue({ rows: [] }),
  fvaKeys: {
    waterfall: (m: number) => ["fva", "waterfall", m],
    roi: (m: number) => ["fva", "roi", m],
    interventions: ["fva", "interventions"],
    snapshotMonths: ["fva", "snapshot-months"],
    snapshotAccuracy: (month: string) => ["fva", "snapshot", month],
    historicalBacktestMonths: ["fva", "historical-backtest-months"],
    historicalBacktestAccuracy: (month: string) => ["fva", "historical-backtest", month],
  },
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

  it("renders the staged FVA ladder and benchmark", async () => {
    const { default: FVATab } = await import("../FVATab");
    render(
      <TestQueryWrapper>
        <FVATab />
      </TestQueryWrapper>
    );
    expect(await screen.findByText("Forecast Value Ladder")).toBeInTheDocument();
    expect(await screen.findByText("Naive Seasonal")).toBeInTheDocument();
    expect(await screen.findByText("AI Adjusted")).toBeInTheDocument();
    expect((await screen.findAllByText("Coming Soon")).length).toBe(2);
    expect(await screen.findByText("Ceiling Benchmark")).toBeInTheDocument();
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
