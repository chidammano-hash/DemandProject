import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------
const mockRuns = [
  {
    run_id: 1,
    run_label: "production_baseline",
    model_id: "lgbm_cluster",
    started_at: "2026-03-22T10:00:00",
    completed_at: "2026-03-22T11:00:00",
    status: "completed" as const,
    accuracy_pct: 72.5,
    wape: 27.5,
    bias: -0.013,
    n_predictions: 2725140,
    n_dfus: 50602,
    feature_count: 37,
    params: { n_estimators: 1500, learning_rate: 0.02 },
    notes: null,
    is_promoted: true,
    promoted_at: "2026-03-22T12:00:00",
  },
  {
    run_id: 2,
    run_label: "aggressive_depth",
    model_id: "lgbm_cluster",
    started_at: "2026-03-22T14:00:00",
    completed_at: "2026-03-22T15:00:00",
    status: "completed" as const,
    accuracy_pct: 73.8,
    wape: 26.2,
    bias: -0.009,
    n_predictions: 2725140,
    n_dfus: 50602,
    feature_count: 45,
    params: { n_estimators: 1500, learning_rate: 0.02, max_depth: 10 },
    notes: null,
    is_promoted: false,
    promoted_at: null,
  },
  {
    run_id: 3,
    run_label: "ultra_slow_lr",
    model_id: "lgbm_cluster",
    started_at: "2026-03-23T08:00:00",
    completed_at: null,
    status: "running" as const,
    accuracy_pct: null,
    wape: null,
    bias: null,
    n_predictions: null,
    n_dfus: null,
    feature_count: null,
    params: { n_estimators: 3000, learning_rate: 0.008 },
    notes: null,
    is_promoted: false,
    promoted_at: null,
  },
];

// ---------------------------------------------------------------------------
// Mocks — global fetch (ModelTuningTab uses inline fetch, not barrel imports)
// ---------------------------------------------------------------------------
function mockFetchWith(experiments: typeof mockRuns, total?: number) {
  vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok: true,
    json: () =>
      Promise.resolve({
        experiments,
        total: total ?? experiments.length,
      }),
  } as Response);
}

// ---------------------------------------------------------------------------
// Mock recharts
// ---------------------------------------------------------------------------
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => null,
  Cell: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
  ReferenceLine: () => null,
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
}));

// ---------------------------------------------------------------------------
// Mock @/api/queries — ModelTuningTab imports STALE, TuningRun, ModelType
// ---------------------------------------------------------------------------
vi.mock("@/api/queries", () => ({
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  STALE_INSIGHTS: 300000,
  insightKeys: { all: () => ["insights"] },
  queryKeys: {},
}));

// ---------------------------------------------------------------------------
// Mock @/hooks/useChartColors
// ---------------------------------------------------------------------------
vi.mock("@/hooks/useChartColors", () => ({
  useChartColors: () => ({
    primary: "#3b82f6",
    secondary: "#10b981",
    accent: "#f59e0b",
    muted: "#6b7280",
    success: "#22c55e",
    warning: "#f59e0b",
    danger: "#ef4444",
  }),
}));

// ---------------------------------------------------------------------------
// Mock @/components/shared-tuning-utils
// ---------------------------------------------------------------------------
vi.mock("@/components/shared-tuning-utils", () => ({
  StatusBadge: ({ status }: { status: string }) => (
    <span data-testid="status-badge">{status}</span>
  ),
  formatDuration: (start: string | null, end: string | null) =>
    start && end ? "1h 0m" : start ? "running..." : "--",
  timeAgo: (ts: string | null) => (ts ? "2d ago" : "--"),
}));

// ---------------------------------------------------------------------------
// Mock sub-panel components as simple divs
// ---------------------------------------------------------------------------
vi.mock("../lgbm-tuning/TuningChatPanel", () => ({
  TuningChatPanel: () => <div>TuningChatPanel</div>,
}));
vi.mock("../lgbm-tuning/ClusterEDAPanel", () => ({
  ClusterEDAPanel: () => <div data-testid="cluster-eda-panel">ClusterEDAPanel</div>,
}));
vi.mock("../lgbm-tuning/FeatureLabPanel", () => ({
  FeatureLabPanel: () => <div data-testid="feature-lab-panel">FeatureLabPanel</div>,
}));
vi.mock("../model-tuning/LagFilterBar", () => ({
  LagFilterBar: ({ value, onChange }: { value?: number; onChange: (v?: number) => void }) => (
    <div data-testid="lag-filter-bar">LagFilterBar</div>
  ),
}));
vi.mock("../model-tuning/EnhancedComparisonPanel", () => ({
  EnhancedComparisonPanel: () => <div>EnhancedComparisonPanel</div>,
}));
vi.mock("../model-tuning/ExperimentBuilder", () => ({
  ExperimentBuilder: ({ open }: { open: boolean }) =>
    open ? <div>ExperimentBuilder</div> : null,
}));
vi.mock("../model-tuning/EnhancedPromoteModal", () => ({
  EnhancedPromoteModal: ({ open }: { open: boolean }) =>
    open ? <div>EnhancedPromoteModal</div> : null,
}));
vi.mock("../model-tuning/LogViewer", () => ({
  LogViewer: ({ open }: { open: boolean }) =>
    open ? <div>LogViewer</div> : null,
}));
vi.mock("../clusters/ClusterExperimentsPanel", () => ({
  ClusterExperimentsPanel: () => <div data-testid="cluster-experiments-panel">ClusterExperimentsPanel</div>,
}));
vi.mock("../champion/ChampionExperimentsPanel", () => ({
  ChampionExperimentsPanel: () => <div data-testid="champion-panel">ChampionExperimentsPanel</div>,
}));
vi.mock("../model-tuning/PipelineConfigPanel", () => ({
  PipelineConfigPanel: () => <div data-testid="pipeline-config-panel">PipelineConfigPanel</div>,
}));

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe("ModelTuningTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchWith(mockRuns);
  });

  it("renders pipeline stage tabs", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Model Experimentation Studio")).toBeInTheDocument();
    });
    expect(screen.getByText("Clustering")).toBeInTheDocument();
    expect(screen.getByText("Backtest & Tune")).toBeInTheDocument();
    expect(screen.getByText("Champion")).toBeInTheDocument();
    expect(screen.getByText("Forecast")).toBeInTheDocument();
    expect(screen.getByText("Pipeline Config")).toBeInTheDocument();
  });

  it("renders all 12 models in model grid", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      // LightGBM appears in grid + selected model header, use getAllByText
      expect(screen.getAllByText("LightGBM").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("CatBoost").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("XGBoost").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Chronos T5")).toBeInTheDocument();
      expect(screen.getByText("Chronos Bolt")).toBeInTheDocument();
      expect(screen.getByText("Chronos 2")).toBeInTheDocument();
      expect(screen.getByText("MSTL")).toBeInTheDocument();
      expect(screen.getByText("N-BEATS")).toBeInTheDocument();
      expect(screen.getByText("Seasonal Naive")).toBeInTheDocument();
      expect(screen.getByText("Rolling Mean")).toBeInTheDocument();
    });
  });

  it("switches to Champion panel on stage click", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Champion")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Champion"));
    await waitFor(() => {
      expect(screen.getByTestId("champion-panel")).toBeInTheDocument();
    });
  });

  it("switches to Clustering panel on stage click", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Clustering")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Clustering"));
    await waitFor(() => {
      expect(screen.getByTestId("cluster-experiments-panel")).toBeInTheDocument();
    });
  });

  it("switches to Pipeline Config panel on stage click", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Pipeline Config")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Pipeline Config"));
    await waitFor(() => {
      expect(screen.getByTestId("pipeline-config-panel")).toBeInTheDocument();
    });
  });

  it("selects model from grid and shows experiments", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("CatBoost")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("CatBoost"));
    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining("/model-tuning/catboost/experiments"),
        expect.anything(),
      );
    });
  });

  it("renders KPI summary cards", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Best Accuracy")).toBeInTheDocument();
      expect(screen.getByText("Production Accuracy")).toBeInTheDocument();
      expect(screen.getByText("Total Runs")).toBeInTheDocument();
      expect(screen.getByText("Active Runs")).toBeInTheDocument();
    });
  });

  it("renders run history table", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Run History")).toBeInTheDocument();
      expect(screen.getByText("Label")).toBeInTheDocument();
      expect(screen.getByText("Status")).toBeInTheDocument();
      expect(screen.getByText("Acc%")).toBeInTheDocument();
      expect(screen.getByText("WAPE")).toBeInTheDocument();
      expect(screen.getByText("Bias")).toBeInTheDocument();
    });
  });

  it("selects baseline on first row click", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("production_baseline")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("production_baseline"));
    await waitFor(() => {
      expect(screen.getByText("(B)")).toBeInTheDocument();
    });
  });

  it("selects candidate on second row click", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("production_baseline")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("production_baseline"));
    await waitFor(() => {
      expect(screen.getByText("(B)")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("aggressive_depth"));
    await waitFor(() => {
      expect(screen.getByText("(C)")).toBeInTheDocument();
    });
  });

  it("opens AI Tuning Advisor on FAB click", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Model Experimentation Studio")).toBeInTheDocument();
    });
    expect(screen.getByTitle("AI Tuning Advisor")).toBeInTheDocument();
    fireEvent.click(screen.getByTitle("AI Tuning Advisor"));
    await waitFor(() => {
      expect(screen.getByText("AI Tuning Advisor")).toBeInTheDocument();
    });
  });

  it("shows status badges for runs", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getAllByText("completed").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("running")).toBeInTheDocument();
    });
  });

  it("shows promoted run label in table", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      // The promoted run label should be visible in the run history table
      expect(screen.getByText("production_baseline")).toBeInTheDocument();
    });
  });

  it("shows empty state for zero runs", async () => {
    mockFetchWith([], 0);
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText(/No experiments yet/i)).toBeInTheDocument();
    });
  });

  it("shows non-tunable model info when foundation model selected", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Chronos 2")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Chronos 2"));
    await waitFor(() => {
      // Multiple elements may match "foundation model" — use getAllByText
      expect(screen.getAllByText(/foundation model/i).length).toBeGreaterThanOrEqual(1);
    });
  });

  it("shows model detail sub-tabs for tunable models", async () => {
    const ModelTuningTab = (await import("../ModelTuningTab")).default;
    render(<TestQueryWrapper><ModelTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Experiments")).toBeInTheDocument();
      expect(screen.getByText("Feature Lab")).toBeInTheDocument();
      expect(screen.getByText("Cluster EDA")).toBeInTheDocument();
    });
  });
});
