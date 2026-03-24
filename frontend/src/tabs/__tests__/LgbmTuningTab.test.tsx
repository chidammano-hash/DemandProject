import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// Mock recharts
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

vi.mock("@/api/queries", () => ({
  lgbmTuningKeys: {
    runs: (p?: Record<string, unknown>) => ["lgbm-tuning-runs", p],
    run: (id: number) => ["lgbm-tuning-run", id],
    compare: (b: number, c: number) => ["lgbm-tuning-compare", b, c],
    comparisons: (n?: number) => ["lgbm-tuning-comparisons", n],
    promoted: () => ["lgbm-tuning-promoted"],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchTuningRuns: vi.fn().mockResolvedValue({
    runs: [
      { run_id: 1, run_label: "baseline", model_id: "lgbm_cluster", started_at: "2026-03-22T10:00:00", completed_at: "2026-03-22T11:00:00", status: "completed", accuracy_pct: 69.34, wape: 30.66, bias: -0.0132, n_predictions: 2725140, n_dfus: 50602, feature_count: 37, params: null, notes: null },
      { run_id: 2, run_label: "v2_features", model_id: "lgbm_cluster", started_at: "2026-03-22T12:00:00", completed_at: "2026-03-22T13:00:00", status: "completed", accuracy_pct: 69.89, wape: 30.11, bias: -0.0098, n_predictions: 2725140, n_dfus: 50602, feature_count: 45, params: null, notes: null },
    ],
    total: 2,
  }),
  fetchTuningComparison: vi.fn().mockResolvedValue({
    baseline: { run_id: 1, run_label: "baseline", accuracy_pct: 69.34, wape: 30.66, bias: -0.0132 },
    candidate: { run_id: 2, run_label: "v2_features", accuracy_pct: 69.89, wape: 30.11, bias: -0.0098 },
    delta_accuracy: 0.55,
    delta_wape: -0.55,
    delta_bias: 0.0034,
    verdict: "improved",
    per_timeframe: [],
  }),
  fetchTuningComparisons: vi.fn().mockResolvedValue({ comparisons: [] }),
  promoteRun: vi.fn().mockResolvedValue({ promoted: true, run_id: 1, run_label: "baseline", accuracy_pct: 69.34, params_written: {}, old_params: {} }),
  fetchPromotedRun: vi.fn().mockResolvedValue({ promoted: null }),
  // Tuning chat mocks
  tuningChatKeys: {
    sessions: () => ["tuning-chat-sessions"],
    session: (id: string) => ["tuning-chat-session", id],
    runStatus: (sid: string, rid: number) => ["tuning-chat-run-status", sid, rid],
  },
  fetchChatSessions: vi.fn().mockResolvedValue({ sessions: [] }),
  fetchChatSession: vi.fn().mockResolvedValue({ session: {}, messages: [] }),
  createChatSession: vi.fn().mockResolvedValue({ session: { session_id: "abc" } }),
  sendTuningChatMessage: vi.fn().mockResolvedValue({ messages: [] }),
  confirmTuningRun: vi.fn().mockResolvedValue({ run_id: 1, status: "started" }),
  fetchRunStatus: vi.fn().mockResolvedValue({ run_id: 1, status: "running" }),
  // Cluster EDA mocks
  clusterEdaKeys: {
    all: ["cluster-eda"],
    profile: () => ["cluster-eda", "profile"],
    errorConcentration: () => ["cluster-eda", "error-concentration"],
    distribution: (id: number) => ["cluster-eda", "distribution", id],
    residuals: (m: string) => ["cluster-eda", "residuals", m],
    seasonalityHeatmap: () => ["cluster-eda", "seasonality-heatmap"],
  },
  fetchClusterProfile: vi.fn().mockResolvedValue({
    rows: [
      { cluster: 0, n_dfus: 8421, mean_demand: 142, cv: 0.45, zero_pct: 12.0, seasonal_amplitude: 0.35, accuracy_pct: 78.2 },
    ],
  }),
  fetchErrorConcentration: vi.fn().mockResolvedValue({
    top_10_pct_share: 47.0,
    worst_months: [],
    worst_clusters: [],
  }),
  fetchClusterDistribution: vi.fn().mockResolvedValue({ cluster: 0, bins: [] }),
  fetchSeasonalityHeatmap: vi.fn().mockResolvedValue({ months: [], rows: [] }),
  // Feature Lab mocks
  featureLabKeys: {
    all: ["feature-lab"],
    importance: (modelId?: string) => ["feature-lab", "importance", modelId ?? "default"],
    stability: () => ["feature-lab", "stability"],
    correlation: (topN?: number) => ["feature-lab", "correlation", topN ?? 20],
    clusterImportance: (cluster: number) => ["feature-lab", "cluster-importance", cluster],
    categories: () => ["feature-lab", "categories"],
  },
  fetchFeatureImportance: vi.fn().mockResolvedValue({
    model_id: "lgbm_cluster",
    features: [{ feature: "lag_1_demand", shap_value: 0.0821, category: "lag", rank: 1 }],
  }),
  fetchFeatureStability: vi.fn().mockResolvedValue({
    features: [{ feature: "lag_1_demand", mean_rank: 1.2, rank_std: 0.4, stability: "stable", n_folds: 5 }],
  }),
  fetchFeatureCorrelation: vi.fn().mockResolvedValue({ top_n: 20, features: [], cells: [] }),
  fetchClusterFeatureImportance: vi.fn().mockResolvedValue({ cluster: 0, features: [] }),
  fetchFeatureCategories: vi.fn().mockResolvedValue({ categories: [{ category: "lag", color: "#3b82f6", count: 6 }] }),
  // Accuracy Budget mocks
  accuracyBudgetKeys: {
    all: ["accuracy-budget"],
    decomposition: (m?: string) => ["accuracy-budget", "decomposition", m ?? "lgbm_cluster"],
    abc: () => ["accuracy-budget", "abc-breakdown"],
    models: () => ["accuracy-budget", "model-comparison"],
    monthly: () => ["accuracy-budget", "monthly-trend"],
    forecastValue: () => ["accuracy-budget", "forecast-value"],
  },
  fetchAccuracyDecomposition: vi.fn().mockResolvedValue({
    current_accuracy: 69.3,
    current_wape: 30.7,
    current_bias: -0.012,
    n_dfus: 50602,
    model_id: "lgbm_cluster",
    oracle_ceiling: 85.0,
    oracle_wape: 15.0,
    naive_baseline: 45.2,
    naive_wape: 54.8,
    forecast_value_added: 24.1,
    addressable_gap: 15.7,
    abc_breakdown: [],
    cluster_breakdown: [],
    components: [],
    irreducible_noise: 6.0,
  }),
  fetchAbcBreakdown: vi.fn().mockResolvedValue({ classes: [] }),
  fetchMonthlyTrend: vi.fn().mockResolvedValue({ months: [], worst_month: null, best_month: null }),
  fetchModelComparison: vi.fn().mockResolvedValue({ models: [], oracle_ceiling: null }),
  fetchForecastValue: vi.fn().mockResolvedValue({ baselines: [], ml_model: null, value_added: null }),
  // Include all other exports that might be needed by barrel import
  queryKeys: {},
  STALE_INSIGHTS: 300000,
  insightKeys: { all: () => ["insights"] },
}));

describe("LgbmTuningTab", () => {
  it("renders the tab heading", async () => {
    const LgbmTuningTab = (await import("../LgbmTuningTab")).default;
    render(<TestQueryWrapper><LgbmTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText(/LGBM Tuning/i)).toBeInTheDocument();
    });
  });

  it("renders run history table", async () => {
    const LgbmTuningTab = (await import("../LgbmTuningTab")).default;
    render(<TestQueryWrapper><LgbmTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("baseline")).toBeInTheDocument();
      expect(screen.getByText("v2_features")).toBeInTheDocument();
    });
  });

  it("shows accuracy values", async () => {
    const LgbmTuningTab = (await import("../LgbmTuningTab")).default;
    render(<TestQueryWrapper><LgbmTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      // formatPct(69.34) renders as "69.3%" in KPI card + table row
      const matches = screen.getAllByText("69.3%");
      expect(matches.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("defaults to runs panel with sub-tab navigation", async () => {
    const LgbmTuningTab = (await import("../LgbmTuningTab")).default;
    render(<TestQueryWrapper><LgbmTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      // Sub-tab buttons visible
      expect(screen.getByText("Runs")).toBeInTheDocument();
      expect(screen.getByText("Cluster EDA")).toBeInTheDocument();
      expect(screen.getByText("Feature Lab")).toBeInTheDocument();
      expect(screen.getByText("Accuracy Budget")).toBeInTheDocument();
    });
    // Run History table is visible (default)
    expect(screen.getByText("Run History")).toBeInTheDocument();
  });

  it("renders cluster-eda panel when selected", async () => {
    const LgbmTuningTab = (await import("../LgbmTuningTab")).default;
    render(<TestQueryWrapper><LgbmTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Cluster EDA")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Cluster EDA"));
    await waitFor(() => {
      // ClusterEDAPanel renders with cluster profile sub-tab (appears in both sub-tab button and CardTitle)
      expect(screen.getAllByText("Cluster Profiles").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders feature-lab panel when selected", async () => {
    const LgbmTuningTab = (await import("../LgbmTuningTab")).default;
    render(<TestQueryWrapper><LgbmTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Feature Lab")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Feature Lab"));
    await waitFor(() => {
      // FeatureLabPanel renders with importance sub-tab (appears in both sub-tab button and CardTitle)
      expect(screen.getAllByText("Feature Importance").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders accuracy-budget panel when selected", async () => {
    const LgbmTuningTab = (await import("../LgbmTuningTab")).default;
    render(<TestQueryWrapper><LgbmTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Accuracy Budget")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Accuracy Budget"));
    await waitFor(() => {
      // AccuracyBudgetPanel renders with waterfall sub-tab (appears in both sub-tab button and CardTitle)
      expect(screen.getAllByText("Accuracy Waterfall").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders Production column and promote buttons for completed runs", async () => {
    const LgbmTuningTab = (await import("../LgbmTuningTab")).default;
    render(<TestQueryWrapper><LgbmTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Production")).toBeInTheDocument();
    });
    // Both completed runs should show Promote buttons
    const promoteButtons = screen.getAllByText("Promote");
    expect(promoteButtons.length).toBe(2);
  });

  it("opens promote modal when Promote button clicked", async () => {
    const LgbmTuningTab = (await import("../LgbmTuningTab")).default;
    render(<TestQueryWrapper><LgbmTuningTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getAllByText("Promote").length).toBeGreaterThanOrEqual(1);
    });
    fireEvent.click(screen.getAllByText("Promote")[0]);
    await waitFor(() => {
      // Modal has both a title and button with "Promote to Production"
      expect(screen.getAllByText("Promote to Production").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText(/Run #1/)).toBeInTheDocument();
    });
  });
});
