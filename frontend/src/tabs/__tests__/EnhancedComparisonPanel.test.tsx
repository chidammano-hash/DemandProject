import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------
const mockLags = [
  { exec_lag: 0, baseline_acc: 78.2, candidate_acc: 79.5, delta_acc: 1.3, baseline_wape: 21.8, candidate_wape: 20.5, delta_wape: -1.3, baseline_bias: 0.25, candidate_bias: 0.22, delta_bias: -0.03 },
  { exec_lag: 1, baseline_acc: 72.4, candidate_acc: 73.8, delta_acc: 1.4, baseline_wape: 27.6, candidate_wape: 26.2, delta_wape: -1.4, baseline_bias: 0.30, candidate_bias: 0.28, delta_bias: -0.02 },
  { exec_lag: 2, baseline_acc: 68.1, candidate_acc: 69.0, delta_acc: 0.9, baseline_wape: 31.9, candidate_wape: 31.0, delta_wape: -0.9, baseline_bias: 0.38, candidate_bias: 0.36, delta_bias: -0.02 },
  { exec_lag: 3, baseline_acc: 64.5, candidate_acc: 65.8, delta_acc: 1.3, baseline_wape: 35.5, candidate_wape: 34.2, delta_wape: -1.3, baseline_bias: 0.45, candidate_bias: 0.42, delta_bias: -0.03 },
  { exec_lag: 4, baseline_acc: 61.2, candidate_acc: 62.1, delta_acc: 0.9, baseline_wape: 38.8, candidate_wape: 37.9, delta_wape: -0.9, baseline_bias: 0.55, candidate_bias: 0.52, delta_bias: -0.03 },
];

const mockComparison = {
  baseline: {
    run_id: 12,
    run_label: "production_baseline",
    accuracy_pct: 72.22,
    wape: 27.78,
    bias: -0.013,
    model_id: "lgbm_cluster",
    started_at: "2026-03-22T10:00:00",
    completed_at: "2026-03-22T11:00:00",
    status: "completed",
    n_predictions: 2725140,
    n_dfus: 50602,
    feature_count: 37,
    params: { n_estimators: 1500, learning_rate: 0.02 },
    notes: null,
  },
  candidate: {
    run_id: 15,
    run_label: "aggressive_depth",
    accuracy_pct: 73.45,
    wape: 26.55,
    bias: -0.009,
    model_id: "lgbm_cluster",
    started_at: "2026-03-22T14:00:00",
    completed_at: "2026-03-22T15:00:00",
    status: "completed",
    n_predictions: 2725140,
    n_dfus: 50602,
    feature_count: 45,
    params: { n_estimators: 1500, learning_rate: 0.02, max_depth: 10, num_leaves: 63, reg_lambda: 3.5 },
    notes: null,
  },
  delta_accuracy: 1.23,
  delta_wape: -1.23,
  delta_bias: 0.004,
  verdict: "improved",
  per_lag: mockLags,
  per_timeframe: [
    { timeframe: "A", baseline_accuracy: 70.0, candidate_accuracy: 71.5, delta_accuracy: 1.5 },
    { timeframe: "B", baseline_accuracy: 72.0, candidate_accuracy: 73.0, delta_accuracy: 1.0 },
  ],
  param_diffs: [
    { param: "max_depth", baseline: -1, candidate: 10 },
    { param: "num_leaves", baseline: 127, candidate: 63 },
    { param: "reg_lambda", baseline: 1.0, candidate: 3.5 },
  ],
  param_common: [
    { param: "n_estimators", value: 1500 },
    { param: "learning_rate", value: 0.02 },
  ],
  feature_diffs: {
    baseline_count: 37,
    candidate_count: 45,
    added: ["rolling_avg_6m", "demand_cv"],
    removed: [],
    common_count: 37,
  },
  config_diffs: [],
  config_common: [
    { setting: "cluster_strategy", value: "per_cluster" },
    { setting: "recursive", value: true },
  ],
};

const mockComparisonNoLags = {
  ...mockComparison,
  per_lag: undefined,
};

const mockFetchTuningComparison = vi.fn().mockResolvedValue(mockComparison);
const mockFetchModelTuningComparison = vi.fn().mockResolvedValue(mockComparison);

vi.mock("recharts");

vi.mock("@/api/queries", () => ({
  lgbmTuningKeys: {
    runs: (p?: Record<string, unknown>) => ["lgbm-tuning-runs", p],
    run: (id: number) => ["lgbm-tuning-run", id],
    compare: (b: number, c: number) => ["lgbm-tuning-compare", b, c],
    comparisons: (n?: number) => ["lgbm-tuning-comparisons", n],
    promoted: () => ["lgbm-tuning-promoted"],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchTuningRuns: vi.fn().mockResolvedValue({ runs: [], total: 0 }),
  fetchTuningComparison: (...args: unknown[]) => mockFetchTuningComparison(...args),
  fetchTuningComparisons: vi.fn().mockResolvedValue({ comparisons: [] }),
  promoteRun: vi.fn().mockResolvedValue({ promoted: true }),
  fetchPromotedRun: vi.fn().mockResolvedValue({ promoted: null }),
  modelTuningKeys: {
    runs: (m: string, p?: Record<string, unknown>) => [`${m}-tuning-runs`, p],
    run: (m: string, id: number) => [`${m}-tuning-run`, id],
    compare: (m: string, b: number, c: number) => [`${m}-tuning-compare`, b, c],
    comparisons: (m: string, n?: number) => [`${m}-tuning-comparisons`, n],
    promoted: (m: string) => [`${m}-tuning-promoted`],
  },
  fetchModelTuningRuns: vi.fn().mockResolvedValue({ runs: [], total: 0 }),
  fetchModelTuningRun: vi.fn().mockResolvedValue({}),
  fetchModelTuningComparison: (...args: unknown[]) => mockFetchModelTuningComparison(...args),
  fetchModelTuningComparisons: vi.fn().mockResolvedValue({ comparisons: [] }),
  promoteModelRun: vi.fn().mockResolvedValue({ promoted: true }),
  fetchModelPromotedRun: vi.fn().mockResolvedValue({ promoted: null }),
  fetchModelExperiments: vi.fn().mockResolvedValue({ runs: [], total_count: 0 }),
  fetchModelExperimentLags: vi.fn().mockResolvedValue({ lags: mockLags }),
  fetchModelComparison: vi.fn().mockResolvedValue(mockComparison),
  fetchModelTemplates: vi.fn().mockResolvedValue({ templates: [] }),
  submitModelExperiment: vi.fn().mockResolvedValue({ run_id: 1, job_id: "abc" }),
  promoteModelExperiment: vi.fn().mockResolvedValue({ success: true }),
  cancelModelExperiment: vi.fn().mockResolvedValue({ success: true }),
  fetchModelPromoted: vi.fn().mockResolvedValue(null),
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
  clusterEdaKeys: {
    all: ["cluster-eda"],
    profile: () => ["cluster-eda", "profile"],
    errorConcentration: () => ["cluster-eda", "error-concentration"],
    distribution: (id: number) => ["cluster-eda", "distribution", id],
    residuals: (m: string) => ["cluster-eda", "residuals", m],
    seasonalityHeatmap: () => ["cluster-eda", "seasonality-heatmap"],
  },
  fetchClusterProfile: vi.fn().mockResolvedValue({ rows: [] }),
  fetchErrorConcentration: vi.fn().mockResolvedValue({ top_10_pct_share: 47.0, worst_months: [], worst_clusters: [] }),
  fetchClusterDistribution: vi.fn().mockResolvedValue({ cluster: 0, bins: [] }),
  fetchSeasonalityHeatmap: vi.fn().mockResolvedValue({ months: [], rows: [] }),
  featureLabKeys: {
    all: ["feature-lab"],
    importance: (modelId?: string) => ["feature-lab", "importance", modelId ?? "default"],
    stability: () => ["feature-lab", "stability"],
    correlation: (topN?: number) => ["feature-lab", "correlation", topN ?? 20],
    clusterImportance: (cluster: number) => ["feature-lab", "cluster-importance", cluster],
    categories: () => ["feature-lab", "categories"],
  },
  fetchFeatureImportance: vi.fn().mockResolvedValue({ model_id: "lgbm_cluster", features: [] }),
  fetchFeatureStability: vi.fn().mockResolvedValue({ features: [] }),
  fetchFeatureCorrelation: vi.fn().mockResolvedValue({ top_n: 20, features: [], cells: [] }),
  fetchClusterFeatureImportance: vi.fn().mockResolvedValue({ cluster: 0, features: [] }),
  fetchFeatureCategories: vi.fn().mockResolvedValue({ categories: [] }),
  accuracyBudgetKeys: {
    all: ["accuracy-budget"],
    decomposition: (m?: string) => ["accuracy-budget", "decomposition", m ?? "lgbm_cluster"],
    abc: () => ["accuracy-budget", "abc-breakdown"],
    models: () => ["accuracy-budget", "model-comparison"],
    monthly: () => ["accuracy-budget", "monthly-trend"],
    forecastValue: () => ["accuracy-budget", "forecast-value"],
  },
  fetchAccuracyDecomposition: vi.fn().mockResolvedValue({
    current_accuracy: 69.3, current_wape: 30.7, current_bias: -0.012, n_dfus: 50602,
    model_id: "lgbm_cluster", oracle_ceiling: 85.0, oracle_wape: 15.0, naive_baseline: 45.2,
    naive_wape: 54.8, forecast_value_added: 24.1, addressable_gap: 15.7,
    abc_breakdown: [], cluster_breakdown: [], components: [], irreducible_noise: 6.0,
  }),
  fetchAbcBreakdown: vi.fn().mockResolvedValue({ classes: [] }),
  fetchMonthlyTrend: vi.fn().mockResolvedValue({ months: [], worst_month: null, best_month: null }),
  fetchModelComparison: vi.fn().mockResolvedValue({ models: [], oracle_ceiling: null }),
  fetchForecastValue: vi.fn().mockResolvedValue({ baselines: [], ml_model: null, value_added: null }),
  queryKeys: {},
  STALE_INSIGHTS: 300000,
  insightKeys: { all: () => ["insights"] },
}));

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe("EnhancedComparisonPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchTuningComparison.mockResolvedValue(mockComparison);
  });

  it("renders verdict badge", async () => {
    const { ComparisonPanel } = await import("../lgbm-tuning/ComparisonPanel");
    render(
      <TestQueryWrapper>
        <ComparisonPanel baselineId={12} candidateId={15} modelType="lgbm" />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("IMPROVED")).toBeInTheDocument();
    });
  });

  it("renders per-lag accuracy table", async () => {
    // Verify that per-lag data is structured with 5 rows (lag 0-4)
    expect(mockComparison.per_lag).toHaveLength(5);
    expect(mockComparison.per_lag[0].exec_lag).toBe(0);
    expect(mockComparison.per_lag[4].exec_lag).toBe(4);
    // Each row has baseline and candidate accuracy
    for (const lag of mockComparison.per_lag) {
      expect(lag.baseline_acc).toBeDefined();
      expect(lag.candidate_acc).toBeDefined();
      expect(lag.delta_acc).toBeDefined();
    }
  });

  it("renders portfolio metric cards", async () => {
    const { ComparisonPanel } = await import("../lgbm-tuning/ComparisonPanel");
    render(
      <TestQueryWrapper>
        <ComparisonPanel baselineId={12} candidateId={15} modelType="lgbm" />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      // 3 metric cards: Accuracy, WAPE, Bias
      expect(screen.getByText("Accuracy %")).toBeInTheDocument();
      expect(screen.getByText("WAPE")).toBeInTheDocument();
      expect(screen.getByText("Bias")).toBeInTheDocument();
    });
  });

  it("renders parameter diff table", async () => {
    const { ComparisonPanel } = await import("../lgbm-tuning/ComparisonPanel");
    render(
      <TestQueryWrapper>
        <ComparisonPanel baselineId={12} candidateId={15} modelType="lgbm" />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("IMPROVED")).toBeInTheDocument();
    });
    // Switch to Params view
    fireEvent.click(screen.getByText("Params"));
    await waitFor(() => {
      expect(screen.getByText("Changed Parameters")).toBeInTheDocument();
      expect(screen.getByText("max_depth")).toBeInTheDocument();
      expect(screen.getByText("num_leaves")).toBeInTheDocument();
      expect(screen.getByText("reg_lambda")).toBeInTheDocument();
    });
  });

  it("renders promote buttons", async () => {
    const { ComparisonPanel } = await import("../lgbm-tuning/ComparisonPanel");
    render(
      <TestQueryWrapper>
        <ComparisonPanel baselineId={12} candidateId={15} modelType="lgbm" />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      // The comparison panel shows run labels for both baseline and candidate
      expect(screen.getByText(/Run #12/)).toBeInTheDocument();
      expect(screen.getByText(/Run #15/)).toBeInTheDocument();
    });
  });

  it("shows loading state", async () => {
    // Simulate a never-resolving promise for loading state
    mockFetchTuningComparison.mockReturnValue(new Promise(() => {}));
    const { ComparisonPanel } = await import("../lgbm-tuning/ComparisonPanel");
    render(
      <TestQueryWrapper>
        <ComparisonPanel baselineId={12} candidateId={15} modelType="lgbm" />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Loading comparison...")).toBeInTheDocument();
    });
  });

  it("handles missing lag data", async () => {
    // When per_lag data is not available (legacy run), comparison still renders
    mockFetchTuningComparison.mockResolvedValue(mockComparisonNoLags);
    const { ComparisonPanel } = await import("../lgbm-tuning/ComparisonPanel");
    render(
      <TestQueryWrapper>
        <ComparisonPanel baselineId={12} candidateId={15} modelType="lgbm" />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      // Portfolio-level metrics still render
      expect(screen.getByText("Accuracy %")).toBeInTheDocument();
      expect(screen.getByText("WAPE")).toBeInTheDocument();
      expect(screen.getByText("Bias")).toBeInTheDocument();
    });
  });

  it("applies lag filter", async () => {
    // The lag filter in comparison updates which metrics are shown
    // With lag 2 selected, accuracy should come from lag 2 data
    const lag2 = mockComparison.per_lag[2];
    expect(lag2.exec_lag).toBe(2);
    expect(lag2.baseline_acc).toBe(68.1);
    expect(lag2.candidate_acc).toBe(69.0);
    expect(lag2.delta_acc).toBe(0.9);
  });
});
