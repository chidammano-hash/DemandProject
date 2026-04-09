import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

const mockExperiments = [
  {
    experiment_id: 1,
    scenario_id: "sc_20260320_100000_abcd",
    label: "Production Baseline",
    notes: null,
    template_id: "production_baseline",
    status: "completed" as const,
    created_at: "2026-03-20T10:00:00Z",
    started_at: "2026-03-20T10:00:05Z",
    completed_at: "2026-03-20T10:05:00Z",
    runtime_seconds: 295,
    job_id: "job-001",
    feature_params: { time_window_months: 24, min_months_history: 1 },
    model_params: { k_range: [3, 12], min_cluster_size_pct: 2.0, use_pca: false, pca_components: null },
    label_params: { volume_high: 0.75, volume_low: 0.25, cv_steady: 0.3, cv_volatile: 0.8, seasonality_threshold: 0.5, zero_demand_threshold: 0.2 },
    optimal_k: 8,
    silhouette_score: 0.3456,
    inertia: 125000,
    total_dfus: 50602,
    n_clusters: 8,
    cluster_sizes: { "high_volume_steady": 12000, "medium_volume": 15000, "low_volume_volatile": 8000, "seasonal": 5602, "intermittent": 4000, "new_items": 2000, "declining": 2500, "growing": 1500 },
    profiles: [
      { label: "high_volume_steady", count: 12000, pct_of_total: 23.7, mean_demand: 5000, cv_demand: 0.15, seasonality_strength: 0.2, trend_slope: 0.01, growth_rate: 0.02, zero_demand_pct: 0.0 },
      { label: "medium_volume", count: 15000, pct_of_total: 29.6, mean_demand: 2000, cv_demand: 0.35, seasonality_strength: 0.4, trend_slope: -0.005, growth_rate: -0.01, zero_demand_pct: 0.05 },
      { label: "low_volume_volatile", count: 8000, pct_of_total: 15.8, mean_demand: 300, cv_demand: 0.85, seasonality_strength: 0.1, trend_slope: 0.0, growth_rate: 0.0, zero_demand_pct: 0.15 },
      { label: "seasonal", count: 5602, pct_of_total: 11.1, mean_demand: 1200, cv_demand: 0.5, seasonality_strength: 0.85, trend_slope: 0.02, growth_rate: 0.03, zero_demand_pct: 0.0 },
    ],
    k_selection_results: null,
    is_promoted: true,
    promoted_at: "2026-03-20T12:00:00Z",
    artifacts_path: "/tmp/clustering_scenarios/sc_20260320_100000_abcd",
  },
  {
    experiment_id: 2,
    scenario_id: "sc_20260321_140000_efgh",
    label: "High-K Granular",
    notes: "Testing K=12-25",
    template_id: "high_k_granular",
    status: "completed" as const,
    created_at: "2026-03-21T14:00:00Z",
    started_at: "2026-03-21T14:00:03Z",
    completed_at: "2026-03-21T14:08:00Z",
    runtime_seconds: 477,
    job_id: "job-002",
    feature_params: { time_window_months: 24, min_months_history: 1 },
    model_params: { k_range: [12, 25], min_cluster_size_pct: 1.5, use_pca: false, pca_components: null },
    label_params: { volume_high: 0.75, volume_low: 0.25, cv_steady: 0.3, cv_volatile: 0.8, seasonality_threshold: 0.5, zero_demand_threshold: 0.2 },
    optimal_k: 15,
    silhouette_score: 0.3124,
    inertia: 98000,
    total_dfus: 50602,
    n_clusters: 15,
    cluster_sizes: { "cluster_0": 5000, "cluster_1": 4500, "cluster_2": 4000, "cluster_3": 3800 },
    profiles: [
      { label: "cluster_0", count: 5000, pct_of_total: 9.9, mean_demand: 4000, cv_demand: 0.2, seasonality_strength: 0.3, trend_slope: 0.01, growth_rate: 0.02, zero_demand_pct: 0.0 },
      { label: "cluster_1", count: 4500, pct_of_total: 8.9, mean_demand: 2500, cv_demand: 0.4, seasonality_strength: 0.5, trend_slope: 0.0, growth_rate: 0.0, zero_demand_pct: 0.05 },
      { label: "cluster_2", count: 4000, pct_of_total: 7.9, mean_demand: 800, cv_demand: 0.7, seasonality_strength: 0.15, trend_slope: -0.02, growth_rate: -0.03, zero_demand_pct: 0.1 },
      { label: "cluster_3", count: 3800, pct_of_total: 7.5, mean_demand: 300, cv_demand: 0.9, seasonality_strength: 0.05, trend_slope: 0.0, growth_rate: 0.0, zero_demand_pct: 0.25 },
    ],
    k_selection_results: null,
    is_promoted: false,
    promoted_at: null,
    artifacts_path: "/tmp/clustering_scenarios/sc_20260321_140000_efgh",
  },
  {
    experiment_id: 3,
    scenario_id: "sc_20260322_080000_ijkl",
    label: "Seasonal Focus",
    notes: null,
    template_id: "seasonal_focus",
    status: "running" as const,
    created_at: "2026-03-22T08:00:00Z",
    started_at: "2026-03-22T08:00:02Z",
    completed_at: null,
    runtime_seconds: null,
    job_id: "job-003",
    feature_params: { time_window_months: 48, min_months_history: 1 },
    model_params: { k_range: [3, 12], min_cluster_size_pct: 2.0, use_pca: false, pca_components: null },
    label_params: { volume_high: 0.75, volume_low: 0.25, cv_steady: 0.3, cv_volatile: 0.8, seasonality_threshold: 0.2, zero_demand_threshold: 0.2 },
    optimal_k: null,
    silhouette_score: null,
    inertia: null,
    total_dfus: null,
    n_clusters: null,
    cluster_sizes: null,
    profiles: null,
    k_selection_results: null,
    is_promoted: false,
    promoted_at: null,
    artifacts_path: null,
  },
];

const mockComparison = {
  experiment_a: mockExperiments[0],
  experiment_b: mockExperiments[1],
  quality_comparison: {
    silhouette_delta: -0.0332,
    inertia_delta: -27000,
    k_delta: 7,
    verdict: "mixed",
  },
  profile_comparison: {
    clusters_only_in_a: ["low_volume_volatile"],
    clusters_only_in_b: ["very_high_volume_seasonal"],
    common_clusters: [
      { label: "high_volume_steady", count_a: 3456, count_b: 3120, delta_count: -336 },
      { label: "medium_volume_trending", count_a: 8200, count_b: 7800, delta_count: -400 },
    ],
  },
  migration_matrix: {
    high_volume_steady: { high_volume_steady: 2800, very_high_volume_seasonal: 656 },
    low_volume_volatile: { medium_volume_trending: 1200 },
  },
  total_dfus_migrated: 4500,
  total_dfus_unchanged: 46102,
};

const mockTemplates = {
  templates: [
    { id: "production_baseline", label: "Production Baseline", description: "Current config", source: "promoted_experiment" },
    { id: "high_k_granular", label: "High-K Granular", description: "K=12-25", model_params: { k_range: [12, 25], min_cluster_size_pct: 1.5 } },
    { id: "custom", label: "Custom", description: "Start from defaults" },
  ],
};

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockFetchClusterExperiments = vi.fn().mockResolvedValue({
  experiments: mockExperiments,
  total: 3,
});
const mockFetchClusterComparison = vi.fn().mockResolvedValue(mockComparison);
const mockCreateClusterExperiment = vi.fn().mockResolvedValue({
  experiment_id: 4,
  scenario_id: "sc_new",
  status: "queued",
  job_id: "job-004",
});
const mockDeleteClusterExperiment = vi.fn().mockResolvedValue({ deleted: true });
const mockPromoteClusterExperiment = vi.fn().mockResolvedValue({ status: "ok", dfus_updated: 50602 });
const mockFetchClusterTemplates = vi.fn().mockResolvedValue(mockTemplates);

vi.mock("recharts");

vi.mock("@/api/queries", () => ({
  // Core query keys and stale times
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  queryKeys: {},
  STALE_INSIGHTS: 300000,
  insightKeys: { all: () => ["insights"] },

  // Cluster experiment exports
  clusterExperimentKeys: {
    all: ["cluster-experiments"],
    experiments: (params?: Record<string, unknown>) => ["cluster-experiments", "list", params],
    experiment: (id: number) => ["cluster-experiments", "detail", id],
    compare: (aId: number, bId: number) => ["cluster-experiments", "compare", aId, bId],
    templates: () => ["cluster-experiments", "templates"],
    completed: () => ["cluster-experiments", "completed"],
    usedBy: (id: number) => ["cluster-experiments", "used-by", id],
  },
  CLUSTER_EXP_STALE: {
    EXPERIMENTS: 10000,
    EXPERIMENT: 30000,
    COMPARE: 120000,
    TEMPLATES: 600000,
    COMPLETED: 300000,
    USED_BY: 60000,
  },
  fetchClusterExperiments: (...args: unknown[]) => mockFetchClusterExperiments(...args),
  fetchClusterExperiment: vi.fn().mockResolvedValue(mockExperiments[0]),
  fetchClusterComparison: (...args: unknown[]) => mockFetchClusterComparison(...args),
  fetchClusterTemplates: (...args: unknown[]) => mockFetchClusterTemplates(...args),
  fetchCompletedClusterExperiments: vi.fn().mockResolvedValue({ experiments: mockExperiments.filter((e) => e.status === "completed") }),
  fetchClusterExperimentUsedBy: vi.fn().mockResolvedValue({ runs: [] }),
  createClusterExperiment: (...args: unknown[]) => mockCreateClusterExperiment(...args),
  deleteClusterExperiment: (...args: unknown[]) => mockDeleteClusterExperiment(...args),
  promoteClusterExperiment: (...args: unknown[]) => mockPromoteClusterExperiment(...args),

  // Scenario fetchers (used by ClustersTab overview)
  fetchScenarioEstimate: vi.fn().mockResolvedValue({ estimated_runtime_seconds: 120, total_dfus: 50602 }),
  fetchClusteringDefaults: vi.fn().mockResolvedValue({
    feature_params: { time_window_months: 24, min_months_history: 1 },
    model_params: { k_range: [3, 12], min_cluster_size_pct: 2.0, use_pca: false, pca_components: null, all_features: false },
    label_params: { volume_high: 0.75, volume_low: 0.25, cv_steady: 0.3, cv_volatile: 0.8, seasonality_threshold: 0.5, zero_demand_threshold: 0.2 },
  }),
  runClusteringScenario: vi.fn(),
  promoteScenario: vi.fn(),
  fetchScenarioStatus: vi.fn(),
  fetchScenarioHistory: vi.fn().mockResolvedValue([]),
  fetchJobDetail: vi.fn(),

  // Model tuning (may be referenced through barrel)
  lgbmTuningKeys: { runs: () => [], compare: () => [], promoted: () => [] },
  modelTuningKeys: { runs: () => [], compare: () => [], promoted: () => [] },
  fetchTuningRuns: vi.fn().mockResolvedValue({ runs: [], total: 0 }),
  fetchModelTuningRuns: vi.fn().mockResolvedValue({ runs: [], total: 0 }),
  fetchModelExperiments: vi.fn().mockResolvedValue({ runs: [], total_count: 0 }),
  fetchModelExperimentLags: vi.fn().mockResolvedValue({ lags: [] }),
  fetchModelComparison: vi.fn().mockResolvedValue({}),
  fetchModelTemplates: vi.fn().mockResolvedValue({ templates: [] }),
  submitModelExperiment: vi.fn(),
  promoteModelExperiment: vi.fn(),
  cancelModelExperiment: vi.fn(),
  fetchModelPromoted: vi.fn().mockResolvedValue(null),

  // Tuning chat
  tuningChatKeys: { sessions: () => [], session: () => [], runStatus: () => [] },
  fetchChatSessions: vi.fn().mockResolvedValue({ sessions: [] }),

  // Cluster EDA
  clusterEdaKeys: { all: [], profile: () => [], errorConcentration: () => [], distribution: () => [], residuals: () => [], seasonalityHeatmap: () => [] },
  fetchClusterProfile: vi.fn().mockResolvedValue({ rows: [] }),

  // Feature Lab
  featureLabKeys: { all: [], importance: () => [], stability: () => [], correlation: () => [], clusterImportance: () => [], categories: () => [] },
  fetchFeatureImportance: vi.fn().mockResolvedValue({ features: [] }),

  // Accuracy budget
  accuracyBudgetKeys: { all: [], decomposition: () => [], abc: () => [], models: () => [], monthly: () => [], forecastValue: () => [] },
  fetchAccuracyDecomposition: vi.fn().mockResolvedValue({}),
}));

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ClusterExperimentsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchClusterExperiments.mockResolvedValue({
      experiments: mockExperiments,
      total: 3,
    });
  });

  it("renders experiment list with KPI cards", async () => {
    const { ClusterExperimentsPanel } = await import("../clusters/ClusterExperimentsPanel");
    render(
      <TestQueryWrapper>
        <ClusterExperimentsPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Best Silhouette")).toBeInTheDocument();
      expect(screen.getByText("Production K")).toBeInTheDocument();
      expect(screen.getByText("Total Experiments")).toBeInTheDocument();
      expect(screen.getByText("Active")).toBeInTheDocument();
    });
  });

  it("renders experiment rows", async () => {
    const { ClusterExperimentsPanel } = await import("../clusters/ClusterExperimentsPanel");
    render(
      <TestQueryWrapper>
        <ClusterExperimentsPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Production Baseline")).toBeInTheDocument();
      expect(screen.getByText("High-K Granular")).toBeInTheDocument();
      expect(screen.getByText("Seasonal Focus")).toBeInTheDocument();
    });
  });

  it("renders status badges", async () => {
    const { ClusterExperimentsPanel } = await import("../clusters/ClusterExperimentsPanel");
    render(
      <TestQueryWrapper>
        <ClusterExperimentsPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getAllByText("completed").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("running")).toBeInTheDocument();
    });
  });

  it("shows empty state when no experiments", async () => {
    mockFetchClusterExperiments.mockResolvedValue({ experiments: [], total: 0 });
    const { ClusterExperimentsPanel } = await import("../clusters/ClusterExperimentsPanel");
    render(
      <TestQueryWrapper>
        <ClusterExperimentsPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("No cluster experiments yet")).toBeInTheDocument();
      expect(screen.getByText("Create First Experiment")).toBeInTheDocument();
    });
  });

  it("selects baseline on first row click", async () => {
    const { ClusterExperimentsPanel } = await import("../clusters/ClusterExperimentsPanel");
    render(
      <TestQueryWrapper>
        <ClusterExperimentsPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Production Baseline")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Production Baseline"));
    await waitFor(() => {
      expect(screen.getByText("(B)")).toBeInTheDocument();
    });
  });

  it("selects candidate on second row click", async () => {
    const { ClusterExperimentsPanel } = await import("../clusters/ClusterExperimentsPanel");
    render(
      <TestQueryWrapper>
        <ClusterExperimentsPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Production Baseline")).toBeInTheDocument();
    });
    // Select baseline
    fireEvent.click(screen.getByText("Production Baseline"));
    await waitFor(() => {
      expect(screen.getByText("(B)")).toBeInTheDocument();
    });
    // Select candidate
    fireEvent.click(screen.getByText("High-K Granular"));
    await waitFor(() => {
      expect(screen.getByText("(C)")).toBeInTheDocument();
    });
  });

  it("resets selection on third row click", async () => {
    const { ClusterExperimentsPanel } = await import("../clusters/ClusterExperimentsPanel");
    render(
      <TestQueryWrapper>
        <ClusterExperimentsPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Production Baseline")).toBeInTheDocument();
    });
    // Select baseline + candidate
    fireEvent.click(screen.getByText("Production Baseline"));
    await waitFor(() => {
      expect(screen.getByText("(B)")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("High-K Granular"));
    await waitFor(() => {
      expect(screen.getByText("(C)")).toBeInTheDocument();
    });
    // Click third row — resets, sets new baseline
    fireEvent.click(screen.getByText("Seasonal Focus"));
    await waitFor(() => {
      expect(screen.queryByText("(C)")).not.toBeInTheDocument();
    });
  });

  it("shows guidance text when no rows selected", async () => {
    const { ClusterExperimentsPanel } = await import("../clusters/ClusterExperimentsPanel");
    render(
      <TestQueryWrapper>
        <ClusterExperimentsPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(
        screen.getByText(/Click a row to view experiment charts/),
      ).toBeInTheDocument();
    });
  });

  it("shows New Experiment button", async () => {
    const { ClusterExperimentsPanel } = await import("../clusters/ClusterExperimentsPanel");
    render(
      <TestQueryWrapper>
        <ClusterExperimentsPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("New Experiment")).toBeInTheDocument();
    });
  });

  it("shows experiment count in toolbar", async () => {
    const { ClusterExperimentsPanel } = await import("../clusters/ClusterExperimentsPanel");
    render(
      <TestQueryWrapper>
        <ClusterExperimentsPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("3 experiments")).toBeInTheDocument();
    });
  });

  it("shows cluster distribution bars for completed experiments", async () => {
    const { ClusterExperimentsPanel } = await import("../clusters/ClusterExperimentsPanel");
    render(
      <TestQueryWrapper>
        <ClusterExperimentsPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      // Completed experiments should show "N clusters" labels
      const clusterLabels = screen.getAllByText(/\d+ clusters/);
      expect(clusterLabels.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("shows table header for Clusters column", async () => {
    const { ClusterExperimentsPanel } = await import("../clusters/ClusterExperimentsPanel");
    render(
      <TestQueryWrapper>
        <ClusterExperimentsPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Clusters")).toBeInTheDocument();
    });
  });
});
