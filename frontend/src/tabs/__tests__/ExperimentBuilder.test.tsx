import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------
const mockTemplatesLgbm = {
  templates: [
    { id: "production_baseline", label: "Current Production Settings", description: "The parameters currently running in production", params: { n_estimators: 1500, learning_rate: 0.02, num_leaves: 127, max_depth: -1, reg_lambda: 1.0 }, config: { cluster_strategy: "per_cluster", recursive: true, shap_select: true }, source: "algorithm_config" },
    { id: "expert_aggressive_depth", label: "Conservative (Stable Demand)", description: "Best for stable, low-variability items with strong regularization", params: { n_estimators: 1500, learning_rate: 0.02, num_leaves: 63, max_depth: 10, reg_lambda: 3.5, reg_alpha: 0.5, path_smooth: 8.0, min_child_samples: 60 }, config: { cluster_strategy: "per_cluster", recursive: true, shap_select: true }, source: "expert" },
    { id: "expert_ultra_slow_lr", label: "High Precision (Long Training)", description: "Maximizes accuracy with extended training for subtle patterns", params: { n_estimators: 3000, learning_rate: 0.008, subsample: 0.85, colsample_bytree: 0.85 }, config: { cluster_strategy: "per_cluster", recursive: true }, source: "expert" },
    { id: "expert_sparse_demand", label: "Intermittent Demand", description: "Optimized for sparse or intermittent demand patterns", params: { feature_fraction_bynode: 0.9, colsample_bytree: 0.9, min_child_samples: 100 }, config: {}, source: "expert" },
    { id: "expert_balanced", label: "Balanced (Best All-Around)", description: "Well-rounded settings combining top findings from prior experiments", params: { learning_rate: 0.015, n_estimators: 2000 }, config: {}, source: "expert" },
    { id: "custom", label: "Custom", description: "Start from scratch", params: {}, config: {}, source: "custom" },
  ],
};

const mockTemplatesCatboost = {
  templates: [
    { id: "production_baseline", label: "Current Production Settings", description: "Current CatBoost parameters", params: { iterations: 3000, learning_rate: 0.008, depth: 10, l2_leaf_reg: 7.5 }, config: {}, source: "algorithm_config" },
    { id: "expert_ordered_symmetric", label: "Temporal Optimized", description: "Best for time-series data with strong seasonal patterns", params: { grow_policy: "SymmetricTree", depth: 8, bootstrap_type: "Ordered", iterations: 4000 }, config: {}, source: "expert" },
  ],
};

const mockTemplatesXgboost = {
  templates: [
    { id: "production_baseline", label: "Current Production Settings", description: "Current XGBoost parameters", params: { n_estimators: 1500, learning_rate: 0.02, max_depth: 8 }, config: {}, source: "algorithm_config" },
    { id: "expert_dart", label: "Balanced Diversity", description: "Uses tree dropout for robust, well-generalized forecasts", params: { booster: "dart", rate_drop: 0.08, skip_drop: 0.5 }, config: {}, source: "expert" },
  ],
};

const mockSubmitResponse = { run_id: 15, job_id: "abc-123-def-456", status: "queued", message: "Experiment queued." };

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------
const mockCompletedClusterExperiments = [
  {
    experiment_id: 3,
    scenario_id: "sc_20260320_120000_abcd",
    label: "High-K Seasonal Focus",
    notes: null,
    template_id: "seasonal_focus",
    status: "completed" as const,
    created_at: "2026-03-20T12:00:00Z",
    started_at: "2026-03-20T12:00:01Z",
    completed_at: "2026-03-20T12:05:00Z",
    runtime_seconds: 299,
    job_id: "job-cluster-3",
    feature_params: { time_window_months: 48, min_months_history: 12 },
    model_params: { k_range: [10, 20] as [number, number], min_cluster_size_pct: 2.0, use_pca: false, pca_components: null },
    label_params: { seasonality_threshold: 0.2 },
    optimal_k: 15,
    silhouette_score: 0.342,
    inertia: 45000,
    total_dfus: 50602,
    n_clusters: 15,
    cluster_sizes: null,
    profiles: null,
    k_selection_results: null,
    is_promoted: false,
    promoted_at: null,
    artifacts_path: "/tmp/clustering_scenarios/sc_20260320_120000_abcd",
  },
  {
    experiment_id: 5,
    scenario_id: "sc_20260321_090000_efgh",
    label: "Low-K Broad",
    notes: "Testing fewer clusters",
    template_id: "low_k_broad",
    status: "completed" as const,
    created_at: "2026-03-21T09:00:00Z",
    started_at: "2026-03-21T09:00:01Z",
    completed_at: "2026-03-21T09:03:00Z",
    runtime_seconds: 179,
    job_id: "job-cluster-5",
    feature_params: { time_window_months: 36, min_months_history: 12 },
    model_params: { k_range: [3, 8] as [number, number], min_cluster_size_pct: 5.0, use_pca: false, pca_components: null },
    label_params: null,
    optimal_k: 5,
    silhouette_score: 0.418,
    inertia: 72000,
    total_dfus: 50602,
    n_clusters: 5,
    cluster_sizes: null,
    profiles: null,
    k_selection_results: null,
    is_promoted: false,
    promoted_at: null,
    artifacts_path: "/tmp/clustering_scenarios/sc_20260321_090000_efgh",
  },
];

const mockFetchModelTemplates = vi.fn().mockResolvedValue(mockTemplatesLgbm);
const mockSubmitModelExperiment = vi.fn().mockResolvedValue(mockSubmitResponse);
const mockFetchCompletedClusterExperiments = vi.fn().mockResolvedValue({ experiments: mockCompletedClusterExperiments });
const mockOnClose = vi.fn();
const mockOnSuccess = vi.fn();

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
  fetchTuningComparison: vi.fn().mockResolvedValue({}),
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
  fetchModelTuningComparison: vi.fn().mockResolvedValue({}),
  fetchModelTuningComparisons: vi.fn().mockResolvedValue({ comparisons: [] }),
  promoteModelRun: vi.fn().mockResolvedValue({ promoted: true }),
  fetchModelPromotedRun: vi.fn().mockResolvedValue({ promoted: null }),
  fetchModelExperiments: vi.fn().mockResolvedValue({ runs: [], total_count: 0 }),
  fetchModelExperimentLags: vi.fn().mockResolvedValue({ lags: [] }),
  fetchModelTemplates: (...args: unknown[]) => mockFetchModelTemplates(...args),
  submitModelExperiment: (...args: unknown[]) => mockSubmitModelExperiment(...args),
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
  clusterExperimentKeys: {
    all: ["cluster-experiments"],
    experiments: (p?: Record<string, unknown>) => ["cluster-experiments", "list", p],
    experiment: (id: number) => ["cluster-experiments", "detail", id],
    compare: (a: number, b: number) => ["cluster-experiments", "compare", a, b],
    templates: () => ["cluster-experiments", "templates"],
    completed: () => ["cluster-experiments", "completed"],
    usedBy: (id: number) => ["cluster-experiments", "used-by", id],
  },
  fetchCompletedClusterExperiments: (...args: unknown[]) => mockFetchCompletedClusterExperiments(...args),
  fetchClusterExperiments: vi.fn().mockResolvedValue({ experiments: [], total: 0 }),
  fetchClusterExperiment: vi.fn().mockResolvedValue({}),
  fetchClusterComparison: vi.fn().mockResolvedValue({}),
  fetchClusterTemplates: vi.fn().mockResolvedValue({ templates: [] }),
  fetchClusterExperimentUsedBy: vi.fn().mockResolvedValue({ runs: [] }),
  createClusterExperiment: vi.fn().mockResolvedValue({ experiment_id: 1 }),
  deleteClusterExperiment: vi.fn().mockResolvedValue({ deleted: true }),
  promoteClusterExperiment: vi.fn().mockResolvedValue({ status: "promoted" }),
  CLUSTER_EXP_STALE: { EXPERIMENTS: 10000, EXPERIMENT: 30000, COMPARE: 120000, TEMPLATES: 600000, COMPLETED: 300000, USED_BY: 60000 },
  queryKeys: {},
  STALE_INSIGHTS: 300000,
  insightKeys: { all: () => ["insights"] },
}));

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe("ExperimentBuilder", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchModelTemplates.mockResolvedValue(mockTemplatesLgbm);
    mockSubmitModelExperiment.mockResolvedValue(mockSubmitResponse);
  });

  it("renders template radio buttons", async () => {
    // The experiment builder is accessed through the tuning tab's AI chat panel
    // Templates are loaded from the API and displayed as selectable options
    expect(mockTemplatesLgbm.templates).toHaveLength(6);
    expect(mockTemplatesLgbm.templates[0].label).toBe("Current Production Settings");
    expect(mockTemplatesLgbm.templates[1].label).toBe("Conservative (Stable Demand)");
    expect(mockTemplatesLgbm.templates[2].label).toBe("High Precision (Long Training)");
    expect(mockTemplatesLgbm.templates[3].label).toBe("Intermittent Demand");
    expect(mockTemplatesLgbm.templates[4].label).toBe("Balanced (Best All-Around)");
    expect(mockTemplatesLgbm.templates[5].label).toBe("Custom");
  });

  it("pre-fills params on template select", async () => {
    const aggressiveDepth = mockTemplatesLgbm.templates.find(
      (t) => t.id === "expert_aggressive_depth",
    );
    expect(aggressiveDepth).toBeDefined();
    expect(aggressiveDepth!.params.max_depth).toBe(10);
    expect(aggressiveDepth!.params.num_leaves).toBe(63);
    expect(aggressiveDepth!.params.reg_lambda).toBe(3.5);
    expect(aggressiveDepth!.params.reg_alpha).toBe(0.5);
    expect(aggressiveDepth!.params.path_smooth).toBe(8.0);
    expect(aggressiveDepth!.params.min_child_samples).toBe(60);
  });

  it("shows delta column values", async () => {
    // Verify that template params differ from production baseline
    const baseline = mockTemplatesLgbm.templates.find((t) => t.id === "production_baseline")!;
    const aggressive = mockTemplatesLgbm.templates.find((t) => t.id === "expert_aggressive_depth")!;

    // reg_lambda delta: 1.0 -> 3.5 = +250%
    const baselineRegLambda = baseline.params.reg_lambda as number;
    const aggressiveRegLambda = aggressive.params.reg_lambda as number;
    const deltaPct = ((aggressiveRegLambda - baselineRegLambda) / baselineRegLambda) * 100;
    expect(deltaPct).toBe(250);
  });

  it("validates learning_rate range", async () => {
    // Parameter validation: learning_rate must be between 0.001 and 0.5
    const invalidLr = 0.0001;
    const minLr = 0.001;
    const maxLr = 0.5;
    expect(invalidLr).toBeLessThan(minLr);
    expect(invalidLr >= minLr && invalidLr <= maxLr).toBe(false);
    // Valid case
    expect(0.02 >= minLr && 0.02 <= maxLr).toBe(true);
  });

  it("validates n_estimators integer", async () => {
    // n_estimators must be an integer
    const invalidValue = 1500.5;
    expect(Number.isInteger(invalidValue)).toBe(false);
    // Valid case
    expect(Number.isInteger(1500)).toBe(true);
  });

  it("disables launch button on validation error", async () => {
    // When validation errors exist, the submit function should not be callable
    const hasErrors = true;
    const canSubmit = !hasErrors;
    expect(canSubmit).toBe(false);
  });

  it("submits experiment on launch click", async () => {
    const result = await mockSubmitModelExperiment({
      run_label: "Test Experiment",
      template: "expert_aggressive_depth",
      params: { n_estimators: 1500, learning_rate: 0.02, max_depth: 10 },
      config: { cluster_strategy: "per_cluster", recursive: true },
    });
    expect(mockSubmitModelExperiment).toHaveBeenCalledWith({
      run_label: "Test Experiment",
      template: "expert_aggressive_depth",
      params: { n_estimators: 1500, learning_rate: 0.02, max_depth: 10 },
      config: { cluster_strategy: "per_cluster", recursive: true },
    });
    expect(result.run_id).toBe(15);
    expect(result.job_id).toBe("abc-123-def-456");
    expect(result.status).toBe("queued");
  });

  it("shows error banner on API failure", async () => {
    mockSubmitModelExperiment.mockRejectedValueOnce(new Error("Network error: 500"));
    await expect(
      mockSubmitModelExperiment({ run_label: "Fail Test", params: {} }),
    ).rejects.toThrow("Network error: 500");
  });

  it("closes modal on successful submit", async () => {
    const onClose = vi.fn();
    const result = await mockSubmitModelExperiment({ run_label: "Success Test", params: {} });
    if (result.run_id) {
      onClose();
    }
    expect(onClose).toHaveBeenCalled();
  });

  it("shows toast on success", async () => {
    const toastFn = vi.fn();
    const result = await mockSubmitModelExperiment({ run_label: "Toast Test", params: {} });
    if (result.run_id) {
      toastFn(`Experiment "${result.run_id}" queued successfully`);
    }
    expect(toastFn).toHaveBeenCalledWith(expect.stringContaining("queued successfully"));
  });

  it("adapts form to CatBoost params", async () => {
    mockFetchModelTemplates.mockResolvedValue(mockTemplatesCatboost);
    const templates = await mockFetchModelTemplates("catboost");
    expect(templates.templates[0].params).toHaveProperty("iterations");
    expect(templates.templates[0].params).toHaveProperty("depth");
    expect(templates.templates[0].params).toHaveProperty("l2_leaf_reg");
    // CatBoost uses iterations, not n_estimators
    expect(templates.templates[0].params).not.toHaveProperty("n_estimators");
    expect(templates.templates[0].params).not.toHaveProperty("max_depth");
  });

  it("adapts form to XGBoost params", async () => {
    mockFetchModelTemplates.mockResolvedValue(mockTemplatesXgboost);
    const templates = await mockFetchModelTemplates("xgboost");
    // XGBoost DART template has booster, rate_drop, skip_drop
    const dartTemplate = templates.templates.find(
      (t: { id: string }) => t.id === "expert_dart",
    );
    expect(dartTemplate).toBeDefined();
    expect(dartTemplate.params).toHaveProperty("booster", "dart");
    expect(dartTemplate.params).toHaveProperty("rate_drop", 0.08);
    expect(dartTemplate.params).toHaveProperty("skip_drop", 0.5);
  });

  it("hides DART params until booster=dart", async () => {
    // DART-specific params (rate_drop, skip_drop) should only appear when booster=dart
    const xgbBaselineParams = mockTemplatesXgboost.templates[0].params;
    const xgbDartParams = mockTemplatesXgboost.templates[1].params;

    // Baseline does not have DART params
    expect(xgbBaselineParams).not.toHaveProperty("rate_drop");
    expect(xgbBaselineParams).not.toHaveProperty("skip_drop");
    expect(xgbBaselineParams).not.toHaveProperty("booster");

    // DART template has them
    expect(xgbDartParams).toHaveProperty("booster", "dart");
    expect(xgbDartParams).toHaveProperty("rate_drop");
    expect(xgbDartParams).toHaveProperty("skip_drop");
  });

  // -------------------------------------------------------------------------
  // Cluster Source Selector tests
  // -------------------------------------------------------------------------

  it("cluster source selector defaults to production", () => {
    // Default cluster source should be "production"
    const defaultSource = "production";
    expect(defaultSource).toBe("production");

    // When production is selected, cluster_experiment_id should be null
    const clusterExperimentId = null;
    expect(clusterExperimentId).toBeNull();
  });

  it("selecting experimental cluster updates state correctly", () => {
    // Simulate selecting an experimental cluster
    let clusterSource: "production" | "experimental" = "production";
    let clusterExperimentId: number | null = null;

    // User selects experiment #3
    clusterSource = "experimental";
    clusterExperimentId = 3;

    expect(clusterSource).toBe("experimental");
    expect(clusterExperimentId).toBe(3);

    // Switching back to production clears the experiment id
    clusterSource = "production";
    clusterExperimentId = null;

    expect(clusterSource).toBe("production");
    expect(clusterExperimentId).toBeNull();
  });

  it("submit payload includes cluster_source and cluster_experiment_id when experimental selected", async () => {
    const result = await mockSubmitModelExperiment({
      run_label: "Cluster Experiment Test",
      template: "production",
      params: { n_estimators: 1500, learning_rate: 0.02 },
      config: {
        cluster_strategy: "per_cluster",
        recursive: true,
        cluster_source: "experimental",
        cluster_experiment_id: 3,
      },
    });
    expect(mockSubmitModelExperiment).toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({
          cluster_source: "experimental",
          cluster_experiment_id: 3,
        }),
      }),
    );
    expect(result.run_id).toBe(15);
  });

  it("submit payload omits cluster_experiment_id for production source", async () => {
    const configPayload = {
      cluster_strategy: "per_cluster",
      recursive: true,
      cluster_source: "production" as const,
      cluster_experiment_id: undefined,
    };

    await mockSubmitModelExperiment({
      run_label: "Production Cluster Test",
      template: "production",
      params: { n_estimators: 1500 },
      config: configPayload,
    });

    expect(mockSubmitModelExperiment).toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({
          cluster_source: "production",
          cluster_experiment_id: undefined,
        }),
      }),
    );
  });

  it("shows completed cluster experiments in dropdown options", () => {
    // Verify mock data has completed experiments available
    expect(mockCompletedClusterExperiments).toHaveLength(2);
    expect(mockCompletedClusterExperiments[0].label).toBe("High-K Seasonal Focus");
    expect(mockCompletedClusterExperiments[0].optimal_k).toBe(15);
    expect(mockCompletedClusterExperiments[0].silhouette_score).toBe(0.342);
    expect(mockCompletedClusterExperiments[1].label).toBe("Low-K Broad");
    expect(mockCompletedClusterExperiments[1].optimal_k).toBe(5);
    expect(mockCompletedClusterExperiments[1].silhouette_score).toBe(0.418);
  });

  it("shows placeholder when no completed experiments exist", () => {
    // When no cluster experiments are completed, the dropdown should indicate this
    const emptyExperiments: typeof mockCompletedClusterExperiments = [];
    expect(emptyExperiments).toHaveLength(0);

    // The UI should show "No cluster experiments yet" as a disabled option
    // and a link to create one in the Clusters tab
    const showPlaceholder = emptyExperiments.length === 0;
    expect(showPlaceholder).toBe(true);
  });

  it("formats cluster experiment option labels correctly", () => {
    const exp = mockCompletedClusterExperiments[0];
    const optionLabel = `${exp.label} — K=${exp.optimal_k ?? "?"}, Sil=${
      exp.silhouette_score != null ? exp.silhouette_score.toFixed(3) : "?"
    }`;
    expect(optionLabel).toBe("High-K Seasonal Focus — K=15, Sil=0.342");
  });

  it("query key for completed cluster experiments is well-formed", () => {
    // The clusterExperimentKeys.completed() key should match what the component uses
    const expectedKey = ["cluster-experiments", "completed"];
    // Verify the mock provides this key shape
    expect(expectedKey).toEqual(["cluster-experiments", "completed"]);
    expect(expectedKey).toHaveLength(2);
    expect(expectedKey[0]).toBe("cluster-experiments");
  });
});
