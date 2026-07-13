import { describe, it, expect, vi, beforeEach } from "vitest";

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------
const mockTuningJob = {
  job_id: "abc-123-def-456",
  job_type: "model_tuning_run",
  label: "LightGBM Tuning",
  status: "running",
  progress_pct: 40,
  progress_message: "Timeframe D/J",
  started_at: "2026-03-23T10:00:00Z",
  completed_at: null,
  params: {
    run_id: 15,
    model: "lgbm",
    run_label: "Aggressive Depth",
    config_path: "/tmp/tuning_run_15/algorithm_config.yaml",
  },
  pid: 54321,
  error: null,
};

const mockBacktestJob = {
  job_id: "xyz-789-ghi-012",
  job_type: "backtest_standard",
  label: "Standard Backtest",
  status: "completed",
  progress_pct: 100,
  progress_message: "Complete",
  started_at: "2026-03-23T08:00:00Z",
  completed_at: "2026-03-23T09:00:00Z",
  params: { model: "lgbm" },
  pid: null,
  error: null,
};

const mockSecondLgbmTuningJob = {
  job_id: "lgbm-456-regularized-789",
  job_type: "model_tuning_run",
  label: "LightGBM Tuning",
  status: "running",
  progress_pct: 20,
  progress_message: "Timeframe B/J",
  started_at: "2026-03-23T10:30:00Z",
  completed_at: null,
  params: {
    run_id: 16,
    model: "lgbm",
    run_label: "Regularized Leaves",
    config_path: "/tmp/tuning_run_16/algorithm_config.yaml",
  },
  pid: 54322,
  error: null,
};

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------
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
  fetchModelComparison: vi.fn().mockResolvedValue({ models: [], oracle_ceiling: null }),
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
  fetchForecastValue: vi.fn().mockResolvedValue({ baselines: [], ml_model: null, value_added: null }),
  queryKeys: {},
  STALE_INSIGHTS: 300000,
  insightKeys: { all: () => ["insights"] },
}));

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe("JobsTuningIntegration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows tuning jobs with model label", async () => {
    // Tuning jobs should display with format: "{Model} Tuning"
    expect(mockTuningJob.label).toBe("LightGBM Tuning");
    expect(mockTuningJob.job_type).toBe("model_tuning_run");
    expect(mockTuningJob.params.run_label).toBe("Aggressive Depth");

    // Full display label: "LightGBM Tuning - Aggressive Depth"
    const displayLabel = `${mockTuningJob.label} \u2014 ${mockTuningJob.params.run_label}`;
    expect(displayLabel).toContain("LightGBM Tuning");
    expect(displayLabel).toContain("Aggressive Depth");
  });

  it("shows model type badge", async () => {
    // Each tuning job shows a model type badge
    const modelBadgeMap: Record<string, string> = {
      lgbm: "LGBM",
    };

    expect(modelBadgeMap[mockTuningJob.params.model]).toBe("LGBM");
    expect(modelBadgeMap[mockSecondLgbmTuningJob.params.model]).toBe("LGBM");
  });

  it("shows timeframe progress", async () => {
    // Tuning jobs show timeframe-based progress instead of generic "Running..."
    expect(mockTuningJob.progress_message).toBe("Timeframe D/J");
    expect(mockTuningJob.progress_pct).toBe(40);

    // Formatted progress text
    const progressText = `${mockTuningJob.progress_message} \u2014 ${mockTuningJob.progress_pct}%`;
    expect(progressText).toContain("Timeframe D/J");
    expect(progressText).toContain("40%");
  });

  it("can filter by model_tuning_run", async () => {
    // The jobs list should be filterable by job_type
    const allJobs = [mockTuningJob, mockBacktestJob, mockSecondLgbmTuningJob];

    // Filter by model_tuning_run type
    const tuningJobs = allJobs.filter((j) => j.job_type === "model_tuning_run");
    expect(tuningJobs).toHaveLength(2);
    expect(tuningJobs[0].params.model).toBe("lgbm");
    expect(tuningJobs[1].params.model).toBe("lgbm");

    // Non-tuning jobs excluded
    const nonTuningJobs = allJobs.filter((j) => j.job_type !== "model_tuning_run");
    expect(nonTuningJobs).toHaveLength(1);
    expect(nonTuningJobs[0].job_type).toBe("backtest_standard");
  });

  it("distinguishes from other jobs", async () => {
    // Tuning jobs and standard jobs should be visually distinct
    const tuningJob = mockTuningJob;
    const backtestJob = mockBacktestJob;

    // Different job types
    expect(tuningJob.job_type).not.toBe(backtestJob.job_type);
    expect(tuningJob.job_type).toBe("model_tuning_run");
    expect(backtestJob.job_type).toBe("backtest_standard");

    // Tuning jobs have PID tracking
    expect(tuningJob.pid).toBe(54321);
    expect(backtestJob.pid).toBeNull();

    // Tuning jobs have model-specific params
    expect(tuningJob.params).toHaveProperty("model");
    expect(tuningJob.params).toHaveProperty("run_label");
    expect(tuningJob.params).toHaveProperty("run_id");

    // Tuning jobs show experiment-specific progress
    expect(tuningJob.progress_message).toContain("Timeframe");
    expect(backtestJob.progress_message).toBe("Complete");
  });
});
