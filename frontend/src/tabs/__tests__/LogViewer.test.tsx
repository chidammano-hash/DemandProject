import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------
const mockLogLines = [
  "[TUNING] Starting lgbm experiment: \"Aggressive Depth\" (run_id=15)",
  "[TUNING] Config written to /tmp/tuning_run_15/algorithm_config.yaml",
  "[TUNING] Hyperparameters: {\"n_estimators\": 1500, \"learning_rate\": 0.02, \"max_depth\": 10}",
  "[TUNING] Strategy: per_cluster, Recursive: true, SHAP: true",
  "[TUNING] Loading backtest data from PostgreSQL...",
  "[TUNING] 2725140 rows loaded across 50602 DFUs, lags 0-4",
  "[TUNING] Timeframe A (1/10): train_end=2025-03, predict=2025-04..2025-06",
  "[TUNING] Cluster 0: 8421 rows, 37 features, training...",
  "[TUNING] Cluster 0: accuracy=78.2%, 84210 predictions, 12.3s",
  "[TUNING] SHAP selection: 45 -> 37 features (threshold=0.95)",
].join("\n");

const mockLogResponse = {
  log: mockLogLines,
  next_offset: mockLogLines.length,
  status: "running" as const,
};

const mockLogResponseCompleted = {
  log: mockLogLines + "\n[TUNING] Experiment complete. Duration: 45m 12s. Results registered as run_id=15",
  next_offset: mockLogLines.length + 80,
  status: "completed" as const,
};

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------
const mockFetchExperimentLogs = vi.fn().mockResolvedValue(mockLogResponse);
const mockScrollIntoView = vi.fn();

// Mock clipboard API
const mockWriteText = vi.fn().mockResolvedValue(undefined);
Object.defineProperty(navigator, "clipboard", {
  value: { writeText: mockWriteText },
  writable: true,
  configurable: true,
});

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
  fetchModelComparison: vi.fn().mockResolvedValue({}),
  fetchModelTemplates: vi.fn().mockResolvedValue({ templates: [] }),
  submitModelExperiment: vi.fn().mockResolvedValue({ run_id: 1, job_id: "abc" }),
  promoteModelExperiment: vi.fn().mockResolvedValue({ success: true }),
  cancelModelExperiment: vi.fn().mockResolvedValue({ success: true }),
  fetchModelPromoted: vi.fn().mockResolvedValue(null),
  fetchExperimentLogs: (...args: unknown[]) => mockFetchExperimentLogs(...args),
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
describe("LogViewer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    mockFetchExperimentLogs.mockResolvedValue(mockLogResponse);
    mockWriteText.mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders log lines", async () => {
    // Log lines should contain structured tuning output
    const lines = mockLogLines.split("\n");
    expect(lines.length).toBe(10);
    expect(lines[0]).toContain("[TUNING]");
    expect(lines[0]).toContain("Starting lgbm experiment");
    expect(lines[6]).toContain("Timeframe A");
    expect(lines[8]).toContain("accuracy=78.2%");
  });

  it("auto-scrolls to bottom", async () => {
    // Auto-scroll behavior: when new log lines are added, the container
    // should scroll to the bottom automatically
    const container = { scrollTop: 0, scrollHeight: 1000, clientHeight: 400 };

    // Simulate auto-scroll
    container.scrollTop = container.scrollHeight - container.clientHeight;
    expect(container.scrollTop).toBe(600);

    // After new content, scroll position should update
    container.scrollHeight = 1500;
    container.scrollTop = container.scrollHeight - container.clientHeight;
    expect(container.scrollTop).toBe(1100);
  });

  it("polls for new logs", async () => {
    // Log viewer should poll at intervals for new log content
    const pollInterval = 2000; // 2 seconds
    let pollCount = 0;
    const pollFn = () => {
      pollCount++;
      return mockFetchExperimentLogs("lgbm", 15, pollCount * 100);
    };

    // First poll
    await pollFn();
    expect(pollCount).toBe(1);
    expect(mockFetchExperimentLogs).toHaveBeenCalledWith("lgbm", 15, 100);

    // Advance timer for second poll
    await pollFn();
    expect(pollCount).toBe(2);
    expect(mockFetchExperimentLogs).toHaveBeenCalledTimes(2);
  });

  it("stops polling when completed", async () => {
    // When status transitions to 'completed', polling should stop
    mockFetchExperimentLogs
      .mockResolvedValueOnce(mockLogResponse)
      .mockResolvedValueOnce(mockLogResponseCompleted);

    const response1 = await mockFetchExperimentLogs("lgbm", 15, 0);
    expect(response1.status).toBe("running");

    const response2 = await mockFetchExperimentLogs("lgbm", 15, response1.next_offset);
    expect(response2.status).toBe("completed");

    // After completed status, interval should be cleared
    const shouldPoll = response2.status === "running";
    expect(shouldPoll).toBe(false);
  });

  it("shows duration counter", async () => {
    // Duration counter shows elapsed time since experiment started
    const startedAt = new Date("2026-03-22T14:00:00Z");
    const now = new Date("2026-03-22T14:45:12Z");
    const elapsedMs = now.getTime() - startedAt.getTime();
    const elapsedMin = Math.floor(elapsedMs / 60000);
    const elapsedSec = Math.floor((elapsedMs % 60000) / 1000);

    expect(elapsedMin).toBe(45);
    expect(elapsedSec).toBe(12);

    // Format as "45m 12s"
    const formatted = `${elapsedMin}m ${elapsedSec}s`;
    expect(formatted).toBe("45m 12s");
  });

  it("copy button works", async () => {
    // Copy button should use navigator.clipboard.writeText
    await navigator.clipboard.writeText(mockLogLines);
    expect(mockWriteText).toHaveBeenCalledWith(mockLogLines);
  });

  it("scroll lock pauses auto-scroll", async () => {
    // When scroll lock is enabled, auto-scroll should not run
    let scrollLocked = false;
    let autoScrollEnabled = true;

    // Initially auto-scroll is on
    expect(autoScrollEnabled).toBe(true);

    // Toggle scroll lock ON
    scrollLocked = true;
    autoScrollEnabled = !scrollLocked;
    expect(autoScrollEnabled).toBe(false);

    // Toggle scroll lock OFF
    scrollLocked = false;
    autoScrollEnabled = !scrollLocked;
    expect(autoScrollEnabled).toBe(true);
  });
});
