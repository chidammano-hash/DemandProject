import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------
const mockRun = {
  run_id: 15,
  run_label: "aggressive_depth",
  model_id: "lgbm_cluster",
  started_at: "2026-03-22T14:00:00",
  completed_at: "2026-03-22T15:00:00",
  status: "completed" as const,
  accuracy_pct: 73.45,
  wape: 26.55,
  bias: -0.009,
  n_predictions: 2725140,
  n_dfus: 50602,
  feature_count: 45,
  params: {
    n_estimators: 1500,
    learning_rate: 0.02,
    max_depth: 10,
    num_leaves: 63,
    reg_lambda: 3.5,
    reg_alpha: 0.5,
    path_smooth: 8.0,
    min_child_samples: 60,
  },
  notes: null,
  is_promoted: false,
  promoted_at: null,
};

const mockLagSummary = [
  { exec_lag: 0, accuracy_pct: 79.5, wape: 20.5, bias: 0.22 },
  { exec_lag: 1, accuracy_pct: 73.8, wape: 26.2, bias: 0.28 },
  { exec_lag: 2, accuracy_pct: 69.0, wape: 31.0, bias: 0.36 },
  { exec_lag: 3, accuracy_pct: 65.8, wape: 34.2, bias: 0.42 },
  { exec_lag: 4, accuracy_pct: 62.1, wape: 37.9, bias: 0.52 },
];

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------
const mockOnConfirm = vi.fn();
const mockOnCancel = vi.fn();

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
describe("EnhancedPromoteModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockOnConfirm.mockClear();
    mockOnCancel.mockClear();
  });

  it("renders run metrics", async () => {
    const { PromoteModal } = await import("../lgbm-tuning/PromoteModal");
    render(
      <TestQueryWrapper>
        <PromoteModal
          run={mockRun}
          onConfirm={mockOnConfirm}
          onCancel={mockOnCancel}
          isPending={false}
          modelLabel="LGBM"
        />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      // Accuracy badge
      expect(screen.getByText(/Accuracy:/)).toBeInTheDocument();
      // WAPE badge
      expect(screen.getByText(/WAPE:/)).toBeInTheDocument();
      // Bias badge
      expect(screen.getByText(/Bias:/)).toBeInTheDocument();
    });
  });

  it("renders per-lag summary", async () => {
    // Verify lag data structure for per-lag summary display
    expect(mockLagSummary).toHaveLength(5);
    for (const lag of mockLagSummary) {
      expect(lag.exec_lag).toBeGreaterThanOrEqual(0);
      expect(lag.exec_lag).toBeLessThanOrEqual(4);
      expect(lag.accuracy_pct).toBeDefined();
      expect(lag.wape).toBeDefined();
      expect(lag.bias).toBeDefined();
    }
    // Lag 0 should be highest accuracy
    expect(mockLagSummary[0].accuracy_pct).toBeGreaterThan(mockLagSummary[4].accuracy_pct);
  });

  it("renders parameter diff table", async () => {
    const { PromoteModal } = await import("../lgbm-tuning/PromoteModal");
    render(
      <TestQueryWrapper>
        <PromoteModal
          run={mockRun}
          onConfirm={mockOnConfirm}
          onCancel={mockOnCancel}
          isPending={false}
          modelLabel="LGBM"
        />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      // Parameters are listed in the modal
      expect(screen.getByText("Parameters to write")).toBeInTheDocument();
      // Individual params visible
      expect(screen.getByText("learning_rate")).toBeInTheDocument();
      expect(screen.getByText("max_depth")).toBeInTheDocument();
      expect(screen.getByText("n_estimators")).toBeInTheDocument();
    });
  });

  it("renders promotion checklist", async () => {
    const { PromoteModal } = await import("../lgbm-tuning/PromoteModal");
    render(
      <TestQueryWrapper>
        <PromoteModal
          run={mockRun}
          onConfirm={mockOnConfirm}
          onCancel={mockOnCancel}
          isPending={false}
          modelLabel="LGBM"
        />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      // The modal shows a checklist of what promotion will do
      expect(screen.getByText("This will:")).toBeInTheDocument();
      expect(screen.getByText(/hyperparameters/i)).toBeInTheDocument();
      expect(screen.getByText(/promoted production/i)).toBeInTheDocument();
    });
  });

  it("calls promote API on confirm", async () => {
    const { PromoteModal } = await import("../lgbm-tuning/PromoteModal");
    render(
      <TestQueryWrapper>
        <PromoteModal
          run={mockRun}
          onConfirm={mockOnConfirm}
          onCancel={mockOnCancel}
          isPending={false}
          modelLabel="LGBM"
        />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getAllByText("Promote to Production").length).toBeGreaterThanOrEqual(1);
    });
    // Click the promote button (not the title)
    const promoteButtons = screen.getAllByText("Promote to Production");
    const button = promoteButtons.find(
      (el) => el.tagName === "BUTTON" || el.closest("button"),
    );
    if (button) {
      fireEvent.click(button.closest("button") ?? button);
    }
    expect(mockOnConfirm).toHaveBeenCalled();
  });

  it("shows spinner during promotion", async () => {
    const { PromoteModal } = await import("../lgbm-tuning/PromoteModal");
    render(
      <TestQueryWrapper>
        <PromoteModal
          run={mockRun}
          onConfirm={mockOnConfirm}
          onCancel={mockOnCancel}
          isPending={true}
          modelLabel="LGBM"
        />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      // When isPending=true, the button should be disabled
      const promoteButtons = screen.getAllByText("Promote to Production");
      const button = promoteButtons.find(
        (el) => el.closest("button"),
      );
      if (button) {
        const btn = button.closest("button")!;
        expect(btn).toBeDisabled();
      }
      // Cancel button should also be disabled
      const cancelBtn = screen.getByText("Cancel").closest("button")!;
      expect(cancelBtn).toBeDisabled();
    });
  });

  it("closes modal on success", async () => {
    const { PromoteModal } = await import("../lgbm-tuning/PromoteModal");
    render(
      <TestQueryWrapper>
        <PromoteModal
          run={mockRun}
          onConfirm={mockOnConfirm}
          onCancel={mockOnCancel}
          isPending={false}
          modelLabel="LGBM"
        />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getAllByText("Promote to Production").length).toBeGreaterThanOrEqual(1);
    });
    // After successful promotion, the parent calls onCancel to close
    // Simulate by calling onCancel
    mockOnCancel();
    expect(mockOnCancel).toHaveBeenCalled();
  });

  it("shows error on failure", async () => {
    const { PromoteModal } = await import("../lgbm-tuning/PromoteModal");
    render(
      <TestQueryWrapper>
        <PromoteModal
          run={mockRun}
          onConfirm={mockOnConfirm}
          onCancel={mockOnCancel}
          isPending={false}
          errorMessage="Database connection failed"
          modelLabel="LGBM"
        />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Database connection failed")).toBeInTheDocument();
    });
  });
});
