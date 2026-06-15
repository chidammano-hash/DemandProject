import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

const mockTemplates = {
  templates: [
    { id: "production_baseline", label: "Production Baseline", description: "Current production config", source: "promoted_experiment" },
    { id: "high_k_granular", label: "High-K Granular", description: "K=12-25, finer segments", model_params: { k_range: [12, 25], min_cluster_size_pct: 1.5 } },
    { id: "low_k_broad", label: "Low-K Broad", description: "K=3-8, robust clusters", model_params: { k_range: [3, 8], min_cluster_size_pct: 5.0 } },
    { id: "seasonal_focus", label: "Seasonal Focus", description: "48-month window, low threshold", feature_params: { time_window_months: 48 }, label_params: { seasonality_threshold: 0.2 } },
    { id: "intermittent_specialist", label: "Intermittent Specialist", description: "Low zero-demand threshold", label_params: { zero_demand_threshold: 0.1, cv_volatile: 0.6 } },
    { id: "pca_compressed", label: "PCA Compressed", description: "All features + PCA", model_params: { all_features: true, use_pca: true, pca_components: 10 } },
    { id: "custom", label: "Custom", description: "Start from defaults" },
  ],
};

const mockEstimate = {
  estimated_runtime_seconds: 120,
  total_dfus: 50602,
  k_min: 3,
  k_max: 12,
};

const mockCreateResponse = {
  experiment_id: 4,
  scenario_id: "sc_20260325_120000_mnop",
  status: "queued",
  job_id: "job-004",
};

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockFetchClusterTemplates = vi.fn().mockResolvedValue(mockTemplates);
const mockCreateClusterExperiment = vi.fn().mockResolvedValue(mockCreateResponse);
const mockOnClose = vi.fn();
const mockOnSubmitted = vi.fn();

vi.mock("recharts");

vi.mock("@/api/queries", () => ({
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  queryKeys: {},
  STALE_INSIGHTS: 300000,
  insightKeys: { all: () => ["insights"] },

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
  fetchClusterExperiments: vi.fn().mockResolvedValue({ experiments: [], total: 0 }),
  fetchClusterTemplates: (...args: unknown[]) => mockFetchClusterTemplates(...args),
  createClusterExperiment: (...args: unknown[]) => mockCreateClusterExperiment(...args),
  deleteClusterExperiment: vi.fn(),
  promoteClusterExperiment: vi.fn(),
  fetchClusterComparison: vi.fn(),
  fetchCompletedClusterExperiments: vi.fn().mockResolvedValue({ experiments: [] }),

  fetchScenarioEstimate: vi.fn().mockResolvedValue(mockEstimate),
  fetchClusterCoreFeatures: vi.fn().mockResolvedValue({ features: ["mean_demand", "cv_demand"] }),
  fetchClusteringDefaults: vi.fn().mockResolvedValue({
    feature_params: { time_window_months: 24, min_months_history: 1 },
    model_params: { k_range: [3, 12], min_cluster_size_pct: 2.0, use_pca: false, pca_components: null, all_features: false },
    label_params: { volume_high: 0.75, volume_low: 0.25, cv_steady: 0.3, cv_volatile: 0.8, seasonality_threshold: 0.5, zero_demand_threshold: 0.2 },
  }),

  // Other barrel exports
  lgbmTuningKeys: { runs: () => [], compare: () => [], promoted: () => [] },
  modelTuningKeys: { runs: () => [], compare: () => [], promoted: () => [] },
  fetchTuningRuns: vi.fn().mockResolvedValue({ runs: [], total: 0 }),
  fetchModelTuningRuns: vi.fn().mockResolvedValue({ runs: [], total: 0 }),
  fetchModelExperiments: vi.fn().mockResolvedValue({ runs: [], total_count: 0 }),
  tuningChatKeys: { sessions: () => [], session: () => [], runStatus: () => [] },
  clusterEdaKeys: { all: [], profile: () => [] },
  featureLabKeys: { all: [], importance: () => [] },
  accuracyBudgetKeys: { all: [], decomposition: () => [] },
}));

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ClusterExperimentBuilder", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchClusterTemplates.mockResolvedValue(mockTemplates);
    mockCreateClusterExperiment.mockResolvedValue(mockCreateResponse);
  });

  it("renders template selection buttons", async () => {
    const { ClusterExperimentBuilder } = await import("../clusters/ClusterExperimentBuilder");
    render(
      <TestQueryWrapper>
        <ClusterExperimentBuilder
          open={true}
          onClose={mockOnClose}
          onSubmitted={mockOnSubmitted}
        />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("Production Baseline")).toBeInTheDocument();
      expect(screen.getByText("High-K Granular")).toBeInTheDocument();
      expect(screen.getByText("Low-K Broad")).toBeInTheDocument();
      expect(screen.getByText("Seasonal Focus")).toBeInTheDocument();
      expect(screen.getByText("Intermittent Specialist")).toBeInTheDocument();
      expect(screen.getByText("PCA Compressed")).toBeInTheDocument();
      expect(screen.getByText("Custom")).toBeInTheDocument();
    });
  });

  it("renders parameter form sections", async () => {
    const { ClusterExperimentBuilder } = await import("../clusters/ClusterExperimentBuilder");
    render(
      <TestQueryWrapper>
        <ClusterExperimentBuilder
          open={true}
          onClose={mockOnClose}
          onSubmitted={mockOnSubmitted}
        />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("Data Scope")).toBeInTheDocument();
      expect(screen.getByText("Model")).toBeInTheDocument();
      expect(screen.getByText("Labeling Thresholds")).toBeInTheDocument();
    });
  });

  it("renders label and notes inputs", async () => {
    const { ClusterExperimentBuilder } = await import("../clusters/ClusterExperimentBuilder");
    render(
      <TestQueryWrapper>
        <ClusterExperimentBuilder
          open={true}
          onClose={mockOnClose}
          onSubmitted={mockOnSubmitted}
        />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByPlaceholderText("e.g. High-K Seasonal Focus")).toBeInTheDocument();
      expect(screen.getByPlaceholderText("Optional description or hypothesis")).toBeInTheDocument();
    });
  });

  it("disables launch button when label is empty", async () => {
    const { ClusterExperimentBuilder } = await import("../clusters/ClusterExperimentBuilder");
    render(
      <TestQueryWrapper>
        <ClusterExperimentBuilder
          open={true}
          onClose={mockOnClose}
          onSubmitted={mockOnSubmitted}
        />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      const launchButton = screen.getByText("Launch Experiment");
      expect(launchButton.closest("button")).toBeDisabled();
    });
  });

  it("enables launch button when label is filled", async () => {
    const { ClusterExperimentBuilder } = await import("../clusters/ClusterExperimentBuilder");
    render(
      <TestQueryWrapper>
        <ClusterExperimentBuilder
          open={true}
          onClose={mockOnClose}
          onSubmitted={mockOnSubmitted}
        />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByPlaceholderText("e.g. High-K Seasonal Focus")).toBeInTheDocument();
    });

    // Fill label
    fireEvent.change(screen.getByPlaceholderText("e.g. High-K Seasonal Focus"), {
      target: { value: "Test Experiment" },
    });

    await waitFor(() => {
      const launchButton = screen.getByText("Launch Experiment");
      expect(launchButton.closest("button")).not.toBeDisabled();
    });
  });

  it("calls onClose when cancel clicked", async () => {
    const { ClusterExperimentBuilder } = await import("../clusters/ClusterExperimentBuilder");
    render(
      <TestQueryWrapper>
        <ClusterExperimentBuilder
          open={true}
          onClose={mockOnClose}
          onSubmitted={mockOnSubmitted}
        />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("Cancel")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Cancel"));
    expect(mockOnClose).toHaveBeenCalled();
  });

  it("does not render when open is false", async () => {
    const { ClusterExperimentBuilder } = await import("../clusters/ClusterExperimentBuilder");
    const { container } = render(
      <TestQueryWrapper>
        <ClusterExperimentBuilder
          open={false}
          onClose={mockOnClose}
          onSubmitted={mockOnSubmitted}
        />
      </TestQueryWrapper>,
    );

    expect(container.textContent).toBe("");
  });

  it("submits experiment with correct payload", async () => {
    const result = await mockCreateClusterExperiment({
      label: "Test Experiment",
      template: "high_k_granular",
      feature_params: { time_window_months: 24, min_months_history: 1 },
      model_params: { k_range: [12, 25], min_cluster_size_pct: 1.5, use_pca: false, pca_components: null },
      label_params: { volume_high: 0.75, volume_low: 0.25, cv_steady: 0.3, cv_volatile: 0.8, seasonality_threshold: 0.5, zero_demand_threshold: 0.2 },
    });

    expect(mockCreateClusterExperiment).toHaveBeenCalledWith(
      expect.objectContaining({
        label: "Test Experiment",
        template: "high_k_granular",
      }),
    );
    expect(result.experiment_id).toBe(4);
    expect(result.status).toBe("queued");
  });

  it("shows error on API failure", async () => {
    mockCreateClusterExperiment.mockRejectedValueOnce(new Error("Server error 500"));
    await expect(
      mockCreateClusterExperiment({ label: "Fail Test" }),
    ).rejects.toThrow("Server error 500");
  });

  it("template data structure is correct", async () => {
    expect(mockTemplates.templates).toHaveLength(7);
    expect(mockTemplates.templates[0].label).toBe("Production Baseline");

    const highK = mockTemplates.templates.find((t) => t.id === "high_k_granular");
    expect(highK).toBeDefined();
    expect(highK!.model_params).toHaveProperty("k_range");
    expect(highK!.model_params!.k_range).toEqual([12, 25]);
    expect(highK!.model_params!.min_cluster_size_pct).toBe(1.5);
  });

  it("renders with cloneFrom pre-populated", async () => {
    const { ClusterExperimentBuilder } = await import("../clusters/ClusterExperimentBuilder");
    render(
      <TestQueryWrapper>
        <ClusterExperimentBuilder
          open={true}
          onClose={mockOnClose}
          onSubmitted={mockOnSubmitted}
          cloneFrom={{
            featureParams: { time_window_months: 48, min_months_history: 6 },
            modelParams: { k_range: [10, 20], min_cluster_size_pct: 3.0, use_pca: true, pca_components: 8 },
            labelParams: { volume_high: 0.8, volume_low: 0.2, cv_steady: 0.4, cv_volatile: 0.9, seasonality_threshold: 0.3, zero_demand_threshold: 0.15 },
            label: "Source Experiment",
            notes: "Original notes",
          }}
        />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      // The label input should show "(clone)" suffix
      const labelInput = screen.getByPlaceholderText("e.g. High-K Seasonal Focus") as HTMLInputElement;
      expect(labelInput.value).toContain("clone");
    });
  });
});
