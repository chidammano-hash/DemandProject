import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

const completedExperiment = {
  experiment_id: 1,
  scenario_id: "sc_20260320_100000_abcd",
  label: "High-K Test",
  notes: "Testing higher K range",
  template_id: "high_k_granular",
  status: "completed" as const,
  created_at: "2026-03-20T10:00:00Z",
  started_at: "2026-03-20T10:00:05Z",
  completed_at: "2026-03-20T10:05:00Z",
  runtime_seconds: 295,
  job_id: "job-001",
  feature_params: { time_window_months: 24, min_months_history: 1 },
  model_params: {
    k_range: [3, 12] as [number, number],
    min_cluster_size_pct: 2.0,
    use_pca: false,
    pca_components: null,
  },
  label_params: {
    volume_high: 0.75,
    volume_low: 0.25,
    cv_steady: 0.3,
    cv_volatile: 0.8,
    seasonality_threshold: 0.5,
    zero_demand_threshold: 0.2,
  },
  optimal_k: 11,
  silhouette_score: 0.2389,
  inertia: 125000,
  total_dfus: 20573,
  n_clusters: 11,
  cluster_sizes: null,
  profiles: [
    {
      label: "high_volume_steady",
      count: 5000,
      pct_of_total: 24.3,
      mean_demand: 1500,
      cv_demand: 0.25,
      seasonality_strength: 0.12,
      trend_slope: 0.002,
      growth_rate: 0.03,
      zero_demand_pct: 0.0,
    },
    {
      label: "low_volume_volatile",
      count: 3200,
      pct_of_total: 15.6,
      mean_demand: 50,
      cv_demand: 1.8,
      seasonality_strength: 0.05,
      trend_slope: -0.001,
      growth_rate: -0.02,
      zero_demand_pct: 0.35,
    },
  ],
  k_selection_results: {
    k_values: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    inertias: [500000, 400000, 320000, 260000, 210000, 175000, 150000, 135000, 125000, 118000],
    silhouette_scores: [0.15, 0.18, 0.20, 0.22, 0.23, 0.235, 0.237, 0.238, 0.2389, 0.235],
    ch_scores: [1200, 1350, 1500, 1550, 1540, 1520, 1510, 1505, 1502, 1490],
    feasible_mask: [true, true, true, true, true, true, true, true, true, true],
    pca_scatter: {
      pc1_variance: 30.14,
      pc2_variance: 17.0,
      points: [
        { pc1: -2.1, pc2: 0.3, cluster: 0 },
        { pc1: 1.5, pc2: -1.2, cluster: 1 },
        { pc1: 0.8, pc2: 2.5, cluster: 0 },
      ],
    },
  },
  is_promoted: false,
  promoted_at: null,
  artifacts_path: "/tmp/clustering_scenarios/sc_20260320_100000_abcd",
};

const runningExperiment = {
  ...completedExperiment,
  experiment_id: 2,
  status: "running" as const,
  optimal_k: null,
  silhouette_score: null,
  inertia: null,
  total_dfus: null,
  profiles: null,
  k_selection_results: null,
  completed_at: null,
  runtime_seconds: null,
};

const failedExperiment = {
  ...completedExperiment,
  experiment_id: 3,
  status: "failed" as const,
  optimal_k: null,
  silhouette_score: null,
  inertia: null,
  total_dfus: null,
  profiles: null,
  k_selection_results: null,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ClusterExperimentDetail", () => {
  it("renders experiment header with label and metrics", async () => {
    const { ClusterExperimentDetail } = await import(
      "../clusters/ClusterExperimentDetail"
    );
    render(
      <TestQueryWrapper>
        <ClusterExperimentDetail experiment={completedExperiment} />
      </TestQueryWrapper>,
    );
    expect(screen.getByText("High-K Test")).toBeInTheDocument();
    expect(screen.getByText("K=11")).toBeInTheDocument();
    expect(screen.getByText(/0\.2389/)).toBeInTheDocument();
    expect(screen.getByText(/20,573 DFUs/)).toBeInTheDocument();
  });

  it("renders profile table with cluster data", async () => {
    const { ClusterExperimentDetail } = await import(
      "../clusters/ClusterExperimentDetail"
    );
    render(
      <TestQueryWrapper>
        <ClusterExperimentDetail experiment={completedExperiment} />
      </TestQueryWrapper>,
    );
    // formatClusterLabel("high_volume_steady") → "MOVR.CALM"
    expect(screen.getByText("MOVR.CALM")).toBeInTheDocument();
    // formatClusterLabel("low_volume_volatile") → "SLOW.WILD"
    expect(screen.getByText("SLOW.WILD")).toBeInTheDocument();
  });

  it("renders charts for completed experiment", async () => {
    const { ClusterExperimentDetail } = await import(
      "../clusters/ClusterExperimentDetail"
    );
    render(
      <TestQueryWrapper>
        <ClusterExperimentDetail experiment={completedExperiment} />
      </TestQueryWrapper>,
    );
    // K Selection 3-panel
    expect(screen.getByText(/K Selection/)).toBeInTheDocument();
    expect(screen.getByText("Elbow Method")).toBeInTheDocument();
    expect(screen.getByText(/Silhouette Score/)).toBeInTheDocument();
    expect(screen.getByText(/Calinski-Harabasz Score/)).toBeInTheDocument();
    // PCA scatter
    expect(screen.getByText(/Cluster Visualization \(2D PCA\)/)).toBeInTheDocument();
    // Existing charts
    expect(screen.getByText("Cluster Size Distribution")).toBeInTheDocument();
    expect(screen.getByText("Cluster Profile Radar")).toBeInTheDocument();
  });

  it("shows running message for in-progress experiments", async () => {
    const { ClusterExperimentDetail } = await import(
      "../clusters/ClusterExperimentDetail"
    );
    render(
      <TestQueryWrapper>
        <ClusterExperimentDetail experiment={runningExperiment} />
      </TestQueryWrapper>,
    );
    expect(
      screen.getByText(/Experiment is still running/),
    ).toBeInTheDocument();
  });

  it("shows failed message for failed experiments", async () => {
    const { ClusterExperimentDetail } = await import(
      "../clusters/ClusterExperimentDetail"
    );
    render(
      <TestQueryWrapper>
        <ClusterExperimentDetail experiment={failedExperiment} />
      </TestQueryWrapper>,
    );
    expect(
      screen.getByText(/Experiment failed/),
    ).toBeInTheDocument();
  });

  it("shows Promote button when not promoted", async () => {
    const onPromote = vi.fn();
    const { ClusterExperimentDetail } = await import(
      "../clusters/ClusterExperimentDetail"
    );
    render(
      <TestQueryWrapper>
        <ClusterExperimentDetail
          experiment={completedExperiment}
          onPromote={onPromote}
        />
      </TestQueryWrapper>,
    );
    expect(screen.getByText("Promote")).toBeInTheDocument();
  });

  it("hides Promote button when already promoted", async () => {
    const promoted = { ...completedExperiment, is_promoted: true };
    const { ClusterExperimentDetail } = await import(
      "../clusters/ClusterExperimentDetail"
    );
    render(
      <TestQueryWrapper>
        <ClusterExperimentDetail experiment={promoted} onPromote={() => {}} />
      </TestQueryWrapper>,
    );
    expect(screen.queryByText("Promote")).not.toBeInTheDocument();
  });

  it("displays experiment notes when present", async () => {
    const { ClusterExperimentDetail } = await import(
      "../clusters/ClusterExperimentDetail"
    );
    render(
      <TestQueryWrapper>
        <ClusterExperimentDetail experiment={completedExperiment} />
      </TestQueryWrapper>,
    );
    expect(screen.getByText("Testing higher K range")).toBeInTheDocument();
  });
});
