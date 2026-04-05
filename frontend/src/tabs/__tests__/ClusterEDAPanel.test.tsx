import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

vi.mock("@/api/queries", () => ({
  clusterEdaKeys: {
    all: ["cluster-eda"],
    profile: () => ["cluster-eda", "profile"],
    errorConcentration: () => ["cluster-eda", "error-concentration"],
    distribution: (id: number) => ["cluster-eda", "distribution", id],
    residuals: (m: string) => ["cluster-eda", "residuals", m],
    seasonalityHeatmap: () => ["cluster-eda", "seasonality-heatmap"],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchClusterProfile: vi.fn().mockResolvedValue({
    rows: [
      { cluster: 0, n_dfus: 8421, mean_demand: 142, cv: 0.45, zero_pct: 12.0, seasonal_amplitude: 0.35, accuracy_pct: 78.2 },
      { cluster: 1, n_dfus: 3102, mean_demand: 891, cv: 0.22, zero_pct: 2.0, seasonal_amplitude: 0.18, accuracy_pct: 82.1 },
      { cluster: 2, n_dfus: 15203, mean_demand: 8, cv: 1.85, zero_pct: 68.0, seasonal_amplitude: 0.91, accuracy_pct: 41.3 },
    ],
  }),
  fetchErrorConcentration: vi.fn().mockResolvedValue({
    top_10_pct_share: 47.0,
    worst_months: [
      { month: "2025-01", error_share: 15.0 },
      { month: "2025-02", error_share: 12.0 },
    ],
    worst_clusters: [
      { cluster: 2, error_share: 35.0, n_dfus: 15203 },
      { cluster: 0, error_share: 25.0, n_dfus: 8421 },
    ],
  }),
  fetchClusterDistribution: vi.fn().mockResolvedValue({
    cluster: 0,
    bins: [{ bin_start: 0, bin_end: 100, count: 3000 }],
  }),
  fetchSeasonalityHeatmap: vi.fn().mockResolvedValue({
    months: ["Jan", "Feb", "Mar"],
    rows: [
      { cluster: 0, values: [21.5, 18.3, 22.1] },
      { cluster: 1, values: [15.2, 14.8, 16.1] },
    ],
  }),
  // Barrel stubs
  queryKeys: {},
  STALE_INSIGHTS: 300000,
  insightKeys: { all: () => ["insights"] },
}));

import { fetchClusterProfile } from "@/api/queries";
import { ClusterEDAPanel } from "@/tabs/lgbm-tuning/ClusterEDAPanel";

describe("ClusterEDAPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders cluster profile table by default with KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <ClusterEDAPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      // "Cluster Profiles" appears in both sub-tab button and CardTitle
      expect(screen.getAllByText("Cluster Profiles").length).toBeGreaterThanOrEqual(1);
    });
    // KPI cards
    expect(screen.getByText("Clusters")).toBeInTheDocument();
    expect(screen.getByText("Total DFUs")).toBeInTheDocument();
    expect(screen.getByText("Avg Accuracy")).toBeInTheDocument();
    expect(screen.getByText("High-Zero Clusters")).toBeInTheDocument();
    // Cluster rows (rendered as C0, C1, C2)
    expect(screen.getByText("C0")).toBeInTheDocument();
    expect(screen.getByText("C1")).toBeInTheDocument();
    expect(screen.getByText("C2")).toBeInTheDocument();
  });

  it("renders sortable table headers", async () => {
    render(
      <TestQueryWrapper>
        <ClusterEDAPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Mean Demand")).toBeInTheDocument();
    });
    expect(screen.getByText("CV")).toBeInTheDocument();
    expect(screen.getByText("Zero %")).toBeInTheDocument();
    expect(screen.getByText("Seasonal Amp")).toBeInTheDocument();
    expect(screen.getByText("Accuracy")).toBeInTheDocument();
  });

  it("shows sub-tab navigation with all sections", async () => {
    render(
      <TestQueryWrapper>
        <ClusterEDAPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getAllByText("Cluster Profiles").length).toBeGreaterThanOrEqual(1);
    });
    // Sub-tab buttons (may have duplicate text with CardTitle for active tab)
    expect(screen.getAllByText("Error Concentration").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Demand Distribution").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Seasonality Heatmap").length).toBeGreaterThanOrEqual(1);
  });

  it("switches to error concentration section", async () => {
    render(
      <TestQueryWrapper>
        <ClusterEDAPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Error Concentration")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Error Concentration"));
    await waitFor(() => {
      expect(screen.getByText("Top 10% DFU Error Share")).toBeInTheDocument();
    });
    expect(screen.getByText("Worst Months")).toBeInTheDocument();
    expect(screen.getByText("Worst Clusters")).toBeInTheDocument();
  });

  it("switches to demand distribution section with cluster selector", async () => {
    render(
      <TestQueryWrapper>
        <ClusterEDAPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Demand Distribution")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Demand Distribution"));
    await waitFor(() => {
      // Cluster selector pills should appear
      expect(screen.getByText("Cluster:")).toBeInTheDocument();
    });
  });

  it("switches to seasonality heatmap section", async () => {
    render(
      <TestQueryWrapper>
        <ClusterEDAPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Seasonality Heatmap")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Seasonality Heatmap"));
    await waitFor(() => {
      expect(screen.getByText(/Seasonal demand amplitude/)).toBeInTheDocument();
    });
  });

  it("renders empty state when no cluster data", async () => {
    (fetchClusterProfile as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ rows: [] });

    render(
      <TestQueryWrapper>
        <ClusterEDAPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText(/No cluster data available/)).toBeInTheDocument();
    });
  });

  it("renders accuracy badges with color coding", async () => {
    render(
      <TestQueryWrapper>
        <ClusterEDAPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      // Cluster 1 has 82.1% accuracy (green badge)
      expect(screen.getByText("82.1%")).toBeInTheDocument();
      // Cluster 2 has 41.3% accuracy (red badge)
      expect(screen.getByText("41.3%")).toBeInTheDocument();
    });
  });
});
