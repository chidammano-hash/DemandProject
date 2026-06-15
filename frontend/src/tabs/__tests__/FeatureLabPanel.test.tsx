import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

vi.mock("@/api/queries", () => ({
  featureLabKeys: {
    all: ["feature-lab"],
    importance: (modelId?: string) => ["feature-lab", "importance", modelId ?? "default"],
    stability: () => ["feature-lab", "stability"],
    correlation: (topN?: number) => ["feature-lab", "correlation", topN ?? 20],
    clusterImportance: (cluster: number) => ["feature-lab", "cluster-importance", cluster],
    categories: () => ["feature-lab", "categories"],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchFeatureImportance: vi.fn().mockResolvedValue({
    model_id: "lgbm_cluster",
    features: [
      { feature: "lag_1_demand", shap_value: 0.0821, category: "lag", rank: 1 },
      { feature: "rolling_mean_3m", shap_value: 0.0654, category: "rolling", rank: 2 },
      { feature: "month_sin", shap_value: 0.0412, category: "calendar", rank: 3 },
      { feature: "price_ratio", shap_value: 0.0298, category: "external", rank: 4 },
    ],
  }),
  fetchFeatureStability: vi.fn().mockResolvedValue({
    features: [
      { feature: "lag_1_demand", mean_rank: 1.2, rank_std: 0.4, stability: "stable", n_folds: 5 },
      { feature: "rolling_mean_3m", mean_rank: 2.8, rank_std: 1.1, stability: "moderate", n_folds: 5 },
      { feature: "month_sin", mean_rank: 5.4, rank_std: 3.2, stability: "unstable", n_folds: 5 },
      { feature: "price_ratio", mean_rank: 4.1, rank_std: 0.9, stability: "stable", n_folds: 5 },
    ],
  }),
  fetchFeatureCorrelation: vi.fn().mockResolvedValue({
    top_n: 20,
    features: ["lag_1_demand", "rolling_mean_3m", "month_sin"],
    cells: [
      { feature_a: "lag_1_demand", feature_b: "rolling_mean_3m", correlation: 0.92 },
      { feature_a: "lag_1_demand", feature_b: "month_sin", correlation: 0.35 },
      { feature_a: "rolling_mean_3m", feature_b: "month_sin", correlation: 0.71 },
    ],
  }),
  fetchClusterFeatureImportance: vi.fn().mockResolvedValue({
    cluster: 0,
    features: [{ feature: "lag_1_demand", shap_value: 0.09, rank: 1 }],
  }),
  fetchFeatureCategories: vi.fn().mockResolvedValue({
    categories: [
      { category: "lag", color: "#3b82f6", count: 6 },
      { category: "rolling", color: "#10b981", count: 4 },
      { category: "calendar", color: "#f59e0b", count: 3 },
      { category: "external", color: "#8b5cf6", count: 2 },
    ],
  }),
  // Per-cluster section fetches the cluster list via the cluster-eda profile
  // fetcher (U6.10 — was a raw fetch).
  clusterEdaKeys: { profile: () => ["cluster-eda", "profile"] },
  fetchClusterProfile: vi.fn().mockResolvedValue({
    rows: [
      { cluster: 0, n_dfus: 100, mean_demand: 50, cv: 0.3, zero_pct: 0.1 },
      { cluster: 1, n_dfus: 80, mean_demand: 20, cv: 0.6, zero_pct: 0.4 },
    ],
  }),
  // Barrel stubs
  queryKeys: {},
  STALE_INSIGHTS: 300000,
  insightKeys: { all: () => ["insights"] },
}));

import {
  fetchFeatureImportance,
  fetchFeatureStability,
  fetchFeatureCategories,
} from "@/api/queries";
import { FeatureLabPanel } from "@/tabs/lgbm-tuning/FeatureLabPanel";

describe("FeatureLabPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders feature importance chart by default", async () => {
    render(
      <TestQueryWrapper>
        <FeatureLabPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      // "Feature Importance" appears in both sub-tab button and CardTitle
      expect(screen.getAllByText("Feature Importance").length).toBeGreaterThanOrEqual(1);
    });
    // Horizontal bar chart should render
    expect(screen.getByTestId("bar-chart")).toBeInTheDocument();
  });

  it("renders feature categories legend", async () => {
    render(
      <TestQueryWrapper>
        <FeatureLabPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("lag")).toBeInTheDocument();
    });
    expect(screen.getByText("(6)")).toBeInTheDocument();
    expect(screen.getByText("rolling")).toBeInTheDocument();
    expect(screen.getByText("(4)")).toBeInTheDocument();
    expect(screen.getByText("calendar")).toBeInTheDocument();
    expect(screen.getByText("external")).toBeInTheDocument();
  });

  it("shows all sub-tab navigation buttons", async () => {
    render(
      <TestQueryWrapper>
        <FeatureLabPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getAllByText("Feature Importance").length).toBeGreaterThanOrEqual(1);
    });
    expect(screen.getAllByText("Feature Stability").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Correlation Heatmap").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Per-Cluster").length).toBeGreaterThanOrEqual(1);
  });

  it("switches to stability tab with stability badges", async () => {
    render(
      <TestQueryWrapper>
        <FeatureLabPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getAllByText("Feature Stability").length).toBeGreaterThanOrEqual(1);
    });
    fireEvent.click(screen.getAllByText("Feature Stability")[0]);
    await waitFor(() => {
      const stableBadges = screen.getAllByText("stable");
      expect(stableBadges.length).toBeGreaterThanOrEqual(2);
    });
    expect(screen.getAllByText("moderate").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("unstable").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("lag_1_demand")).toBeInTheDocument();
    expect(screen.getByText("rolling_mean_3m")).toBeInTheDocument();
  });

  it("switches to correlation heatmap tab", async () => {
    render(
      <TestQueryWrapper>
        <FeatureLabPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getAllByText("Correlation Heatmap").length).toBeGreaterThanOrEqual(1);
    });
    fireEvent.click(screen.getAllByText("Correlation Heatmap")[0]);
    await waitFor(() => {
      expect(screen.getByText(/Red cells indicate/)).toBeInTheDocument();
    });
  });

  it("renders empty state when no SHAP data available", async () => {
    (fetchFeatureImportance as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      model_id: "lgbm_cluster",
      features: [],
    });
    (fetchFeatureStability as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      features: [],
    });

    render(
      <TestQueryWrapper>
        <FeatureLabPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText(/No feature data available/)).toBeInTheDocument();
    });
  });

  it("calls all fetch functions on mount", async () => {
    render(
      <TestQueryWrapper>
        <FeatureLabPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(fetchFeatureImportance).toHaveBeenCalled();
      expect(fetchFeatureStability).toHaveBeenCalled();
      expect(fetchFeatureCategories).toHaveBeenCalled();
    });
  });
});
