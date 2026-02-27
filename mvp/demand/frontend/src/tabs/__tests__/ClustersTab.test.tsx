import { describe, it, expect, vi } from "vitest";
import { render, waitFor, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";
import { ScenarioNotificationProvider } from "@/context/ScenarioNotificationContext";

vi.mock("@/api/queries", () => ({
  queryKeys: {
    dfuClusters: (s: string) => ["dfu-clusters", s],
    clusterProfiles: () => ["cluster-profiles"],
    clusteringDefaults: () => ["clustering-defaults"],
    clusteringScenario: (id: string) => ["clustering-scenario", id],
    scenarioEstimate: (p: Record<string, unknown>) => ["scenario-estimate", p],
    scenarioStatus: (id: string) => ["scenario-status", id],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchDfuClusters: vi.fn().mockResolvedValue({
    domain: "dfu",
    total_assigned: 10,
    clusters: [
      { cluster_id: "1", label: "high_volume_steady", count: 5, pct_of_total: 50, avg_demand: 1000, cv_demand: 0.3 },
    ],
  }),
  fetchClusterProfiles: vi.fn().mockResolvedValue({
    profiles: [],
    metadata: { optimal_k: 5, silhouette_score: 0.45, inertia: 12345 },
  }),
  fetchClusteringDefaults: vi.fn().mockResolvedValue({
    feature_params: { time_window_months: 24, min_months_history: 1 },
    model_params: { k_range: [3, 12], min_cluster_size_pct: 2.0, use_pca: false, pca_components: null, skip_gap: true, all_features: false },
    label_params: { volume_high: 0.75, volume_low: 0.25, cv_steady: 0.3, cv_volatile: 0.8, seasonality_threshold: 0.5, zero_demand_threshold: 0.2 },
  }),
  fetchScenarioEstimate: vi.fn().mockResolvedValue({
    estimated_seconds: 45,
    dfu_count: 1200,
    k_range: 10,
    skip_gap: true,
  }),
  fetchScenarioStatus: vi.fn().mockResolvedValue({
    scenario_id: "sc_test",
    status: "running",
    elapsed_seconds: 5,
  }),
  runClusteringScenario: vi.fn(),
  promoteScenario: vi.fn(),
}));

const ClustersTab = (await import("@/tabs/ClustersTab")).default;

function renderTab() {
  return render(
    <TestQueryWrapper>
      <ScenarioNotificationProvider>
        <ClustersTab domain="dfu" onDomainChange={vi.fn()} theme="light" />
      </ScenarioNotificationProvider>
    </TestQueryWrapper>
  );
}

describe("ClustersTab", () => {
  it("renders without crashing", async () => {
    renderTab();
    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });

  it("renders cluster summary table", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("high_volume_steady")).toBeDefined();
    });
  });

  it("renders What-If Scenarios section", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("What-If Scenarios")).toBeDefined();
    });
  });
});
