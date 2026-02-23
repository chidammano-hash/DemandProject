import { describe, it, expect, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", () => ({
  queryKeys: {
    dfuClusters: (s: string) => ["dfu-clusters", s],
    clusterProfiles: () => ["cluster-profiles"],
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
}));

const ClustersTab = (await import("@/tabs/ClustersTab")).default;

describe("ClustersTab", () => {
  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <ClustersTab domain="dfu" onDomainChange={vi.fn()} theme="light" />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });
});
