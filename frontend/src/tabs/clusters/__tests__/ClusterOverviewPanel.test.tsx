import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";

// F4.1 — the Clusters Overview defaults to the ML source, which is genuinely
// empty (dim_sku.ml_cluster is 100% NULL), while 310K SKUs ARE assigned via the
// Source (sku.txt) path. The old copy said only "No cluster assignments yet",
// implying the platform has no clustering — false and misleading. When the ML
// payload is empty but the source payload is non-empty, the panel must name the
// populated alternative so the planner knows clustering data exists.

const fetchSkuClusters = vi.fn();
const fetchClusterProfiles = vi.fn();
const submitJob = vi.fn();

vi.mock("@/api/queries", () => ({
  queryKeys: {
    skuClusters: (source: string) => ["sku-clusters", source],
    clusterProfiles: () => ["cluster-profiles"],
  },
  STALE: { FIVE_MIN: 0, TEN_MIN: 0 },
  fetchSkuClusters: (source: string) => fetchSkuClusters(source),
  fetchClusterProfiles: () => fetchClusterProfiles(),
  submitJob: (...a: unknown[]) => submitJob(...a),
}));

import ClusterOverviewPanel from "@/tabs/clusters/ClusterOverviewPanel";

describe("ClusterOverviewPanel empty-ML / populated-source hint (F4.1)", () => {
  beforeEach(() => {
    fetchSkuClusters.mockReset();
    fetchClusterProfiles.mockReset();
    fetchClusterProfiles.mockResolvedValue({ metadata: null });
  });

  it("names the populated Source alternative when ML is empty but source has assignments", async () => {
    fetchSkuClusters.mockImplementation((source: string) =>
      source === "ml"
        ? Promise.resolve({ total_assigned: 0, clusters: [] })
        : Promise.resolve({
            total_assigned: 310558,
            clusters: [{ label: "L2_3", count: 109555, pct_of_total: 35, avg_demand: 100, cv_demand: 1 }],
          }),
    );

    render(<ClusterOverviewPanel onDomainChange={() => {}} />, { wrapper: TestQueryWrapper });

    await waitFor(() => {
      // The empty-state hint must name the populated alternative: a single
      // paragraph that contains the 310,558 count AND the "Source (sku.txt)"
      // label — not the bare "No cluster assignments yet" message alone.
      const hint = screen.getByText(
        (_content, el) =>
          el?.tagName === "P" &&
          /310,558/.test(el.textContent ?? "") &&
          /Source \(sku\.txt\)/i.test(el.textContent ?? ""),
      );
      expect(hint).toBeInTheDocument();
    });
  });

  it("shows the plain bare empty message when BOTH ML and source are empty", async () => {
    fetchSkuClusters.mockResolvedValue({ total_assigned: 0, clusters: [] });

    render(<ClusterOverviewPanel onDomainChange={() => {}} />, { wrapper: TestQueryWrapper });

    await waitFor(() => {
      expect(screen.getByText(/Run the clustering pipeline/i)).toBeInTheDocument();
    });
    expect(screen.queryByText(/310,558/)).not.toBeInTheDocument();
  });
});
