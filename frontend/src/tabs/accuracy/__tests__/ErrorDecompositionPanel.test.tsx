import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";
import { ErrorDecompositionPanel } from "../ErrorDecompositionPanel";

// STALE is the only symbol the panel pulls from the queries barrel.
vi.mock("@/api/queries", () => ({
  STALE: { TWO_MIN: 120000 },
}));

const fetchAccuracyDecomposition = vi.fn();
const fetchErrorContributors = vi.fn();

vi.mock("@/api/queries/accuracy", () => ({
  fetchAccuracyDecomposition: (...args: unknown[]) => fetchAccuracyDecomposition(...args),
  fetchErrorContributors: (...args: unknown[]) => fetchErrorContributors(...args),
  accuracyDecompositionKeys: { list: (p: unknown) => ["accuracy-decomposition", p] },
  errorContributorsKeys: { list: (p: unknown) => ["error-contributors", p] },
}));

const DECOMP_PAYLOAD = {
  group_by: "seasonality_profile",
  lag_filter: -1,
  models: ["champion"],
  source: "agg_accuracy_by_dfu",
  rows: [
    {
      bucket: "seasonal",
      by_model: {
        champion: {
          volume_weighted: {
            accuracy_pct: 65, wape: 35, bias: -0.1,
            sum_forecast: 210, sum_actual: 200, sku_count: 2,
          },
          unweighted: { n_dfus: 2, n_undefined: 1, mean_accuracy_pct: 70, median_accuracy_pct: 75 },
          error_contribution_pct: 100,
          n_dfus: 2,
        },
      },
    },
  ],
};

const CONTRIB_PAYLOAD = {
  models: ["champion"],
  lag_filter: -1,
  limit: 15,
  total_abs_error: 125,
  total_dfus: 5,
  source: "agg_accuracy_by_dfu",
  contributors: [
    {
      item_id: "SKU-ALPHA", customer_group: "CG", loc: "L1",
      cluster_assignment: "clusterA", region: "R1", abc_vol: "A",
      seasonality_profile: "seasonal", sum_actual: 900, sum_abs_error: 60,
      accuracy_pct: 40, wape: 60, bias: -0.2, bias_direction: "under",
      error_contribution_pct: 48, cumulative_contribution_pct: 48,
    },
  ],
};

describe("ErrorDecompositionPanel", () => {
  it("renders both weightings, error share, and the top error contributor", async () => {
    fetchAccuracyDecomposition.mockResolvedValue(DECOMP_PAYLOAD);
    fetchErrorContributors.mockResolvedValue(CONTRIB_PAYLOAD);

    render(
      <ErrorDecompositionPanel models="champion" lag={-1} monthFrom="" enabled />,
      { wrapper: TestQueryWrapper },
    );

    // Decomposition row: bucket titleCased + the three accuracy figures.
    await waitFor(() => expect(screen.getByText("Seasonal")).toBeInTheDocument());
    expect(screen.getByText("65%")).toBeInTheDocument();   // volume-weighted
    expect(screen.getByText("70%")).toBeInTheDocument();   // per-DFU mean
    expect(screen.getByText("75%")).toBeInTheDocument();   // per-DFU median

    // Pareto: the top contributor's item id and its under-forecast bias badge.
    await waitFor(() => expect(screen.getByText("SKU-ALPHA")).toBeInTheDocument());
    expect(screen.getByText("under")).toBeInTheDocument();
    // Contribution share appears in both tables; at least one 48% present.
    expect(screen.getAllByText("48%").length).toBeGreaterThanOrEqual(1);
  });

  it("does not fetch when disabled", () => {
    fetchAccuracyDecomposition.mockClear();
    fetchErrorContributors.mockClear();
    render(
      <ErrorDecompositionPanel models="champion" lag={-1} monthFrom="" enabled={false} />,
      { wrapper: TestQueryWrapper },
    );
    expect(fetchAccuracyDecomposition).not.toHaveBeenCalled();
    expect(fetchErrorContributors).not.toHaveBeenCalled();
  });
});
