import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// ---------------------------------------------------------------------------
// Use vi.hoisted to ensure mocks are available before vi.mock factory runs
// ---------------------------------------------------------------------------
const { mockFetchDfuShap, mockFetchShapSummary } = vi.hoisted(() => ({
  mockFetchDfuShap: vi.fn(),
  mockFetchShapSummary: vi.fn(),
}));

vi.mock("@/api/queries/core", () => ({
  fetchDfuShap: (...args: unknown[]) => mockFetchDfuShap(...args),
  fetchShapSummary: (...args: unknown[]) => mockFetchShapSummary(...args),
}));

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------
const _SHAP_PAYLOAD = {
  item_no: "100320",
  loc: "1401-BULK",
  model_id: "lgbm_cluster",
  cluster_id: "0",
  top_n: 3,
  computed_at: "2026-01-01T00:00:00Z",
  future_lag_model_id: "lgbm_cluster",
  points: [
    {
      month: "2024-01-01",
      is_future: false,
      base_value: 120.0,
      other_shap: 2.1,
      features: [
        { name: "qty_lag_1", value: 10.5 },
        { name: "rolling_mean_3m", value: -3.2 },
        { name: "month", value: 1.1 },
      ],
    },
    {
      month: "2024-02-01",
      is_future: false,
      base_value: 120.0,
      other_shap: 1.9,
      features: [
        { name: "qty_lag_1", value: 9.0 },
        { name: "rolling_mean_3m", value: -2.8 },
        { name: "month", value: 0.9 },
      ],
    },
    {
      month: "2026-04-01",
      is_future: true,
      base_value: 120.0,
      other_shap: 3.0,
      features: [
        { name: "qty_lag_1", value: 12.0 },
        { name: "rolling_mean_3m", value: -4.0 },
        { name: "month", value: 1.5 },
      ],
    },
  ],
};

const _SHAP_SUMMARY = {
  model_id: "lgbm_cluster",
  total_features: 2,
  features: [
    { feature: "qty_lag_1", mean_abs_shap_across_timeframes: 8.5 },
    { feature: "rolling_mean_3m", mean_abs_shap_across_timeframes: 3.2 },
  ],
};

import { DfuShapPanel } from "@/tabs/dfu-analysis/DfuShapPanel";

describe("DfuShapPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // -------------------------------------------------------------------------
  // 1. Placeholder when selectedModel is null
  // -------------------------------------------------------------------------
  it("shows placeholder when no model is selected", () => {
    render(
      <TestQueryWrapper>
        <DfuShapPanel
          selectedModel={null}
          itemNo="100320"
          loc="1401-BULK"
          dfuMode="item_location"
          visibleMonths={[]}
        />
      </TestQueryWrapper>
    );

    expect(
      screen.getByText(/Click a forecast line above to explore SHAP feature contributions/i)
    ).toBeDefined();
  });

  // -------------------------------------------------------------------------
  // 2. Note shown for non item_location modes
  // -------------------------------------------------------------------------
  it("shows mode restriction note when dfuMode is not item_location", () => {
    render(
      <TestQueryWrapper>
        <DfuShapPanel
          selectedModel="lgbm_cluster"
          itemNo="100320"
          loc="1401-BULK"
          dfuMode="all_items_at_location"
          visibleMonths={[]}
        />
      </TestQueryWrapper>
    );

    expect(
      screen.getByText(/Per-DFU SHAP requires single item \+ location mode/i)
    ).toBeDefined();
  });

  // -------------------------------------------------------------------------
  // 3. Card title visible while loading
  // -------------------------------------------------------------------------
  it("shows card title while fetching (loading state)", async () => {
    // Return a promise that never resolves → stays in loading state
    mockFetchDfuShap.mockReturnValue(new Promise(() => {}));

    render(
      <TestQueryWrapper>
        <DfuShapPanel
          selectedModel="lgbm_cluster"
          itemNo="100320"
          loc="1401-BULK"
          dfuMode="item_location"
          visibleMonths={[]}
        />
      </TestQueryWrapper>
    );

    // Card title renders immediately
    await waitFor(() => {
      expect(screen.getByText(/SHAP Feature Contributions/i)).toBeDefined();
    });
  });

  // -------------------------------------------------------------------------
  // 4. Chart rendered on successful fetch
  // -------------------------------------------------------------------------
  it("renders cluster info and future note on successful fetch", async () => {
    mockFetchDfuShap.mockResolvedValue(_SHAP_PAYLOAD);

    render(
      <TestQueryWrapper>
        <DfuShapPanel
          selectedModel="lgbm_cluster"
          itemNo="100320"
          loc="1401-BULK"
          dfuMode="item_location"
          visibleMonths={["2024-01-01", "2024-02-01", "2026-04-01"]}
        />
      </TestQueryWrapper>
    );

    // Cluster badge and future months note should appear after data loads
    expect(await screen.findByText(/cluster: 0/i)).toBeDefined();
    expect(await screen.findByText(/Faded bars = future forecast months/i)).toBeDefined();
  });

  // -------------------------------------------------------------------------
  // 5. Fallback to cluster-level summary on 404
  // -------------------------------------------------------------------------
  it("falls back to cluster-level summary on 404 error", async () => {
    mockFetchDfuShap.mockRejectedValue(new Error("HTTP 404 Not Found"));
    mockFetchShapSummary.mockResolvedValue(_SHAP_SUMMARY);

    render(
      <TestQueryWrapper>
        <DfuShapPanel
          selectedModel="lgbm_cluster"
          itemNo="100320"
          loc="1401-BULK"
          dfuMode="item_location"
          visibleMonths={[]}
        />
      </TestQueryWrapper>
    );

    // Fallback warning text should appear
    expect(
      await screen.findByText(/Showing cluster-level SHAP/i)
    ).toBeDefined();
  });
});
