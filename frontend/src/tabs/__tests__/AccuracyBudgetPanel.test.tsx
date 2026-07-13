import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

vi.mock("@/api/queries", () => ({
  accuracyBudgetKeys: {
    all: ["accuracy-budget"],
    decomposition: (m?: string) => ["accuracy-budget", "decomposition", m ?? "lgbm_cluster"],
    abc: () => ["accuracy-budget", "abc-breakdown"],
    models: () => ["accuracy-budget", "model-comparison"],
    monthly: () => ["accuracy-budget", "monthly-trend"],
    forecastValue: () => ["accuracy-budget", "forecast-value"],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchAccuracyDecomposition: vi.fn().mockResolvedValue({
    current_accuracy: 69.3,
    current_wape: 30.7,
    current_bias: -0.012,
    n_dfus: 50602,
    model_id: "lgbm_cluster",
    oracle_ceiling: 85.0,
    oracle_wape: 15.0,
    naive_baseline: 45.2,
    naive_wape: 54.8,
    forecast_value_added: 24.1,
    addressable_gap: 15.7,
    abc_breakdown: [],
    cluster_breakdown: [],
    components: [
      { name: "Intermittent demand", estimated_gain_pp: 2.5, rationale: "Zero-inflated DFUs" },
      { name: "Seasonality mismatch", estimated_gain_pp: 1.5, rationale: "Seasonal patterns not captured" },
      { name: "New product launches", estimated_gain_pp: 1.0, rationale: "No history for new items" },
    ],
    irreducible_noise: 6.0,
  }),
  fetchAbcBreakdown: vi.fn().mockResolvedValue({
    classes: [
      { abc_class: "A", accuracy_pct: 78.5, wape: 21.5, bias: -0.01, n_dfus: 5000, volume_share: 0.72, error_share: 0.35 },
      { abc_class: "B", accuracy_pct: 65.2, wape: 34.8, bias: 0.02, n_dfus: 12000, volume_share: 0.20, error_share: 0.28 },
      { abc_class: "C", accuracy_pct: 42.1, wape: 57.9, bias: 0.05, n_dfus: 33000, volume_share: 0.08, error_share: 0.37 },
    ],
  }),
  fetchMonthlyTrend: vi.fn().mockResolvedValue({
    months: [
      { month: 1, accuracy: 68.1, wape: 31.9, bias: -0.01, n_dfus: 50602, flag: null },
      { month: 2, accuracy: 69.8, wape: 30.2, bias: 0.02, n_dfus: 50602, flag: null },
      { month: 3, accuracy: 71.2, wape: 28.8, bias: -0.005, n_dfus: 50602, flag: null },
    ],
    worst_month: { month: 1, accuracy: 68.1, wape: 31.9, bias: -0.01, n_dfus: 50602, flag: null },
    best_month: { month: 3, accuracy: 71.2, wape: 28.8, bias: -0.005, n_dfus: 50602, flag: null },
  }),
  fetchModelComparison: vi.fn().mockResolvedValue({
    models: [
      { model_id: "lgbm_cluster", accuracy_pct: 69.3, wape: 30.7, bias: -0.012, n_dfus: 50602 },
      { model_id: "nhits", accuracy_pct: 67.1, wape: 32.9, bias: 0.008, n_dfus: 50602 },
      { model_id: "chronos2_enriched", accuracy_pct: 66.5, wape: 33.5, bias: -0.015, n_dfus: 50602 },
    ],
    oracle_ceiling: { accuracy: 85.0, wape: 15.0 },
  }),
  fetchForecastValue: vi.fn().mockResolvedValue({
    baselines: [
      { name: "seasonal_naive", description: "Same month last year", accuracy: 45.2, wape: 54.8 },
      { name: "rolling_3m_avg", description: "3-month rolling average", accuracy: 42.0, wape: 58.0 },
    ],
    ml_model: { name: "lgbm_cluster", accuracy: 69.3, wape: 30.7 },
    value_added: { vs_seasonal_naive: 24.1, vs_rolling_3m: 27.3, vs_flat: 30.0 },
  }),
  // Barrel stubs
  queryKeys: {},
  STALE_INSIGHTS: 300000,
  insightKeys: { all: () => ["insights"] },
}));

import {
  fetchAccuracyDecomposition,
  fetchForecastValue,
} from "@/api/queries";
import { AccuracyBudgetPanel } from "@/tabs/lgbm-tuning/AccuracyBudgetPanel";

describe("AccuracyBudgetPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders summary KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <AccuracyBudgetPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Naive Baseline")).toBeInTheDocument();
    });
    expect(screen.getByText("ML Model")).toBeInTheDocument();
    expect(screen.getByText("Oracle Ceiling")).toBeInTheDocument();
    expect(screen.getByText("Addressable Gap")).toBeInTheDocument();
    expect(screen.getByText("Value Added")).toBeInTheDocument();
  });

  it("renders waterfall sub-tab by default with chart", async () => {
    render(
      <TestQueryWrapper>
        <AccuracyBudgetPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getAllByText("Accuracy Waterfall").length).toBeGreaterThanOrEqual(1);
    });
    // Value-added cards from waterfall section
    await waitFor(() => {
      expect(screen.getByText(/Same month last year/)).toBeInTheDocument();
    });
  });

  it("shows all sub-tab navigation buttons", async () => {
    render(
      <TestQueryWrapper>
        <AccuracyBudgetPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getAllByText("Accuracy Waterfall").length).toBeGreaterThanOrEqual(1);
    });
    expect(screen.getAllByText("Gap Decomposition").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("ABC Targets").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Monthly Trend").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Model Comparison").length).toBeGreaterThanOrEqual(1);
  });

  it("switches to gap decomposition tab", async () => {
    render(
      <TestQueryWrapper>
        <AccuracyBudgetPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Gap Decomposition")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Gap Decomposition"));
    await waitFor(() => {
      expect(screen.getByText("Intermittent demand")).toBeInTheDocument();
    });
    expect(screen.getByText("Seasonality mismatch")).toBeInTheDocument();
    expect(screen.getByText("New product launches")).toBeInTheDocument();
  });

  it("switches to ABC targets tab with status badges", async () => {
    render(
      <TestQueryWrapper>
        <AccuracyBudgetPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("ABC Targets")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("ABC Targets"));
    await waitFor(() => {
      expect(screen.getByText("A")).toBeInTheDocument();
    });
    expect(screen.getByText("B")).toBeInTheDocument();
    expect(screen.getByText("C")).toBeInTheDocument();
    // All 3 are below target
    const belowBadges = screen.getAllByText("Below Target");
    expect(belowBadges.length).toBe(3);
  });

  it("switches to monthly trend tab with line chart", async () => {
    render(
      <TestQueryWrapper>
        <AccuracyBudgetPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Monthly Trend")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Monthly Trend"));
    await waitFor(() => {
      expect(screen.getByTestId("line-chart")).toBeInTheDocument();
    });
  });

  it("switches to model comparison tab", async () => {
    render(
      <TestQueryWrapper>
        <AccuracyBudgetPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Model Comparison")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Model Comparison"));
    await waitFor(() => {
      expect(screen.getByText("lgbm_cluster")).toBeInTheDocument();
    });
    expect(screen.getByText("N-HiTS")).toBeInTheDocument();
    expect(screen.getByText("Chronos 2E")).toBeInTheDocument();
    expect(screen.getByText("Best")).toBeInTheDocument();
  });

  it("renders empty state when no data", async () => {
    (fetchAccuracyDecomposition as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      current_accuracy: null,
    });

    render(
      <TestQueryWrapper>
        <AccuracyBudgetPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText(/No accuracy budget data/)).toBeInTheDocument();
    });
  });

  it("calls decomposition fetch on mount", async () => {
    render(
      <TestQueryWrapper>
        <AccuracyBudgetPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(fetchAccuracyDecomposition).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(fetchForecastValue).toHaveBeenCalled();
    });
  });
});
