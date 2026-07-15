import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { UnifiedChartPanel } from "../UnifiedChartPanel";
import type { SkuAnalysisPayload } from "@/types";
import type { CandidateForecastsPayload } from "@/api/queries/production-forecast";

// Recharts leaf components render null in the shared mock; assertions target the
// pill toolbar (which carries the model labels), not the chart lines.
vi.mock("recharts");
vi.mock("@/hooks/useChartColors", async () => {
  const { PALETTE } = await import("@/constants/palette");
  const charts = PALETTE.light.charts;
  return {
    useChartColors: () => ({
      theme: "light",
      chartColors: { grid: "#eee", axis: "#333", tooltip_bg: "#fff", tooltip_border: "#ccc" },
      roles: charts.roles,
      series: [...charts.series],
      fallback: [...charts.fallback],
      heatmap: [...charts.heatmapScale],
      trendColors: [...charts.series],
      okabeIto: [...charts.series],
    }),
  };
});

const skuData: SkuAnalysisPayload = {
  mode: "item_location",
  item: "127630",
  location: "1401-BULK",
  points: 36,
  models: ["lgbm_cluster"],
  series: [
    { month: "2025-01-01", sales_qty: 100, backtest_lgbm_cluster: 98 },
    { month: "2025-02-01", sales_qty: 110, backtest_lgbm_cluster: 112 },
  ],
  model_monthly: {},
  dfu_attributes: [],
};

const candidate: CandidateForecastsPayload = {
  item_id: "127630",
  loc: "1401-BULK",
  models: {
    lgbm_cluster: [
      {
        forecast_month: "2025-01-01",
        forecast_qty: 98,
        forecast_qty_lower: null,
        forecast_qty_upper: null,
        actual_qty: 100,
        accuracy_pct: 98,
        wape: 2,
        bias: -0.02,
        horizon_months: 1,
        cluster_id: "c1",
      },
    ],
  },
};

function renderPanel(props: Partial<React.ComponentProps<typeof UnifiedChartPanel>> = {}) {
  return render(
    <UnifiedChartPanel
      skuData={skuData}
      skuFilteredSeries={skuData.series as Record<string, unknown>[]}
      skuMonths={["2025-01-01", "2025-02-01"]}
      skuTimeStart=""
      setSkuTimeStart={() => {}}
      skuTimeEnd=""
      setSkuTimeEnd={() => {}}
      skuDefaultStart=""
      skuVisibleSeries={new Set(["forecast_lgbm_cluster", "sales_qty"])}
      setSkuVisibleSeries={() => {}}
      {...props}
    />,
  );
}

describe("UnifiedChartPanel — backtest overlay", () => {
  it("hides the Backtest pill row when no candidate data exists", () => {
    renderPanel({ candidateForecastData: null });
    expect(screen.queryByText(/Backtest/)).toBeNull();
  });

  it("shows the Backtest pill row when candidate (past backtest) data exists", () => {
    renderPanel({ candidateForecastData: candidate });
    // The Backtest toggle-all control appears (label is "Backtest −" when the
    // models start visible, mirroring the Staging row).
    expect(screen.getByText(/Backtest/)).toBeDefined();
  });
});
