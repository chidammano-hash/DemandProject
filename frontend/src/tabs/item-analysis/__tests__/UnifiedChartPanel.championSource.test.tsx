import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { UnifiedChartPanel } from "../UnifiedChartPanel";
import type { SkuAnalysisPayload } from "@/types";

// Recharts is heavy and DOM-dimension dependent; the project mocks it in tests.
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

function makeData(overrides: Partial<SkuAnalysisPayload> = {}): SkuAnalysisPayload {
  return {
    mode: "item_location",
    item: "127630",
    location: "1401-BULK",
    points: 36,
    models: ["champion"],
    series: [{ month: "2026-04-01", forecast_champion: 110, champion_source: "nbeats" }],
    model_monthly: {},
    dfu_attributes: [],
    ...overrides,
  };
}

function renderPanel(data: SkuAnalysisPayload) {
  return render(
    <UnifiedChartPanel
      skuData={data}
      skuFilteredSeries={data.series as Record<string, unknown>[]}
      skuMonths={["2026-04-01"]}
      skuTimeStart=""
      setSkuTimeStart={() => {}}
      skuTimeEnd=""
      setSkuTimeEnd={() => {}}
      skuDefaultStart=""
      skuVisibleSeries={new Set(["forecast_champion"])}
      setSkuVisibleSeries={() => {}}
    />,
  );
}

describe("UnifiedChartPanel — champion source label", () => {
  it("labels the champion pill with its winning model when a source is known", () => {
    renderPanel(makeData({ champion_dominant_source: "nbeats" }));
    expect(screen.getByText("champion (N-BEATS)")).toBeDefined();
  });

  it("falls back to a bare 'champion' label when no source is known", () => {
    renderPanel(makeData({ champion_dominant_source: null }));
    expect(screen.getByText("champion")).toBeDefined();
    expect(screen.queryByText(/champion \(/)).toBeNull();
  });
});
