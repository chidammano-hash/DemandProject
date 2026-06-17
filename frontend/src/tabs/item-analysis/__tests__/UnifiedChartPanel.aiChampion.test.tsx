import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { UnifiedChartPanel } from "../UnifiedChartPanel";
import type { SkuAnalysisPayload } from "@/types";

// Recharts is heavy and DOM-dimension dependent; the project mocks it in tests.
vi.mock("recharts");

// useChartColors reads the theme context; stub it so the panel renders standalone.
vi.mock("@/hooks/useChartColors", () => ({
  useChartColors: () => ({
    chartColors: {
      grid: "#eee",
      axis: "#333",
      tooltip_bg: "#fff",
      tooltip_border: "#ccc",
    },
  }),
}));

const skuData: SkuAnalysisPayload = {
  mode: "item_location",
  item: "127630",
  location: "1401-BULK",
  points: 36,
  models: ["champion"],
  series: [
    { month: "2026-03-01", sales_qty: 100 },
    { month: "2026-04-01", forecast_champion: 110 },
  ],
  model_monthly: {},
  dfu_attributes: [],
};

function renderPanel(props: Partial<React.ComponentProps<typeof UnifiedChartPanel>> = {}) {
  return render(
    <UnifiedChartPanel
      skuData={skuData}
      skuFilteredSeries={skuData.series as Record<string, unknown>[]}
      skuMonths={["2026-03-01", "2026-04-01"]}
      skuTimeStart=""
      setSkuTimeStart={() => {}}
      skuTimeEnd=""
      setSkuTimeEnd={() => {}}
      skuDefaultStart=""
      skuVisibleSeries={new Set(["forecast_champion", "ai_champion"])}
      setSkuVisibleSeries={() => {}}
      {...props}
    />,
  );
}

describe("UnifiedChartPanel — AI Champion overlay", () => {
  it("hides the AI Champion pill and rationale when no saved adjustment exists", () => {
    renderPanel({ hasAiChampion: false });
    expect(screen.queryByText("AI Champion")).toBeNull();
  });

  it("shows the AI Champion pill and rationale once a saved adjustment exists", () => {
    renderPanel({
      hasAiChampion: true,
      aiChampionRecCode: "SCALE_UP",
      aiChampionRationale: "Recent actuals trend above the champion baseline.",
    });
    // Pill label
    expect(screen.getByText("AI Champion")).toBeDefined();
    // Rationale caption carries the recommendation code + the reason text
    expect(screen.getByText(/SCALE_UP/)).toBeDefined();
    expect(
      screen.getByText(/Recent actuals trend above the champion baseline\./),
    ).toBeDefined();
  });
});
