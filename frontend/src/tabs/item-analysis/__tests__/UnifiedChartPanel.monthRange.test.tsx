import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { UnifiedChartPanel } from "../UnifiedChartPanel";
import type { SkuAnalysisPayload } from "@/types";

vi.mock("recharts");
vi.mock("@/hooks/useChartColors", () => ({
  useChartColors: () => ({
    chartColors: { grid: "#eee", axis: "#333", tooltip_bg: "#fff", tooltip_border: "#ccc" },
  }),
}));

const skuData: SkuAnalysisPayload = {
  mode: "item_location",
  item: "127630",
  location: "1401-BULK",
  points: 36,
  models: ["lgbm_cluster"],
  series: [
    { month: "2026-05-01", sales_qty: 100 },
    { month: "2026-06-01", sales_qty: 110 },
  ],
  model_monthly: {},
  dfu_attributes: [],
};

function renderPanel(props: Partial<React.ComponentProps<typeof UnifiedChartPanel>> = {}) {
  return render(
    <UnifiedChartPanel
      skuData={skuData}
      skuFilteredSeries={skuData.series as Record<string, unknown>[]}
      skuMonths={["2026-05-01", "2026-06-01"]}
      skuTimeStart=""
      setSkuTimeStart={() => {}}
      skuTimeEnd=""
      setSkuTimeEnd={() => {}}
      skuDefaultStart=""
      skuVisibleSeries={new Set(["sales_qty"])}
      setSkuVisibleSeries={() => {}}
      {...props}
    />,
  );
}

describe("UnifiedChartPanel — TO range covers forecast months", () => {
  it("offers future forecast months as TO options", () => {
    renderPanel({ skuFutureMonths: ["2026-07-01", "2027-06-01"] });

    const toSelect = screen.getByRole("combobox", { name: /to/i });
    const options = [...toSelect.querySelectorAll("option")].map((o) => o.textContent);
    expect(options).toContain("Jul 2026");
    expect(options).toContain("Jun 2027");
  });

  it("shows the full-horizon end as the displayed TO when no bound is set", () => {
    renderPanel({ skuFutureMonths: ["2026-07-01", "2027-06-01"] });

    const toSelect = screen.getByRole("combobox", { name: /to/i }) as HTMLSelectElement;
    expect(toSelect.value).toBe("2027-06-01");
  });

  it("keeps history-only behavior when there are no future months", () => {
    renderPanel();

    const toSelect = screen.getByRole("combobox", { name: /to/i }) as HTMLSelectElement;
    expect(toSelect.value).toBe("2026-06-01");
  });
});
