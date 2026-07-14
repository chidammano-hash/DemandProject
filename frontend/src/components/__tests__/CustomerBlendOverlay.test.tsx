import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("recharts", () => ({
  Line: ({ dataKey, legendType, name }: { dataKey: string; legendType: string; name: string }) => (
    <span data-key={dataKey} data-legend-type={legendType}>
      {name}
    </span>
  ),
}));

vi.mock("@/hooks/useChartColors", () => ({
  useChartColors: () => ({
    chartColors: { axis: "neutral" },
    okabeIto: ["orange", "sky", "green", "yellow", "blue", "vermillion", "purple"],
  }),
}));

import { CustomerBlendLegend, CustomerBlendLines } from "../CustomerBlendOverlay";
import {
  CUSTOMER_BLEND_KEYS,
  mergeCustomerBlendOverlay,
  toCustomerBlendChartPoints,
} from "@/lib/customer-blend-overlay";

const months = [
  {
    forecast_month: "2026-07-01",
    raw_customer_demand_qty: 120,
    normalized_customer_qty: 90,
    champion_qty: 100,
    blended_qty: 95,
    lower_bound: 75,
    upper_bound: 125,
    fulfillment_ratio: 0.75,
    effective_customer_weight: 0.5,
    coverage_status: "blended" as const,
    interval_method: "champion_width_shift" as const,
  },
  {
    forecast_month: "2026-08-01",
    raw_customer_demand_qty: null,
    normalized_customer_qty: null,
    champion_qty: 80,
    blended_qty: 80,
    lower_bound: 70,
    upper_bound: 90,
    fulfillment_ratio: null,
    effective_customer_weight: 0,
    coverage_status: "champion_fallback" as const,
    interval_method: "champion_passthrough" as const,
  },
];

describe("CustomerBlendOverlay", () => {
  it("maps normalized customer demand, source champion, and blend into chart fields", () => {
    const points = toCustomerBlendChartPoints(months);

    expect(points[0]).toMatchObject({
      month: "2026-07-01",
      [CUSTOMER_BLEND_KEYS.bottomUp]: 90,
      [CUSTOMER_BLEND_KEYS.sourceChampion]: 100,
      [CUSTOMER_BLEND_KEYS.blend]: 95,
    });
    expect(points[1][CUSTOMER_BLEND_KEYS.bottomUp]).toBeNull();
  });

  it("overlays matching months and appends missing future months in order", () => {
    const merged = mergeCustomerBlendOverlay(
      [
        { month: "2026-06-01", actual_qty: 70 },
        { month: "2026-07-15", actual_qty: 75 },
      ],
      toCustomerBlendChartPoints(months)
    );

    expect(merged.map((point) => point.month)).toEqual(["2026-06-01", "2026-07-15", "2026-08-01"]);
    expect(merged[1][CUSTOMER_BLEND_KEYS.blend]).toBe(95);
    expect(merged[2][CUSTOMER_BLEND_KEYS.sourceChampion]).toBe(80);
  });

  it("renders the three comparison labels and an explicit fallback count", () => {
    render(
      <CustomerBlendLegend
        months={months}
        status="ready"
        runId="blend-run-1"
        planningMonth="2026-07-01"
      />
    );

    const legend = screen.getByRole("status");
    expect(within(legend).getByText("Customer Bottom-Up")).toBeInTheDocument();
    expect(within(legend).getByText("Source Champion")).toBeInTheDocument();
    expect(within(legend).getByText("Customer Blend")).toBeInTheDocument();
    expect(screen.getByText("Customer forecast blend")).toBeInTheDocument();
    expect(screen.getByText("Customer signal normalized to sales")).toBeInTheDocument();
    expect(screen.getByText("Champion fallback · 1 of 2 months")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Champion fallback · 1 of 2 months");
    expect(within(legend).getByText("Customer Bottom-Up").previousElementSibling).toHaveClass(
      "border-dashed"
    );
    expect(within(legend).getByText("Source Champion").previousElementSibling).toHaveClass(
      "border-dotted"
    );
    expect(within(legend).getByText("Customer Blend").previousElementSibling).toHaveClass(
      "border-solid"
    );
    const table = screen.getByRole("table", {
      name: "Monthly customer forecast blend quantities · Jul 2026 · run blend-run-1",
    });
    expect(
      within(table).getByRole("row", { name: /Jul 2026 90 100 95 Blended/ })
    ).toBeInTheDocument();
    expect(
      within(table).getByRole("row", { name: /Aug 2026 — 80 80 Champion fallback/ })
    ).toBeInTheDocument();
  });

  it("keeps the unavailable-blend message distinct", () => {
    render(<CustomerBlendLegend months={[]} status="empty" />);

    expect(
      screen.getByText("No customer blend is available for this item and location.")
    ).toBeInTheDocument();
  });

  it("uses reusable named chart lines", () => {
    render(<CustomerBlendLines yAxisId="left" />);

    expect(screen.getByText("Customer Bottom-Up")).toHaveAttribute(
      "data-key",
      CUSTOMER_BLEND_KEYS.bottomUp
    );
    expect(screen.getByText("Source Champion")).toHaveAttribute(
      "data-key",
      CUSTOMER_BLEND_KEYS.sourceChampion
    );
    expect(screen.getByText("Customer Blend")).toHaveAttribute(
      "data-key",
      CUSTOMER_BLEND_KEYS.blend
    );
    for (const line of screen.getAllByText(/Customer Bottom-Up|Source Champion|Customer Blend/)) {
      expect(line).toHaveAttribute("data-legend-type", "none");
    }
  });
});
