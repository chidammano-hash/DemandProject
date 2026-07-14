import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import { ThemeProvider } from "@/context/ThemeContext";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SkuAnalysisPayload } from "@/types";
import { CustomerBlendUnifiedChartPanel } from "../CustomerBlendUnifiedChartPanel";

vi.mock("recharts");

afterEach(() => {
  vi.unstubAllGlobals();
});

const skuData: SkuAnalysisPayload = {
  mode: "item_location",
  item: "ITEM-1",
  location: "LOC-1",
  points: 1,
  models: ["champion"],
  series: [{ month: "2026-06-01", sales_qty: 70 }],
  model_monthly: {},
  dfu_attributes: [],
};

function stubCustomerBlend() {
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            run_id: "blend-1",
            status: "ready",
            planning_month: "2026-07-01",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            run_id: "blend-1",
            customer_run_id: "customer-1",
            source_run_id: "source-1",
            source_production_run_id: "production-1",
            item_id: "ITEM-1",
            location_id: "LOC-1",
            months: [
              {
                forecast_month: "2026-07-01",
                raw_customer_demand_qty: null,
                normalized_customer_qty: null,
                champion_qty: 80,
                blended_qty: 80,
                lower_bound: 70,
                upper_bound: 90,
                fulfillment_ratio: null,
                effective_customer_weight: 0,
                coverage_status: "champion_fallback",
                interval_method: "champion_passthrough",
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      )
  );
}

function renderPanel(skuTimeEnd = "") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <ThemeProvider value={{ theme: "light" }}>
      <QueryClientProvider client={client}>
        <CustomerBlendUnifiedChartPanel
          skuData={skuData}
          skuFilteredSeries={skuData.series as Record<string, unknown>[]}
          skuMonths={["2026-06-01"]}
          skuTimeStart=""
          setSkuTimeStart={() => {}}
          skuTimeEnd={skuTimeEnd}
          setSkuTimeEnd={() => {}}
          skuDefaultStart=""
          skuVisibleSeries={new Set(["sales_qty"])}
          setSkuVisibleSeries={() => {}}
        />
      </QueryClientProvider>
    </ThemeProvider>
  );
}

describe("CustomerBlendUnifiedChartPanel", () => {
  it("adds customer blend months and labels champion fallback on Item Analysis", async () => {
    stubCustomerBlend();
    renderPanel();

    const legendLabel = await screen.findByText("Customer Bottom-Up", { selector: "span" });
    const legend = legendLabel.closest('[role="status"]') as HTMLElement;
    expect(legend).not.toBeNull();
    expect(within(legend).getByText("Customer Bottom-Up")).toBeInTheDocument();
    expect(within(legend).getByText("Source Champion")).toBeInTheDocument();
    expect(within(legend).getByText("Customer Blend")).toBeInTheDocument();
    expect(screen.getByLabelText("Blend vintage Jul 2026, run blend-1")).toBeInTheDocument();
    expect(screen.getByText("Champion fallback · 1 of 1 month")).toBeInTheDocument();
    const toSelect = screen.getByRole("combobox", { name: /to/i });
    expect([...toSelect.querySelectorAll("option")].map((option) => option.value)).toContain(
      "2026-07-01"
    );
  });

  it("keeps fallback status aligned with the selected chart range", async () => {
    stubCustomerBlend();
    renderPanel("2026-06-01");

    expect(
      await screen.findByText("No customer blend falls within the selected range.")
    ).toBeInTheDocument();
    expect(screen.queryByText(/Champion fallback/)).toBeNull();
  });
});
