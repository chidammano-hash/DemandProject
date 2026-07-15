import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import { ThemeProvider } from "@/context/ThemeContext";
import { afterEach, describe, expect, it, vi } from "vitest";

import { customerForecastKeys } from "@/api/queries/customerForecast";
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

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function stubCustomerBlend({ trendFails = false } = {}) {
  let lineage = 1;
  const fetchMock = vi.fn((input: string | URL | Request) => {
    const url = String(input);
    const blendRunId = `blend-${lineage}`;
    const shadowRunId = `shadow-${lineage}`;
    if (url === "/customer-forecast/blend/latest") {
      return Promise.resolve(
        jsonResponse({
          run_id: blendRunId,
          status: "ready",
          planning_month: "2026-07-01",
          bottom_up_staging_run_id: shadowRunId,
        })
      );
    }
    if (url.startsWith("/customer-forecast/blend/series?")) {
      return Promise.resolve(
        jsonResponse({
          run_id: blendRunId,
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
        })
      );
    }
    if (url.startsWith("/customer-forecast/blend/trend?")) {
      if (trendFails) {
        return Promise.resolve(jsonResponse({ detail: "trend lookup failed" }, 500));
      }
      return Promise.resolve(
        jsonResponse({
          run_id: blendRunId,
          planning_month: "2026-07-01",
          bottom_up_staging_run_id: shadowRunId,
          months: [
            {
              month: "2026-06-01",
              phase: "backtest",
              actual_qty: 70,
              customer_bottom_up_qty: 68,
              source_champion_qty: 72,
              customer_blend_qty: 70,
            },
          ],
        })
      );
    }
    if (url.startsWith("/forecast/production/staging?")) {
      return Promise.resolve(
        jsonResponse({
          item_id: "ITEM-1",
          loc: "LOC-1",
          models: {
            customer_bottom_up: [
              {
                source_model_id: "customer_bottom_up",
                source_run_id: shadowRunId,
                forecast_month: "2026-07-01",
                forecast_qty: 75,
              },
            ],
            customer_bottom_up_blend: [
              {
                source_model_id: "customer_bottom_up_blend",
                source_run_id: blendRunId,
                forecast_month: "2026-07-01",
                forecast_qty: 80,
              },
            ],
          },
        })
      );
    }
    throw new Error(`Unexpected request: ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return {
    fetchMock,
    advanceLineage() {
      lineage += 1;
    },
  };
}

function renderPanel(skuTimeEnd = "") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return {
    client,
    ...render(
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
    ),
  };
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

  it("adds exact customer backtest models to the Item Analysis history controls", async () => {
    stubCustomerBlend();
    renderPanel();

    expect(
      (await screen.findAllByRole("button", { name: "Customer Bottom-Up" })).length
    ).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Source Champion" })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Customer Blend" }).length).toBeGreaterThan(0);
  });

  it("keeps exact bottom-up staging visible when the historical trend fails", async () => {
    stubCustomerBlend({ trendFails: true });
    renderPanel();

    expect(await screen.findByText("Staging −")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Customer Bottom-Up" })).toBeInTheDocument();
  });

  it("refetches exact staging when the current blend lineage changes in place", async () => {
    const { fetchMock, advanceLineage } = stubCustomerBlend();
    const { client } = renderPanel();

    await screen.findByText("Staging −");
    expect(
      fetchMock.mock.calls.filter(([url]) =>
        String(url).startsWith("/forecast/production/staging?")
      )
    ).toHaveLength(1);

    await act(async () => {
      advanceLineage();
      await client.invalidateQueries({ queryKey: customerForecastKeys.latestBlend });
    });

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.filter(([url]) =>
          String(url).startsWith("/forecast/production/staging?")
        )
      ).toHaveLength(2)
    );
  });
});
