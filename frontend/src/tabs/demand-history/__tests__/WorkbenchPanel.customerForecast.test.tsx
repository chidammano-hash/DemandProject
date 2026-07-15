import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";

const { fetchCustomerForecastSeries, fetchProductionForecast, useWorkbench } = vi.hoisted(() => ({
  fetchCustomerForecastSeries: vi.fn(),
  fetchProductionForecast: vi.fn(),
  useWorkbench: vi.fn(),
}));

vi.mock("recharts");
vi.mock("@/api/queries/demand-history", () => ({
  useWorkbench,
}));
vi.mock("@/api/queries/customerForecast", () => ({
  customerForecastKeys: {
    series: (filters: unknown) => ["customer-forecast", "series", filters],
  },
  fetchCustomerForecastSeries,
}));
vi.mock("@/api/queries/production-forecast", () => ({
  fetchProductionForecast,
}));
vi.mock("@/tabs/DemandHistoryTab", () => ({
  useDemandHistorySelection: () => ({
    itemId: "84587",
    loc: "1401-BULK",
    setSelection: vi.fn(),
  }),
}));

import { DemandWorkbenchPanel } from "../WorkbenchPanel";

describe("DemandWorkbenchPanel customer forecast", () => {
  beforeEach(() => {
    useWorkbench.mockReturnValue({
      data: {
        grain: "item_loc_customer",
        series: [
          {
            key: "84587||1401-BULK||5726",
            label: "ABC LIQUORS#145(ABC WAREHOUSE)",
            total_demand: 130153,
            months: [{ month: "2026-06-01", demand_qty: 10000 }],
          },
        ],
        hierarchy_children: null,
        total: 1,
      },
      isLoading: false,
      isError: false,
    });
    fetchCustomerForecastSeries.mockResolvedValue({
      forecast: [
        {
          month: "2026-07-01",
          forecast_qty: 5420.4119,
          lower_bound: null,
          upper_bound: null,
          model_id: "croston",
        },
      ],
    });
    fetchProductionForecast.mockResolvedValue({ forecasts: [] });
  });

  it("enables Forecast and loads the selected customer series", async () => {
    render(
      <TestQueryWrapper>
        <DemandWorkbenchPanel />
      </TestQueryWrapper>
    );

    fireEvent.click(screen.getByRole("button", { name: "Item + Loc + Cust" }));
    fireEvent.click(await screen.findByText("ABC LIQUORS#145(ABC WAREHOUSE)"));

    const checkbox = screen.getByRole("checkbox", { name: "Forecast" });
    expect(checkbox).toBeEnabled();
    fireEvent.click(checkbox);

    await waitFor(() =>
      expect(fetchCustomerForecastSeries).toHaveBeenCalledWith({
        item_id: "84587",
        location_id: "1401-BULK",
        customer_no: "5726",
      })
    );
    expect(fetchProductionForecast).not.toHaveBeenCalled();
  });
});
