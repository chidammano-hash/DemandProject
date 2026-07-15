import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  fetchProductionForecast,
  fetchStagingForecasts,
  fetchCandidateForecasts,
  fetchAiChampionSaved,
} = vi.hoisted(() => ({
  fetchProductionForecast: vi.fn(),
  fetchStagingForecasts: vi.fn(),
  fetchCandidateForecasts: vi.fn(),
  fetchAiChampionSaved: vi.fn(),
}));

vi.mock("@/api/queries/production-forecast", () => ({
  fetchProductionForecast,
  fetchStagingForecasts,
  fetchCandidateForecasts,
}));

vi.mock("@/api/queries/ai-champion", () => ({
  aiChampionKeys: { saved: (item: string, loc: string) => ["ai", item, loc] },
  fetchAiChampionSaved,
}));

import { useItemForecastOverlays } from "../useItemForecastOverlays";

function Wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useItemForecastOverlays", () => {
  beforeEach(() => {
    fetchProductionForecast.mockReset().mockResolvedValue({ forecasts: [] });
    fetchStagingForecasts.mockReset().mockResolvedValue({
      item_id: "ITEM-1",
      loc: "LOC-1",
      models: {
        customer_bottom_up: [{ forecast_month: "2026-07-01", forecast_qty: 90 }],
      },
    });
    fetchCandidateForecasts.mockReset().mockResolvedValue({
      item_id: "ITEM-1",
      loc: "LOC-1",
      models: {
        customer_bottom_up: [{ forecast_month: "2026-06-01", forecast_qty: 92 }],
      },
    });
    fetchAiChampionSaved.mockReset().mockResolvedValue({ total: 0, rows: [] });
  });

  it("loads and composes staged and backtest overlays for one DFU", async () => {
    const { result } = renderHook(
      () =>
        useItemForecastOverlays({
          itemId: " ITEM-1 ",
          locationId: " LOC-1 ",
          historyMonths: ["2026-06-01"],
          filteredSeries: [{ month: "2026-06-01", sales_qty: 100 }],
          timeEnd: "",
          aiEnabled: true,
        }),
      { wrapper: Wrapper }
    );

    await waitFor(() => expect(result.current.skuFutureMonths).toEqual(["2026-07-01"]));
    expect(result.current.mergedFilteredSeries[0]).toMatchObject({
      backtest_customer_bottom_up: 92,
    });
    expect(result.current.mergedFilteredSeries[1]).toMatchObject({
      staging_customer_bottom_up: 90,
    });
    expect(fetchStagingForecasts).toHaveBeenCalledWith({
      item_id: "ITEM-1",
      loc: "LOC-1",
    });
  });

  it("hides the previous DFU payload while a new selection is loading", async () => {
    const { result, rerender } = renderHook(
      ({ itemId, locationId }) =>
        useItemForecastOverlays({
          itemId,
          locationId,
          historyMonths: ["2026-06-01"],
          filteredSeries: [{ month: "2026-06-01", sales_qty: 100 }],
          timeEnd: "",
          aiEnabled: false,
        }),
      {
        initialProps: { itemId: "ITEM-1", locationId: "LOC-1" },
        wrapper: Wrapper,
      }
    );

    await waitFor(() => expect(result.current.stagingForecastData?.loc).toBe("LOC-1"));
    expect(result.current.mergedFilteredSeries[1]).toMatchObject({
      staging_customer_bottom_up: 90,
    });

    const pending = new Promise<never>(() => {});
    fetchProductionForecast.mockReturnValueOnce(pending);
    fetchStagingForecasts.mockReturnValueOnce(pending);
    fetchCandidateForecasts.mockReturnValueOnce(pending);
    rerender({ itemId: "ITEM-2", locationId: "LOC-2" });

    expect(result.current.prodForecastData).toBeNull();
    expect(result.current.stagingForecastData).toBeNull();
    expect(result.current.candidateForecastData).toBeNull();
    expect(result.current.mergedFilteredSeries).toEqual([{ month: "2026-06-01", sales_qty: 100 }]);
  });
});
