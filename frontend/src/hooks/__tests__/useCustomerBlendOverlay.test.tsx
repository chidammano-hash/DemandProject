import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { customerForecastKeys } from "@/api/queries/customerForecast";

import { useCustomerBlendOverlay } from "../useCustomerBlendOverlay";

function createHarness() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  function QueryWrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  }
  return { client, wrapper: QueryWrapper };
}

function createWrapper() {
  return createHarness().wrapper;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useCustomerBlendOverlay", () => {
  it("loads the latest blend for an enabled exact item-location", async () => {
    const fetchMock = vi
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
            item_id: "ITEM 1",
            location_id: "LOC/1",
            months: [],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      );
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useCustomerBlendOverlay(" ITEM 1 ", " LOC/1 ", true), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.status).toBe("empty"));
    expect(fetchMock.mock.calls[0][0]).toBe("/customer-forecast/blend/latest");
    expect(fetchMock.mock.calls[1][0]).toBe(
      "/customer-forecast/blend/series?item_id=ITEM+1&location_id=LOC%2F1&run_id=blend-1"
    );
    expect(result.current.runId).toBe("blend-1");
    expect(result.current.planningMonth).toBe("2026-07-01");
  });

  it("treats a missing blend as an empty overlay instead of a screen error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Customer blend series not found" }), {
          status: 404,
          headers: { "Content-Type": "application/json" },
        })
      )
    );

    const { result } = renderHook(() => useCustomerBlendOverlay("ITEM-1", "LOC-1", true), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.status).toBe("empty"));
    expect(result.current.points).toEqual([]);
  });

  it("does not request series for a generating current blend", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          run_id: "blend-generating",
          status: "generating",
          planning_month: "2026-07-01",
          job_id: "job-1",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useCustomerBlendOverlay("ITEM-1", "LOC-1", true), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.runId).toBe("blend-generating"));
    expect(result.current.status).toBe("loading");
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("clears a mounted overlay when current lineage no longer resolves", async () => {
    const fetchMock = vi
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
                blended_qty: 100,
                champion_qty: 95,
                coverage_status: "blended",
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: "No current customer blend run found" }), {
          status: 404,
          headers: { "Content-Type": "application/json" },
        })
      );
    vi.stubGlobal("fetch", fetchMock);
    const { client, wrapper } = createHarness();

    const { result } = renderHook(() => useCustomerBlendOverlay("ITEM-1", "LOC-1", true), {
      wrapper,
    });

    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.points).toHaveLength(1);

    await client.invalidateQueries({ queryKey: customerForecastKeys.latestBlend });

    await waitFor(() => expect(result.current.status).toBe("empty"));
    expect(result.current.points).toEqual([]);
    expect(result.current.runId).toBeNull();
  });

  it("does not query aggregated or incomplete selections", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useCustomerBlendOverlay("ITEM-1", "", false), {
      wrapper: createWrapper(),
    });

    expect(result.current.status).toBe("idle");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
