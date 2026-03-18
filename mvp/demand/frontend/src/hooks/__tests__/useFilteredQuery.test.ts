import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { createElement } from "react";
import { useFilteredQuery, useItemLocationQuery } from "@/hooks/useFilteredQuery";
import type { GlobalFilters } from "@/types/theme";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client }, children);
}

const BASE_FILTERS: GlobalFilters = {
  item: [],
  location: [],
      cluster: [],
  brand: [],
  category: [],
  market: [],
  channel: [],
  timeGrain: "month",
};

// ---------------------------------------------------------------------------
// Mock GlobalFilterContext — overridden per-suite via vi.mock hoisting
// ---------------------------------------------------------------------------

let mockFilters: GlobalFilters = { ...BASE_FILTERS };

vi.mock("@/context/GlobalFilterContext", () => ({
  useGlobalFilterContext: () => ({
    filters: mockFilters,
    setFilters: vi.fn(),
    resetFilters: vi.fn(),
    hasActiveFilters: false,
    planningDate: null,
  }),
}));

beforeEach(() => {
  mockFilters = { ...BASE_FILTERS };
});

// ---------------------------------------------------------------------------
// useFilteredQuery — core behaviour
// ---------------------------------------------------------------------------

describe("useFilteredQuery", () => {
  it("always includes baseParams in effectiveParams", async () => {
    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(async (p: { months: number }) => {
      capturedParams.push({ ...p });
      return { ok: true };
    });

    const { result } = renderHook(
      () =>
        useFilteredQuery({
          baseParams: { months: 12 },
          queryKey: (p) => ["test", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).toMatchObject({ months: 12 });
  });

  it("maps a single item filter value into effectiveParams", async () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320"] };

    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(async (p: { months: number; item?: string }) => {
      capturedParams.push({ ...p });
      return { ok: true };
    });

    const { result } = renderHook(
      () =>
        useFilteredQuery({
          filterMapping: {
            item: (v) => (Array.isArray(v) && v.length === 1 ? { item: v[0] } : null),
          },
          baseParams: { months: 6 },
          queryKey: (p) => ["test-item", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).toMatchObject({ months: 6, item: "100320" });
  });

  it("returns null from mapper when item filter has 0 values — param is not added", async () => {
    mockFilters = { ...BASE_FILTERS, item: [] };

    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(async (p: { months: number; item?: string }) => {
      capturedParams.push({ ...p });
      return { ok: true };
    });

    const { result } = renderHook(
      () =>
        useFilteredQuery({
          filterMapping: {
            item: (v) => (Array.isArray(v) && v.length === 1 ? { item: v[0] } : null),
          },
          baseParams: { months: 6 },
          queryKey: (p) => ["test-no-item", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).not.toHaveProperty("item");
  });

  it("returns null from mapper when item filter has 2+ values — param is not added", async () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320", "100321"] };

    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(async (p: { months: number; item?: string }) => {
      capturedParams.push({ ...p });
      return { ok: true };
    });

    const { result } = renderHook(
      () =>
        useFilteredQuery({
          filterMapping: {
            item: (v) => (Array.isArray(v) && v.length === 1 ? { item: v[0] } : null),
          },
          baseParams: { months: 6 },
          queryKey: (p) => ["test-multi-item", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).not.toHaveProperty("item");
  });

  it("maps a single location filter value into effectiveParams", async () => {
    mockFilters = { ...BASE_FILTERS, location: ["1401-BULK"] };

    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(async (p: { months: number; location?: string }) => {
      capturedParams.push({ ...p });
      return { ok: true };
    });

    const { result } = renderHook(
      () =>
        useFilteredQuery({
          filterMapping: {
            location: (v) =>
              Array.isArray(v) && v.length === 1 ? { location: v[0] } : null,
          },
          baseParams: { months: 3 },
          queryKey: (p) => ["test-loc", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).toMatchObject({ months: 3, location: "1401-BULK" });
  });

  it("ignores filter keys not listed in filterMapping", async () => {
    mockFilters = { ...BASE_FILTERS, brand: ["Nike"], item: ["100320"] };

    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(async (p: { months: number; item?: string }) => {
      capturedParams.push({ ...p });
      return { ok: true };
    });

    const { result } = renderHook(
      () =>
        useFilteredQuery({
          filterMapping: {
            item: (v) => (Array.isArray(v) && v.length === 1 ? { item: v[0] } : null),
          },
          baseParams: { months: 6 },
          queryKey: (p) => ["test-ignore-brand", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).toMatchObject({ months: 6, item: "100320" });
    expect(capturedParams[0]).not.toHaveProperty("brand");
  });

  it("merges multiple filter mappings together", async () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320"], location: ["1401-BULK"] };

    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(
      async (p: { months: number; item?: string; location?: string }) => {
        capturedParams.push({ ...p });
        return { ok: true };
      },
    );

    const { result } = renderHook(
      () =>
        useFilteredQuery({
          filterMapping: {
            item: (v) => (Array.isArray(v) && v.length === 1 ? { item: v[0] } : null),
            location: (v) =>
              Array.isArray(v) && v.length === 1 ? { location: v[0] } : null,
          },
          baseParams: { months: 12 },
          queryKey: (p) => ["test-both", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).toMatchObject({
      months: 12,
      item: "100320",
      location: "1401-BULK",
    });
  });

  it("works with no filterMapping at all — only baseParams are passed", async () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320"] };

    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(async (p: { mode: string }) => {
      capturedParams.push({ ...p });
      return { ok: true };
    });

    const { result } = renderHook(
      () =>
        useFilteredQuery({
          baseParams: { mode: "summary" },
          queryKey: (p) => ["test-no-mapping", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).toEqual({ mode: "summary" });
  });

  it("respects enabled: false — queryFn is not called", () => {
    const queryFn = vi.fn(async () => ({ value: 1 }));

    const { result } = renderHook(
      () =>
        useFilteredQuery({
          baseParams: { x: 1 },
          queryKey: (p) => ["test-disabled", p] as const,
          queryFn,
          enabled: false,
        }),
      { wrapper: makeWrapper() },
    );

    expect(result.current.isPending).toBe(true);
    expect(queryFn).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// useItemLocationQuery — convenience wrapper
// ---------------------------------------------------------------------------

describe("useItemLocationQuery", () => {
  it("maps single item filter to { item } param", async () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320"] };

    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(async (p: { months: number; item?: string; location?: string }) => {
      capturedParams.push({ ...p });
      return { rows: [] };
    });

    const { result } = renderHook(
      () =>
        useItemLocationQuery({
          baseParams: { months: 6 },
          queryKey: (p) => ["il-test", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).toMatchObject({ months: 6, item: "100320" });
    expect(capturedParams[0]).not.toHaveProperty("location");
  });

  it("maps single location filter to { location } param", async () => {
    mockFilters = { ...BASE_FILTERS, location: ["1401-BULK"] };

    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(async (p: { months: number; item?: string; location?: string }) => {
      capturedParams.push({ ...p });
      return { rows: [] };
    });

    const { result } = renderHook(
      () =>
        useItemLocationQuery({
          baseParams: { months: 6 },
          queryKey: (p) => ["il-loc-test", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).toMatchObject({ months: 6, location: "1401-BULK" });
    expect(capturedParams[0]).not.toHaveProperty("item");
  });

  it("maps both item and location when each has exactly 1 value", async () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320"], location: ["1401-BULK"] };

    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(async (p: { months: number; item?: string; location?: string }) => {
      capturedParams.push({ ...p });
      return { rows: [] };
    });

    const { result } = renderHook(
      () =>
        useItemLocationQuery({
          baseParams: { months: 12 },
          queryKey: (p) => ["il-both", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).toMatchObject({
      months: 12,
      item: "100320",
      location: "1401-BULK",
    });
  });

  it("does not map item when filter is empty", async () => {
    mockFilters = { ...BASE_FILTERS, item: [] };

    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(async (p: { months: number; item?: string; location?: string }) => {
      capturedParams.push({ ...p });
      return { rows: [] };
    });

    const { result } = renderHook(
      () =>
        useItemLocationQuery({
          baseParams: { months: 6 },
          queryKey: (p) => ["il-empty-item", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).not.toHaveProperty("item");
  });

  it("does not map item when filter has 2+ values", async () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320", "100321"] };

    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(async (p: { months: number; item?: string; location?: string }) => {
      capturedParams.push({ ...p });
      return { rows: [] };
    });

    const { result } = renderHook(
      () =>
        useItemLocationQuery({
          baseParams: { months: 6 },
          queryKey: (p) => ["il-multi-item", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).not.toHaveProperty("item");
  });

  it("always includes baseParams regardless of filter state", async () => {
    mockFilters = { ...BASE_FILTERS };

    const capturedParams: Record<string, unknown>[] = [];
    const queryFn = vi.fn(
      async (p: { months: number; sortBy: string; item?: string; location?: string }) => {
        capturedParams.push({ ...p });
        return { rows: [] };
      },
    );

    const { result } = renderHook(
      () =>
        useItemLocationQuery({
          baseParams: { months: 3, sortBy: "qty_on_hand" },
          queryKey: (p) => ["il-base", p] as const,
          queryFn,
        }),
      { wrapper: makeWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(capturedParams[0]).toMatchObject({ months: 3, sortBy: "qty_on_hand" });
  });
});
