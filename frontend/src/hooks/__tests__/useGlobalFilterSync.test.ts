import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import type { GlobalFilters } from "@/types/theme";

// ---------------------------------------------------------------------------
// Mock GlobalFilterContext — overridden per test via mockFilters
// ---------------------------------------------------------------------------

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

// Import after mock so the hook picks up the mocked context
import {
  useGlobalFilterSync,
  useItemFilterSync,
  useItemLocationFilterSync,
} from "@/hooks/useGlobalFilterSync";

beforeEach(() => {
  mockFilters = { ...BASE_FILTERS };
});

// ---------------------------------------------------------------------------
// useGlobalFilterSync — core behaviour
// ---------------------------------------------------------------------------

describe("useGlobalFilterSync", () => {
  it("returns empty string values when global filters are empty", () => {
    const { result } = renderHook(() =>
      useGlobalFilterSync({ item: true, location: true }),
    );

    expect(result.current.item).toBe("");
    expect(result.current.location).toBe("");
  });

  it("provides setItem and setLocation setters", () => {
    const { result } = renderHook(() =>
      useGlobalFilterSync({ item: true, location: true }),
    );

    expect(typeof result.current.setItem).toBe("function");
    expect(typeof result.current.setLocation).toBe("function");
  });

  it("provides resetAll function", () => {
    const { result } = renderHook(() =>
      useGlobalFilterSync({ item: true, location: true }),
    );

    expect(typeof result.current.resetAll).toBe("function");
  });

  it("syncs item when global filter has exactly 1 item", () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320"] };

    const { result } = renderHook(() =>
      useGlobalFilterSync({ item: true, location: true }),
    );

    expect(result.current.item).toBe("100320");
    expect(result.current.location).toBe("");
  });

  it("syncs location when global filter has exactly 1 location", () => {
    mockFilters = { ...BASE_FILTERS, location: ["1401-BULK"] };

    const { result } = renderHook(() =>
      useGlobalFilterSync({ item: true, location: true }),
    );

    expect(result.current.item).toBe("");
    expect(result.current.location).toBe("1401-BULK");
  });

  it("syncs both item and location when each has exactly 1 value", () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320"], location: ["1401-BULK"] };

    const { result } = renderHook(() =>
      useGlobalFilterSync({ item: true, location: true }),
    );

    expect(result.current.item).toBe("100320");
    expect(result.current.location).toBe("1401-BULK");
  });

  it("does not sync item when global filter has 0 values", () => {
    mockFilters = { ...BASE_FILTERS, item: [] };

    const { result } = renderHook(() =>
      useGlobalFilterSync({ item: true }),
    );

    expect(result.current.item).toBe("");
  });

  it("does not sync item when global filter has 2+ values", () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320", "100321"] };

    const { result } = renderHook(() =>
      useGlobalFilterSync({ item: true }),
    );

    expect(result.current.item).toBe("");
  });

  it("does not sync location when global filter has 2+ values", () => {
    mockFilters = {
      ...BASE_FILTERS,
      item: ["100320"],
      location: ["LOC-A", "LOC-B"],
    };

    const { result } = renderHook(() =>
      useGlobalFilterSync({ item: true, location: true }),
    );

    expect(result.current.item).toBe("100320");
    expect(result.current.location).toBe("");
  });

  it("updates local state via setItem setter", () => {
    const { result } = renderHook(() =>
      useGlobalFilterSync({ item: true, location: true }),
    );

    act(() => {
      result.current.setItem("MANUAL-ITEM");
    });

    expect(result.current.item).toBe("MANUAL-ITEM");
    expect(result.current.location).toBe("");
  });

  it("updates local state via setLocation setter", () => {
    const { result } = renderHook(() =>
      useGlobalFilterSync({ item: true, location: true }),
    );

    act(() => {
      result.current.setLocation("MANUAL-LOC");
    });

    expect(result.current.location).toBe("MANUAL-LOC");
    expect(result.current.item).toBe("");
  });

  it("resetAll clears all synced filters back to empty strings", () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320"], location: ["1401-BULK"] };

    const { result } = renderHook(() =>
      useGlobalFilterSync({ item: true, location: true }),
    );

    expect(result.current.item).toBe("100320");
    expect(result.current.location).toBe("1401-BULK");

    act(() => {
      result.current.resetAll();
    });

    expect(result.current.item).toBe("");
    expect(result.current.location).toBe("");
  });

  it("re-syncs when global filter changes", () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320"] };

    const { result, rerender } = renderHook(() =>
      useGlobalFilterSync({ item: true }),
    );

    expect(result.current.item).toBe("100320");

    // Simulate global filter change
    mockFilters = { ...BASE_FILTERS, item: ["200400"] };
    rerender();

    expect(result.current.item).toBe("200400");
  });

  it("does not re-sync when global filter key has not changed (dedup)", () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320"] };

    const { result, rerender } = renderHook(() =>
      useGlobalFilterSync({ item: true }),
    );

    expect(result.current.item).toBe("100320");

    // Manually change local state
    act(() => {
      result.current.setItem("CUSTOM");
    });
    expect(result.current.item).toBe("CUSTOM");

    // Re-render with same global filter — should NOT overwrite manual change
    rerender();
    expect(result.current.item).toBe("CUSTOM");
  });

  it("supports item-only config (no location in result)", () => {
    mockFilters = { ...BASE_FILTERS, item: ["SKU-1"] };

    const { result } = renderHook(() =>
      useGlobalFilterSync({ item: true }),
    );

    expect(result.current.item).toBe("SKU-1");
    expect(typeof result.current.setItem).toBe("function");
    // location-related properties should not be present
    expect((result.current as Record<string, unknown>).location).toBeUndefined();
    expect((result.current as Record<string, unknown>).setLocation).toBeUndefined();
  });

  it("supports custom initial values", () => {
    const { result } = renderHook(() =>
      useGlobalFilterSync({
        item: { initialValue: "DEFAULT-ITEM" },
        location: { initialValue: "DEFAULT-LOC" },
      }),
    );

    expect(result.current.item).toBe("DEFAULT-ITEM");
    expect(result.current.location).toBe("DEFAULT-LOC");
  });

  it("resetAll restores custom initial values", () => {
    mockFilters = { ...BASE_FILTERS, item: ["SYNCED"] };

    const { result } = renderHook(() =>
      useGlobalFilterSync({
        item: { initialValue: "DEFAULT-ITEM" },
      }),
    );

    expect(result.current.item).toBe("SYNCED");

    act(() => {
      result.current.resetAll();
    });

    expect(result.current.item).toBe("DEFAULT-ITEM");
  });

  it("supports syncing other filter keys like brand and cluster", () => {
    mockFilters = { ...BASE_FILTERS, brand: ["Nike"], cluster: ["C3"] };

    const { result } = renderHook(() =>
      useGlobalFilterSync({ brand: true, cluster: true }),
    );

    expect(result.current.brand).toBe("Nike");
    expect(result.current.cluster).toBe("C3");
    expect(typeof result.current.setBrand).toBe("function");
    expect(typeof result.current.setCluster).toBe("function");
  });

  it("does not sync brand when it has 2+ values", () => {
    mockFilters = { ...BASE_FILTERS, brand: ["Nike", "Adidas"] };

    const { result } = renderHook(() =>
      useGlobalFilterSync({ brand: true }),
    );

    expect(result.current.brand).toBe("");
  });

  it("setter identity is stable across renders", () => {
    mockFilters = { ...BASE_FILTERS, item: ["100320"] };

    const { result, rerender } = renderHook(() =>
      useGlobalFilterSync({ item: true }),
    );

    const firstSetItem = result.current.setItem;
    rerender();
    expect(result.current.setItem).toBe(firstSetItem);
  });
});

// ---------------------------------------------------------------------------
// useItemFilterSync — convenience wrapper
// ---------------------------------------------------------------------------

describe("useItemFilterSync", () => {
  it("syncs item from global filter", () => {
    mockFilters = { ...BASE_FILTERS, item: ["SKU-99"] };

    const { result } = renderHook(() => useItemFilterSync());

    expect(result.current.item).toBe("SKU-99");
    expect(typeof result.current.setItem).toBe("function");
    expect(typeof result.current.resetAll).toBe("function");
  });

  it("does not include location", () => {
    mockFilters = { ...BASE_FILTERS, item: ["SKU-99"], location: ["LOC-1"] };

    const { result } = renderHook(() => useItemFilterSync());

    expect(result.current.item).toBe("SKU-99");
    expect((result.current as Record<string, unknown>).location).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// useItemLocationFilterSync — convenience wrapper
// ---------------------------------------------------------------------------

describe("useItemLocationFilterSync", () => {
  it("syncs both item and location from global filter", () => {
    mockFilters = { ...BASE_FILTERS, item: ["SKU-1"], location: ["LOC-A"] };

    const { result } = renderHook(() => useItemLocationFilterSync());

    expect(result.current.item).toBe("SKU-1");
    expect(result.current.location).toBe("LOC-A");
    expect(typeof result.current.setItem).toBe("function");
    expect(typeof result.current.setLocation).toBe("function");
    expect(typeof result.current.resetAll).toBe("function");
  });

  it("returns empty strings when global filters are empty", () => {
    const { result } = renderHook(() => useItemLocationFilterSync());

    expect(result.current.item).toBe("");
    expect(result.current.location).toBe("");
  });

  it("syncs on global filter change", () => {
    mockFilters = { ...BASE_FILTERS, item: ["A"], location: ["B"] };

    const { result, rerender } = renderHook(() => useItemLocationFilterSync());

    expect(result.current.item).toBe("A");
    expect(result.current.location).toBe("B");

    mockFilters = { ...BASE_FILTERS, item: ["X"], location: ["Y"] };
    rerender();

    expect(result.current.item).toBe("X");
    expect(result.current.location).toBe("Y");
  });
});
