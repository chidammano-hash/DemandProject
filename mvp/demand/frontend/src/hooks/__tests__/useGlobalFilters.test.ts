import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useGlobalFilters } from "@/hooks/useGlobalFilters";

// Mock window.history.replaceState to avoid jsdom errors
const replaceStateSpy = vi.fn();
Object.defineProperty(window, "history", {
  value: { ...window.history, replaceState: replaceStateSpy },
  writable: true,
});

describe("useGlobalFilters", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    // Reset URL search params
    Object.defineProperty(window, "location", {
      value: { ...window.location, search: "", href: "http://localhost/" },
      writable: true,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("defaults to empty filters", () => {
    const { result } = renderHook(() => useGlobalFilters());
    expect(result.current.filters).toEqual({
      brand: [],
      category: [],
      market: [],
      channel: [],
      item: [],
      location: [],
      timeGrain: "month",
    });
    expect(result.current.hasActiveFilters).toBe(false);
  });

  it("setFilters updates state with partial filters", () => {
    const { result } = renderHook(() => useGlobalFilters());

    act(() => {
      result.current.setFilters({ brand: ["BrandA", "BrandB"] });
    });
    expect(result.current.filters.brand).toEqual(["BrandA", "BrandB"]);
    expect(result.current.hasActiveFilters).toBe(true);
    // Other filters remain unchanged
    expect(result.current.filters.category).toEqual([]);
    expect(result.current.filters.timeGrain).toBe("month");
  });

  it("setFilters updates timeGrain", () => {
    const { result } = renderHook(() => useGlobalFilters());

    act(() => {
      result.current.setFilters({ timeGrain: "quarter" });
    });
    expect(result.current.filters.timeGrain).toBe("quarter");
  });

  it("setFilters can update multiple filter keys at once", () => {
    const { result } = renderHook(() => useGlobalFilters());

    act(() => {
      result.current.setFilters({ brand: ["X"], market: ["CA"] });
    });
    expect(result.current.filters.brand).toEqual(["X"]);
    expect(result.current.filters.market).toEqual(["CA"]);
  });

  it("resetFilters clears all filters", () => {
    const { result } = renderHook(() => useGlobalFilters());

    // Set some filters first
    act(() => {
      result.current.setFilters({
        brand: ["BrandA"],
        category: ["Cat1"],
        market: ["NY"],
        channel: ["Online"],
        timeGrain: "quarter",
      });
    });
    expect(result.current.hasActiveFilters).toBe(true);

    // Reset
    act(() => {
      result.current.resetFilters();
    });
    expect(result.current.filters).toEqual({
      brand: [],
      category: [],
      market: [],
      channel: [],
      item: [],
      location: [],
      timeGrain: "month",
    });
    expect(result.current.hasActiveFilters).toBe(false);
  });

  it("hasActiveFilters is false when only timeGrain is non-default", () => {
    const { result } = renderHook(() => useGlobalFilters());

    act(() => {
      result.current.setFilters({ timeGrain: "quarter" });
    });
    // hasActiveFilters only checks brand/category/market/channel arrays
    expect(result.current.hasActiveFilters).toBe(false);
  });

  it("hasActiveFilters is true when any filter array is non-empty", () => {
    const { result } = renderHook(() => useGlobalFilters());

    act(() => {
      result.current.setFilters({ channel: ["Retail"] });
    });
    expect(result.current.hasActiveFilters).toBe(true);
  });

  it("setFilters updates item filter", () => {
    const { result } = renderHook(() => useGlobalFilters());

    act(() => {
      result.current.setFilters({ item: ["100320"] });
    });
    expect(result.current.filters.item).toEqual(["100320"]);
    expect(result.current.hasActiveFilters).toBe(true);
  });

  it("setFilters updates location filter", () => {
    const { result } = renderHook(() => useGlobalFilters());

    act(() => {
      result.current.setFilters({ location: ["1401-BULK"] });
    });
    expect(result.current.filters.location).toEqual(["1401-BULK"]);
    expect(result.current.hasActiveFilters).toBe(true);
  });

  it("hasActiveFilters is true when item array is non-empty", () => {
    const { result } = renderHook(() => useGlobalFilters());

    act(() => {
      result.current.setFilters({ item: ["100320", "100321"] });
    });
    expect(result.current.hasActiveFilters).toBe(true);
  });
});
