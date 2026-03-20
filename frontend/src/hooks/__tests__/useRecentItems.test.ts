import { describe, it, expect, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useRecentItems } from "../useRecentItems";

// Provide a minimal localStorage mock if not present
const store: Record<string, string> = {};
if (typeof globalThis.localStorage === "undefined" || typeof globalThis.localStorage.getItem !== "function") {
  Object.defineProperty(globalThis, "localStorage", {
    value: {
      getItem: (key: string) => store[key] ?? null,
      setItem: (key: string, value: string) => { store[key] = value; },
      removeItem: (key: string) => { delete store[key]; },
      clear: () => { for (const k of Object.keys(store)) delete store[k]; },
    },
    writable: true,
  });
}

describe("useRecentItems", () => {
  beforeEach(() => {
    localStorage.removeItem("demand-studio-recent-items");
  });

  it("starts empty", () => {
    const { result } = renderHook(() => useRecentItems());
    expect(result.current.recentItems).toHaveLength(0);
  });

  it("adds an item", () => {
    const { result } = renderHook(() => useRecentItems());
    act(() => result.current.addRecentItem({ itemNo: "100", label: "100" }));
    expect(result.current.recentItems).toHaveLength(1);
    expect(result.current.recentItems[0].itemNo).toBe("100");
  });

  it("deduplicates", () => {
    const { result } = renderHook(() => useRecentItems());
    act(() => result.current.addRecentItem({ itemNo: "100", label: "100" }));
    act(() => result.current.addRecentItem({ itemNo: "100", label: "100" }));
    expect(result.current.recentItems).toHaveLength(1);
  });

  it("limits to 5", () => {
    const { result } = renderHook(() => useRecentItems());
    for (let i = 0; i < 7; i++) {
      act(() => result.current.addRecentItem({ itemNo: String(i), label: String(i) }));
    }
    expect(result.current.recentItems).toHaveLength(5);
  });

  it("clears items", () => {
    const { result } = renderHook(() => useRecentItems());
    act(() => result.current.addRecentItem({ itemNo: "100", label: "100" }));
    act(() => result.current.clearRecentItems());
    expect(result.current.recentItems).toHaveLength(0);
  });
});
