import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useTabVisibility } from "@/hooks/useTabVisibility";

describe("useTabVisibility", () => {
  // Save and restore the original document.hidden descriptor
  const originalDescriptor = Object.getOwnPropertyDescriptor(Document.prototype, "hidden")!;

  let hiddenValue = false;

  beforeEach(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => hiddenValue,
    });
    hiddenValue = false;
  });

  afterEach(() => {
    Object.defineProperty(document, "hidden", originalDescriptor);
  });

  it("returns true when the tab is initially visible", () => {
    hiddenValue = false;
    const { result } = renderHook(() => useTabVisibility());
    expect(result.current).toBe(true);
  });

  it("returns false when the tab is initially hidden", () => {
    hiddenValue = true;
    const { result } = renderHook(() => useTabVisibility());
    expect(result.current).toBe(false);
  });

  it("updates to false when the tab becomes hidden", () => {
    hiddenValue = false;
    const { result } = renderHook(() => useTabVisibility());
    expect(result.current).toBe(true);

    act(() => {
      hiddenValue = true;
      document.dispatchEvent(new Event("visibilitychange"));
    });

    expect(result.current).toBe(false);
  });

  it("updates to true when the tab becomes visible again", () => {
    hiddenValue = true;
    const { result } = renderHook(() => useTabVisibility());
    expect(result.current).toBe(false);

    act(() => {
      hiddenValue = false;
      document.dispatchEvent(new Event("visibilitychange"));
    });

    expect(result.current).toBe(true);
  });

  it("removes the event listener on unmount", () => {
    const removeSpy = vi.spyOn(document, "removeEventListener");
    const { unmount } = renderHook(() => useTabVisibility());
    unmount();
    expect(removeSpy).toHaveBeenCalledWith("visibilitychange", expect.any(Function));
    removeSpy.mockRestore();
  });
});
