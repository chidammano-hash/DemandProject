import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { usePanelToggles } from "@/hooks/usePanelToggles";

const STORAGE_KEY = "test:panels";
const DEFAULTS = { overlay: true, shap: true, variability: false };

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

describe("usePanelToggles", () => {
  beforeEach(() => {
    // Clean our storage key specifically
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* no-op */ }
  });

  it("returns defaults when no stored state", () => {
    const { result } = renderHook(() => usePanelToggles(STORAGE_KEY, DEFAULTS));
    expect(result.current.panels).toEqual(DEFAULTS);
  });

  it("reads from localStorage on mount", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ overlay: false, shap: true, variability: true }));
    const { result } = renderHook(() => usePanelToggles(STORAGE_KEY, DEFAULTS));
    expect(result.current.panels.overlay).toBe(false);
    expect(result.current.panels.variability).toBe(true);
  });

  it("merges stored state with defaults for new keys", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ overlay: false }));
    const { result } = renderHook(() => usePanelToggles(STORAGE_KEY, DEFAULTS));
    expect(result.current.panels.overlay).toBe(false);
    expect(result.current.panels.shap).toBe(true);
    expect(result.current.panels.variability).toBe(false);
  });

  it("toggle flips a panel and persists to localStorage", () => {
    const { result } = renderHook(() => usePanelToggles(STORAGE_KEY, DEFAULTS));
    expect(result.current.panels.overlay).toBe(true);

    act(() => result.current.toggle("overlay"));
    expect(result.current.panels.overlay).toBe(false);

    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY)!);
    expect(stored.overlay).toBe(false);
  });

  it("toggle twice returns to original state", () => {
    const { result } = renderHook(() => usePanelToggles(STORAGE_KEY, DEFAULTS));

    act(() => result.current.toggle("variability"));
    expect(result.current.panels.variability).toBe(true);

    act(() => result.current.toggle("variability"));
    expect(result.current.panels.variability).toBe(false);
  });

  it("resetDefaults restores defaults and clears storage", () => {
    const { result } = renderHook(() => usePanelToggles(STORAGE_KEY, DEFAULTS));

    act(() => result.current.toggle("overlay"));
    expect(result.current.panels.overlay).toBe(false);

    act(() => result.current.resetDefaults());
    expect(result.current.panels).toEqual(DEFAULTS);
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("handles corrupt localStorage gracefully", () => {
    localStorage.setItem(STORAGE_KEY, "not valid json");
    const { result } = renderHook(() => usePanelToggles(STORAGE_KEY, DEFAULTS));
    expect(result.current.panels).toEqual(DEFAULTS);
  });
});
