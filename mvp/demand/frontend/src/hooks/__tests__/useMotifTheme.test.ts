import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";

// Import the boot file to register all 5 motifs before any test runs.
import "@/constants/motifs";
import { useMotifTheme } from "@/hooks/useMotifTheme";

// Provide a localStorage mock (same pattern as useTheme.test.ts)
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
    get length() {
      return Object.keys(store).length;
    },
    key: (i: number) => Object.keys(store)[i] ?? null,
  };
})();

Object.defineProperty(window, "localStorage", { value: localStorageMock });

describe("useMotifTheme", () => {
  beforeEach(() => {
    localStorageMock.clear();
    document.documentElement.removeAttribute("data-motif");
  });

  it('defaults to "periodic" when no localStorage', () => {
    const { result } = renderHook(() => useMotifTheme());
    expect(result.current.motifId).toBe("periodic");
    expect(result.current.motifConfig.id).toBe("periodic");
  });

  it("reads motif from localStorage", () => {
    localStorageMock.setItem("ds-motif", "spirits");
    const { result } = renderHook(() => useMotifTheme());
    expect(result.current.motifId).toBe("spirits");
    expect(result.current.motifConfig.displayName).toBe("The Cellar");
  });

  it('falls back to "periodic" for invalid stored value', () => {
    localStorageMock.setItem("ds-motif", "nonexistent");
    const { result } = renderHook(() => useMotifTheme());
    expect(result.current.motifId).toBe("periodic");
  });

  it("persists motif to localStorage on change", () => {
    const { result } = renderHook(() => useMotifTheme());
    act(() => result.current.setMotif("f1"));
    expect(localStorageMock.getItem("ds-motif")).toBe("f1");
  });

  it("sets data-motif attribute on document root", () => {
    const { result } = renderHook(() => useMotifTheme());
    // Initial render sets the attribute
    expect(document.documentElement.getAttribute("data-motif")).toBe(
      "periodic",
    );
    act(() => result.current.setMotif("zen"));
    expect(document.documentElement.getAttribute("data-motif")).toBe("zen");
  });

  it("cycleMotif cycles through all motifs", () => {
    const { result } = renderHook(() => useMotifTheme());
    // Start at "periodic". Cycling should advance to the next registered motif.
    const initialId = result.current.motifId;
    expect(initialId).toBe("periodic");

    act(() => result.current.cycleMotif());
    const secondId = result.current.motifId;
    expect(secondId).not.toBe(initialId);

    // Keep cycling through all 5 to return to "periodic"
    // We are at index 1 after 1 cycle. Need 4 more to wrap: 1->2->3->4->0
    act(() => result.current.cycleMotif());
    act(() => result.current.cycleMotif());
    act(() => result.current.cycleMotif());
    act(() => result.current.cycleMotif());
    // After 5 total cycles we should be back at the start
    expect(result.current.motifId).toBe(initialId);
  });

  it("getTile returns correct tile for tab key", () => {
    const { result } = renderHook(() => useMotifTheme());
    const tile = result.current.getTile("explorer");
    expect(tile).toBeDefined();
    expect(tile.primary).toBe("Dx"); // periodic motif explorer tile
    expect(tile.label).toBe("Explorer");
  });

  it("getTile returns previewTile for unknown tab key", () => {
    const { result } = renderHook(() => useMotifTheme());
    const tile = result.current.getTile("nonexistent_tab_key");
    const previewTile = result.current.motifConfig.previewTile;
    expect(tile).toEqual(previewTile);
  });

  it("setMotif ignores invalid motif IDs", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { result } = renderHook(() => useMotifTheme());
    act(() => result.current.setMotif("bogus" as any));
    expect(result.current.motifId).toBe("periodic");
    warnSpy.mockRestore();
  });
});
