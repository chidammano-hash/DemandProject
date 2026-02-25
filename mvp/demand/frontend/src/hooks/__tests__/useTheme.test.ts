import { describe, it, expect, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useTheme } from "@/hooks/useTheme";

const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => { store[key] = value; },
    removeItem: (key: string) => { delete store[key]; },
    clear: () => { store = {}; },
    get length() { return Object.keys(store).length; },
    key: (i: number) => Object.keys(store)[i] ?? null,
  };
})();

Object.defineProperty(window, "localStorage", { value: localStorageMock });

describe("useTheme", () => {
  beforeEach(() => {
    localStorageMock.clear();
    document.documentElement.classList.remove("light", "dark", "midnight");
    document.documentElement.removeAttribute("data-transitioning");
    document.documentElement.removeAttribute("data-theme");
  });

  it("defaults to light / general when no localStorage", () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe("light");
    expect(result.current.themeId).toBe("general");
  });

  it("reads product theme from localStorage", () => {
    localStorageMock.setItem("ds-product-theme", "obsidian");
    localStorageMock.setItem("ds-color-mode", "dark");
    const { result } = renderHook(() => useTheme());
    expect(result.current.themeId).toBe("obsidian");
    expect(result.current.theme).toBe("dark");
  });

  it("falls back to general for invalid stored value", () => {
    localStorageMock.setItem("ds-product-theme", "invalid");
    const { result } = renderHook(() => useTheme());
    expect(result.current.themeId).toBe("general");
  });

  it("persists theme to localStorage on change", () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setProductTheme("wine-spirits"));
    expect(localStorageMock.getItem("ds-product-theme")).toBe("wine-spirits");
  });

  it("applies theme class to document root", () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setColorMode("dark"));
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(document.documentElement.classList.contains("light")).toBe(false);
  });

  it("obsidian forces dark mode", () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setProductTheme("obsidian"));
    expect(result.current.colorMode).toBe("dark");
    expect(result.current.effectiveClass).toBe("dark");
    // Trying to set light should be ignored
    act(() => result.current.setColorMode("light"));
    expect(result.current.effectiveClass).toBe("dark");
  });

  it("cycles themes in order", () => {
    const { result } = renderHook(() => useTheme());
    // Default is general
    expect(result.current.themeId).toBe("general");
    act(() => result.current.cycleTheme());
    expect(result.current.themeId).toBe("obsidian");
    act(() => result.current.cycleTheme());
    expect(result.current.themeId).toBe("wine-spirits");
    act(() => result.current.cycleTheme());
    expect(result.current.themeId).toBe("general");
  });

  it("returns trendColors and chartColors", () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.trendColors).toBeDefined();
    expect(Array.isArray(result.current.trendColors)).toBe(true);
    expect(result.current.chartColors).toBeDefined();
    expect(result.current.chartColors.grid).toBeDefined();
  });

  it("sets data-theme attribute", () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setProductTheme("wine-spirits"));
    expect(document.documentElement.getAttribute("data-theme")).toBe("wine-spirits");
  });
});
