import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useTheme } from "@/hooks/useTheme";

// Provide a localStorage mock for jsdom environments where it may not support .clear()
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
  });

  it("defaults to light when no localStorage", () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe("light");
  });

  it("reads theme from localStorage", () => {
    localStorageMock.setItem("ds-theme", "dark");
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe("dark");
  });

  it("falls back to light for invalid stored value", () => {
    localStorageMock.setItem("ds-theme", "invalid");
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe("light");
  });

  it("persists theme to localStorage on change", () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setTheme("midnight"));
    expect(localStorageMock.getItem("ds-theme")).toBe("midnight");
  });

  it("applies theme class to document root", () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setTheme("dark"));
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(document.documentElement.classList.contains("light")).toBe(false);
  });

  it("removes previous theme class", () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setTheme("dark"));
    act(() => result.current.setTheme("midnight"));
    expect(document.documentElement.classList.contains("midnight")).toBe(true);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });

  it("returns trendColors and chartColors", () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.trendColors).toBeDefined();
    expect(result.current.chartColors).toBeDefined();
  });
});
