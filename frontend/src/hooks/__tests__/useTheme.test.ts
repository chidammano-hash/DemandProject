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
    document.documentElement.classList.remove("light", "soft", "dark");
    document.documentElement.removeAttribute("data-transitioning");
    document.documentElement.removeAttribute("data-theme");
    document.documentElement.removeAttribute("data-mode");
  });

  it("defaults to light / general when no localStorage", () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe("light");
    expect(result.current.themeId).toBe("general");
  });

  it("reads color mode from localStorage", () => {
    localStorageMock.setItem("ds-color-mode", "dark");
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe("dark");
    expect(result.current.colorMode).toBe("dark");
  });

  it("reads soft color mode from localStorage", () => {
    localStorageMock.setItem("ds-color-mode", "soft");
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe("soft");
    expect(result.current.colorMode).toBe("soft");
  });

  it("falls back to light for invalid stored color mode", () => {
    localStorageMock.setItem("ds-color-mode", "invalid");
    const { result } = renderHook(() => useTheme());
    expect(result.current.colorMode).toBe("light");
  });

  it("persists color mode to localStorage on change", () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setColorMode("dark"));
    expect(localStorageMock.getItem("ds-color-mode")).toBe("dark");
  });

  it("persists soft mode to localStorage", () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setColorMode("soft"));
    expect(localStorageMock.getItem("ds-color-mode")).toBe("soft");
  });

  it("applies theme class to document root", () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setColorMode("dark"));
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(document.documentElement.classList.contains("light")).toBe(false);
    expect(document.documentElement.classList.contains("soft")).toBe(false);
    expect(document.documentElement.getAttribute("data-mode")).toBe("dark");
  });

  it("soft mode applies light + soft CSS classes (not dark)", () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setColorMode("soft"));
    expect(document.documentElement.classList.contains("light")).toBe(true);
    expect(document.documentElement.classList.contains("soft")).toBe(true);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(document.documentElement.getAttribute("data-mode")).toBe("soft");
  });

  it("light mode does not carry the soft class", () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.setColorMode("soft"));
    act(() => result.current.setColorMode("light"));
    expect(document.documentElement.classList.contains("light")).toBe(true);
    expect(document.documentElement.classList.contains("soft")).toBe(false);
    expect(document.documentElement.getAttribute("data-mode")).toBe("light");
  });

  it("toggleColorMode cycles light → soft → dark → light", () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.colorMode).toBe("light");
    act(() => result.current.toggleColorMode());
    expect(result.current.colorMode).toBe("soft");
    act(() => result.current.toggleColorMode());
    expect(result.current.colorMode).toBe("dark");
    act(() => result.current.toggleColorMode());
    expect(result.current.colorMode).toBe("light");
  });

  it("returns trendColors and chartColors", () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.trendColors).toBeDefined();
    expect(Array.isArray(result.current.trendColors)).toBe(true);
    expect(result.current.chartColors).toBeDefined();
    expect(result.current.chartColors.grid).toBeDefined();
  });

  it("sets data-theme attribute to general", () => {
    renderHook(() => useTheme());
    expect(document.documentElement.getAttribute("data-theme")).toBe("general");
  });

  it("effectiveClass is light for both light and soft modes", () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.effectiveClass).toBe("light");
    act(() => result.current.setColorMode("soft"));
    expect(result.current.effectiveClass).toBe("light");
    act(() => result.current.setColorMode("dark"));
    expect(result.current.effectiveClass).toBe("dark");
  });
});
