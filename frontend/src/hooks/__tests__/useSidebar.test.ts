import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useSidebar } from "@/hooks/useSidebar";

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

describe("useSidebar", () => {
  beforeEach(() => {
    localStorageMock.clear();
  });

  it("default state based on window width - collapsed when < 1440", () => {
    // jsdom default innerWidth is 1024 which is < 1440
    Object.defineProperty(window, "innerWidth", { value: 1024, writable: true });
    const { result } = renderHook(() => useSidebar());
    expect(result.current.collapsed).toBe(true);
  });

  it("default state based on window width - expanded when >= 1440", () => {
    Object.defineProperty(window, "innerWidth", { value: 1440, writable: true });
    localStorageMock.clear(); // ensure no stored value
    const { result } = renderHook(() => useSidebar());
    expect(result.current.collapsed).toBe(false);
  });

  it("toggle works - flips collapsed state", () => {
    Object.defineProperty(window, "innerWidth", { value: 1440, writable: true });
    localStorageMock.clear();
    const { result } = renderHook(() => useSidebar());
    const initialState = result.current.collapsed;

    act(() => result.current.toggle());
    expect(result.current.collapsed).toBe(!initialState);

    act(() => result.current.toggle());
    expect(result.current.collapsed).toBe(initialState);
  });

  it("localStorage persistence - saves collapsed state", () => {
    Object.defineProperty(window, "innerWidth", { value: 1440, writable: true });
    localStorageMock.clear();
    const { result } = renderHook(() => useSidebar());

    // Initial state should be expanded (1440px)
    expect(localStorageMock.getItem("ds-sidebar")).toBe("expanded");

    act(() => result.current.toggle());
    expect(localStorageMock.getItem("ds-sidebar")).toBe("collapsed");

    act(() => result.current.toggle());
    expect(localStorageMock.getItem("ds-sidebar")).toBe("expanded");
  });

  it("reads collapsed state from localStorage", () => {
    localStorageMock.setItem("ds-sidebar", "collapsed");
    const { result } = renderHook(() => useSidebar());
    expect(result.current.collapsed).toBe(true);
  });

  it("reads expanded state from localStorage", () => {
    localStorageMock.setItem("ds-sidebar", "expanded");
    const { result } = renderHook(() => useSidebar());
    expect(result.current.collapsed).toBe(false);
  });

  it("provides mobileOpen state and callbacks", () => {
    const { result } = renderHook(() => useSidebar());
    expect(result.current.mobileOpen).toBe(false);

    act(() => result.current.openMobile());
    expect(result.current.mobileOpen).toBe(true);

    act(() => result.current.closeMobile());
    expect(result.current.mobileOpen).toBe(false);
  });

  it("setCollapsed directly sets collapsed state", () => {
    Object.defineProperty(window, "innerWidth", { value: 1440, writable: true });
    localStorageMock.clear();
    const { result } = renderHook(() => useSidebar());

    act(() => result.current.setCollapsed(true));
    expect(result.current.collapsed).toBe(true);

    act(() => result.current.setCollapsed(false));
    expect(result.current.collapsed).toBe(false);
  });
});
