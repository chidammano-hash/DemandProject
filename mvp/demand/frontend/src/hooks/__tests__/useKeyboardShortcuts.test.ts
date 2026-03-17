import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";

function fireKey(key: string, opts: Partial<KeyboardEventInit> = {}) {
  window.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, ...opts }));
}

describe("useKeyboardShortcuts", () => {
  const defaultConfig = {
    onTabSwitch: vi.fn(),
    onFocusSearch: vi.fn(),
    onClosePanel: vi.fn(),
    onPrevPage: vi.fn(),
    onNextPage: vi.fn(),
    onToggleFields: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("switches tabs with 1-9 keys", () => {
    renderHook(() => useKeyboardShortcuts(defaultConfig));

    act(() => fireKey("1"));
    expect(defaultConfig.onTabSwitch).toHaveBeenCalledWith("aiPlanner");

    act(() => fireKey("2"));
    expect(defaultConfig.onTabSwitch).toHaveBeenCalledWith("controlTower");

    act(() => fireKey("3"));
    expect(defaultConfig.onTabSwitch).toHaveBeenCalledWith("aggregateAnalysis");

    act(() => fireKey("4"));
    expect(defaultConfig.onTabSwitch).toHaveBeenCalledWith("itemAnalysis");

    act(() => fireKey("5"));
    expect(defaultConfig.onTabSwitch).toHaveBeenCalledWith("exceptions");

    act(() => fireKey("6"));
    expect(defaultConfig.onTabSwitch).toHaveBeenCalledWith("invPlanning");

    act(() => fireKey("7"));
    expect(defaultConfig.onTabSwitch).toHaveBeenCalledWith("jobs");
  });

  it("focuses search on /", () => {
    renderHook(() => useKeyboardShortcuts(defaultConfig));
    act(() => fireKey("/"));
    expect(defaultConfig.onFocusSearch).toHaveBeenCalled();
  });

  it("calls onClosePanel on Escape", () => {
    renderHook(() => useKeyboardShortcuts(defaultConfig));
    act(() => fireKey("Escape"));
    expect(defaultConfig.onClosePanel).toHaveBeenCalled();
  });

  it("navigates pages with arrow keys", () => {
    renderHook(() => useKeyboardShortcuts(defaultConfig));
    act(() => fireKey("ArrowLeft"));
    expect(defaultConfig.onPrevPage).toHaveBeenCalled();

    act(() => fireKey("ArrowRight"));
    expect(defaultConfig.onNextPage).toHaveBeenCalled();
  });

  it("toggles fields with Ctrl+E", () => {
    renderHook(() => useKeyboardShortcuts(defaultConfig));
    act(() => fireKey("e", { ctrlKey: true }));
    expect(defaultConfig.onToggleFields).toHaveBeenCalled();
  });

  it("toggles help with ?", () => {
    const { result } = renderHook(() => useKeyboardShortcuts(defaultConfig));
    expect(result.current.showHelp).toBe(false);

    act(() => fireKey("?"));
    expect(result.current.showHelp).toBe(true);

    act(() => fireKey("?"));
    expect(result.current.showHelp).toBe(false);
  });

  it("closes help with Escape", () => {
    const { result } = renderHook(() => useKeyboardShortcuts(defaultConfig));

    // Open help first
    act(() => fireKey("?"));
    expect(result.current.showHelp).toBe(true);

    // Escape closes it
    act(() => fireKey("Escape"));
    expect(result.current.showHelp).toBe(false);
  });

  it("closeHelp callback works", () => {
    const { result } = renderHook(() => useKeyboardShortcuts(defaultConfig));
    act(() => fireKey("?"));
    expect(result.current.showHelp).toBe(true);

    act(() => result.current.closeHelp());
    expect(result.current.showHelp).toBe(false);
  });
});
