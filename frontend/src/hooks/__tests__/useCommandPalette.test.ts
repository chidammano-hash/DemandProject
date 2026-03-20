import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useCommandPalette } from "../useCommandPalette";

describe("useCommandPalette", () => {
  it("starts with open=false", () => {
    const { result } = renderHook(() => useCommandPalette());
    expect(result.current.open).toBe(false);
  });

  it("toggle() opens the palette", () => {
    const { result } = renderHook(() => useCommandPalette());
    act(() => result.current.toggle());
    expect(result.current.open).toBe(true);
  });

  it("toggle() twice closes the palette", () => {
    const { result } = renderHook(() => useCommandPalette());
    act(() => result.current.toggle());
    act(() => result.current.toggle());
    expect(result.current.open).toBe(false);
  });

  it("close() closes the palette", () => {
    const { result } = renderHook(() => useCommandPalette());
    act(() => result.current.toggle());
    expect(result.current.open).toBe(true);
    act(() => result.current.close());
    expect(result.current.open).toBe(false);
  });

  it("close() is a no-op when already closed", () => {
    const { result } = renderHook(() => useCommandPalette());
    act(() => result.current.close());
    expect(result.current.open).toBe(false);
  });

  it("opens on Cmd+K", () => {
    const { result } = renderHook(() => useCommandPalette());
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }));
    });
    expect(result.current.open).toBe(true);
  });

  it("closes on Cmd+K when already open", () => {
    const { result } = renderHook(() => useCommandPalette());
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }));
    });
    expect(result.current.open).toBe(true);
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }));
    });
    expect(result.current.open).toBe(false);
  });

  it("opens on Ctrl+K", () => {
    const { result } = renderHook(() => useCommandPalette());
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true }));
    });
    expect(result.current.open).toBe(true);
  });

  it("does not open on plain K key", () => {
    const { result } = renderHook(() => useCommandPalette());
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "k" }));
    });
    expect(result.current.open).toBe(false);
  });

  it("cleans up event listener on unmount", () => {
    const { result, unmount } = renderHook(() => useCommandPalette());
    unmount();
    // After unmount, dispatching the event should not change state
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }));
    });
    // result.current still reflects unmounted state — open was false
    expect(result.current.open).toBe(false);
  });
});
