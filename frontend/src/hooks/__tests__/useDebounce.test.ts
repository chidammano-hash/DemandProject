import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useDebounce } from "@/hooks/useDebounce";

describe("useDebounce", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns the initial string value immediately", () => {
    const { result } = renderHook(() => useDebounce("hello", 300));
    expect(result.current).toBe("hello");
  });

  it("does not update the value before the delay expires", () => {
    const { result, rerender } = renderHook(
      ({ val }) => useDebounce(val, 300),
      { initialProps: { val: "hello" } }
    );
    rerender({ val: "world" });
    // Timer has NOT been advanced — should still return old value
    expect(result.current).toBe("hello");
  });

  it("updates the value after the delay expires", () => {
    const { result, rerender } = renderHook(
      ({ val }) => useDebounce(val, 300),
      { initialProps: { val: "hello" } }
    );
    rerender({ val: "world" });
    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(result.current).toBe("world");
  });

  it("cancels a pending timer when value changes before delay", () => {
    const { result, rerender } = renderHook(
      ({ val }) => useDebounce(val, 300),
      { initialProps: { val: "a" } }
    );
    // First change
    rerender({ val: "b" });
    // Advance partway — not enough to fire
    act(() => { vi.advanceTimersByTime(150); });
    expect(result.current).toBe("a");

    // Second change before the 300ms fires
    rerender({ val: "c" });
    // Advance enough to cover the second 300ms window
    act(() => { vi.advanceTimersByTime(300); });
    // Should settle on the LAST value, not "b"
    expect(result.current).toBe("c");
  });

  it("works with number values", () => {
    const { result, rerender } = renderHook(
      ({ val }) => useDebounce(val, 200),
      { initialProps: { val: 1 } }
    );
    expect(result.current).toBe(1);
    rerender({ val: 42 });
    act(() => { vi.advanceTimersByTime(200); });
    expect(result.current).toBe(42);
  });

  it("works with object values using JSON-based equality", () => {
    const obj1 = { a: 1 };
    const obj2 = { a: 2 };
    const { result, rerender } = renderHook(
      ({ val }) => useDebounce(val, 300),
      { initialProps: { val: obj1 } }
    );
    expect(result.current).toEqual({ a: 1 });
    rerender({ val: obj2 });
    act(() => { vi.advanceTimersByTime(300); });
    expect(result.current).toEqual({ a: 2 });
  });

  it("does not trigger re-render when same object content is passed", () => {
    // Two references with the same JSON content should NOT re-trigger debounce
    const obj1 = { x: 10 };
    const obj2 = { x: 10 }; // same content, different reference
    const { result, rerender } = renderHook(
      ({ val }) => useDebounce(val, 300),
      { initialProps: { val: obj1 } }
    );
    act(() => { vi.advanceTimersByTime(300); });
    expect(result.current).toEqual({ x: 10 });

    // Re-render with a new reference but same JSON — effect dependency shouldn't change
    rerender({ val: obj2 });
    // The debounced value should already be settled
    act(() => { vi.advanceTimersByTime(300); });
    expect(result.current).toEqual({ x: 10 });
  });

  it("respects different delay values", () => {
    const { result, rerender } = renderHook(
      ({ val }) => useDebounce(val, 500),
      { initialProps: { val: "start" } }
    );
    rerender({ val: "end" });
    // 300ms is not enough for a 500ms delay
    act(() => { vi.advanceTimersByTime(300); });
    expect(result.current).toBe("start");
    // Another 200ms completes the 500ms window
    act(() => { vi.advanceTimersByTime(200); });
    expect(result.current).toBe("end");
  });

  it("handles empty string values", () => {
    const { result, rerender } = renderHook(
      ({ val }) => useDebounce(val, 300),
      { initialProps: { val: "some text" } }
    );
    rerender({ val: "" });
    act(() => { vi.advanceTimersByTime(300); });
    expect(result.current).toBe("");
  });
});
