import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useUndoable, __fireUndoForTest } from "@/hooks/useUndoable";
import { toast } from "@/components/Toaster";

vi.mock("@/components/Toaster", () => ({
  toast: {
    info: vi.fn(),
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    dismiss: vi.fn(),
  },
}));

describe("useUndoable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("surfaces an undo-capable info toast with the given message", () => {
    const { result } = renderHook(() => useUndoable());
    act(() => {
      result.current("Marked resolved", vi.fn());
    });
    expect(toast.info).toHaveBeenCalledWith(
      expect.stringContaining("Marked resolved"),
    );
    expect(toast.info).toHaveBeenCalledWith(
      expect.stringContaining("click to undo"),
    );
  });

  it("invokes the undo callback and shows a success toast when fired", () => {
    const onUndo = vi.fn();
    const { result } = renderHook(() => useUndoable());
    let key = "";
    act(() => {
      key = result.current("Deleted", onUndo);
    });
    expect(onUndo).not.toHaveBeenCalled();
    act(() => {
      __fireUndoForTest(key);
    });
    expect(onUndo).toHaveBeenCalledTimes(1);
    expect(toast.success).toHaveBeenCalledWith("Undone.");
  });

  it("does NOT double-invoke onUndo if the event fires twice", () => {
    const onUndo = vi.fn();
    const { result } = renderHook(() => useUndoable());
    let key = "";
    act(() => {
      key = result.current("Ack", onUndo);
    });
    act(() => {
      __fireUndoForTest(key);
    });
    // Second fire after listener removal should have no effect.
    act(() => {
      __fireUndoForTest(key);
    });
    expect(onUndo).toHaveBeenCalledTimes(1);
  });
});
