/**
 * Tests for the minimal toast system (Gen-4 roadmap UX-7).
 */
import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
import { Toaster, toast, __test__ } from "@/components/Toaster";

function clearAllToasts() {
  for (const t of [...__test__.toastStack]) __test__.dismiss(t.id);
}

describe("Toaster", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    clearAllToasts();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders nothing when no toasts are queued", () => {
    const { container } = render(<Toaster />);
    expect(container.querySelector("[role='status']")).toBeNull();
  });

  it("displays an error toast with matching text", () => {
    render(<Toaster />);
    act(() => {
      toast.error("Save failed.");
    });
    expect(screen.getByText("Save failed.")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite");
  });

  it("stacks multiple toasts in insertion order", () => {
    render(<Toaster />);
    act(() => {
      toast.info("First");
      toast.success("Second");
    });
    const buttons = screen.getAllByRole("button");
    expect(buttons[0]).toHaveTextContent("First");
    expect(buttons[1]).toHaveTextContent("Second");
  });

  it("auto-dismisses toasts after timeout", () => {
    render(<Toaster />);
    act(() => {
      toast.warning("Bye soon");
    });
    expect(screen.getByText("Bye soon")).toBeInTheDocument();
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(screen.queryByText("Bye soon")).toBeNull();
  });

  it("dismisses on click", () => {
    render(<Toaster />);
    act(() => {
      toast.error("click me");
    });
    const btn = screen.getByText("click me");
    act(() => {
      btn.click();
    });
    expect(screen.queryByText("click me")).toBeNull();
  });
});
