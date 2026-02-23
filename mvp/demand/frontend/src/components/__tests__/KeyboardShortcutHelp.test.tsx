import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { KeyboardShortcutHelp } from "@/components/KeyboardShortcutHelp";

describe("KeyboardShortcutHelp", () => {
  it("renders the title", () => {
    render(<KeyboardShortcutHelp onClose={vi.fn()} />);
    expect(screen.getByText("Keyboard Shortcuts")).toBeInTheDocument();
  });

  it("lists all shortcuts", () => {
    render(<KeyboardShortcutHelp onClose={vi.fn()} />);
    expect(screen.getByText("Explorer tab")).toBeInTheDocument();
    expect(screen.getByText("Clusters tab")).toBeInTheDocument();
    expect(screen.getByText("DFU Analysis tab")).toBeInTheDocument();
    expect(screen.getByText("Accuracy tab")).toBeInTheDocument();
    expect(screen.getByText("Market Intel tab")).toBeInTheDocument();
    expect(screen.getByText("Focus search")).toBeInTheDocument();
    expect(screen.getByText("Close panel / help")).toBeInTheDocument();
    expect(screen.getByText("Toggle column fields")).toBeInTheDocument();
  });

  it("calls onClose when backdrop clicked", async () => {
    const onClose = vi.fn();
    const { container } = render(<KeyboardShortcutHelp onClose={onClose} />);
    // Click the backdrop (outermost div)
    const backdrop = container.firstChild as HTMLElement;
    await userEvent.click(backdrop);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not call onClose when content clicked", async () => {
    const onClose = vi.fn();
    render(<KeyboardShortcutHelp onClose={onClose} />);
    // Click on the title (inside the modal content)
    await userEvent.click(screen.getByText("Keyboard Shortcuts"));
    expect(onClose).not.toHaveBeenCalled();
  });

  it("renders kbd elements for keys", () => {
    const { container } = render(<KeyboardShortcutHelp onClose={vi.fn()} />);
    const kbdElements = container.querySelectorAll("kbd");
    expect(kbdElements.length).toBeGreaterThan(10);
  });
});
