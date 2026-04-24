import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { KeyboardShortcutHelp } from "@/components/KeyboardShortcutHelp";

describe("KeyboardShortcutHelp", () => {
  it("renders as an accessible dialog with a title", () => {
    render(<KeyboardShortcutHelp onClose={vi.fn()} />);
    const dialog = screen.getByRole("dialog");
    expect(dialog).toBeInTheDocument();
    expect(screen.getByText("Keyboard Shortcuts")).toBeInTheDocument();
  });

  it("lists all shortcuts", () => {
    render(<KeyboardShortcutHelp onClose={vi.fn()} />);
    expect(screen.getByText("AI Planner")).toBeInTheDocument();
    expect(screen.getByText("Control Tower")).toBeInTheDocument();
    expect(screen.getByText("DFU Analysis")).toBeInTheDocument();
    expect(screen.getByText("Accuracy")).toBeInTheDocument();
    expect(screen.getByText("Inv. Planning")).toBeInTheDocument();
    expect(screen.getByText("Focus search")).toBeInTheDocument();
    expect(screen.getByText("Close panel / help")).toBeInTheDocument();
    expect(screen.getByText("Toggle column fields")).toBeInTheDocument();
  });

  it("calls onClose when Escape is pressed (Radix Dialog default)", async () => {
    const onClose = vi.fn();
    render(<KeyboardShortcutHelp onClose={onClose} />);
    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("renders kbd elements for keys", () => {
    render(<KeyboardShortcutHelp onClose={vi.fn()} />);
    // Radix renders into a portal — query the document, not the test container.
    const kbdElements = document.querySelectorAll("kbd");
    expect(kbdElements.length).toBeGreaterThan(10);
  });
});
