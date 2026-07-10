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

  it("lists live tab labels (U6.1 — no retired pre-restructure names)", () => {
    render(<KeyboardShortcutHelp onClose={vi.fn()} />);
    // Numeric rows are derived from NAV_ITEMS — show the current IA labels.
    expect(screen.getByText("Command Center")).toBeInTheDocument();
    expect(screen.getByText("Portfolio")).toBeInTheDocument();
    expect(screen.getByText("Item Analysis")).toBeInTheDocument();
    expect(screen.getByText("Inv. Planning")).toBeInTheDocument();
    expect(screen.getByText("S&OP")).toBeInTheDocument();
    expect(screen.getByText("Workflows")).toBeInTheDocument();
    // global shortcuts
    expect(screen.getByText("Focus search")).toBeInTheDocument();
    expect(screen.getByText("Close panel / help")).toBeInTheDocument();
    expect(screen.getByText("Toggle column fields")).toBeInTheDocument();
    // retired labels must NOT appear
    expect(screen.queryByText("Control Tower")).not.toBeInTheDocument();
    expect(screen.queryByText("AI Planner")).not.toBeInTheDocument();
    expect(screen.queryByText("DFU Analysis")).not.toBeInTheDocument();
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
