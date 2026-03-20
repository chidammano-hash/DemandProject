import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { CommandPalette } from "../CommandPalette";

// jsdom doesn't implement scrollIntoView
Element.prototype.scrollIntoView = vi.fn();

vi.mock("../AppSidebar", () => ({
  NAV_ITEMS: [
    { key: "commandCenter", label: "Command Center", icon: () => null, section: "command", shortcut: "1" },
    { key: "aggregateAnalysis", label: "Portfolio", icon: () => null, section: "demand", shortcut: "2" },
    { key: "itemAnalysis", label: "Item Analysis", icon: () => null, section: "demand", shortcut: "3" },
    { key: "invPlanning", label: "Inv. Planning", icon: () => null, section: "supply", shortcut: "4" },
  ],
}));

describe("CommandPalette", () => {
  const defaultProps = {
    open: true,
    onClose: vi.fn(),
    onNavigate: vi.fn(),
    activeTab: "commandCenter",
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders dialog when open", () => {
    render(<CommandPalette {...defaultProps} />);
    expect(screen.getByRole("dialog")).toBeDefined();
  });

  it("does not render when closed", () => {
    render(<CommandPalette {...defaultProps} open={false} />);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("renders search input", () => {
    render(<CommandPalette {...defaultProps} />);
    expect(screen.getByLabelText("Search commands")).toBeDefined();
  });

  it("shows all nav items when no search query", () => {
    render(<CommandPalette {...defaultProps} />);
    expect(screen.getByText("Command Center")).toBeDefined();
    expect(screen.getByText("Portfolio")).toBeDefined();
    expect(screen.getByText("Item Analysis")).toBeDefined();
    expect(screen.getByText("Inv. Planning")).toBeDefined();
  });

  it("shows quick action items", () => {
    render(<CommandPalette {...defaultProps} />);
    expect(screen.getByText("Toggle dark mode")).toBeDefined();
    expect(screen.getByText("Toggle sidebar")).toBeDefined();
    expect(screen.getByText("Keyboard shortcuts")).toBeDefined();
  });

  it("filters items by search query", () => {
    render(<CommandPalette {...defaultProps} />);
    const input = screen.getByLabelText("Search commands");
    fireEvent.change(input, { target: { value: "portfolio" } });
    expect(screen.getByText("Portfolio")).toBeDefined();
    expect(screen.queryByText("Item Analysis")).toBeNull();
    expect(screen.queryByText("Inv. Planning")).toBeNull();
  });

  it("shows no results message for unmatched query", () => {
    render(<CommandPalette {...defaultProps} />);
    const input = screen.getByLabelText("Search commands");
    fireEvent.change(input, { target: { value: "zzzznonexistent" } });
    expect(screen.getByText("No results found.")).toBeDefined();
  });

  it("calls onClose on backdrop click", () => {
    render(<CommandPalette {...defaultProps} />);
    fireEvent.click(screen.getByTestId("command-palette-backdrop"));
    expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
  });

  it("calls onNavigate when a navigation item is clicked", () => {
    render(<CommandPalette {...defaultProps} />);
    fireEvent.click(screen.getByText("Portfolio"));
    expect(defaultProps.onNavigate).toHaveBeenCalledWith("aggregateAnalysis");
  });

  it("calls onClose after executing a navigation item", () => {
    render(<CommandPalette {...defaultProps} />);
    fireEvent.click(screen.getByText("Portfolio"));
    expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
  });

  it("calls onToggleDarkMode for dark mode action", () => {
    const onToggleDarkMode = vi.fn();
    render(<CommandPalette {...defaultProps} onToggleDarkMode={onToggleDarkMode} />);
    fireEvent.click(screen.getByText("Toggle dark mode"));
    expect(onToggleDarkMode).toHaveBeenCalledTimes(1);
  });

  it("calls onToggleSidebar for sidebar action", () => {
    const onToggleSidebar = vi.fn();
    render(<CommandPalette {...defaultProps} onToggleSidebar={onToggleSidebar} />);
    fireEvent.click(screen.getByText("Toggle sidebar"));
    expect(onToggleSidebar).toHaveBeenCalledTimes(1);
  });

  it("calls onShowKeyboardHelp for keyboard shortcuts action", () => {
    const onShowKeyboardHelp = vi.fn();
    render(<CommandPalette {...defaultProps} onShowKeyboardHelp={onShowKeyboardHelp} />);
    fireEvent.click(screen.getByText("Keyboard shortcuts"));
    expect(onShowKeyboardHelp).toHaveBeenCalledTimes(1);
  });

  it("marks active tab with (current) indicator", () => {
    render(<CommandPalette {...defaultProps} activeTab="commandCenter" />);
    expect(screen.getByText("(current)")).toBeDefined();
  });

  it("renders keyboard shortcut badges", () => {
    render(<CommandPalette {...defaultProps} />);
    expect(screen.getByText("1")).toBeDefined();
    expect(screen.getByText("2")).toBeDefined();
    expect(screen.getByText("3")).toBeDefined();
  });

  it("renders footer hints", () => {
    render(<CommandPalette {...defaultProps} />);
    expect(screen.getByText("to select")).toBeDefined();
    expect(screen.getByText("esc to close")).toBeDefined();
  });

  it("filters by keywords", () => {
    render(<CommandPalette {...defaultProps} />);
    const input = screen.getByLabelText("Search commands");
    fireEvent.change(input, { target: { value: "dark" } });
    expect(screen.getByText("Toggle dark mode")).toBeDefined();
    // Nav items without "dark" keyword should be gone
    expect(screen.queryByText("Command Center")).toBeNull();
  });
});
