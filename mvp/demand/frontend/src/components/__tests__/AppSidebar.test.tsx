import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { AppSidebar, NAV_ITEMS } from "@/components/AppSidebar";
import { JobNotificationProvider } from "@/context/JobNotificationContext";

function renderSidebar(props = {}) {
  const defaultProps = {
    activeTab: "overview",
    onNavigate: vi.fn(),
    collapsed: false,
    onToggle: vi.fn(),
    appName: "Demand Studio",
    ...props,
  };
  return render(
    <JobNotificationProvider>
      <AppSidebar {...defaultProps} />
    </JobNotificationProvider>
  );
}

describe("AppSidebar", () => {
  it("renders all nav items", () => {
    renderSidebar();
    // Each nav item renders as a button with the item label text
    for (const item of NAV_ITEMS) {
      expect(screen.getByText(item.label)).toBeInTheDocument();
    }
    expect(NAV_ITEMS.length).toBe(14);
  });

  it("active item has aria-current='page'", () => {
    renderSidebar({ activeTab: "explorer" });
    const activeButton = screen.getByText("Explorer").closest("button");
    expect(activeButton).toHaveAttribute("aria-current", "page");

    // Non-active items should not have aria-current
    const overviewButton = screen.getByText("Overview").closest("button");
    expect(overviewButton).not.toHaveAttribute("aria-current");
  });

  it("collapsed state hides labels", () => {
    renderSidebar({ collapsed: true });
    expect(screen.queryByText("Demand Studio")).not.toBeInTheDocument();
    // The nav item labels should not be rendered as visible text
    expect(screen.queryByText("Overview")).not.toBeInTheDocument();
    expect(screen.queryByText("Explorer")).not.toBeInTheDocument();
  });

  it("toggle button calls onToggle", () => {
    const onToggle = vi.fn();
    renderSidebar({ onToggle });
    const toggleBtn = screen.getByLabelText("Collapse sidebar");
    fireEvent.click(toggleBtn);
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it("toggle button shows 'Expand sidebar' when collapsed", () => {
    renderSidebar({ collapsed: true });
    expect(screen.getByLabelText("Expand sidebar")).toBeInTheDocument();
  });

  it("navigation calls onNavigate with item key", () => {
    const onNavigate = vi.fn();
    renderSidebar({ onNavigate, collapsed: false });
    fireEvent.click(screen.getByText("Explorer"));
    expect(onNavigate).toHaveBeenCalledWith("explorer");
  });

  it("navigation calls onNavigate for each item", () => {
    const onNavigate = vi.fn();
    renderSidebar({ onNavigate, collapsed: false });
    fireEvent.click(screen.getByText("Accuracy"));
    expect(onNavigate).toHaveBeenCalledWith("accuracy");
    fireEvent.click(screen.getByText("Chat"));
    expect(onNavigate).toHaveBeenCalledWith("chat");
  });

  it("renders app name when not collapsed", () => {
    renderSidebar({ appName: "Test App" });
    expect(screen.getByText("Test App")).toBeInTheDocument();
  });

  it("renders theme footer when provided", () => {
    renderSidebar({ themeFooter: <div data-testid="theme-footer">Footer</div> });
    expect(screen.getByTestId("theme-footer")).toBeInTheDocument();
  });

  it("does not render theme footer when not provided", () => {
    renderSidebar();
    expect(screen.queryByTestId("theme-footer")).not.toBeInTheDocument();
  });

  it("renders mobile toggle button", () => {
    renderSidebar();
    expect(screen.getByLabelText("Toggle navigation")).toBeInTheDocument();
  });
});
