import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { AppSidebar, NAV_ITEMS } from "@/components/AppSidebar";
import { JobNotificationProvider } from "@/context/JobNotificationContext";

function renderSidebar(props = {}) {
  const defaultProps = {
    activeTab: "aggregateAnalysis",
    onNavigate: vi.fn(),
    collapsed: false,
    onToggle: vi.fn(),
    appName: "Vrantis",
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
    expect(NAV_ITEMS.length).toBe(18);
  });

  it("sections appear in Tower -> Demand -> Supply -> Operations -> System order (UX-1)", () => {
    // First occurrence of each section in NAV_ITEMS.
    const firstIndexBySection: Record<string, number> = {};
    NAV_ITEMS.forEach((item, i) => {
      if (firstIndexBySection[item.section] === undefined) {
        firstIndexBySection[item.section] = i;
      }
    });
    expect(firstIndexBySection.tower).toBeLessThan(firstIndexBySection.demand);
    expect(firstIndexBySection.demand).toBeLessThan(firstIndexBySection.supply);
    expect(firstIndexBySection.supply).toBeLessThan(firstIndexBySection.operations);
    expect(firstIndexBySection.operations).toBeLessThan(firstIndexBySection.system);
  });

  it("active item has aria-current='page'", () => {
    renderSidebar({ activeTab: "explorer" });
    const activeButton = screen.getByText("Explorer").closest("button");
    expect(activeButton).toHaveAttribute("aria-current", "page");

    // Non-active items should not have aria-current
    const aggButton = screen.getByText("Portfolio").closest("button");
    expect(aggButton).not.toHaveAttribute("aria-current");
  });

  it("collapsed state hides labels", () => {
    renderSidebar({ collapsed: true });
    expect(screen.queryByText("Vrantis")).not.toBeInTheDocument();
    // The nav item labels should not be rendered as visible text
    expect(screen.queryByText("Portfolio")).not.toBeInTheDocument();
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
    fireEvent.click(screen.getByText("Portfolio"));
    expect(onNavigate).toHaveBeenCalledWith("aggregateAnalysis");
    fireEvent.click(screen.getByText("Command Center"));
    expect(onNavigate).toHaveBeenCalledWith("commandCenter");
  });

  it("renders Vrantis wordmark when not collapsed", () => {
    renderSidebar({ collapsed: false });
    expect(screen.getByText("Vrantis")).toBeInTheDocument();
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

  // U5.2 — "FVA & ROI" and "AI FVA Backtest" sat on consecutive Demand rows
  // with the IDENTICAL BarChart3 icon, indistinguishable when the sidebar is
  // collapsed to icon-only. They must render different glyphs.
  it("FVA & ROI and AI FVA Backtest use distinct icons (U5.2)", () => {
    const fva = NAV_ITEMS.find((i) => i.key === "fva");
    const aiFva = NAV_ITEMS.find((i) => i.key === "aiPlannerFva");
    expect(fva).toBeDefined();
    expect(aiFva).toBeDefined();
    expect(fva!.icon).not.toBe(aiFva!.icon);
  });
});
