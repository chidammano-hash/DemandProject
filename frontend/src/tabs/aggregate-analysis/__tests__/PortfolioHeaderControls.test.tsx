import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { EMPTY_FILTERS, PANEL_DEFAULTS, type LocalFilters } from "../aggregateShared";

vi.mock("../FilterDropdowns", () => ({
  FilterDropdown: ({
    config,
    onSelect,
  }: {
    config: { label: string };
    onSelect: (values: string[]) => void;
  }) => <button onClick={() => onSelect([config.label])}>Filter by {config.label}</button>,
  SearchableFilterDropdown: ({
    config,
    onSelect,
  }: {
    config: { label: string };
    onSelect: (values: string[]) => void;
  }) => <button onClick={() => onSelect([config.label])}>Filter by {config.label}</button>,
  TimeGrainToggle: ({ onChange }: { onChange: (value: "month" | "quarter") => void }) => (
    <button onClick={() => onChange("quarter")}>Quarter</button>
  ),
}));

const { PortfolioHeaderControls } = await import("../PortfolioHeaderControls");

function Harness({ onTogglePanel }: { onTogglePanel: (key: string) => void }) {
  const [filters, setFilters] = useState<LocalFilters>({
    ...EMPTY_FILTERS,
    brand: ["Brand A"],
  });
  return (
    <PortfolioHeaderControls
      filters={filters}
      setFilters={setFilters}
      onFilterChange={(key, values) => setFilters((current) => ({ ...current, [key]: values }))}
      planningDate="2026-07-01"
      skuCount={1_234}
      visiblePanels={PANEL_DEFAULTS}
      onTogglePanel={onTogglePanel}
    />
  );
}

describe("PortfolioHeaderControls", () => {
  it("renders planning context, filters, and panel controls", () => {
    render(<Harness onTogglePanel={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "Portfolio Analysis" })).toBeInTheDocument();
    expect(screen.getByText(/Plan as of/)).toBeInTheDocument();
    expect(screen.getByText("1,234 SKUs")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Filter by Brand" })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Toggle Forecast vs Actual" })).toBeChecked();
  });

  it("resets active filters and delegates panel toggles", () => {
    const onTogglePanel = vi.fn();
    render(<Harness onTogglePanel={onTogglePanel} />);

    fireEvent.click(screen.getByRole("checkbox", { name: "Toggle Heatmap" }));
    expect(onTogglePanel).toHaveBeenCalledWith("heatmap");

    fireEvent.click(screen.getByRole("button", { name: /Reset/ }));
    expect(screen.queryByText("1,234 SKUs")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Reset/ })).not.toBeInTheDocument();
  });
});
