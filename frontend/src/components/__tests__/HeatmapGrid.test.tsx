import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { HeatmapGrid, makeHeatmapScale } from "@/components/HeatmapGrid";

const sampleRows = [
  { label: "Category A", values: [92.5, 78.3, 55.0] },
  { label: "Category B", values: [88.1, 63.4, 97.2] },
];
const sampleColumns = ["Jan", "Feb", "Mar"];
const colorScale = (value: number) => (value > 80 ? "#22c55e" : value > 60 ? "#eab308" : "#ef4444");

describe("HeatmapGrid", () => {
  it("renders cells with correct aria labels", () => {
    render(
      <HeatmapGrid
        rows={sampleRows}
        columnLabels={sampleColumns}
        colorScale={colorScale}
      />
    );
    // Check aria labels for specific cells
    expect(screen.getByLabelText("Category A, Jan: 92.5%")).toBeInTheDocument();
    expect(screen.getByLabelText("Category A, Feb: 78.3%")).toBeInTheDocument();
    expect(screen.getByLabelText("Category A, Mar: 55.0%")).toBeInTheDocument();
    expect(screen.getByLabelText("Category B, Jan: 88.1%")).toBeInTheDocument();
    expect(screen.getByLabelText("Category B, Feb: 63.4%")).toBeInTheDocument();
    expect(screen.getByLabelText("Category B, Mar: 97.2%")).toBeInTheDocument();
  });

  it("renders empty state message when no rows", () => {
    render(
      <HeatmapGrid
        rows={[]}
        columnLabels={sampleColumns}
        colorScale={colorScale}
      />
    );
    expect(screen.getByText("No data available")).toBeInTheDocument();
  });

  it("renders grid role with accessible label", () => {
    render(
      <HeatmapGrid
        rows={sampleRows}
        columnLabels={sampleColumns}
        colorScale={colorScale}
      />
    );
    expect(screen.getByRole("grid", { name: "Performance heatmap" })).toBeInTheDocument();
  });

  it("renders gridcell role for each data cell", () => {
    render(
      <HeatmapGrid
        rows={sampleRows}
        columnLabels={sampleColumns}
        colorScale={colorScale}
      />
    );
    const gridcells = screen.getAllByRole("gridcell");
    // 2 rows * 3 columns = 6 cells
    expect(gridcells.length).toBe(6);
  });

  it("renders column labels", () => {
    render(
      <HeatmapGrid
        rows={sampleRows}
        columnLabels={sampleColumns}
        colorScale={colorScale}
      />
    );
    expect(screen.getByText("Jan")).toBeInTheDocument();
    expect(screen.getByText("Feb")).toBeInTheDocument();
    expect(screen.getByText("Mar")).toBeInTheDocument();
  });

  it("renders row labels", () => {
    render(
      <HeatmapGrid
        rows={sampleRows}
        columnLabels={sampleColumns}
        colorScale={colorScale}
      />
    );
    expect(screen.getByText("Category A")).toBeInTheDocument();
    expect(screen.getByText("Category B")).toBeInTheDocument();
  });

  it("calls onCellClick with row and col labels", () => {
    const onCellClick = vi.fn();
    render(
      <HeatmapGrid
        rows={sampleRows}
        columnLabels={sampleColumns}
        colorScale={colorScale}
        onCellClick={onCellClick}
      />
    );
    const cell = screen.getByLabelText("Category A, Feb: 78.3%");
    fireEvent.click(cell);
    expect(onCellClick).toHaveBeenCalledWith("Category A", "Feb");
  });

  it("shows DFU count inline on hover when counts provided", () => {
    const rowsWithCounts = [
      { label: "Category A", values: [92.5], counts: [42] },
    ];
    render(
      <HeatmapGrid
        rows={rowsWithCounts}
        columnLabels={["Jan"]}
        colorScale={colorScale}
      />
    );
    const cell = screen.getByLabelText("Category A, Jan: 92.5%, 42 SKUs");
    fireEvent.mouseEnter(cell);
    expect(screen.getByText("(42)")).toBeInTheDocument();
  });

  it("does not show count when counts not provided", () => {
    render(
      <HeatmapGrid
        rows={sampleRows}
        columnLabels={sampleColumns}
        colorScale={colorScale}
      />
    );
    const cell = screen.getByLabelText("Category A, Jan: 92.5%");
    fireEvent.mouseEnter(cell);
    expect(screen.queryByText(/\(\d+\)/)).not.toBeInTheDocument();
  });

  it("uses custom valueFormat when provided", () => {
    const customFormat = (v: number) => `$${v.toFixed(0)}`;
    render(
      <HeatmapGrid
        rows={[{ label: "Row1", values: [42.7] }]}
        columnLabels={["Col1"]}
        colorScale={colorScale}
        valueFormat={customFormat}
      />
    );
    expect(screen.getByText("$43")).toBeInTheDocument();
    expect(screen.getByLabelText("Row1, Col1: $43")).toBeInTheDocument();
  });
});

describe("HeatmapGrid legend", () => {
  it("does not render legend by default", () => {
    const { container } = render(
      <HeatmapGrid
        rows={sampleRows}
        columnLabels={sampleColumns}
        colorScale={colorScale}
      />
    );
    expect(container.querySelector("[aria-label='Heatmap legend']")).not.toBeInTheDocument();
  });

  it("renders legend when showLegend is true", () => {
    const { container } = render(
      <HeatmapGrid
        rows={sampleRows}
        columnLabels={sampleColumns}
        colorScale={colorScale}
        showLegend
      />
    );
    const legend = container.querySelector("[aria-label='Heatmap legend']");
    expect(legend).toBeInTheDocument();
    expect(screen.getByText("0%")).toBeInTheDocument();
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("uses custom min/max labels", () => {
    render(
      <HeatmapGrid
        rows={sampleRows}
        columnLabels={sampleColumns}
        colorScale={colorScale}
        showLegend
        minLabel="Low"
        maxLabel="High"
      />
    );
    expect(screen.getByText("Low")).toBeInTheDocument();
    expect(screen.getByText("High")).toBeInTheDocument();
  });
});

describe("HeatmapGrid zero-DFU cells", () => {
  it("renders zero-DFU cells as empty with no-data aria label", () => {
    const rows = [
      { label: "CatA", values: [90, 0], counts: [5, 0] },
      { label: "CatB", values: [0, 85], counts: [0, 3] },
    ];
    render(
      <HeatmapGrid rows={rows} columnLabels={["Jan", "Feb"]} colorScale={colorScale} />
    );
    // Cells with SKUs render value
    expect(screen.getByLabelText("CatA, Jan: 90.0%, 5 SKUs")).toBeInTheDocument();
    expect(screen.getByLabelText("CatB, Feb: 85.0%, 3 SKUs")).toBeInTheDocument();
    // Cells without SKUs render as no data
    expect(screen.getByLabelText("CatA, Feb: no data")).toBeInTheDocument();
    expect(screen.getByLabelText("CatB, Jan: no data")).toBeInTheDocument();
  });
});

describe("makeHeatmapScale", () => {
  const scale = ["#00ff00", "#88ff00", "#ffff00", "#ff8800", "#ff0000"];
  const getColor = makeHeatmapScale(scale);

  it("returns excellent color for values >= 95", () => {
    expect(getColor(95)).toBe("#00ff00");
    expect(getColor(100)).toBe("#00ff00");
  });

  it("returns good color for values 85-94", () => {
    expect(getColor(85)).toBe("#88ff00");
    expect(getColor(94)).toBe("#88ff00");
  });

  it("returns warning color for values 70-84", () => {
    expect(getColor(70)).toBe("#ffff00");
    expect(getColor(84)).toBe("#ffff00");
  });

  it("returns poor color for values 50-69", () => {
    expect(getColor(50)).toBe("#ff8800");
    expect(getColor(69)).toBe("#ff8800");
  });

  it("returns critical color for values < 50", () => {
    expect(getColor(49)).toBe("#ff0000");
    expect(getColor(0)).toBe("#ff0000");
  });
});
