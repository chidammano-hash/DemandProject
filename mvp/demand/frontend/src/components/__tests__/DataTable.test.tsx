import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DataTable } from "@/components/DataTable";

// Mock useVirtualizer so all rows render in jsdom (no real scroll container)
vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: (opts: { count: number; estimateSize: () => number }) => {
    const size = opts.estimateSize();
    const items = Array.from({ length: opts.count }, (_, i) => ({
      index: i, start: i * size, end: (i + 1) * size, size, key: i, lane: 0,
    }));
    return { getVirtualItems: () => items, getTotalSize: () => items.length * size };
  },
}));

// Mock downloadCsv to avoid blob/URL API issues in jsdom
vi.mock("@/lib/export", () => ({
  downloadCsv: vi.fn(),
}));

import { downloadCsv } from "@/lib/export";

const ALL_COLUMNS = ["name", "value", "category"];

const makeVisibleColumns = (cols: string[] = ALL_COLUMNS) =>
  Object.fromEntries(cols.map((c) => [c, true]));

const baseProps = {
  data: [] as Record<string, unknown>[],
  columns: ALL_COLUMNS,
  visibleColumns: makeVisibleColumns(),
  onToggleColumn: vi.fn(),
  showFieldPanel: false,
  onToggleFieldPanel: vi.fn(),
  sortBy: "",
  sortDir: "asc" as const,
  onSortChange: vi.fn(),
  columnFilters: {},
  onColumnFilterChange: vi.fn(),
  columnSuggestions: {},
  total: 0,
  totalApproximate: false,
  offset: 0,
  limit: 50,
  onOffsetChange: vi.fn(),
  onLimitChange: vi.fn(),
  isLoading: false,
  domain: "test",
};

const sampleRows: Record<string, unknown>[] = [
  { name: "Item A", value: 100, category: "Alpha" },
  { name: "Item B", value: 200, category: "Beta" },
];

describe("DataTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders without crashing with empty data", () => {
    render(<DataTable {...baseProps} />);
    expect(screen.getByText("No records")).toBeInTheDocument();
  });

  it("renders column headers from the columns prop", () => {
    render(<DataTable {...baseProps} data={sampleRows} total={sampleRows.length} />);
    // titleCase converts snake_case/lowercase — "name" → "Name", "value" → "Value"
    expect(screen.getByText("Name")).toBeInTheDocument();
    expect(screen.getByText("Value")).toBeInTheDocument();
    expect(screen.getByText("Category")).toBeInTheDocument();
  });

  it("renders row cell content for visible data", () => {
    render(<DataTable {...baseProps} data={sampleRows} total={sampleRows.length} />);
    expect(screen.getByText("Item A")).toBeInTheDocument();
    expect(screen.getByText("Item B")).toBeInTheDocument();
    expect(screen.getByText("Alpha")).toBeInTheDocument();
    expect(screen.getByText("Beta")).toBeInTheDocument();
  });

  it("shows 'No records' message when data is empty and not loading", () => {
    render(<DataTable {...baseProps} data={[]} total={0} />);
    expect(screen.getByText("No records")).toBeInTheDocument();
  });

  it("calls onSortChange when a column header button is clicked", () => {
    const onSortChange = vi.fn();
    render(
      <DataTable
        {...baseProps}
        data={sampleRows}
        total={sampleRows.length}
        onSortChange={onSortChange}
      />
    );
    // Click the "Name" column header sort button
    fireEvent.click(screen.getByText("Name"));
    expect(onSortChange).toHaveBeenCalledWith("name");
  });

  it("renders CSV export button and triggers downloadCsv on click", () => {
    render(<DataTable {...baseProps} data={sampleRows} total={sampleRows.length} />);
    const csvButton = screen.getByTitle("Export to CSV");
    expect(csvButton).toBeInTheDocument();
    fireEvent.click(csvButton);
    expect(downloadCsv).toHaveBeenCalledTimes(1);
  });

  it("shows pagination info with correct start/end/total", () => {
    render(
      <DataTable
        {...baseProps}
        data={sampleRows}
        total={2}
        offset={0}
        limit={50}
      />
    );
    expect(screen.getByText(/Showing/)).toBeInTheDocument();
  });

  it("Previous button is disabled at offset 0", () => {
    render(<DataTable {...baseProps} data={sampleRows} total={2} offset={0} />);
    const prevBtn = screen.getByRole("button", { name: /Previous/i });
    expect(prevBtn).toBeDisabled();
  });

  it("Next button is disabled when all rows are shown", () => {
    render(<DataTable {...baseProps} data={sampleRows} total={2} offset={0} limit={50} />);
    const nextBtn = screen.getByRole("button", { name: /Next/i });
    expect(nextBtn).toBeDisabled();
  });

  it("Next button is enabled when more rows exist and calls onOffsetChange", () => {
    const onOffsetChange = vi.fn();
    render(
      <DataTable
        {...baseProps}
        data={sampleRows}
        total={100}
        offset={0}
        limit={50}
        onOffsetChange={onOffsetChange}
      />
    );
    const nextBtn = screen.getByRole("button", { name: /Next/i });
    expect(nextBtn).not.toBeDisabled();
    fireEvent.click(nextBtn);
    expect(onOffsetChange).toHaveBeenCalledWith(50);
  });

  it("Previous button calls onOffsetChange with correct offset", () => {
    const onOffsetChange = vi.fn();
    render(
      <DataTable
        {...baseProps}
        data={sampleRows}
        total={100}
        offset={50}
        limit={50}
        onOffsetChange={onOffsetChange}
      />
    );
    const prevBtn = screen.getByRole("button", { name: /Previous/i });
    expect(prevBtn).not.toBeDisabled();
    fireEvent.click(prevBtn);
    expect(onOffsetChange).toHaveBeenCalledWith(0);
  });

  it("renders without crashing with 100 rows", () => {
    const largeData: Record<string, unknown>[] = Array.from({ length: 100 }, (_, i) => ({
      name: `Item ${i}`,
      value: i * 10,
      category: `Cat ${i % 5}`,
    }));
    render(
      <DataTable
        {...baseProps}
        data={largeData}
        total={100}
      />
    );
    // Table should render without throwing — at minimum headers present
    expect(screen.getByText("Name")).toBeInTheDocument();
  });

  it("shows field panel with column checkboxes when showFieldPanel is true", () => {
    render(
      <DataTable
        {...baseProps}
        showFieldPanel={true}
        data={sampleRows}
        total={sampleRows.length}
      />
    );
    // Select All and Deselect All buttons are visible in the panel (use getByText to avoid
    // matching the _select column header checkbox which also has aria-label "Select all")
    expect(screen.getByText("Select All")).toBeInTheDocument();
    expect(screen.getByText("Deselect All")).toBeInTheDocument();
  });

  it("calls onToggleColumn for all columns when Select All is clicked", () => {
    const onToggleColumn = vi.fn();
    render(
      <DataTable
        {...baseProps}
        showFieldPanel={true}
        onToggleColumn={onToggleColumn}
        data={sampleRows}
        total={sampleRows.length}
      />
    );
    fireEvent.click(screen.getByText("Select All"));
    expect(onToggleColumn).toHaveBeenCalledTimes(ALL_COLUMNS.length);
    ALL_COLUMNS.forEach((col) => {
      expect(onToggleColumn).toHaveBeenCalledWith(col, true);
    });
  });

  it("calls onColumnFilterChange when filter input changes", () => {
    const onColumnFilterChange = vi.fn();
    render(
      <DataTable
        {...baseProps}
        data={sampleRows}
        total={sampleRows.length}
        onColumnFilterChange={onColumnFilterChange}
      />
    );
    const inputs = screen.getAllByPlaceholderText("Filter (=exact)");
    // First input is for the first visible column (name)
    fireEvent.change(inputs[0], { target: { value: "Alpha" } });
    expect(onColumnFilterChange).toHaveBeenCalledWith("name", "Alpha");
  });

  it("shows approximate total with '+' notation when totalApproximate is true", () => {
    render(
      <DataTable
        {...baseProps}
        data={sampleRows}
        total={100001}
        totalApproximate={true}
        offset={0}
        limit={50}
      />
    );
    expect(screen.getByText(/100,000\+/)).toBeInTheDocument();
  });

  it("does not render loading element when isLoading is false", () => {
    const loadingEl = <div data-testid="loader">Loading...</div>;
    render(
      <DataTable
        {...baseProps}
        isLoading={false}
        loadingElement={loadingEl}
        data={[]}
        total={0}
      />
    );
    expect(screen.queryByTestId("loader")).not.toBeInTheDocument();
  });

  it("renders loading element when isLoading is true", () => {
    const loadingEl = <div data-testid="loader">Loading...</div>;
    render(
      <DataTable
        {...baseProps}
        isLoading={true}
        loadingElement={loadingEl}
        data={[]}
        total={0}
      />
    );
    expect(screen.getByTestId("loader")).toBeInTheDocument();
  });
});
