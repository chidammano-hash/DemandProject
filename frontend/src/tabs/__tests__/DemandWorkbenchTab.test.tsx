import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

vi.mock("@/api/queries/demand-history", () => ({
  demandHistoryKeys: {
    workbench: (...args: unknown[]) => ["demand-history-workbench", ...args],
  },
  useWorkbench: vi.fn().mockReturnValue({
    data: {
      grain: "item",
      series: [
        {
          key: "ITEM_A__LOC_1",
          label: "Item A (Loc 1)",
          total_demand: 5000,
          months: [
            { month: "2025-01", demand_qty: 400 },
            { month: "2025-02", demand_qty: 600 },
          ],
        },
        {
          key: "ITEM_B__LOC_2",
          label: "Item B (Loc 2)",
          total_demand: 3000,
          months: [
            { month: "2025-01", demand_qty: 200 },
            { month: "2025-02", demand_qty: 800 },
          ],
        },
      ],
      hierarchy_children: "item_loc",
      total: 2,
    },
    isLoading: false,
    isError: false,
  }),
}));

// Need to wrap in DemandHistoryTab for context, or provide context directly
// Since WorkbenchPanel uses useDemandHistorySelection, we need the parent
import DemandHistoryTab from "../DemandHistoryTab";

describe("DemandWorkbenchPanel (via DemandHistoryTab)", () => {
  it("renders series in the tree", () => {
    render(
      <TestQueryWrapper>
        <DemandHistoryTab />
      </TestQueryWrapper>,
    );

    expect(screen.getByText("Item A (Loc 1)")).toBeInTheDocument();
    expect(screen.getByText("Item B (Loc 2)")).toBeInTheDocument();
  });

  it("shows total demand for each series", () => {
    render(
      <TestQueryWrapper>
        <DemandHistoryTab />
      </TestQueryWrapper>,
    );

    expect(screen.getByText("5,000")).toBeInTheDocument();
    expect(screen.getByText("3,000")).toBeInTheDocument();
  });

  it("shows placeholder when no series selected", () => {
    render(
      <TestQueryWrapper>
        <DemandHistoryTab />
      </TestQueryWrapper>,
    );

    expect(screen.getByText("No series selected")).toBeInTheDocument();
  });

  it("shows chart when a series is selected", () => {
    render(
      <TestQueryWrapper>
        <DemandHistoryTab />
      </TestQueryWrapper>,
    );

    fireEvent.click(screen.getByText("Item A (Loc 1)"));
    expect(screen.getByTestId("composed-chart")).toBeInTheDocument();
  });

  it("filters series by search input", () => {
    render(
      <TestQueryWrapper>
        <DemandHistoryTab />
      </TestQueryWrapper>,
    );

    const search = screen.getByPlaceholderText("Search items...");
    fireEvent.change(search, { target: { value: "Item A" } });

    expect(screen.getByText("Item A (Loc 1)")).toBeInTheDocument();
    expect(screen.queryByText("Item B (Loc 2)")).not.toBeInTheDocument();
  });

  it("renders grain selector buttons", () => {
    render(
      <TestQueryWrapper>
        <DemandHistoryTab />
      </TestQueryWrapper>,
    );

    expect(screen.getByText("Item")).toBeInTheDocument();
    expect(screen.getByText("Item + Loc")).toBeInTheDocument();
    expect(screen.getByText("Item + Loc + Cust")).toBeInTheDocument();
  });
});
