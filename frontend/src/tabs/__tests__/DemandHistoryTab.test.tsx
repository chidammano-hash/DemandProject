import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

// Mock all demand-history queries
vi.mock("@/api/queries/demand-history", () => ({
  demandHistoryKeys: {
    reference: (...args: unknown[]) => ["demand-history-reference", ...args],
    decomposition: (...args: unknown[]) => ["demand-history-decomposition", ...args],
    comparison: (...args: unknown[]) => ["demand-history-comparison", ...args],
    workbench: (...args: unknown[]) => ["demand-history-workbench", ...args],
    matrix: (...args: unknown[]) => ["demand-history-matrix", ...args],
    matrixDrill: (...args: unknown[]) => ["demand-history-matrix-drill", ...args],
  },
  useReference: vi.fn().mockReturnValue({ data: null, isLoading: false, isError: false }),
  useDecomposition: vi.fn().mockReturnValue({ data: null, isLoading: false, isError: false }),
  useComparison: vi.fn().mockReturnValue({ data: null, isLoading: false, isError: false }),
  useWorkbench: vi.fn().mockReturnValue({
    data: { grain: "item", series: [], hierarchy_children: [], total_count: 0 },
    isLoading: false,
    isError: false,
  }),
  useMatrix: vi.fn().mockReturnValue({ data: null, isLoading: false, isError: false }),
  useMatrixDrill: vi.fn().mockReturnValue({ data: null, isLoading: false, isError: false }),
  fetchReference: vi.fn(),
  fetchDecomposition: vi.fn(),
  fetchComparison: vi.fn(),
  fetchWorkbench: vi.fn(),
  fetchMatrix: vi.fn(),
  fetchMatrixDrill: vi.fn(),
}));

import DemandHistoryTab from "../DemandHistoryTab";

describe("DemandHistoryTab", () => {
  it("renders sub-navigation buttons", () => {
    render(
      <TestQueryWrapper>
        <DemandHistoryTab />
      </TestQueryWrapper>,
    );

    expect(screen.getByText("Workbench")).toBeInTheDocument();
    expect(screen.getByText("Decomposition")).toBeInTheDocument();
    expect(screen.getByText("Comparison")).toBeInTheDocument();
    expect(screen.getByText("Matrix")).toBeInTheDocument();
  });

  it("defaults to Workbench panel", () => {
    render(
      <TestQueryWrapper>
        <DemandHistoryTab />
      </TestQueryWrapper>,
    );

    // Workbench shows the grain selector buttons
    expect(screen.getByText("item")).toBeInTheDocument();
  });

  it("switches to Decomposition panel on click", () => {
    render(
      <TestQueryWrapper>
        <DemandHistoryTab />
      </TestQueryWrapper>,
    );

    fireEvent.click(screen.getByText("Decomposition"));
    expect(screen.getByPlaceholderText("Item ID")).toBeInTheDocument();
  });

  it("switches to Comparison panel on click", () => {
    render(
      <TestQueryWrapper>
        <DemandHistoryTab />
      </TestQueryWrapper>,
    );

    fireEvent.click(screen.getByText("Comparison"));
    expect(screen.getByPlaceholderText("Item ID")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Location")).toBeInTheDocument();
  });

  it("switches to Matrix panel on click", () => {
    render(
      <TestQueryWrapper>
        <DemandHistoryTab />
      </TestQueryWrapper>,
    );

    fireEvent.click(screen.getByText("Matrix"));
    expect(screen.getByText("Rows:")).toBeInTheDocument();
    expect(screen.getByText("Cols:")).toBeInTheDocument();
  });
});
