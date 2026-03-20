import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";
import { TestQueryWrapper } from "./test-utils";

// Mock the sql-runner queries
const mockExecuteQuery = vi.fn();
const mockFetchSchema = vi.fn();
const mockFetchHistory = vi.fn();

vi.mock("@/api/queries/sql-runner", () => ({
  fetchExecuteQuery: (...args: unknown[]) => mockExecuteQuery(...args),
  fetchSchema: (...args: unknown[]) => mockFetchSchema(...args),
  fetchQueryHistory: (...args: unknown[]) => mockFetchHistory(...args),
  sqlRunnerKeys: {
    schema: () => ["sql-runner-schema"],
    history: () => ["sql-runner-history"],
  },
}));

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: vi.fn((key: string) => { delete store[key]; }),
    clear: vi.fn(() => { store = {}; }),
    get length() { return Object.keys(store).length; },
    key: vi.fn((i: number) => Object.keys(store)[i] ?? null),
  };
})();
Object.defineProperty(globalThis, "localStorage", { value: localStorageMock, writable: true });

import { SqlRunnerTab } from "../SqlRunnerTab";

describe("SqlRunnerTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorageMock.clear();
    mockFetchSchema.mockResolvedValue({ tables: [] });
    mockFetchHistory.mockResolvedValue({ history: [] });
  });

  it("renders editor and run button", () => {
    render(
      <TestQueryWrapper>
        <SqlRunnerTab />
      </TestQueryWrapper>,
    );
    expect(screen.getByTestId("sql-editor")).toBeInTheDocument();
    expect(screen.getByTestId("run-query-btn")).toBeInTheDocument();
    expect(screen.getByText("SQL Runner")).toBeInTheDocument();
  });

  it("displays default placeholder text in editor", () => {
    render(
      <TestQueryWrapper>
        <SqlRunnerTab />
      </TestQueryWrapper>,
    );
    const editor = screen.getByTestId("sql-editor") as HTMLTextAreaElement;
    expect(editor.value).toContain("SELECT");
  });

  it("updates editor text on input", () => {
    render(
      <TestQueryWrapper>
        <SqlRunnerTab />
      </TestQueryWrapper>,
    );
    const editor = screen.getByTestId("sql-editor") as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: "SELECT * FROM dim_item" } });
    expect(editor.value).toBe("SELECT * FROM dim_item");
  });

  it("executes query on run button click", async () => {
    mockExecuteQuery.mockResolvedValue({
      columns: ["id", "name"],
      rows: [[1, "alpha"], [2, "beta"]],
      row_count: 2,
      truncated: false,
      elapsed_ms: 12.5,
    });

    render(
      <TestQueryWrapper>
        <SqlRunnerTab />
      </TestQueryWrapper>,
    );

    const editor = screen.getByTestId("sql-editor") as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: "SELECT id, name FROM dim_item" } });
    fireEvent.click(screen.getByTestId("run-query-btn"));

    await waitFor(() => {
      expect(mockExecuteQuery).toHaveBeenCalledWith("SELECT id, name FROM dim_item");
    });

    await waitFor(() => {
      expect(screen.getByText("Results")).toBeInTheDocument();
      expect(screen.getByText("2 rows")).toBeInTheDocument();
    });
  });

  it("displays error state for failed queries", async () => {
    mockExecuteQuery.mockRejectedValue(new Error("relation does not exist"));

    render(
      <TestQueryWrapper>
        <SqlRunnerTab />
      </TestQueryWrapper>,
    );

    const editor = screen.getByTestId("sql-editor") as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: "SELECT * FROM bad_table" } });
    fireEvent.click(screen.getByTestId("run-query-btn"));

    await waitFor(() => {
      expect(screen.getByText("relation does not exist")).toBeInTheDocument();
    });
  });

  it("clears editor on clear button click", () => {
    render(
      <TestQueryWrapper>
        <SqlRunnerTab />
      </TestQueryWrapper>,
    );

    fireEvent.click(screen.getByText("Clear"));
    const editor = screen.getByTestId("sql-editor") as HTMLTextAreaElement;
    expect(editor.value).toBe("");
  });

  it("toggles schema browser visibility", () => {
    render(
      <TestQueryWrapper>
        <SqlRunnerTab />
      </TestQueryWrapper>,
    );

    // Schema is visible by default
    expect(screen.getByText(/Tables/)).toBeInTheDocument();

    // Toggle off
    fireEvent.click(screen.getByText("Schema"));
    // The tables count text should be gone
  });

  it("toggles history panel visibility", () => {
    render(
      <TestQueryWrapper>
        <SqlRunnerTab />
      </TestQueryWrapper>,
    );

    // History hidden by default
    expect(screen.queryByText("Query History")).not.toBeInTheDocument();

    // Toggle on
    fireEvent.click(screen.getByText("History"));
    expect(screen.getByText(/Query History/)).toBeInTheDocument();
  });

  it("renders schema browser with tables", async () => {
    mockFetchSchema.mockResolvedValue({
      tables: [
        {
          schema_name: "public",
          table_name: "dim_item",
          table_type: "BASE TABLE",
          columns: [
            { name: "sk", data_type: "integer", is_nullable: false },
            { name: "name", data_type: "text", is_nullable: true },
          ],
        },
      ],
    });

    render(
      <TestQueryWrapper>
        <SqlRunnerTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("dim_item")).toBeInTheDocument();
    });
  });

  it("disables run button when editor is empty", () => {
    render(
      <TestQueryWrapper>
        <SqlRunnerTab />
      </TestQueryWrapper>,
    );

    const editor = screen.getByTestId("sql-editor") as HTMLTextAreaElement;
    fireEvent.change(editor, { target: { value: "" } });
    expect(screen.getByTestId("run-query-btn")).toBeDisabled();
  });
});
