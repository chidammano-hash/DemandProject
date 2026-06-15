import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", async () => {
  const actual = await vi.importActual("@/api/queries");
  return {
    ...actual,
    exceptionKeys: {
      summary: (p?: unknown) => ["exc-summary", p],
      list: (p?: unknown) => ["exc-list", p],
    },
    STALE: { FIVE_MIN: 300000, ONE_MIN: 60000, TEN_MIN: 600000, TWO_MIN: 120000 },
    fetchExceptionSummary: vi.fn(),
    fetchExceptions: vi.fn(),
    acknowledgeException: vi.fn(),
    updateExceptionStatus: vi.fn(),
    generateExceptions: vi.fn(),
  };
});

import {
  fetchExceptionSummary,
  fetchExceptions,
  acknowledgeException,
} from "@/api/queries";
import { ExceptionQueuePanel } from "@/tabs/inv-planning/ExceptionQueuePanel";

const mockSummary = {
  open_count: 25,
  by_severity: { critical: 3, high: 8, medium: 10, low: 4 },
  total_recommended_order_value: 150000,
};

const mockException = {
  exception_id: "exc-001",
  item_id: "100320",
  loc: "1401-BULK",
  exception_type: "below_rop",
  severity: "critical",
  status: "open",
  current_qty_on_hand: 50,
  ss_combined: 200,
  recommended_order_qty: 500,
  recommended_order_by: "2026-03-15",
};

beforeEach(() => {
  vi.clearAllMocks();
  (fetchExceptionSummary as any).mockResolvedValue(mockSummary);
  (fetchExceptions as any).mockResolvedValue({ total: 1, rows: [mockException] });
});

describe("ExceptionQueuePanel", () => {
  it("renders KPI cards", async () => {
    render(
      <TestQueryWrapper>
        <ExceptionQueuePanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Open Issues")).toBeInTheDocument();
    });
    // "Urgent" appears as KPI label; severity filter pills still show "Critical" and "High"
    expect(screen.getAllByText("Urgent").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("High Priority")).toBeInTheDocument();
    expect(screen.getByText("Value at Risk")).toBeInTheDocument();
  });

  it("renders exception table rows", async () => {
    render(
      <TestQueryWrapper>
        <ExceptionQueuePanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("100320")).toBeInTheDocument();
    expect(await screen.findByText("URGENT")).toBeInTheDocument();
  });

  it("makes the drill-in exception row keyboard-operable (U6.2)", async () => {
    render(
      <TestQueryWrapper>
        <ExceptionQueuePanel />
      </TestQueryWrapper>,
    );
    // The clickable <tr> exposes button role + tab focusability.
    const itemCell = await screen.findByText("100320");
    const row = itemCell.closest("tr") as HTMLTableRowElement;
    expect(row).toBeTruthy();
    expect(row.getAttribute("role")).toBe("button");
    expect(row.getAttribute("tabindex")).toBe("0");
    expect(row.getAttribute("aria-expanded")).toBe("false");

    // Enter activates the same handler as a click — row expands.
    fireEvent.keyDown(row, { key: "Enter" });
    await waitFor(() => {
      expect(row.getAttribute("aria-expanded")).toBe("true");
    });
  });

  it("renders Generate Exceptions button", async () => {
    render(
      <TestQueryWrapper>
        <ExceptionQueuePanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Generate Exceptions")).toBeInTheDocument();
  });

  it("renders Acknowledge button for open exceptions", async () => {
    render(
      <TestQueryWrapper>
        <ExceptionQueuePanel />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText("Acknowledge")).toBeInTheDocument();
  });

  it("optimistically flips status to 'acknowledged' before the server responds (UX-7)", async () => {
    // Hold the ack mutation open so we can observe the pre-resolve state.
    let resolveAck: (v: unknown) => void = () => {};
    (acknowledgeException as unknown as { mockImplementation: (fn: () => Promise<unknown>) => void }).mockImplementation(
      () => new Promise((res) => { resolveAck = res; }),
    );

    render(
      <TestQueryWrapper>
        <ExceptionQueuePanel />
      </TestQueryWrapper>,
    );

    // The button's text node is "Acknowledge"; closest() finds the button element.
    const textNode = await screen.findByText("Acknowledge");
    const btn = textNode.closest("button") as HTMLButtonElement;
    expect(btn).toBeTruthy();
    fireEvent.click(btn);

    // Optimistic update: row status visibly flipped to "acknowledged" in the
    // status badge column, and the ack button is no longer rendered for it.
    await waitFor(() => {
      expect(screen.getByText("acknowledged")).toBeInTheDocument();
    });

    // Clean up.
    resolveAck({ ...mockException, status: "acknowledged" });
  });
});
