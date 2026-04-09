import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
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
});
