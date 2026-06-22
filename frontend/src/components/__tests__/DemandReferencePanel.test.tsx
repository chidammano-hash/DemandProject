import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "../../tabs/__tests__/test-utils";

vi.mock("recharts");

vi.mock("@/api/queries/demand-history", () => ({
  demandHistoryKeys: {
    reference: (...args: unknown[]) => ["demand-history-reference", ...args],
  },
  useReference: vi.fn().mockReturnValue({
    data: {
      item_id: "ITEM_001",
      loc: "LOC_A",
      item_description: "Widget Alpha",
      location_name: "Warehouse East",
      history: [
        { month: "2025-01", demand_qty: 100, sales_qty: 90 },
        { month: "2025-02", demand_qty: 120, sales_qty: 110 },
      ],
      top_customers: [
        { customer_no: "C001", customer_name: "Acme Corp", demand_qty: 500, pct_share: 45.2 },
        { customer_no: "C002", customer_name: "Beta Inc", demand_qty: 300, pct_share: 27.1 },
      ],
      trend_mom_pct: 3.5,
      current_inventory: 1200,
      avg_lead_time: 14,
      forecast_accuracy: 82.3,
    },
    isLoading: false,
    isError: false,
  }),
}));

import { DemandReferencePanel } from "../DemandReferencePanel";

describe("DemandReferencePanel", () => {
  const onClose = vi.fn();

  it("renders nothing when closed", () => {
    const { container } = render(
      <TestQueryWrapper>
        <DemandReferencePanel itemId="ITEM_001" loc="LOC_A" open={false} onClose={onClose} />
      </TestQueryWrapper>,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders item description and location when open", () => {
    render(
      <TestQueryWrapper>
        <DemandReferencePanel itemId="ITEM_001" loc="LOC_A" open={true} onClose={onClose} />
      </TestQueryWrapper>,
    );

    expect(screen.getByText("Widget Alpha")).toBeInTheDocument();
    expect(screen.getByText("Warehouse East")).toBeInTheDocument();
  });

  it("shows KPI cards", () => {
    render(
      <TestQueryWrapper>
        <DemandReferencePanel itemId="ITEM_001" loc="LOC_A" open={true} onClose={onClose} />
      </TestQueryWrapper>,
    );

    expect(screen.getByText("+3.5%")).toBeInTheDocument();
    expect(screen.getByText("82.3%")).toBeInTheDocument();
    // current_inventory: 1200 → compact notation ("1.2K"), per kpiFmt in DemandReferencePanel.
    expect(screen.getByText("1.2K")).toBeInTheDocument();
    expect(screen.getByText("14d")).toBeInTheDocument();
  });

  it("renders top customers heading", () => {
    render(
      <TestQueryWrapper>
        <DemandReferencePanel itemId="ITEM_001" loc="LOC_A" open={true} onClose={onClose} />
      </TestQueryWrapper>,
    );

    expect(screen.getByText("Top Customers")).toBeInTheDocument();
    // Customer names rendered inside mocked BarChart YAxis (mocked as null),
    // so we verify the bar chart container renders
    expect(screen.getByTestId("bar-chart")).toBeInTheDocument();
  });

  it("calls onClose when close button clicked", () => {
    render(
      <TestQueryWrapper>
        <DemandReferencePanel itemId="ITEM_001" loc="LOC_A" open={true} onClose={onClose} />
      </TestQueryWrapper>,
    );

    fireEvent.click(screen.getByLabelText("Close panel"));
    expect(onClose).toHaveBeenCalled();
  });

  it("calls onClose on Escape key", () => {
    render(
      <TestQueryWrapper>
        <DemandReferencePanel itemId="ITEM_001" loc="LOC_A" open={true} onClose={onClose} />
      </TestQueryWrapper>,
    );

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });

  it("renders charts", () => {
    render(
      <TestQueryWrapper>
        <DemandReferencePanel itemId="ITEM_001" loc="LOC_A" open={true} onClose={onClose} />
      </TestQueryWrapper>,
    );

    expect(screen.getByTestId("area-chart")).toBeInTheDocument();
    expect(screen.getByTestId("bar-chart")).toBeInTheDocument();
  });
});