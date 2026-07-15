import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const useCustomerBlendOverlay = vi.fn();

vi.mock("recharts");
vi.mock("@/api/queries/demand-history", () => ({
  useComparison: () => ({
    data: {
      item_id: "ITEM-1",
      loc: "LOC-1",
      comparison: [{ month: "2026-06-01", actual_qty: 70 }],
    },
    isLoading: false,
    isError: false,
  }),
}));
vi.mock("@/hooks/useCustomerBlendOverlay", () => ({
  useCustomerBlendOverlay: (...args: unknown[]) => useCustomerBlendOverlay(...args),
}));
vi.mock("@/hooks/useChartColors", () => ({
  useChartColors: () => ({
    chartColors: {
      grid: "grid",
      axis: "axis",
      tooltip_bg: "tooltip-bg",
      tooltip_border: "tooltip-border",
    },
    okabeIto: ["orange", "sky", "green", "yellow", "blue", "vermillion", "purple"],
  }),
}));

import { CUSTOMER_BLEND_KEYS } from "@/lib/customer-blend-overlay";
import { ComparisonPanel } from "../ComparisonPanel";

describe("ComparisonPanel customer blend overlay", () => {
  it("shows all blend series and fallback status for the exact comparison selection", () => {
    useCustomerBlendOverlay.mockReturnValue({
      status: "ready",
      months: [
        {
          forecast_month: "2026-07-01",
          normalized_customer_qty: null,
          champion_qty: 80,
          blended_qty: 80,
          coverage_status: "champion_fallback",
        },
      ],
      points: [
        {
          month: "2026-07-01",
          [CUSTOMER_BLEND_KEYS.bottomUp]: null,
          [CUSTOMER_BLEND_KEYS.sourceChampion]: 80,
          [CUSTOMER_BLEND_KEYS.blend]: 80,
        },
      ],
      runId: "blend-1",
      planningMonth: "2026-07-01",
      invalidReason: null,
    });

    render(<ComparisonPanel />);
    fireEvent.change(screen.getByPlaceholderText("Item ID"), {
      target: { value: "ITEM-1" },
    });
    fireEvent.change(screen.getByPlaceholderText("Location"), {
      target: { value: "LOC-1" },
    });

    const legend = screen.getByRole("status");
    expect(within(legend).getByText("Customer Bottom-Up")).toBeInTheDocument();
    expect(within(legend).getByText("Source Champion")).toBeInTheDocument();
    expect(within(legend).getByText("Customer Blend")).toBeInTheDocument();
    expect(screen.getByText("Hierarchy Bottom-Up")).toBeInTheDocument();
    expect(screen.getByText("Hierarchy Top-Down")).toBeInTheDocument();
    expect(screen.getByText("Hierarchy Reconciled")).toBeInTheDocument();
    expect(screen.getByText("Champion fallback · 1 of 1 month")).toBeInTheDocument();
    expect(screen.getByLabelText("Blend vintage Jul 2026, run blend-1")).toBeInTheDocument();
    expect(useCustomerBlendOverlay).toHaveBeenLastCalledWith("ITEM-1", "LOC-1", true);
  });

  it("explains when no staged blend draft exists instead of blaming the item", () => {
    useCustomerBlendOverlay.mockReturnValue({
      status: "empty",
      months: [],
      points: [],
      runId: null,
      planningMonth: null,
      invalidReason: null,
    });

    render(<ComparisonPanel />);
    fireEvent.change(screen.getByPlaceholderText("Item ID"), {
      target: { value: "ITEM-1" },
    });
    fireEvent.change(screen.getByPlaceholderText("Location"), {
      target: { value: "LOC-1" },
    });

    expect(
      screen.getByText("No staged customer blend draft exists yet. Generate a blend draft first.")
    ).toBeInTheDocument();
    expect(
      screen.queryByText("No customer blend is available for this item and location.")
    ).not.toBeInTheDocument();
  });
});
