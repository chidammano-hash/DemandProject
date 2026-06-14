/**
 * U9.3: Customer-Map MoM delta badges must expose an accessible label naming
 * the comparison period, so a screen reader announces "up 28.1% month-over-month
 * vs prior month" rather than a bare "↑ 28.1% MoM" with no period anchor.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "../../__tests__/test-utils";
import { KpiSummaryCards } from "../KpiSummaryCards";

vi.mock("@/api/queries/customer-analytics", () => ({
  customerAnalyticsKeys: {
    kpis: (f: unknown) => ["customer-analytics", "kpis", f],
  },
  fetchCustomerAnalyticsKpis: vi.fn().mockResolvedValue({
    kpis: [
      { key: "total_demand", value: 23_000_000, delta: 28.1 },
      { key: "oos_volume", value: 461_000, delta: 42.9 },
    ],
  }),
}));

describe("KpiSummaryCards MoM delta accessibility (U9.3)", () => {
  it("each delta exposes an aria-label naming the month-over-month comparison", async () => {
    render(<KpiSummaryCards filters={{} as never} />, { wrapper: TestQueryWrapper });
    // The +28.1% total-demand delta must carry a labeled, period-anchored aria.
    await waitFor(() => {
      const labeled = screen.getByLabelText(/28\.1% month-over-month vs prior month/i);
      expect(labeled).toBeInTheDocument();
    });
    // The OOS delta is likewise labeled.
    expect(
      screen.getByLabelText(/42\.9% month-over-month vs prior month/i),
    ).toBeInTheDocument();
  });
});
