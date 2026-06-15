import { describe, it, expect, vi } from "vitest";
import type { ReactNode } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { CustomerHeatmap } from "../CustomerHeatmap";
import { ThemeProvider } from "@/context/ThemeContext";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";

// ECharts is heavy / canvas-bound in jsdom; stub it to a placeholder div so the
// surrounding toggle pills render and can be queried.
vi.mock("@/components/echarts-modular", () => ({
  ModularReactECharts: () => <div data-testid="echarts" />,
}));

vi.mock("@/api/queries/customer-analytics", async () => {
  const actual = await vi.importActual<typeof import("@/api/queries/customer-analytics")>(
    "@/api/queries/customer-analytics",
  );
  return {
    ...actual,
    fetchCustomerAnalyticsHeatmap: vi.fn().mockResolvedValue({
      items: [{ item_id: "1", item_desc: "Item One" }],
      states: ["CA"],
      cells: [{ item_id: "1", state: "CA", demand_qty: 100, customer_count: 5, fill_rate: 95 }],
    }),
  };
});

function wrap(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <ThemeProvider value={{ theme: "light" }}>
      <QueryClientProvider client={qc}>{node}</QueryClientProvider>
    </ThemeProvider>
  );
}

const filters = {} as CustomerAnalyticsFilters;

describe("CustomerHeatmap toggle accessibility (U3.3)", () => {
  it("exposes the active metric via aria-pressed=true and inactive via false", async () => {
    render(wrap(<CustomerHeatmap filters={filters} metric="demand_qty" topN={10} />));
    await waitFor(() => expect(screen.getByTestId("echarts")).toBeInTheDocument());

    const active = screen.getByRole("button", { name: "Demand" });
    const inactive = screen.getByRole("button", { name: "Customers" });
    expect(active).toHaveAttribute("aria-pressed", "true");
    expect(inactive).toHaveAttribute("aria-pressed", "false");
  });
});
