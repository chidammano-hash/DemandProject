import { describe, it, expect, vi } from "vitest";
import type { ReactNode } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { CustomerDemandMap } from "../CustomerDemandMap";
import { ThemeProvider } from "@/context/ThemeContext";
import { DashboardFilterProvider } from "../DashboardFilterContext";
import { formatCompactKMB } from "@/lib/formatters";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";

vi.mock("leaflet/dist/leaflet.css", () => ({}));
vi.mock("react-leaflet", () => ({
  MapContainer: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  TileLayer: () => null,
  GeoJSON: () => <div data-testid="geojson-layer" />,
  CircleMarker: ({ children }: { children?: ReactNode }) => <div>{children}</div>,
  Tooltip: ({ children }: { children: ReactNode }) => <span>{children}</span>,
}));
vi.mock("@/assets/us-states.json", () => ({ default: { type: "FeatureCollection", features: [] } }));

// Inlined (not a top-level const) because vi.mock is hoisted above module init.
vi.mock("@/api/queries/customer-analytics", async () => {
  const actual = await vi.importActual<typeof import("@/api/queries/customer-analytics")>(
    "@/api/queries/customer-analytics",
  );
  return {
    ...actual,
    fetchCustomerAnalyticsMap: vi.fn().mockResolvedValue({
      total_customers: 12345,
      total_demand: 22986295,
      locations: [
        { label: "CA", state: "CA", customer_count: 100, demand_qty: 22986295, sales_qty: 1, oos_qty: 0, fill_rate: 95, lat: 36, lon: -119 },
      ],
    }),
  };
});

const TOTAL_DEMAND = 22986295;

function wrap(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <ThemeProvider value={{ theme: "light" }}>
      <QueryClientProvider client={qc}>
        <DashboardFilterProvider>{node}</DashboardFilterProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

const filters = {} as CustomerAnalyticsFilters;

describe("CustomerDemandMap footer demand format (U3.2)", () => {
  it("renders total demand with the same compact formatter the KPI tile uses", async () => {
    render(
      wrap(
        <CustomerDemandMap
          filters={filters}
          metric="demand_qty"
          groupBy="state"
          onMetricChange={() => {}}
          onGroupByChange={() => {}}
        />,
      ),
    );
    // The footer must read the compact form (e.g. "23.0M"), NOT the bare
    // full-digit "22,986,295" — matching the "Total Demand" KPI tile.
    const compact = formatCompactKMB(TOTAL_DEMAND); // "23.0M"
    await waitFor(() =>
      expect(screen.getByText(new RegExp(`${compact} cases total demand`))).toBeInTheDocument(),
    );
    expect(screen.queryByText(/22,986,295/)).toBeNull();
  });
});
