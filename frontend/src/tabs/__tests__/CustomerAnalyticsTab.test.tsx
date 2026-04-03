import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// Mock leaflet CSS
vi.mock("leaflet/dist/leaflet.css", () => ({}));

// Mock react-leaflet
vi.mock("react-leaflet", () => ({
  MapContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="map-container">{children}</div>
  ),
  TileLayer: () => null,
  GeoJSON: () => <div data-testid="geojson-layer" />,
  CircleMarker: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="circle-marker">{children}</div>
  ),
  Tooltip: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
}));

// Mock echarts-for-react
vi.mock("echarts-for-react", () => ({
  default: (props: { style?: React.CSSProperties }) => (
    <div data-testid="echart" style={props.style} />
  ),
}));

// Mock US states GeoJSON
vi.mock("@/assets/us-states.json", () => ({
  default: {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { name: "California" }, geometry: { type: "Polygon", coordinates: [] } },
    ],
  },
}));

// Mock customer-analytics queries
vi.mock("@/api/queries/customer-analytics", () => ({
  customerAnalyticsKeys: {
    map: (...args: unknown[]) => ["customer-analytics-map", ...args],
    treemap: (...args: unknown[]) => ["customer-analytics-treemap", ...args],
    heatmap: (...args: unknown[]) => ["customer-analytics-heatmap", ...args],
    channelMix: (...args: unknown[]) => ["customer-analytics-channel-mix", ...args],
    segmentTrends: (...args: unknown[]) => ["customer-analytics-segment-trends", ...args],
    ranking: (...args: unknown[]) => ["customer-analytics-ranking", ...args],
    oosImpact: (...args: unknown[]) => ["customer-analytics-oos-impact", ...args],
    items: (...args: unknown[]) => ["customer-analytics-items", ...args],
  },
  fetchCustomerAnalyticsMap: vi.fn().mockResolvedValue({
    locations: [
      { label: "CA", state: "CA", customer_count: 500, demand_qty: 10000, sales_qty: 9000, oos_qty: 1000, fill_rate: 90, lat: 36.78, lon: -119.42 },
    ],
    group_by: "state",
    metric: "demand_qty",
    total_demand: 10000,
    total_customers: 500,
  }),
  fetchCustomerAnalyticsTreemap: vi.fn().mockResolvedValue({
    tree: [{ name: "CA", value: 10000, children: [{ name: "On Premise", value: 6000, children: [] }] }],
  }),
  fetchCustomerAnalyticsHeatmap: vi.fn().mockResolvedValue({
    items: [{ item_id: "ITEM001", item_desc: "Widget A" }],
    states: ["CA", "TX"],
    cells: [{ item_id: "ITEM001", state: "CA", demand_qty: 5000, customer_count: 20, fill_rate: 95 }],
    metric: "demand_qty",
  }),
  fetchCustomerAnalyticsChannelMix: vi.fn().mockResolvedValue({
    tree: [{ name: "On Premise", value: 8000, children: [] }],
  }),
  fetchCustomerAnalyticsSegmentTrends: vi.fn().mockResolvedValue({
    segments: [
      {
        segment: "On Premise",
        total_demand: 5000,
        total_customers: 100,
        fill_rate: 92,
        mom_change: 2.5,
        trend: [{ month: "2026-01-01", demand_qty: 2500, sales_qty: 2300, fill_rate: 92 }],
      },
    ],
    segment_by: "rpt_channel_desc",
  }),
  fetchCustomerAnalyticsRanking: vi.fn().mockResolvedValue({
    customers: [
      { customer_no: "C001", customer_name: "Acme Corp", state: "CA", channel: "On Premise", demand_qty: 10000, sales_qty: 9000, oos_qty: 1000, fill_rate: 90 },
    ],
    sort: "demand_desc",
    top_n: 20,
  }),
  fetchCustomerAnalyticsOosImpact: vi.fn().mockResolvedValue({
    bubbles: [
      { label: "Acme Corp", state: "CA", channel: "On Premise", demand_qty: 10000, sales_qty: 9000, oos_qty: 1000, fill_rate: 90 },
    ],
    grain: "customer",
  }),
  fetchCustomerAnalyticsItems: vi.fn().mockResolvedValue({
    items: [{ item_id: "ITEM001", item_desc: "Widget A" }],
  }),
}));

const { CustomerAnalyticsTab } = await import("@/tabs/CustomerAnalyticsTab");

function renderTab() {
  return render(
    <TestQueryWrapper>
      <CustomerAnalyticsTab />
    </TestQueryWrapper>,
  );
}

describe("CustomerAnalyticsTab", () => {
  it("renders without crashing", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Customer Demand Map")).toBeDefined();
    });
  });

  it("renders filter bar with item search", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByPlaceholderText("Search item...")).toBeDefined();
    });
  });

  it("renders map panel with metric buttons", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Customers")).toBeDefined();
      expect(screen.getByText("Demand (cases)")).toBeDefined();
    });
  });

  it("renders treemap panel", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Customer Concentration")).toBeDefined();
    });
  });

  it("renders heatmap panel", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Item x State Heatmap")).toBeDefined();
    });
  });

  it("renders channel mix sunburst", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Channel Mix")).toBeDefined();
    });
  });

  it("renders segment trends with segment-by buttons", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Segment Trends")).toBeDefined();
      // "Channel" and "Store Type" appear both as filter labels and segment buttons
      expect(screen.getAllByText("Channel").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Store Type").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders OOS impact bubble chart", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("OOS Impact Analysis")).toBeDefined();
    });
  });

  it("renders customer ranking panel", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Customer Ranking")).toBeDefined();
      expect(screen.getByText("Top by Demand")).toBeDefined();
      expect(screen.getByText("Worst Fill Rate")).toBeDefined();
    });
  });

  it("renders map container", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByTestId("map-container")).toBeDefined();
    });
  });

  it("renders ECharts panels", async () => {
    renderTab();
    await waitFor(() => {
      const charts = screen.getAllByTestId("echart");
      // treemap + heatmap + sunburst + oos bubble = 4
      expect(charts.length).toBeGreaterThanOrEqual(4);
    });
  });

  it("shows total customer count", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText(/500 customers/)).toBeDefined();
    });
  });

  it("renders Clear button", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Clear")).toBeDefined();
    });
  });

  it("renders date range inputs", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("From")).toBeDefined();
      expect(screen.getByText("To")).toBeDefined();
    });
  });
});
