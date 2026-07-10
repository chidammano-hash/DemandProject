import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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

// Mock the modular ECharts wrapper used by CA panels (it pulls in
// echarts/core + canvas renderer, which jsdom can't satisfy).
vi.mock("@/components/echarts-modular", () => ({
  ModularReactECharts: (props: { style?: React.CSSProperties }) => (
    <div data-testid="echart" style={props.style} />
  ),
  default: (props: { style?: React.CSSProperties }) => (
    <div data-testid="echart" style={props.style} />
  ),
}));

vi.mock("recharts");

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
    kpis: (...args: unknown[]) => ["customer-analytics-kpis", ...args],
    filterOptions: () => ["customer-analytics-filter-options"],
    lifecycle: (...args: unknown[]) => ["customer-analytics-lifecycle", ...args],
    demandAtRisk: (...args: unknown[]) => ["customer-analytics-demand-at-risk", ...args],
    affinity: (...args: unknown[]) => ["customer-analytics-affinity", ...args],
    orderPatterns: (...args: unknown[]) => ["customer-analytics-order-patterns", ...args],
    demandFlow: (...args: unknown[]) => ["customer-analytics-demand-flow", ...args],
  },
  askCustomerAnalytics: vi.fn().mockResolvedValue({
    answer: "Fill rate is under pressure in the selected scope.",
    provider: "codex",
    model: "gpt-5.5",
    tier: "deep",
    evidence: ["KPIs", "Top demand customers", "Lowest fill-rate customers"],
  }),
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
    tree: [{ name: "CA", value: 10000, fill_rate: 90, children: [{ name: "On Premise", value: 6000, fill_rate: 92, children: [] }] }],
  }),
  fetchCustomerAnalyticsHeatmap: vi.fn().mockResolvedValue({
    items: [{ item_id: "ITEM001", item_desc: "Widget A" }],
    states: ["CA", "TX"],
    cells: [{ item_id: "ITEM001", state: "CA", demand_qty: 5000, customer_count: 20, fill_rate: 95 }],
    metric: "demand_qty",
  }),
  fetchCustomerAnalyticsChannelMix: vi.fn().mockResolvedValue({
    tree: [{ name: "On Premise", value: 8000, customer_count: 200, children: [] }],
    grand_total: 8000,
    total_customers: 200,
    top_channel: "On Premise",
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
  fetchCustomerAnalyticsKpis: vi.fn().mockResolvedValue({
    total_demand: { value: 500000, delta: 3.2 },
    fill_rate: { value: 91.5, delta: -0.8 },
    lost_sales_oos: { value: 42000, delta: -5.1 },
    active_customers: { value: 1250, delta: 1.3 },
    demand_concentration: { value: 78.5, delta: 0.5 },
    order_to_demand_ratio: { value: 0.95, delta: 0.02 },
  }),
  fetchCustomerAnalyticsFilterOptions: vi.fn().mockResolvedValue({
    channels: ["On Premise", "Off Premise"],
    store_types: ["Chain", "Independent"],
    states: ["CA", "TX", "NY"],
  }),
  fetchCustomerAnalyticsLifecycle: vi.fn().mockResolvedValue({
    cohort_heatmap: [
      { cohort_month: "2025-01", months_since: 0, retention_pct: 100 },
      { cohort_month: "2025-01", months_since: 1, retention_pct: 85 },
    ],
    cohort_months: ["2025-01", "2025-02"],
    max_months_since: 3,
    waterfall: [
      { label: "New", value: 120, type: "new" },
      { label: "Churned", value: -30, type: "churned" },
      { label: "Net", value: 90, type: "net" },
    ],
  }),
  fetchCustomerAnalyticsDemandAtRisk: vi.fn().mockResolvedValue({
    bars: [
      { label: "Total Demand", value: 500000, type: "total" },
      { label: "Concentration Risk", value: -50000, type: "risk" },
      { label: "OOS Risk", value: -42000, type: "risk" },
      { label: "Churn Risk", value: -18000, type: "risk" },
      { label: "Secure Demand", value: 390000, type: "secure" },
    ],
  }),
  fetchCustomerAnalyticsAffinity: vi.fn().mockResolvedValue({
    customers: ["Acme Corp", "Beta Inc"],
    items: ["Widget A", "Widget B"],
    cells: [
      { customer: "Acme Corp", item: "Widget A", demand_qty: 5000 },
      { customer: "Beta Inc", item: "Widget B", demand_qty: 3000 },
    ],
  }),
  fetchCustomerAnalyticsOrderPatterns: vi.fn().mockResolvedValue({
    frequency: [
      { bin: "1-5", count: 50 },
      { bin: "6-10", count: 30 },
      { bin: "11+", count: 20 },
    ],
    regularity: [
      { customer: "Acme Corp", avg_interval: 14.5, cv: 0.3, total_orders: 24 },
    ],
  }),
  fetchCustomerAnalyticsDemandFlow: vi.fn().mockResolvedValue({
    nodes: [
      { name: "Warehouse A" },
      { name: "CA" },
      { name: "On Premise" },
    ],
    links: [
      { source: "Warehouse A", target: "CA", value: 10000 },
      { source: "CA", target: "On Premise", value: 8000 },
    ],
  }),
  triggerRecalculateCustomerAnalytics: vi
    .fn()
    .mockResolvedValue({ job_id: "job-ca-1", status: "queued" }),
}));

// RecalculateButton in the header polls the jobs API; stub it out.
vi.mock("@/api/queries/jobs", () => ({
  fetchActiveJobs: vi.fn().mockResolvedValue({ jobs: [] }),
  fetchJobDetail: vi.fn().mockResolvedValue(null),
}));

const { CustomerAnalyticsTab } = await import("@/tabs/CustomerAnalyticsTab");

function renderTab() {
  return render(
    <TestQueryWrapper>
      <CustomerAnalyticsTab />
    </TestQueryWrapper>,
  );
}

async function openView(name: string) {
  fireEvent.click(await screen.findByRole("tab", { name }));
}

describe("CustomerAnalyticsTab", () => {
  it("opens as a focused overview instead of rendering every analysis at once", async () => {
    renderTab();
    expect(await screen.findByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(await screen.findByText("Customer Demand Map")).toBeInTheDocument();
    expect(screen.queryByText("Customer Ranking")).not.toBeInTheDocument();
    expect(screen.queryByText("Customer-Item Affinity")).not.toBeInTheDocument();
  });

  it("switches between task-oriented analytics views", async () => {
    renderTab();
    fireEvent.click(await screen.findByRole("tab", { name: "Customers" }));
    expect(await screen.findByText("Customer Ranking")).toBeInTheDocument();
    expect(screen.queryByText("Customer Demand Map")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Buying behavior" }));
    expect(await screen.findByText("Customer-Item Affinity")).toBeInTheDocument();
  });

  it("supports arrow-key navigation between analytics views", async () => {
    renderTab();
    const overview = await screen.findByRole("tab", { name: "Overview" });
    fireEvent.keyDown(overview, { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "Customers" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("embeds a grounded Customer Intelligence question flow", async () => {
    renderTab();
    const input = await screen.findByPlaceholderText(/Ask about demand, service, or customers/i);
    fireEvent.change(input, { target: { value: "Why is fill rate falling?" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask customer intelligence" }));
    expect(
      await screen.findByText("Fill rate is under pressure in the selected scope."),
    ).toBeInTheDocument();
    expect(screen.getByText(/gpt-5.5/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Customers" }));
    expect(
      screen.queryByText("Fill rate is under pressure in the selected scope."),
    ).not.toBeInTheDocument();
  });

  it("U4.4: opens with a page heading matching its sidebar label + a description", async () => {
    renderTab();
    // The tab must lead with a title+description header like every other tab,
    // not a bare wall of KPI tiles. The heading text must match the sidebar
    // label ("Customer Analytics").
    const heading = await screen.findByRole("heading", { name: /Customer Analytics/i });
    expect(heading).toBeInTheDocument();
    expect(
      screen.getByText(/Move from demand footprint to customer risk/i),
    ).toBeInTheDocument();
  });

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

  it("renders filter dropdowns", async () => {
    renderTab();
    await waitFor(() => {
      // State dropdown
      expect(screen.getByText("All states")).toBeDefined();
      // Channel is now a searchable combobox (U7.11) with the placeholder shown
      // on the input, matching the Store Type affordance beside it.
      expect(screen.getByRole("combobox", { name: "Channel" })).toBeDefined();
      expect(screen.getByPlaceholderText("All channels")).toBeDefined();
      // Store Type is a searchable combobox (U5.11) with the placeholder shown
      // on the input, not a native <option>.
      expect(screen.getByRole("combobox", { name: "Store Type" })).toBeDefined();
      expect(screen.getByPlaceholderText("All types")).toBeDefined();
    });
  });

  it("U7.11: Channel filter is a searchable combobox matching Store Type (consistent affordance)", async () => {
    renderTab();
    await waitFor(() => {
      // Both Channel and Store Type expose the combobox role — same control type
      // sitting side by side (previously Channel was a raw native <select>).
      expect(screen.getByRole("combobox", { name: "Channel" })).toBeDefined();
      expect(screen.getByRole("combobox", { name: "Store Type" })).toBeDefined();
    });
  });

  it("renders KPI cards", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Total Demand")).toBeDefined();
      // "Fill Rate" appears in KPI, heatmap, and ranking — use getAllByText
      expect(screen.getAllByText("Fill Rate").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Lost Sales (OOS)")).toBeDefined();
      expect(screen.getByText("Active Customers")).toBeDefined();
      expect(screen.getByText("Demand Concentration")).toBeDefined();
      expect(screen.getByText("Order-to-Demand Ratio")).toBeDefined();
    });
  });

  it("renders map panel with metric buttons", async () => {
    renderTab();
    await waitFor(() => {
      // "Customers" appears in map buttons, heatmap headers, and segment table
      expect(screen.getAllByText("Customers").length).toBeGreaterThanOrEqual(1);
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
    await openView("Segments");
    await waitFor(() => {
      expect(screen.getByText("Item x State Heatmap")).toBeDefined();
    });
  });

  it("renders channel mix sunburst", async () => {
    renderTab();
    await openView("Segments");
    await waitFor(() => {
      expect(screen.getByText("Channel Mix")).toBeDefined();
    });
  });

  it("renders segment trends with segment-by buttons", async () => {
    renderTab();
    await openView("Segments");
    await waitFor(() => {
      expect(screen.getByText("Segment Trends")).toBeDefined();
      expect(screen.getAllByText("Channel").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Store Type").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders OOS impact bubble chart", async () => {
    renderTab();
    await openView("Service risk");
    await waitFor(() => {
      expect(screen.getByText("OOS Impact Analysis")).toBeDefined();
    });
  });

  it("renders customer ranking panel", async () => {
    renderTab();
    await openView("Customers");
    await waitFor(() => {
      expect(screen.getByText("Customer Ranking")).toBeDefined();
      expect(screen.getByText("Top by Demand")).toBeDefined();
      expect(screen.getByText("Worst Fill Rate")).toBeDefined();
    });
  });

  it("renders lifecycle panel", async () => {
    renderTab();
    await openView("Customers");
    await waitFor(() => {
      expect(screen.getByText("Customer Lifecycle")).toBeDefined();
    });
  });

  it("renders demand at risk", async () => {
    renderTab();
    await openView("Service risk");
    await waitFor(() => {
      expect(screen.getByText("Demand at Risk")).toBeDefined();
    });
  });

  it("renders affinity heatmap", async () => {
    renderTab();
    await openView("Buying behavior");
    await waitFor(() => {
      expect(screen.getByText("Customer-Item Affinity")).toBeDefined();
    });
  });

  it("renders order patterns", async () => {
    renderTab();
    await openView("Buying behavior");
    await waitFor(() => {
      expect(screen.getByText("Order Patterns")).toBeDefined();
    });
  });

  it("renders demand flow sankey", async () => {
    renderTab();
    await openView("Buying behavior");
    await waitFor(() => {
      expect(screen.getByText("Demand Flow")).toBeDefined();
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
    await openView("Segments");
    await waitFor(() => {
      const charts = screen.getAllByTestId("echart");
      expect(charts.length).toBeGreaterThanOrEqual(2);
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
      expect(screen.getByText("Clear filters")).toBeDefined();
    });
  });

  it("renders date range inputs", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("From")).toBeDefined();
      expect(screen.getByText("To")).toBeDefined();
    });
  });

  it("renders KPI summary cards container with aria-label", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByLabelText("KPI summary cards")).toBeDefined();
    });
  });

  it("renders sunburst metric toggle buttons", async () => {
    renderTab();
    await openView("Segments");
    await waitFor(() => {
      expect(screen.getByText("Demand Volume")).toBeDefined();
      expect(screen.getByText("Customer Count")).toBeDefined();
    });
  });

  it("renders customer search in ranking panel", async () => {
    renderTab();
    await openView("Customers");
    await waitFor(() => {
      expect(screen.getByPlaceholderText("Search customer...")).toBeDefined();
    });
  });
});
