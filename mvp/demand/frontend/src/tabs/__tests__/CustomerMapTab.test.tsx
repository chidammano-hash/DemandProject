import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// Mock leaflet CSS import
vi.mock("leaflet/dist/leaflet.css", () => ({}));

// Mock react-leaflet (including GeoJSON for choropleth)
vi.mock("react-leaflet", () => ({
  MapContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="map-container">{children}</div>
  ),
  TileLayer: () => null,
  GeoJSON: ({ onEachFeature, data }: { onEachFeature?: unknown; data?: unknown }) => (
    <div data-testid="geojson-layer" data-features={data ? "loaded" : "none"} />
  ),
  CircleMarker: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="circle-marker">{children}</div>
  ),
  Tooltip: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
}));

// Mock bundled GeoJSON asset
vi.mock("@/assets/us-states.json", () => ({
  default: {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { name: "California" }, geometry: { type: "Polygon", coordinates: [] } },
      { type: "Feature", properties: { name: "Texas" }, geometry: { type: "Polygon", coordinates: [] } },
    ],
  },
}));

vi.mock("@/api/queries", () => ({
  queryKeys: {
    customerMap: (groupBy: string) => ["customer-map", groupBy],
  },
  fetchCustomerMap: vi.fn().mockResolvedValue({
    locations: [
      { label: "CA", customer_count: 500, lat: 36.11, lon: -119.68 },
      { label: "TX", customer_count: 350, lat: 31.05, lon: -97.56 },
    ],
    group_by: "state",
    total: 850,
  }),
  STALE: { FIVE_MIN: 300000 },
}));

const { CustomerMapTab } = await import("@/tabs/CustomerMapTab");

function renderTab() {
  return render(
    <TestQueryWrapper>
      <CustomerMapTab />
    </TestQueryWrapper>,
  );
}

describe("CustomerMapTab", () => {
  it("renders without crashing", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Customer Map")).toBeDefined();
    });
  });

  it("renders group-by selector with state/city/zip buttons", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "state" })).toBeDefined();
      expect(screen.getByRole("button", { name: "city" })).toBeDefined();
      expect(screen.getByRole("button", { name: "zip" })).toBeDefined();
    });
  });

  it("renders map container", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByTestId("map-container")).toBeDefined();
    });
  });

  it("renders GeoJSON choropleth layer", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByTestId("geojson-layer")).toBeDefined();
    });
  });

  it("shows total customer count in description", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText(/850 customers/)).toBeDefined();
    });
  });

  it("does not render circle markers in state view", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Customer Map")).toBeDefined();
    });
    expect(screen.queryAllByTestId("circle-marker")).toHaveLength(0);
  });
});
