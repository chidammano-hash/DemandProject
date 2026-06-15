import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { SegmentSparklines, sparklineColors } from "../SegmentSparklines";
import { ThemeProvider } from "@/context/ThemeContext";
import { TREND_COLORS_BY_THEME } from "@/constants/colors";
import type { Theme } from "@/types";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";

// recharts ResponsiveContainer needs a measurable parent in jsdom; stub it so the
// sparkline area renders without a 0x0 warning loop.
vi.mock("recharts", async () => {
  const actual = await vi.importActual<typeof import("recharts")>("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: ReactNode }) => (
      <div style={{ width: 100, height: 24 }}>{children}</div>
    ),
  };
});

vi.mock("@/api/queries/customer-analytics", async () => {
  const actual = await vi.importActual<typeof import("@/api/queries/customer-analytics")>(
    "@/api/queries/customer-analytics",
  );
  return {
    ...actual,
    fetchCustomerAnalyticsSegmentTrends: vi.fn().mockResolvedValue({
      segment_by: "store_type_desc",
      segments: [
        {
          // The dominant real-world row: store_type with no code is emitted as
          // the literal string "null" and is ~86% of all demand.
          segment: "null",
          total_customers: 12000,
          total_demand: 19755775,
          fill_rate: 95,
          mom_change: 3,
          trend: [{ month: "2026-01", demand_qty: 100 }],
        },
        {
          segment: "Chain Grocery Store",
          total_customers: 800,
          total_demand: 1529557,
          fill_rate: 97,
          mom_change: -2,
          trend: [{ month: "2026-01", demand_qty: 50 }],
        },
      ],
    }),
  };
});

function wrap(node: ReactNode, theme: Theme = "light") {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <ThemeProvider value={{ theme }}>
      <QueryClientProvider client={qc}>{node}</QueryClientProvider>
    </ThemeProvider>
  );
}

const filters = {} as CustomerAnalyticsFilters;

describe("SegmentSparklines (U5.10 — nullish segment must not render as literal 'null')", () => {
  it("renders a nullish segment as 'Unclassified', never the literal 'null'", async () => {
    render(
      <>
        {wrap(
          <SegmentSparklines
            filters={filters}
            segmentBy="store_type_desc"
            onSegmentByChange={() => {}}
          />,
        )}
      </>,
    );
    await waitFor(() => expect(screen.getByText("Chain Grocery Store")).toBeInTheDocument());
    expect(screen.getByText("Unclassified")).toBeInTheDocument();
    expect(screen.queryByText("null")).toBeNull();
  });

  it("shows a footnote explaining the Unclassified row when one is present", async () => {
    render(
      wrap(
        <SegmentSparklines
          filters={filters}
          segmentBy="store_type_desc"
          onSegmentByChange={() => {}}
        />,
      ),
    );
    await waitFor(() => expect(screen.getByText("Chain Grocery Store")).toBeInTheDocument());
    // "Unclassified" appears both as the row label and in the footnote.
    expect(screen.getAllByText(/unclassified/i).length).toBeGreaterThanOrEqual(2);
    // The footnote text (distinct from the row label) must explain the bucket.
    const footnote = screen.getByText(/no store-type code|no .* code/i);
    expect(footnote).toBeInTheDocument();
  });

  it("does NOT drop the Unclassified row (it is the majority of demand)", async () => {
    render(
      wrap(
        <SegmentSparklines
          filters={filters}
          segmentBy="store_type_desc"
          onSegmentByChange={() => {}}
        />,
      ),
    );
    await waitFor(() => expect(screen.getByText("Unclassified")).toBeInTheDocument());
    // Its demand value (19.8M compact) is still rendered.
    expect(screen.getByText("19.8M")).toBeInTheDocument();
  });
});

describe("SegmentSparklines (U5.12 — sparkline color must be theme-derived, not hardcoded hex)", () => {
  it("derives the stroke from the theme series color (not '#6366f1')", () => {
    const dark = sparklineColors(TREND_COLORS_BY_THEME.dark[0]);
    expect(dark.stroke).toBe(TREND_COLORS_BY_THEME.dark[0]);
    expect(dark.stroke).not.toBe("#6366f1");
    // Fill is the same hue at low opacity, so dark vs light differ.
    const light = sparklineColors(TREND_COLORS_BY_THEME.light[0]);
    expect(light.fill).not.toBe(dark.fill);
    expect(dark.fill).not.toBe("#e0e7ff");
  });

  it("mounts under the dark theme without throwing (consumes useChartColors)", async () => {
    render(
      wrap(
        <SegmentSparklines
          filters={filters}
          segmentBy="store_type_desc"
          onSegmentByChange={() => {}}
        />,
        "dark",
      ),
    );
    await waitFor(() => expect(screen.getByText("Chain Grocery Store")).toBeInTheDocument());
  });

  it("has no hardcoded chart hex literals left in the panel source", () => {
    const src = readFileSync(
      resolve(__dirname, "../SegmentSparklines.tsx"),
      "utf8",
    );
    // The two removed sparkline hex literals must be gone.
    expect(src).not.toContain("#6366f1");
    expect(src).not.toContain("#e0e7ff");
  });
});
