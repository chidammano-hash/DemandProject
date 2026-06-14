import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { CustomerTreemap } from "../CustomerTreemap";
import type { CustomerAnalyticsFilters } from "@/api/queries/customer-analytics";

// Capture the props handed to the ECharts wrapper so we can assert the chart is
// given a non-zero drawable width (U6.3 — height-only style collapses the treemap)
// AND that the ECharts option no longer relies on the broken
// `visualMap.dimension: "fill_rate"` binding (U7.1 — that resolves every node
// out-of-range and paints the treemap blank).
type ChartProps = { style?: React.CSSProperties; option?: Record<string, unknown> };
const chartProps: Array<ChartProps> = [];
vi.mock("@/components/echarts-modular", () => ({
  ModularReactECharts: (props: ChartProps) => {
    chartProps.push(props);
    return <div data-testid="treemap-echart" style={props.style} />;
  },
  default: (props: ChartProps) => {
    chartProps.push(props);
    return <div data-testid="treemap-echart" style={props.style} />;
  },
}));

vi.mock("@/api/queries/customer-analytics", async () => {
  const actual = await vi.importActual<typeof import("@/api/queries/customer-analytics")>(
    "@/api/queries/customer-analytics",
  );
  return {
    ...actual,
    fetchCustomerAnalyticsTreemap: vi.fn().mockResolvedValue({
      tree: [
        {
          name: "FL",
          value: 12088589.5,
          children: [{ name: "Off Premise Chains", value: 9000000, fill_rate: 98 }],
        },
      ],
    }),
  };
});

function wrap(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{node}</QueryClientProvider>;
}

const filters = {} as CustomerAnalyticsFilters;

describe("CustomerTreemap (U6.3 — must not collapse to zero width)", () => {
  it("gives the ECharts chart an explicit 100% width so the treemap has drawable area", async () => {
    chartProps.length = 0;
    render(wrap(<CustomerTreemap filters={filters} />));
    await waitFor(() => expect(screen.getByTestId("treemap-echart")).toBeInTheDocument());
    const last = chartProps[chartProps.length - 1];
    expect(last.style?.width).toBe("100%");
    expect(last.style?.height).toBe(360);
  });

  it("wraps the chart in a full-width container (role=img)", async () => {
    render(wrap(<CustomerTreemap filters={filters} />));
    await waitFor(() => expect(screen.getByTestId("treemap-echart")).toBeInTheDocument());
    const imgWrapper = document.querySelector('[role="img"]');
    expect(imgWrapper).not.toBeNull();
    expect(imgWrapper?.className).toContain("w-full");
  });
});

describe("CustomerTreemap (U7.1 — must not use the broken visualMap.dimension binding)", () => {
  it("does not bind visualMap to a string dimension (which paints every node out-of-range)", async () => {
    chartProps.length = 0;
    render(wrap(<CustomerTreemap filters={filters} />));
    await waitFor(() => expect(screen.getByTestId("treemap-echart")).toBeInTheDocument());
    const option = chartProps[chartProps.length - 1].option as
      | { visualMap?: { dimension?: unknown } }
      | undefined;
    // The bug: `visualMap.dimension: "fill_rate"` — a string is not a valid
    // dimension index for a scalar-value treemap, so ECharts paints the
    // out-of-range (transparent) color and the rectangles vanish.
    expect(option?.visualMap?.dimension).toBeUndefined();
  });

  it("colors each leaf node via itemStyle.color mapped from its fill_rate", async () => {
    chartProps.length = 0;
    render(wrap(<CustomerTreemap filters={filters} />));
    await waitFor(() => expect(screen.getByTestId("treemap-echart")).toBeInTheDocument());
    const option = chartProps[chartProps.length - 1].option as {
      series?: Array<{ data?: Array<{ name: string; children?: Array<{ itemStyle?: { color?: string } }> }> }>;
    };
    const root = option.series?.[0]?.data?.[0];
    const child = root?.children?.[0];
    // The fill_rate=98 child must carry an explicit color (green end of the ramp)
    // so the treemap actually draws, independent of any visualMap.
    expect(child?.itemStyle?.color).toBeTruthy();
    expect(typeof child?.itemStyle?.color).toBe("string");
  });
});
