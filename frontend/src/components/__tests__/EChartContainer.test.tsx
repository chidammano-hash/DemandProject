import { describe, it, expect, vi } from "vitest";
import { render as rtlRender } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";
import { ThemeProvider } from "@/context/ThemeContext";

const ThemeWrapper = ({ children }: { children: ReactNode }) => (
  <ThemeProvider value={{ theme: "light" }}>{children}</ThemeProvider>
);

const render = (ui: ReactElement) => rtlRender(ui, { wrapper: ThemeWrapper });

// Mock echarts-for-react to avoid canvas rendering in jsdom
vi.mock("echarts-for-react/lib/core", () => ({
  default: ({ style, className }: any) => (
    <div data-testid="echarts-mock" style={style} className={className} />
  ),
}));

vi.mock("echarts/core", () => ({
  use: vi.fn(),
}));

vi.mock("echarts/charts", () => ({
  LineChart: {},
}));

vi.mock("echarts/components", () => ({
  GridComponent: {},
  TooltipComponent: {},
  LegendComponent: {},
  DataZoomComponent: {},
}));

vi.mock("echarts/renderers", () => ({
  CanvasRenderer: {},
}));

// Import after mocks
import { EChartContainer } from "@/components/EChartContainer";

describe("EChartContainer", () => {
  const baseOption = {
    xAxis: { type: "category" as const, data: ["A", "B", "C"] },
    yAxis: { type: "value" as const },
    series: [{ name: "sales", type: "line" as const, data: [1, 2, 3] }],
  };

  it("renders chart container", () => {
    const { getByTestId } = render(
      <EChartContainer option={baseOption} theme="light" />
    );
    expect(getByTestId("echarts-mock")).toBeInTheDocument();
  });

  it("applies default height to inner chart", () => {
    const { getByTestId } = render(
      <EChartContainer option={baseOption} theme="light" />
    );
    const el = getByTestId("echarts-mock");
    expect(el.style.height).toBe("380px");
  });

  it("applies custom height", () => {
    const { getByTestId } = render(
      <EChartContainer option={baseOption} theme="dark" height={500} />
    );
    const el = getByTestId("echarts-mock");
    expect(el.style.height).toBe("500px");
  });

  it("applies className to wrapper", () => {
    const { getByRole } = render(
      <EChartContainer option={baseOption} theme="light" className="my-chart" />
    );
    const wrapper = getByRole("img");
    expect(wrapper.className).toContain("my-chart");
  });

  it("wraps chart with role=img for accessibility", () => {
    const { getByRole } = render(
      <EChartContainer option={baseOption} theme="light" />
    );
    const wrapper = getByRole("img");
    expect(wrapper).toBeInTheDocument();
    expect(wrapper.getAttribute("aria-label")).toMatch(/chart/i);
  });

  it("computes aria-label from series names", () => {
    const { getByRole } = render(
      <EChartContainer option={baseOption} theme="light" />
    );
    const wrapper = getByRole("img");
    const label = wrapper.getAttribute("aria-label") ?? "";
    expect(label).toContain("sales");
    expect(label).toContain("3 data points");
  });

  it("respects explicit ariaLabel override", () => {
    const { getByRole } = render(
      <EChartContainer option={baseOption} theme="light" ariaLabel="Monthly sales trend" />
    );
    expect(getByRole("img").getAttribute("aria-label")).toBe("Monthly sales trend");
  });

  it("renders sr-only data table fallback", () => {
    const { container } = render(
      <EChartContainer option={baseOption} theme="light" />
    );
    const table = container.querySelector("table.sr-only");
    expect(table).not.toBeNull();
    // Header + 3 data rows
    expect(table?.querySelectorAll("tbody tr").length).toBe(3);
  });
});
