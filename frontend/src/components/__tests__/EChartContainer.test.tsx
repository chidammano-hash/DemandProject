import { describe, it, expect, vi } from "vitest";
import { render } from "@testing-library/react";

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
    series: [{ type: "line" as const, data: [1, 2, 3] }],
  };

  it("renders chart container", () => {
    const { getByTestId } = render(
      <EChartContainer option={baseOption} theme="light" />
    );
    expect(getByTestId("echarts-mock")).toBeInTheDocument();
  });

  it("applies default height", () => {
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

  it("applies className", () => {
    const { getByTestId } = render(
      <EChartContainer option={baseOption} theme="light" className="my-chart" />
    );
    expect(getByTestId("echarts-mock").className).toContain("my-chart");
  });
});
