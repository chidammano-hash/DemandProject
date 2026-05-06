import { describe, it, expect, vi } from "vitest";
import { render as rtlRender, screen } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";
import { ThemeProvider } from "@/context/ThemeContext";

// ForecastTrendChart now renders via recharts. The shared mock at
// frontend/__mocks__/recharts.tsx renders a <div data-testid="composed-chart">.
vi.mock("recharts");

const ThemeWrapper = ({ children }: { children: ReactNode }) => (
  <ThemeProvider value={{ theme: "light" }}>{children}</ThemeProvider>
);

const render = (ui: ReactElement) => rtlRender(ui, { wrapper: ThemeWrapper });

import { ForecastTrendChart } from "@/components/ForecastTrendChart";

const defaultChartColors = {
  grid: "#e5e5e5",
  axis: "#737373",
  tooltip: "#ffffff",
};

const defaultSeriesColors = ["#6366f1", "#f59e0b"];

const sampleData = [
  { month: "2024-01", forecast: 1000, actual: 950 },
  { month: "2024-02", forecast: 1100, actual: 1050 },
  { month: "2024-03", forecast: 1200, actual: 1180 },
];

describe("ForecastTrendChart", () => {
  it("renders without crashing with empty data array", () => {
    render(
      <ForecastTrendChart
        data={[]}
        theme="light"
        chartColors={defaultChartColors}
        seriesColors={defaultSeriesColors}
      />
    );
  });

  it("shows empty state message when data is empty", () => {
    render(
      <ForecastTrendChart
        data={[]}
        theme="light"
        chartColors={defaultChartColors}
        seriesColors={defaultSeriesColors}
      />
    );
    expect(screen.getByText("No forecast data available")).toBeInTheDocument();
  });

  it("does NOT show empty state when data is provided", () => {
    render(
      <ForecastTrendChart
        data={sampleData}
        theme="light"
        chartColors={defaultChartColors}
        seriesColors={defaultSeriesColors}
      />
    );
    expect(screen.queryByText("No forecast data available")).not.toBeInTheDocument();
  });

  it("renders a recharts ComposedChart container when data is provided", () => {
    render(
      <ForecastTrendChart
        data={sampleData}
        theme="light"
        chartColors={defaultChartColors}
        seriesColors={defaultSeriesColors}
      />
    );
    expect(screen.getByTestId("composed-chart")).toBeInTheDocument();
  });

  it("renders with dark theme without crashing", () => {
    render(
      <ForecastTrendChart
        data={sampleData}
        theme="dark"
        chartColors={{ grid: "#333", axis: "#aaa", tooltip: "#222" }}
        seriesColors={["#818cf8", "#fbbf24"]}
      />
    );
    expect(screen.getByTestId("composed-chart")).toBeInTheDocument();
  });

  it("renders with a single data point without crashing", () => {
    render(
      <ForecastTrendChart
        data={[{ month: "2024-01", forecast: 500, actual: 480 }]}
        theme="light"
        chartColors={defaultChartColors}
        seriesColors={defaultSeriesColors}
      />
    );
    expect(screen.getByTestId("composed-chart")).toBeInTheDocument();
  });

  it("renders with many data points without crashing", () => {
    const largeData = Array.from({ length: 24 }, (_, i) => ({
      month: `2023-${String(i % 12 + 1).padStart(2, "0")}`,
      forecast: 1000 + i * 50,
      actual: 980 + i * 45,
    }));
    render(
      <ForecastTrendChart
        data={largeData}
        theme="light"
        chartColors={defaultChartColors}
        seriesColors={defaultSeriesColors}
      />
    );
    expect(screen.getByTestId("composed-chart")).toBeInTheDocument();
  });

  it("renders with zero values in data without crashing", () => {
    render(
      <ForecastTrendChart
        data={[
          { month: "2024-01", forecast: 0, actual: 0 },
          { month: "2024-02", forecast: 0, actual: 100 },
        ]}
        theme="light"
        chartColors={defaultChartColors}
        seriesColors={defaultSeriesColors}
      />
    );
    expect(screen.getByTestId("composed-chart")).toBeInTheDocument();
  });

  it("renders with includeCI + quantile band data without crashing (UX-3)", () => {
    const dataWithCI = [
      { month: "2024-01", forecast: 1000, actual: 950, lower_80: 900, upper_80: 1100 },
      { month: "2024-02", forecast: 1100, actual: 1050, lower_80: 990, upper_80: 1210 },
      { month: "2024-03", forecast: 1200, actual: 1180, lower_80: 1080, upper_80: 1320 },
    ];
    render(
      <ForecastTrendChart
        data={dataWithCI}
        theme="light"
        chartColors={defaultChartColors}
        seriesColors={defaultSeriesColors}
        includeCI
      />,
    );
    expect(screen.getByTestId("composed-chart")).toBeInTheDocument();
  });

  it("ignores includeCI flag when data lacks quantile fields", () => {
    // includeCI=true but data doesn't have lower_80/upper_80 — should
    // degrade gracefully to the regular forecast/actual view.
    render(
      <ForecastTrendChart
        data={sampleData}
        theme="light"
        chartColors={defaultChartColors}
        seriesColors={defaultSeriesColors}
        includeCI
      />,
    );
    expect(screen.getByTestId("composed-chart")).toBeInTheDocument();
  });
});
