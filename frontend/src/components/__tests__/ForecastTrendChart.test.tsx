import { describe, it, expect, vi } from "vitest";
import { render as rtlRender, screen } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";
import { ThemeProvider } from "@/context/ThemeContext";

const ThemeWrapper = ({ children }: { children: ReactNode }) => (
  <ThemeProvider value={{ theme: "light" }}>{children}</ThemeProvider>
);

const render = (ui: ReactElement) => rtlRender(ui, { wrapper: ThemeWrapper });

// recharts is replaced by the shared mock at frontend/__mocks__/recharts.tsx,
// which renders ComposedChart as <div data-testid="composed-chart"> and leaf
// series as null. We assert the container / empty-state, not pixels.
vi.mock("recharts");

import { ForecastTrendChart } from "@/components/ForecastTrendChart";

const sample = [
  { month: "2026-01", forecast: 1200, actual: 1100 },
  { month: "2026-02", forecast: 1300, actual: 1250 },
];

describe("ForecastTrendChart", () => {
  it("renders a chart when data is provided", () => {
    render(<ForecastTrendChart data={sample} />);
    expect(screen.getByTestId("composed-chart")).toBeTruthy();
  });

  it("shows an empty state and no chart when there is no data", () => {
    render(<ForecastTrendChart data={[]} />);
    expect(screen.getByText("No forecast data available")).toBeTruthy();
    expect(screen.queryByTestId("composed-chart")).toBeNull();
  });

  it("still renders when includeCI is set and 80% quantiles are present", () => {
    const withCI = [
      { month: "2026-01", forecast: 1200, actual: 1100, lower_80: 1000, upper_80: 1400 },
      { month: "2026-02", forecast: 1300, actual: 1250, lower_80: 1100, upper_80: 1500 },
    ];
    render(<ForecastTrendChart data={withCI} includeCI />);
    expect(screen.getByTestId("composed-chart")).toBeTruthy();
  });
});
