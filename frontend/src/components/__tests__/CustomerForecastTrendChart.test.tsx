import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import type { CustomerBlendTrend } from "@/api/queries/customerForecast";
import { ThemeProvider } from "@/context/ThemeContext";
import { CustomerForecastTrendChart } from "../CustomerForecastTrendChart";

vi.mock("recharts");

const trend: CustomerBlendTrend = {
  run_id: "blend-run-1",
  status: "ready",
  planning_month: "2026-07-01",
  completed_at: "2026-07-14T12:00:00Z",
  backtest_run_id: "backtest-run-1",
  bottom_up_staging_run_id: "bottom-up-run-1",
  backtest_gate: { passed: true },
  filters_applied: {},
  filter_notes: [],
  accuracy: {
    common_actual_qty: 100,
    customer_bottom_up_wape_pct: 10,
    source_champion_wape_pct: 5,
    customer_blend_wape_pct: 2.5,
  },
  coverage: {
    blended_rows: 18,
    champion_fallback_rows: 2,
    global_customer_only_excluded_count: 1,
  },
  months: [
    {
      month: "2026-06-01",
      phase: "backtest",
      actual_qty: 100,
      customer_bottom_up_qty: 90,
      source_champion_qty: 105,
      customer_blend_qty: 97.5,
      lower_bound: null,
      upper_bound: null,
      blended_dfu_count: 1,
      fallback_dfu_count: 0,
    },
    {
      month: "2026-07-01",
      phase: "staged",
      actual_qty: null,
      customer_bottom_up_qty: 92,
      source_champion_qty: 104,
      customer_blend_qty: 98,
      lower_bound: 80,
      upper_bound: 120,
      blended_dfu_count: 1,
      fallback_dfu_count: 0,
    },
  ],
};

function Wrapper({ children }: { children: ReactNode }) {
  return <ThemeProvider value={{ theme: "light" }}>{children}</ThemeProvider>;
}

describe("CustomerForecastTrendChart", () => {
  it("shows exact-lineage backtest and staged customer forecasts", () => {
    render(<CustomerForecastTrendChart trend={trend} />, { wrapper: Wrapper });

    expect(screen.getByTestId("composed-chart")).toBeInTheDocument();
    expect(screen.getByText("Customer Bottom-Up WAPE 10.0%")).toBeInTheDocument();
    expect(screen.getByText("Customer Blend WAPE 2.5%")).toBeInTheDocument();
    expect(screen.getByText("18 blended · 2 fallback")).toBeInTheDocument();
    expect(screen.getByText("Jul 2026 · run blend-r")).toBeInTheDocument();
    expect(
      screen.getByRole("table", { name: "Customer forecast monthly comparison" })
    ).toBeInTheDocument();
  });

  it("shows an empty state when no comparison rows exist", () => {
    render(<CustomerForecastTrendChart trend={{ ...trend, months: [] }} />, { wrapper: Wrapper });

    expect(
      screen.getByText("No customer blend trend is available for these filters.")
    ).toBeInTheDocument();
  });
});
