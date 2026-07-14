import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";

const mockReadiness = vi.fn();
const mockLatestRun = vi.fn();
const mockLatestCompletedRun = vi.fn();
const mockGenerate = vi.fn();
const mockCancel = vi.fn();
const mockSeries = vi.fn();

vi.mock("@/api/queries/customerForecast", () => ({
  customerForecastKeys: {
    readiness: ["customer-forecast", "readiness"],
    latestRun: ["customer-forecast", "latest-run"],
    latestCompletedRun: ["customer-forecast", "latest-completed-run"],
    series: (filters: unknown) => ["customer-forecast", "series", filters],
  },
  fetchCustomerForecastReadiness: () => mockReadiness(),
  fetchLatestCustomerForecastRun: (completedOnly = false) =>
    completedOnly ? mockLatestCompletedRun() : mockLatestRun(),
  generateCustomerForecast: () => mockGenerate(),
  cancelCustomerForecastRun: (runId: string) => mockCancel(runId),
  fetchCustomerForecastSeries: (filters: unknown) => mockSeries(filters),
  customerForecastExportUrl: () => "/customer-forecast/export?run_id=run-1",
}));

vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Line: ({ name }: { name: string }) => <div>{name}</div>,
  CartesianGrid: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  Legend: () => null,
}));

const readiness = {
  ready: true,
  planning_month: "2026-07-01",
  history_start: "2025-01-01",
  history_end: "2026-06-30",
  forecast_start: "2026-07-01",
  forecast_end: "2027-12-31",
  history_months: 18,
  horizon_months: 18,
  source_latest_month: "2026-06-01",
  total_series: 12,
  eligible_series: 10,
  fallback_series: 2,
  forecastable_series: 12,
  skipped_series: 0,
  invalid_key_rows: 0,
  duplicate_grains: 0,
  negative_rows: 0,
  blockers: [],
};

describe("CustomerForecastPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockReadiness.mockResolvedValue(readiness);
    mockLatestRun.mockResolvedValue(null);
    mockLatestCompletedRun.mockResolvedValue(null);
    mockGenerate.mockResolvedValue({ run_id: "run-1", job_id: "job-1", status: "queued" });
    mockSeries.mockResolvedValue({
      run: null,
      item_id: "ITEM-1",
      location_id: "LOC-1",
      customer_no: "CUST-1",
      history: [{ month: "2026-06-01", actual_qty: 8 }],
      forecast: [
        {
          month: "2026-07-01",
          forecast_qty: 9,
          lower_bound: null,
          upper_bound: null,
          model_id: "croston",
        },
      ],
    });
  });

  it("explains the rolling windows and starts generation", async () => {
    const { CustomerForecastPanel } = await import("../CustomerForecastPanel");
    render(
      <TestQueryWrapper>
        <CustomerForecastPanel />
      </TestQueryWrapper>
    );

    expect(await screen.findByText("Customer Forecast Generation")).toBeInTheDocument();
    expect(await screen.findByText(/Jan 2025.*Jun 2026/)).toBeInTheDocument();
    expect(screen.getByText(/Jul 2026.*Dec 2027/)).toBeInTheDocument();
    expect(screen.getByText("12 forecastable series")).toBeInTheDocument();
    expect(screen.getByText(/10 Chronos 2E.*2 Croston/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Generate Customer Forecasts" }));
    await waitFor(() => expect(mockGenerate).toHaveBeenCalledOnce());
  });

  it("shows failure guidance and retries with a new generation", async () => {
    mockLatestRun.mockResolvedValue({
      run_id: "run-failed",
      job_id: "job-failed",
      status: "failed",
      planning_month: "2026-07-01",
      history_start: "2025-01-01",
      history_end: "2026-06-30",
      forecast_start: "2026-07-01",
      forecast_end: "2027-12-31",
      eligible_series: 0,
      row_count: 0,
      skipped_series: 0,
      model_id: "chronos2_enriched",
      created_at: "2026-07-13T12:00:00Z",
      started_at: "2026-07-13T12:01:00Z",
      completed_at: "2026-07-13T12:02:00Z",
      error_summary: "Chronos model loading failed",
      skip_reason_counts: {},
      model_route_counts: {},
    });
    const { CustomerForecastPanel } = await import("../CustomerForecastPanel");
    render(
      <TestQueryWrapper>
        <CustomerForecastPanel />
      </TestQueryWrapper>
    );

    expect(await screen.findByText("Chronos model loading failed")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry Generation" }));
    await waitFor(() => expect(mockGenerate).toHaveBeenCalledOnce());
  });

  it("loads a selected customer series and exposes export", async () => {
    const completedRun = {
      run_id: "run-1",
      job_id: "job-1",
      status: "completed",
      planning_month: "2026-07-01",
      history_start: "2025-01-01",
      history_end: "2026-06-30",
      forecast_start: "2026-07-01",
      forecast_end: "2027-12-31",
      eligible_series: 12,
      row_count: 216,
      skipped_series: 0,
      model_id: "chronos2_enriched",
      created_at: "2026-07-13T12:00:00Z",
      started_at: "2026-07-13T12:01:00Z",
      completed_at: "2026-07-13T12:02:00Z",
      error_summary: null,
      skip_reason_counts: {},
      model_route_counts: { chronos2_enriched: 10, croston: 2 },
    };
    mockLatestRun.mockResolvedValue(completedRun);
    mockLatestCompletedRun.mockResolvedValue(completedRun);
    const { CustomerForecastPanel } = await import("../CustomerForecastPanel");
    render(
      <TestQueryWrapper>
        <CustomerForecastPanel />
      </TestQueryWrapper>
    );

    fireEvent.change(await screen.findByLabelText("Item"), { target: { value: "ITEM-1" } });
    fireEvent.change(screen.getByLabelText("Location"), { target: { value: "LOC-1" } });
    fireEvent.change(screen.getByLabelText("Customer"), { target: { value: "CUST-1" } });
    fireEvent.click(screen.getByRole("button", { name: "Load Series" }));

    expect((await screen.findAllByText("Actual demand")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("Customer forecast").length).toBeGreaterThan(0);
    expect(screen.getByText("Croston/SBA")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Export CSV" })).toHaveAttribute(
      "href",
      "/customer-forecast/export?run_id=run-1"
    );
  });
});
