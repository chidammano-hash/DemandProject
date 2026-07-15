import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";

const mockReadiness = vi.fn();
const mockLatestRun = vi.fn();
const mockLatestCompletedRun = vi.fn();
const mockGenerate = vi.fn();
const mockCancel = vi.fn();
const mockRetry = vi.fn();
const mockSeries = vi.fn();
const mockLatestBacktest = vi.fn();
const mockBlendReadiness = vi.fn();
const mockLatestBlend = vi.fn();
const mockBlendSeries = vi.fn();

vi.mock("@/api/queries/customerForecast", () => ({
  customerForecastKeys: {
    readiness: ["customer-forecast", "readiness"],
    latestRun: ["customer-forecast", "latest-run"],
    latestCompletedRun: ["customer-forecast", "latest-completed-run"],
    series: (filters: unknown) => ["customer-forecast", "series", filters],
    latestBacktest: ["customer-forecast", "backtest", "latest"],
    blendReadiness: (customerRunId?: string) => [
      "customer-forecast",
      "blend",
      "readiness",
      customerRunId ?? null,
    ],
    latestBlend: ["customer-forecast", "blend", "latest"],
    blendSeries: (filters: unknown) => ["customer-forecast", "blend", "series", filters],
  },
  fetchCustomerForecastReadiness: () => mockReadiness(),
  fetchLatestCustomerForecastRun: (completedOnly = false) =>
    completedOnly ? mockLatestCompletedRun() : mockLatestRun(),
  generateCustomerForecast: () => mockGenerate(),
  cancelCustomerForecastRun: (runId: string) => mockCancel(runId),
  retryCustomerForecastRun: (runId: string) => mockRetry(runId),
  fetchCustomerForecastSeries: (filters: unknown) => mockSeries(filters),
  customerForecastExportUrl: (filters: { run_id?: string }) =>
    `/customer-forecast/export?run_id=${filters.run_id ?? "run-1"}`,
  fetchLatestCustomerForecastBacktest: () => mockLatestBacktest(),
  generateCustomerForecastBacktest: vi.fn(),
  fetchCustomerBlendReadiness: (customerRunId?: string) => mockBlendReadiness(customerRunId),
  fetchLatestCustomerBlend: () => mockLatestBlend(),
  generateCustomerBlend: vi.fn(),
  fetchCustomerBlendSeries: (filters: unknown) => mockBlendSeries(filters),
}));

vi.mock("@/api/queries/jobs", () => ({
  fetchJobDetail: vi.fn(),
  jobKeys: {
    detail: (jobId: string | null) => ["jobs", "detail", jobId],
  },
}));

vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Bar: ({ name }: { name: string }) => <div>{name}</div>,
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
  moving_average_series: 3,
  seasonal_repeat_series: 4,
  croston_series: 3,
  model_route_counts: {
    moving_average_3: 3,
    seasonal_repeat_12: 4,
    croston: 3,
  },
  dormant_series: 2,
  forecastable_series: 10,
  skipped_series: 2,
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
    mockLatestBacktest.mockResolvedValue(null);
    mockBlendReadiness.mockResolvedValue({
      ready: true,
      blockers: [],
      customer_run_id: "run-1",
      backtest_gate_passed: true,
      promotion_enabled: true,
      promotion_reason: "Validated customer blend",
    });
    mockLatestBlend.mockResolvedValue(null);
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
          model_id: "moving_average_3",
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
    expect(screen.getByText("10 forecastable series")).toBeInTheDocument();
    expect(screen.getByText(/3 3-Month Moving Average/)).toBeInTheDocument();
    expect(screen.getByText(/4 12-Month Seasonal Repeat/)).toBeInTheDocument();
    expect(screen.getByText(/3 Croston\/SBA/)).toBeInTheDocument();
    expect(screen.getByText(/Demand starting in the latest six months/)).toBeInTheDocument();
    expect(screen.queryByText(/Chronos 2E/)).not.toBeInTheDocument();
    expect(screen.getByText(/2 customer-SKUs ignored/)).toBeInTheDocument();
    expect(screen.getByText("Bottom-Up Blend Validation")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Generate Customer Forecasts" }));
    await waitFor(() => expect(mockGenerate).toHaveBeenCalledOnce());
  });

  it("keeps zero-count rule routes visible in coverage and run composition", async () => {
    const zeroRouteReadiness = {
      ...readiness,
      moving_average_series: 10,
      seasonal_repeat_series: 0,
      croston_series: 0,
      model_route_counts: {
        moving_average_3: 10,
        seasonal_repeat_12: 0,
        croston: 0,
      },
    };
    const completedRun = {
      run_id: "run-zero-routes",
      job_id: "job-zero-routes",
      status: "completed",
      planning_month: "2026-07-01",
      history_start: "2025-01-01",
      history_end: "2026-06-30",
      forecast_start: "2026-07-01",
      forecast_end: "2027-12-31",
      eligible_series: 10,
      row_count: 180,
      skipped_series: 0,
      model_id: "customer_rule_router",
      created_at: "2026-07-13T12:00:00Z",
      started_at: "2026-07-13T12:01:00Z",
      completed_at: "2026-07-13T12:02:00Z",
      error_summary: null,
      skip_reason_counts: {},
      model_route_counts: { moving_average_3: 10 },
      total_series: 10,
      completed_series: 10,
      total_batches: 1,
      completed_batches: 1,
      progress_pct: 100,
      eta_seconds: 0,
    };
    mockReadiness.mockResolvedValue(zeroRouteReadiness);
    mockLatestRun.mockResolvedValue(completedRun);
    mockLatestCompletedRun.mockResolvedValue(completedRun);

    const { CustomerForecastPanel } = await import("../CustomerForecastPanel");
    render(
      <TestQueryWrapper>
        <CustomerForecastPanel />
      </TestQueryWrapper>
    );

    expect(
      await screen.findByText(
        "10 3-Month Moving Average · 0 12-Month Seasonal Repeat · 0 Croston/SBA"
      )
    ).toBeInTheDocument();
    expect(screen.getByText("3-Month Moving Average (10)")).toBeInTheDocument();
    expect(screen.getByText("12-Month Seasonal Repeat (0)")).toBeInTheDocument();
    expect(screen.getByText("Croston/SBA (0)")).toBeInTheDocument();
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
      model_id: "customer_rule_router",
      created_at: "2026-07-13T12:00:00Z",
      started_at: "2026-07-13T12:01:00Z",
      completed_at: "2026-07-13T12:02:00Z",
      error_summary: "Croston generation failed",
      skip_reason_counts: {},
      model_route_counts: {},
      total_series: 10,
      completed_series: 5,
      total_batches: 2,
      completed_batches: 1,
      progress_pct: 54,
      eta_seconds: 600,
    });
    const { CustomerForecastPanel } = await import("../CustomerForecastPanel");
    render(
      <TestQueryWrapper>
        <CustomerForecastPanel />
      </TestQueryWrapper>
    );

    expect(await screen.findByRole("alert")).toHaveTextContent("Croston generation failed");
    fireEvent.click(screen.getByRole("button", { name: "Resume Saved Batches" }));
    await waitFor(() => expect(mockRetry).toHaveBeenCalledWith("run-failed"));
    expect(screen.getByText(/5 \/ 10 customer-SKUs completed/)).toBeInTheDocument();
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
      model_id: "customer_rule_router",
      created_at: "2026-07-13T12:00:00Z",
      started_at: "2026-07-13T12:01:00Z",
      completed_at: "2026-07-13T12:02:00Z",
      error_summary: null,
      skip_reason_counts: {},
      model_route_counts: {
        moving_average_3: 2,
        seasonal_repeat_12: 7,
        croston: 3,
      },
    };
    mockLatestRun.mockResolvedValue(completedRun);
    mockLatestCompletedRun.mockResolvedValue(completedRun);
    const { CustomerForecastPanel } = await import("../CustomerForecastPanel");
    render(
      <TestQueryWrapper>
        <CustomerForecastPanel />
      </TestQueryWrapper>
    );

    fireEvent.change(await screen.findByRole("textbox", { name: "Item" }), {
      target: { value: "ITEM-1" },
    });
    fireEvent.change(screen.getByRole("textbox", { name: "Location" }), {
      target: { value: "LOC-1" },
    });
    fireEvent.change(screen.getByRole("textbox", { name: "Customer" }), {
      target: { value: "CUST-1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Load Series" }));

    expect((await screen.findAllByText("Actual demand")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("Customer forecast").length).toBeGreaterThan(0);
    expect(screen.getByText("Customer Demand and Forecast")).toBeInTheDocument();
    expect(
      screen.getByText("Monthly actual demand and customer forecast", { selector: "caption" })
    ).toBeInTheDocument();
    expect(screen.getByText("3-Month Moving Average")).toBeInTheDocument();
    expect(screen.getByText("3-Month Moving Average (2)")).toBeInTheDocument();
    expect(screen.getByText("12-Month Seasonal Repeat (7)")).toBeInTheDocument();
    expect(screen.getByText("Croston/SBA (3)")).toBeInTheDocument();
    expect(screen.getByText(/Policy: Customer Rule Router/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Export CSV" })).toHaveAttribute(
      "href",
      "/customer-forecast/export?run_id=run-1"
    );
  });

  it("keeps the prior completed result available when a new generation fails", async () => {
    const completedRun = {
      run_id: "run-completed",
      job_id: "job-completed",
      status: "completed",
      planning_month: "2026-07-01",
      history_start: "2025-01-01",
      history_end: "2026-06-30",
      forecast_start: "2026-07-01",
      forecast_end: "2027-12-31",
      eligible_series: 12,
      row_count: 216,
      skipped_series: 0,
      model_id: "croston",
      created_at: "2026-07-13T12:00:00Z",
      started_at: "2026-07-13T12:01:00Z",
      completed_at: "2026-07-13T12:02:00Z",
      error_summary: null,
      skip_reason_counts: {},
      model_route_counts: { croston: 12 },
    };
    const failedRun = {
      ...completedRun,
      run_id: "run-failed",
      job_id: "job-failed",
      status: "failed",
      row_count: 0,
      eligible_series: 0,
      completed_at: "2026-07-13T12:04:00Z",
      error_summary: "Replacement generation failed",
      model_route_counts: {},
    };
    mockLatestRun.mockResolvedValueOnce(completedRun).mockResolvedValue(failedRun);
    mockLatestCompletedRun.mockResolvedValue(null);

    const { CustomerForecastPanel } = await import("../CustomerForecastPanel");
    render(
      <TestQueryWrapper>
        <CustomerForecastPanel />
      </TestQueryWrapper>
    );

    expect(await screen.findByRole("link", { name: "Export CSV" })).toHaveAttribute(
      "href",
      "/customer-forecast/export?run_id=run-completed"
    );
    fireEvent.click(screen.getByRole("button", { name: "Generate Again" }));

    expect(await screen.findByText("Replacement generation failed")).toHaveAttribute(
      "role",
      "alert"
    );
    expect(screen.getByRole("link", { name: "Export CSV" })).toHaveAttribute(
      "href",
      "/customer-forecast/export?run_id=run-completed"
    );
  });

  it("preserves the newest completed result after a later generation fails", async () => {
    const runA = {
      run_id: "run-a",
      job_id: "job-a",
      status: "completed",
      planning_month: "2026-07-01",
      history_start: "2025-01-01",
      history_end: "2026-06-30",
      forecast_start: "2026-07-01",
      forecast_end: "2027-12-31",
      eligible_series: 10,
      row_count: 180,
      skipped_series: 0,
      model_id: "croston",
      created_at: "2026-07-13T12:00:00Z",
      started_at: "2026-07-13T12:01:00Z",
      completed_at: "2026-07-13T12:02:00Z",
      error_summary: null,
      skip_reason_counts: {},
      model_route_counts: { croston: 10 },
      total_series: 10,
      completed_series: 10,
      total_batches: 1,
      completed_batches: 1,
      progress_pct: 100,
      eta_seconds: 0,
    };
    const runB = {
      ...runA,
      run_id: "run-b",
      job_id: "job-b",
      completed_at: "2026-07-13T12:04:00Z",
      row_count: 198,
      eligible_series: 11,
      total_series: 11,
      completed_series: 11,
      model_route_counts: { croston: 11 },
    };
    const runC = {
      ...runB,
      run_id: "run-c",
      job_id: "job-c",
      status: "failed",
      completed_at: "2026-07-13T12:06:00Z",
      row_count: 0,
      eligible_series: 0,
      completed_series: 0,
      error_summary: "Third generation failed",
      model_route_counts: {},
    };
    mockLatestRun.mockResolvedValueOnce(runA).mockResolvedValueOnce(runB).mockResolvedValue(runC);
    mockLatestCompletedRun.mockResolvedValue(runA);

    const { CustomerForecastPanel } = await import("../CustomerForecastPanel");
    render(
      <TestQueryWrapper>
        <CustomerForecastPanel />
      </TestQueryWrapper>
    );

    expect(await screen.findByText("Latest run · run-a")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Generate Again" }));
    expect(await screen.findByText("Latest run · run-b")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Generate Again" }));
    expect(await screen.findByText("Latest run · run-c")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Export CSV" })).toHaveAttribute(
      "href",
      "/customer-forecast/export?run_id=run-b"
    );
  });

  it("labels retained historical Chronos results without presenting them as Croston", async () => {
    const legacyRun = {
      run_id: "legacy-run",
      job_id: "legacy-job",
      status: "completed",
      planning_month: "2026-06-01",
      history_start: "2024-12-01",
      history_end: "2026-05-31",
      forecast_start: "2026-06-01",
      forecast_end: "2027-11-30",
      eligible_series: 1,
      row_count: 18,
      skipped_series: 0,
      model_id: "hybrid_customer_v1",
      created_at: "2026-06-13T12:00:00Z",
      started_at: "2026-06-13T12:01:00Z",
      completed_at: "2026-06-13T12:02:00Z",
      error_summary: null,
      skip_reason_counts: {},
      model_route_counts: { chronos2_enriched: 1 },
    };
    mockLatestRun.mockResolvedValue(legacyRun);
    mockLatestCompletedRun.mockResolvedValue(legacyRun);
    mockBlendReadiness.mockResolvedValue({
      ready: false,
      blockers: ["Generate a new rule-routed customer forecast with current configuration"],
      customer_run_id: "legacy-run",
      backtest_gate_passed: false,
    });
    mockSeries.mockResolvedValue({
      run: legacyRun,
      item_id: "ITEM-1",
      location_id: "LOC-1",
      customer_no: "CUST-1",
      history: [],
      forecast: [
        {
          month: "2026-06-01",
          forecast_qty: 9,
          lower_bound: 8,
          upper_bound: 10,
          model_id: "chronos2_enriched",
        },
      ],
    });
    const { CustomerForecastPanel } = await import("../CustomerForecastPanel");
    render(
      <TestQueryWrapper>
        <CustomerForecastPanel />
      </TestQueryWrapper>
    );

    fireEvent.change(await screen.findByRole("textbox", { name: "Item" }), {
      target: { value: "ITEM-1" },
    });
    fireEvent.change(screen.getByRole("textbox", { name: "Location" }), {
      target: { value: "LOC-1" },
    });
    fireEvent.change(screen.getByRole("textbox", { name: "Customer" }), {
      target: { value: "CUST-1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Load Series" }));

    expect(await screen.findByText("Chronos 2E (1)")).toBeInTheDocument();
    expect(await screen.findByText("Chronos 2E")).toBeInTheDocument();
    expect(
      screen.getByText("Generate a new rule-routed customer forecast with current configuration")
    ).toBeInTheDocument();
  });
});
