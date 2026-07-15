import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { backtestMgmtKeys } from "@/api/queries/backtest-management";
import { customerForecastKeys } from "@/api/queries/customerForecast";
import { ThemeProvider } from "@/context/ThemeContext";
import { createTestQueryClient, TestQueryWrapper } from "@/tabs/__tests__/test-utils";

const mockLatestBacktest = vi.fn();
const mockGenerateBacktest = vi.fn();
const mockBlendReadiness = vi.fn();
const mockLatestBlend = vi.fn();
const mockGenerateBlend = vi.fn();
const mockBlendSeries = vi.fn();
const mockJobDetail = vi.fn();

vi.mock("@/api/queries/customerForecast", () => ({
  customerForecastKeys: {
    latestBacktest: ["customer-forecast", "backtest", "latest"],
    blendReadiness: (customerRunId?: string) => [
      "customer-forecast",
      "blend",
      "readiness",
      customerRunId ?? null,
    ],
    latestBlend: ["customer-forecast", "blend", "latest"],
    blendSeriesAll: ["customer-forecast", "blend", "series"],
    blendSeries: (filters: unknown) => ["customer-forecast", "blend", "series", filters],
  },
  fetchLatestCustomerForecastBacktest: () => mockLatestBacktest(),
  generateCustomerForecastBacktest: () => mockGenerateBacktest(),
  fetchCustomerBlendReadiness: (customerRunId?: string) => mockBlendReadiness(customerRunId),
  fetchLatestCustomerBlend: () => mockLatestBlend(),
  generateCustomerBlend: (customerRunId?: string) => mockGenerateBlend(customerRunId),
  fetchCustomerBlendSeries: (filters: unknown) => mockBlendSeries(filters),
}));

vi.mock("@/api/queries/jobs", () => ({
  fetchJobDetail: (jobId: string) => mockJobDetail(jobId),
  jobKeys: {
    detail: (jobId: string | null) => ["jobs", "detail", jobId],
  },
}));

vi.mock("@/api/queries/backtest-management", () => ({
  backtestMgmtKeys: {
    stagingSummary: ["backtest-management", "staging-summary"],
  },
}));

vi.mock("@/hooks/useChartColors", () => ({
  useChartColors: () => ({
    chartColors: {
      grid: "grid",
      axis: "axis",
      tooltip_bg: "tooltip-bg",
      tooltip_border: "tooltip-border",
    },
    okabeIto: ["orange", "sky", "green", "yellow", "blue", "vermillion"],
  }),
}));

vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Bar: ({ name }: { name: string }) => <div>{name}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Line: ({ name }: { name: string }) => <div>{name}</div>,
  CartesianGrid: () => null,
  Legend: () => null,
  Tooltip: () => null,
  XAxis: () => null,
  YAxis: () => null,
}));

const currentBacktest = {
  run_id: "backtest-1",
  job_id: "job-backtest-1",
  status: "completed",
  customer_run_id: "customer-run-1",
  planning_month: "2026-07-01",
  common_months: 6,
  common_dfus: 1200,
  common_rows: 7200,
  actual_qty: 150000,
  component_checksum: "a".repeat(64),
  completed_at: "2026-07-14T16:00:00Z",
  gate_passed: true,
  gate_reason: "passed",
  blend_wape_degradation_pct: -1,
  min_common_months: 6,
  min_common_dfus: 1000,
  max_wape_degradation_pct: 0,
  error_summary: null,
  metrics: [
    {
      model_id: "champion",
      observations: 7200,
      actual_qty: 150000,
      absolute_error: 30000,
      mae: 4.1667,
      wape_pct: 20,
      bias_pct: 2,
      accuracy_pct: 80,
    },
    {
      model_id: "customer_bottom_up",
      observations: 7200,
      actual_qty: 150000,
      absolute_error: 33000,
      mae: 4.5833,
      wape_pct: 22,
      bias_pct: -3,
      accuracy_pct: 78,
    },
    {
      model_id: "customer_bottom_up_blend",
      observations: 7200,
      actual_qty: 150000,
      absolute_error: 28500,
      mae: 3.9583,
      wape_pct: 19,
      bias_pct: -0.5,
      accuracy_pct: 81,
    },
  ],
};

const latestBlend = {
  run_id: "blend-1",
  status: "ready",
  planning_month: "2026-07-01",
  horizon_months: 24,
  row_count: 240,
  dfu_count: 10,
  completed_at: "2026-07-14T16:30:00Z",
  model_id: "customer_bottom_up_blend",
  customer_run_id: "customer-run-1",
  source_run_id: "source-run-1",
  source_production_run_id: "production-run-1",
  source_promotion_id: 42,
  backtest_run_id: "backtest-1",
  blended_row_count: 180,
  champion_fallback_row_count: 60,
  customer_only_excluded_count: 7,
  promotion_enabled: true,
  backtest_gate: { passed: true },
  job_id: "job-blend-1",
  invalid_reason: null,
};

function renderPanelWithClient(children: React.ReactNode) {
  const client = createTestQueryClient();
  const view = render(
    <ThemeProvider value={{ theme: "light" }}>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </ThemeProvider>
  );
  return { client, ...view };
}

describe("CustomerBottomUpBlendPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockLatestBacktest.mockResolvedValue(currentBacktest);
    mockBlendReadiness.mockResolvedValue({
      ready: true,
      blockers: [],
      customer_run_id: "customer-run-1",
      source_promotion_id: 42,
      source_run_id: "source-run-1",
      source_production_run_id: "production-run-1",
      backtest_run_id: "backtest-1",
      backtest_gate_passed: true,
      promotion_enabled: true,
      promotion_reason: "Validated customer blend",
    });
    mockLatestBlend.mockResolvedValue(latestBlend);
    mockGenerateBacktest.mockResolvedValue({
      run_id: "backtest-2",
      job_id: "job-backtest-2",
      status: "queued",
    });
    mockGenerateBlend.mockResolvedValue({
      run_id: "blend-2",
      job_id: "job-blend-2",
      status: "queued",
    });
    mockJobDetail.mockResolvedValue({
      job_id: "job-blend-2",
      status: "running",
    });
    mockBlendSeries.mockResolvedValue({
      run_id: "blend-1",
      customer_run_id: "customer-run-1",
      source_run_id: "source-run-1",
      source_production_run_id: "production-run-1",
      item_id: "ITEM-1",
      location_id: "LOC-1",
      months: [
        {
          forecast_month: "2026-07-01",
          raw_customer_demand_qty: 120,
          normalized_customer_qty: 90,
          champion_qty: 100,
          blended_qty: 95,
          lower_bound: 75,
          upper_bound: 125,
          fulfillment_ratio: 0.75,
          effective_customer_weight: 0.5,
          coverage_status: "blended",
          interval_method: "champion_width_shift",
        },
        {
          forecast_month: "2028-01-01",
          raw_customer_demand_qty: null,
          normalized_customer_qty: null,
          champion_qty: 80,
          blended_qty: 80,
          lower_bound: 70,
          upper_bound: 90,
          fulfillment_ratio: null,
          effective_customer_weight: 0,
          coverage_status: "champion_fallback",
          interval_method: "champion_passthrough",
        },
      ],
    });
  });

  it("shows rule-routed common-cohort accuracy and queues a current qualified draft", async () => {
    const { CustomerBottomUpBlendPanel } = await import("../CustomerBottomUpBlendPanel");
    render(
      <TestQueryWrapper>
        <CustomerBottomUpBlendPanel customerRunId="customer-run-1" />
      </TestQueryWrapper>
    );

    expect(await screen.findByText("Bottom-Up Blend Validation")).toBeInTheDocument();
    expect((await screen.findAllByText("Source Champion")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("Customer Bottom-Up").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Customer Blend").length).toBeGreaterThan(0);
    expect(screen.getByText("81.0% accuracy")).toBeInTheDocument();
    expect(screen.getByText("Common-Cohort Accuracy Comparison")).toBeInTheDocument();
    expect(screen.getByText(/6 common months.*1,200 DFUs.*7,200 observations/)).toBeInTheDocument();
    expect(screen.getByText("Backtest gate passed")).toBeInTheDocument();
    expect(screen.getByText(/Blend WAPE delta -1.0 pp.*limit 0.0 pp/)).toBeInTheDocument();
    expect(screen.getByText(/180 blended rows/)).toBeInTheDocument();
    expect(screen.getByText(/60 champion fallback rows/)).toBeInTheDocument();
    expect(screen.getByText(/7 customer-only DFUs excluded/)).toBeInTheDocument();
    expect(screen.getByText("Promotion policy enabled")).toBeInTheDocument();

    const draftButton = screen.getByRole("button", { name: "Generate Draft" });
    fireEvent.click(draftButton);
    await waitFor(() => expect(mockGenerateBlend).toHaveBeenCalledWith("customer-run-1"));
    await waitFor(() => expect(draftButton).toBeDisabled());

    fireEvent.click(screen.getByRole("button", { name: "Run Blend Backtest" }));
    await waitFor(() => expect(mockGenerateBacktest).toHaveBeenCalledOnce());
    await waitFor(() => expect(mockBlendReadiness.mock.calls.length).toBeGreaterThan(1));
    await waitFor(() => expect(draftButton).toBeDisabled());
  });

  it("loads an exact item-location comparison and labels fallback months", async () => {
    const { CustomerBottomUpBlendPanel } = await import("../CustomerBottomUpBlendPanel");
    render(
      <TestQueryWrapper>
        <CustomerBottomUpBlendPanel customerRunId="customer-run-1" />
      </TestQueryWrapper>
    );

    fireEvent.change(await screen.findByLabelText("Blend item"), {
      target: { value: "ITEM-1" },
    });
    fireEvent.change(screen.getByLabelText("Blend location"), {
      target: { value: "LOC-1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Load Blend Comparison" }));

    await waitFor(() =>
      expect(mockBlendSeries).toHaveBeenCalledWith({
        item_id: "ITEM-1",
        location_id: "LOC-1",
        run_id: "blend-1",
      })
    );
    expect((await screen.findAllByText("Raw Customer Demand")).length).toBeGreaterThan(0);
    expect(screen.getByText("Item-Location Blend Comparison")).toBeInTheDocument();
    expect(screen.getAllByText("Normalized Customer Bottom-Up").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Source Champion").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Customer Blend").length).toBeGreaterThan(0);
    expect(screen.getByText("Blended")).toBeInTheDocument();
    expect(screen.getByText("Champion fallback")).toBeInTheDocument();
    expect(screen.getByText("75.0%")).toBeInTheDocument();
    expect(screen.getByText("50.0%")).toBeInTheDocument();
    expect(
      screen.getByText("Monthly customer blend components and coverage", { selector: "caption" })
    ).toBeInTheDocument();
    expect(
      screen.getByText(/1 month use the source champion only/).closest('[role="status"]')
    ).toHaveClass("border-warning/30", "bg-warning/10");
  });

  it("blocks draft generation until the completed backtest matches the customer run", async () => {
    mockLatestBacktest.mockResolvedValue({
      ...currentBacktest,
      customer_run_id: "older-customer-run",
    });
    mockBlendReadiness.mockResolvedValue({
      ready: false,
      blockers: ["Complete the current customer bottom-up accuracy backtest"],
      customer_run_id: "customer-run-1",
      backtest_gate_passed: false,
    });
    const { CustomerBottomUpBlendPanel } = await import("../CustomerBottomUpBlendPanel");
    render(
      <TestQueryWrapper>
        <CustomerBottomUpBlendPanel customerRunId="customer-run-1" />
      </TestQueryWrapper>
    );

    const draftButton = await screen.findByRole("button", { name: "Generate Draft" });
    expect(draftButton).toBeDisabled();
    const blocker = await screen.findByText(
      "Complete the current customer bottom-up accuracy backtest"
    );
    expect(blocker).toBeInTheDocument();
    expect(blocker.closest('[role="alert"]')).toHaveClass("border-warning/30", "bg-warning/10");
  });

  it("blocks draft generation when the current backtest fails the accuracy gate", async () => {
    mockLatestBacktest.mockResolvedValue({
      ...currentBacktest,
      gate_passed: false,
      gate_reason: "Blend WAPE is 1.5 points worse than the source champion.",
      blend_wape_degradation_pct: 1.5,
    });
    mockBlendReadiness.mockResolvedValue({
      ready: false,
      blockers: [
        "Customer blend backtest gate failed: Blend WAPE is 1.5 points worse than the source champion.",
      ],
      customer_run_id: "customer-run-1",
      backtest_gate_passed: false,
    });
    const { CustomerBottomUpBlendPanel } = await import("../CustomerBottomUpBlendPanel");
    render(
      <TestQueryWrapper>
        <CustomerBottomUpBlendPanel customerRunId="customer-run-1" />
      </TestQueryWrapper>
    );

    const draftButton = await screen.findByRole("button", { name: "Generate Draft" });
    expect(draftButton).toBeDisabled();
    expect(await screen.findByText("Backtest gate blocked")).toBeInTheDocument();
    expect(
      await screen.findByText("Blend WAPE is 1.5 points worse than the source champion.")
    ).toBeInTheDocument();
  });

  it("uses prior matching passed evidence when the newest retry failed", async () => {
    mockLatestBacktest.mockResolvedValue({
      ...currentBacktest,
      run_id: "backtest-retry",
      status: "failed",
      gate_passed: null,
      gate_reason: null,
      metrics: [],
    });
    const { CustomerBottomUpBlendPanel } = await import("../CustomerBottomUpBlendPanel");
    render(
      <TestQueryWrapper>
        <CustomerBottomUpBlendPanel customerRunId="customer-run-1" />
      </TestQueryWrapper>
    );

    const draftButton = await screen.findByRole("button", { name: "Generate Draft" });
    await waitFor(() => expect(draftButton).toBeEnabled());
  });

  it("blocks draft generation when current server lineage is no longer ready", async () => {
    mockBlendReadiness.mockResolvedValue({
      ready: false,
      blockers: ["Promote a fresh unblended champion before creating another customer blend"],
      customer_run_id: "customer-run-1",
      backtest_gate_passed: true,
    });
    const { CustomerBottomUpBlendPanel } = await import("../CustomerBottomUpBlendPanel");
    render(
      <TestQueryWrapper>
        <CustomerBottomUpBlendPanel customerRunId="customer-run-1" />
      </TestQueryWrapper>
    );

    const draftButton = await screen.findByRole("button", { name: "Generate Draft" });
    expect(draftButton).toBeDisabled();
    expect(
      await screen.findByText(
        "Promote a fresh unblended champion before creating another customer blend"
      )
    ).toBeInTheDocument();
  });

  it("stops the draft spinner and reports a terminal job failure", async () => {
    mockJobDetail.mockResolvedValue({
      job_id: "job-blend-2",
      status: "failed",
      error: "Customer blend evidence changed during generation.",
    });
    const { CustomerBottomUpBlendPanel } = await import("../CustomerBottomUpBlendPanel");
    render(
      <TestQueryWrapper>
        <CustomerBottomUpBlendPanel customerRunId="customer-run-1" />
      </TestQueryWrapper>
    );

    const draftButton = await screen.findByRole("button", { name: "Generate Draft" });
    await waitFor(() => expect(draftButton).toBeEnabled());
    fireEvent.click(draftButton);

    expect(
      await screen.findByText("Customer blend evidence changed during generation.")
    ).toBeInTheDocument();
    await waitFor(() => {
      const resetButton = screen.getByRole("button", { name: "Generate Draft" });
      expect(resetButton).toBeEnabled();
    });
  });

  it("refreshes readiness when an observed current-customer backtest becomes terminal", async () => {
    mockLatestBacktest.mockResolvedValue({
      ...currentBacktest,
      status: "generating",
      gate_passed: null,
      gate_reason: null,
      metrics: [],
      error_summary: null,
    });
    mockBlendReadiness
      .mockResolvedValueOnce({
        ready: false,
        blockers: ["Complete the current customer bottom-up accuracy backtest"],
        customer_run_id: "customer-run-1",
        backtest_gate_passed: false,
      })
      .mockResolvedValue({
        ready: true,
        blockers: [],
        customer_run_id: "customer-run-1",
        backtest_gate_passed: true,
      });
    const { CustomerBottomUpBlendPanel } = await import("../CustomerBottomUpBlendPanel");
    const { client } = renderPanelWithClient(
      <CustomerBottomUpBlendPanel customerRunId="customer-run-1" />
    );

    await waitFor(() => expect(mockBlendReadiness).toHaveBeenCalledOnce());
    act(() => {
      client.setQueryData(["customer-forecast", "backtest", "latest"], {
        ...currentBacktest,
        error_summary: null,
      });
    });

    await waitFor(() => expect(mockBlendReadiness.mock.calls.length).toBeGreaterThan(1));
    expect(await screen.findByRole("button", { name: "Generate Draft" })).toBeEnabled();
  });

  it("invalidates blend overlays and release staging when a blend job completes", async () => {
    mockJobDetail.mockResolvedValue({
      job_id: "job-blend-2",
      status: "completed",
      error: null,
    });
    const { CustomerBottomUpBlendPanel } = await import("../CustomerBottomUpBlendPanel");
    const { client } = renderPanelWithClient(
      <CustomerBottomUpBlendPanel customerRunId="customer-run-1" />
    );
    const invalidate = vi.spyOn(client, "invalidateQueries");

    const draftButton = await screen.findByRole("button", { name: "Generate Draft" });
    await waitFor(() => expect(draftButton).toBeEnabled());
    fireEvent.click(draftButton);

    await waitFor(() =>
      expect(invalidate).toHaveBeenCalledWith({
        queryKey: customerForecastKeys.blendSeriesAll,
      })
    );
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: backtestMgmtKeys.stagingSummary,
    });
  });

  it("displays terminal backtest failure details", async () => {
    mockLatestBacktest.mockResolvedValue({
      ...currentBacktest,
      status: "failed",
      gate_passed: null,
      gate_reason: null,
      metrics: [],
      error_summary: "Customer backtest worker failed.",
    });
    const { CustomerBottomUpBlendPanel } = await import("../CustomerBottomUpBlendPanel");
    render(
      <TestQueryWrapper>
        <CustomerBottomUpBlendPanel customerRunId="customer-run-1" />
      </TestQueryWrapper>
    );

    expect(await screen.findByText("Customer backtest worker failed.")).toHaveAttribute(
      "role",
      "alert"
    );
  });

  it("recovers job polling for a generating blend after remount", async () => {
    mockLatestBlend.mockResolvedValueOnce({
      ...latestBlend,
      run_id: "blend-recovered",
      status: "generating",
      job_id: "job-recovered",
      completed_at: null,
    });
    const { CustomerBottomUpBlendPanel } = await import("../CustomerBottomUpBlendPanel");
    render(
      <TestQueryWrapper>
        <CustomerBottomUpBlendPanel customerRunId="customer-run-1" />
      </TestQueryWrapper>
    );

    expect(await screen.findByRole("button", { name: "Draft Generating" })).toBeDisabled();
    await waitFor(() => expect(mockJobDetail).toHaveBeenCalledWith("job-recovered"));
  });

  it("surfaces an invalid blend reason and disables comparison loading", async () => {
    mockLatestBlend.mockResolvedValue({
      ...latestBlend,
      run_id: "blend-invalid",
      status: "invalid",
      job_id: "job-recovered",
      invalid_reason: "Source champion changed during generation.",
    });
    const { CustomerBottomUpBlendPanel } = await import("../CustomerBottomUpBlendPanel");
    render(
      <TestQueryWrapper>
        <CustomerBottomUpBlendPanel customerRunId="customer-run-1" />
      </TestQueryWrapper>
    );

    expect(await screen.findByText("Source champion changed during generation.")).toHaveAttribute(
      "role",
      "alert"
    );
    fireEvent.change(screen.getByLabelText("Blend item"), { target: { value: "ITEM-1" } });
    fireEvent.change(screen.getByLabelText("Blend location"), { target: { value: "LOC-1" } });
    expect(screen.getByRole("button", { name: "Load Blend Comparison" })).toBeDisabled();
  });
});
