import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

const mockRunDQChecks = vi.fn().mockResolvedValue({ triggered: 5, message: "ok" });
const mockFetchDQFixPreview = vi.fn().mockResolvedValue({
  items: [
    { id: 0, fix_type: "range", description: "Clamp fact_sales.qty to [0, 10000000]", affected_rows: 500, recommendation: null, status: "pending" },
    { id: 1, fix_type: "completeness", description: "Impute dim_item.brand_name NULLs with median", affected_rows: 120, recommendation: null, status: "pending" },
    { id: 2, fix_type: "orphans", description: "Orphan keys: fact_sales(dmdunit,loc) → dim_dfu", affected_rows: 42, recommendation: "Reload dimension: make normalize-all && make load-all", status: "pending" },
  ],
  total: 3,
});
const mockApplyDQFixes = vi.fn().mockResolvedValue({
  applied: [{ id: 0, fix_type: "range", description: "Clamp fact_sales.qty", affected_rows: 500, recommendation: null, status: "applied", rows_fixed: 500 }],
  skipped: [],
  total_applied: 1,
  total_skipped: 0,
  total_rows_fixed: 500,
});

const mockDomains = [
  { domain: "sales", score: 95, passed: 19, failed: 1, warnings: 0, total: 20 },
  { domain: "forecast", score: 40, passed: 4, failed: 5, warnings: 1, total: 10 },
];

const mockChecks = [
  { check_id: 1, check_name: "Null check", check_type: "completeness", domain: "sales", table_name: "fact_sales_monthly", severity: "critical", enabled: true, last_status: "pass", last_value: 0.0, last_run: "2026-03-17T10:00:00" },
  { check_id: 2, check_name: "Range check", check_type: "validity", domain: "forecast", table_name: "fact_external_forecast_monthly", severity: "warning", enabled: true, last_status: "fail", last_value: 12.5, last_run: "2026-03-17T09:00:00" },
  { check_id: 3, check_name: "Freshness check", check_type: "timeliness", domain: "forecast", table_name: "fact_external_forecast_monthly", severity: "high", enabled: true, last_status: null, last_value: null, last_run: null },
];

const mockHistoryEntries = [
  { check_id: 2, check_name: "range_forecast_basefcst_pref", check_type: "range", domain: "forecast", table_name: "fact_external_forecast_monthly", severity: "warning", status: "fail", metric_value: 12.5, details: { threshold: 10, actual: 12.5, outliers: 500, outlier_pct: 0.12, min: -1000000, max: 100000000 }, run_ts: "2026-03-17T09:00:00" },
  { check_id: 4, check_name: "uniqueness_sales", check_type: "uniqueness", domain: "sales", table_name: "fact_sales_monthly", severity: "critical", status: "error", metric_value: null, details: { error: "Connection timeout during check execution" }, run_ts: "2026-03-16T22:00:00" },
  { check_id: 1, check_name: "completeness_sales_qty", check_type: "completeness", domain: "sales", table_name: "fact_sales_monthly", severity: "critical", status: "pass", metric_value: 0.0, details: null, run_ts: "2026-03-17T10:00:00" },
];

vi.mock("@/api/queries", () => ({
  fetchDQDashboard: vi.fn().mockResolvedValue({ domains: [] }),
  fetchDQChecks: vi.fn().mockResolvedValue({ checks: [] }),
  fetchDQFreshness: vi.fn().mockResolvedValue({ tables: [] }),
  fetchDQHistory: vi.fn().mockResolvedValue({ entries: [] }),
  runDQChecks: mockRunDQChecks,
  fetchDQFixPreview: mockFetchDQFixPreview,
  applyDQFixes: mockApplyDQFixes,
  dqKeys: {
    dashboard: ["dq", "dashboard"],
    checks: ["dq", "checks"],
    freshness: ["dq", "freshness"],
    history: (domain?: string) => ["dq", "history", domain ?? ""],
    fixPreview: ["dq", "fix", "preview"],
  },
  STALE_PLATFORM: 300000,
  // Medallion lineage mocks
  fetchBatches: vi.fn().mockResolvedValue({ batches: [], total: 0 }),
  fetchCorrections: vi.fn().mockResolvedValue({ corrections: [], total: 0 }),
  fetchQuarantine: vi.fn().mockResolvedValue({ quarantine: [], total: 0 }),
  resolveQuarantine: vi.fn().mockResolvedValue({ quarantine_id: 1, resolved: true }),
  lineageKeys: {
    batches: ["lineage", "batches"],
    corrections: ["lineage", "corrections"],
    quarantine: ["lineage", "quarantine"],
  },
  STALE_LINEAGE: 30000,
}));

beforeEach(() => {
  vi.clearAllMocks();
});

describe("DataQualityTab", () => {
  it("renders without crashing", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    expect(screen.getByText("Data Quality & Observability")).toBeInTheDocument();
  });

  it("shows empty state when no checks run", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    expect(screen.getByText(/No data quality checks have been run yet/)).toBeInTheDocument();
  });

  it("renders pipeline freshness section", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    expect(screen.getByText("Pipeline Freshness")).toBeInTheDocument();
  });

  it("renders check catalog section", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    expect(screen.getByText(/Check Catalog/)).toBeInTheDocument();
  });

  it("renders Run Checks Now button", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    expect(screen.getByRole("button", { name: /Run Checks Now/ })).toBeInTheDocument();
  });

  it("calls runDQChecks when button is clicked", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    fireEvent.click(screen.getByRole("button", { name: /Run Checks Now/ }));
    await waitFor(() => {
      expect(mockRunDQChecks).toHaveBeenCalledTimes(1);
    });
  });

  it("shows loading state while mutation is pending", async () => {
    let resolveRun!: (v: { triggered: number; message: string }) => void;
    mockRunDQChecks.mockImplementation(
      () => new Promise((res) => { resolveRun = res; })
    );

    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    fireEvent.click(screen.getByRole("button", { name: /Run Checks Now/ }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Running/ })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Running/ })).toBeDisabled();
    });

    resolveRun({ triggered: 5, message: "ok" });
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Run Checks Now/ })).toBeInTheDocument();
    });
  });

  it("shows success banner after mutation completes", async () => {
    mockRunDQChecks.mockResolvedValue({ triggered: 5, message: "ok" });

    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    fireEvent.click(screen.getByRole("button", { name: /Run Checks Now/ }));

    await waitFor(() => {
      expect(screen.getByText(/completed successfully/)).toBeInTheDocument();
    });
  });

  it("shows error banner when mutation fails", async () => {
    mockRunDQChecks.mockRejectedValue(new Error("DQ run failed: 500"));

    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    fireEvent.click(screen.getByRole("button", { name: /Run Checks Now/ }));

    await waitFor(() => {
      expect(screen.getByText(/DQ run failed: 500/)).toBeInTheDocument();
    });
  });

  it("invalidates queries on successful mutation", async () => {
    mockRunDQChecks.mockResolvedValue({ triggered: 5, message: "ok" });

    const { default: DataQualityTab } = await import("../DataQualityTab");
    const queries = await import("@/api/queries");
    const fetchDashboardSpy = queries.fetchDQDashboard as ReturnType<typeof vi.fn>;
    const fetchChecksSpy = queries.fetchDQChecks as ReturnType<typeof vi.fn>;
    const fetchFreshnessSpy = queries.fetchDQFreshness as ReturnType<typeof vi.fn>;

    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(fetchDashboardSpy).toHaveBeenCalled();
    });

    fetchDashboardSpy.mockClear();
    fetchChecksSpy.mockClear();
    fetchFreshnessSpy.mockClear();

    fireEvent.click(screen.getByRole("button", { name: /Run Checks Now/ }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Run Checks Now/ })).not.toBeDisabled();
    });

    await waitFor(() => {
      expect(fetchDashboardSpy).toHaveBeenCalled();
      expect(fetchChecksSpy).toHaveBeenCalled();
      expect(fetchFreshnessSpy).toHaveBeenCalled();
    });
  });

  /* ---- New tests for redesigned tab ---- */

  it("renders summary KPI bar with overall health", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchDQDashboard as ReturnType<typeof vi.fn>).mockResolvedValue({ domains: mockDomains });

    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("Overall Health")).toBeInTheDocument();
      expect(screen.getByText("Total Checks")).toBeInTheDocument();
      expect(screen.getByText("Passed")).toBeInTheDocument();
      expect(screen.getByText("Failed")).toBeInTheDocument();
      expect(screen.getByText("Warnings")).toBeInTheDocument();
      // "Last Run" appears in both KPI bar and table header
      expect(screen.getAllByText(/Last Run/).length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders domain health cards as clickable buttons", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchDQDashboard as ReturnType<typeof vi.fn>).mockResolvedValue({ domains: mockDomains });

    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /sales/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /forecast/i })).toBeInTheDocument();
    });
  });

  it("renders recent issues section", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/Recent Issues/)).toBeInTheDocument();
    });
  });

  it("displays recent issue details from history", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchDQHistory as ReturnType<typeof vi.fn>).mockResolvedValue({ entries: mockHistoryEntries });

    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    // Only fail + error entries show in recent issues (not the pass entry)
    await waitFor(() => {
      // The fail entry check_name
      expect(screen.getByText("range_forecast_basefcst_pref")).toBeInTheDocument();
      // The error entry check_name
      expect(screen.getByText("uniqueness_sales")).toBeInTheDocument();
      // AI summary for range issue (always visible)
      expect(screen.getByText(/500 rows.*outside the expected range/)).toBeInTheDocument();
      // AI summary for error issue (always visible)
      expect(screen.getByText(/SQL error executing check/)).toBeInTheDocument();
    });
  });

  it("shows empty issue state when all checks pass", async () => {
    const queries = await import("@/api/queries");
    // History has only passing entries
    (queries.fetchDQHistory as ReturnType<typeof vi.fn>).mockResolvedValue({
      entries: [{ check_id: 1, check_name: "OK", domain: "sales", table_name: "t", severity: "low", status: "pass", metric_value: 0, details: null, run_ts: "2026-03-17T10:00:00" }],
    });

    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/All checks are passing/)).toBeInTheDocument();
    });
  });

  it("filters recent issues by severity", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchDQHistory as ReturnType<typeof vi.fn>).mockResolvedValue({
      entries: [
        { check_id: 1, check_name: "freshness_sales", check_type: "freshness", domain: "sales", table_name: "fact_sales_monthly", severity: "critical", status: "fail", metric_value: 100, details: { hours_since_load: 100 }, run_ts: "2026-03-17T10:00:00" },
        { check_id: 2, check_name: "range_sales_qty", check_type: "range", domain: "sales", table_name: "fact_sales_monthly", severity: "warning", status: "fail", metric_value: 5, details: { outliers: 5, outlier_pct: 0.01, min: 0, max: 10000000 }, run_ts: "2026-03-17T09:00:00" },
      ],
    });

    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    // Both issues visible initially
    await waitFor(() => {
      expect(screen.getByText("freshness_sales")).toBeInTheDocument();
      expect(screen.getByText("range_sales_qty")).toBeInTheDocument();
    });

    // Filter to critical only
    const select = screen.getByLabelText("Filter issues by severity");
    fireEvent.change(select, { target: { value: "critical" } });

    await waitFor(() => {
      expect(screen.getByText("freshness_sales")).toBeInTheDocument();
      expect(screen.queryByText("range_sales_qty")).not.toBeInTheDocument();
    });
  });

  it("renders check catalog with status icons and severity badges", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchDQChecks as ReturnType<typeof vi.fn>).mockResolvedValue({ checks: mockChecks });

    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("Null check")).toBeInTheDocument();
      expect(screen.getByText("Range check")).toBeInTheDocument();
      expect(screen.getByText("Freshness check")).toBeInTheDocument();
    });
  });

  it("filters checks when domain card is clicked", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchDQDashboard as ReturnType<typeof vi.fn>).mockResolvedValue({ domains: mockDomains });
    (queries.fetchDQChecks as ReturnType<typeof vi.fn>).mockResolvedValue({ checks: mockChecks });

    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    // Wait for data to render
    await waitFor(() => {
      expect(screen.getByText("Null check")).toBeInTheDocument();
    });

    // Click the forecast domain card to filter
    fireEvent.click(screen.getByRole("button", { name: /forecast/i }));

    await waitFor(() => {
      // Forecast checks should still be visible
      expect(screen.getByText("Range check")).toBeInTheDocument();
      expect(screen.getByText("Freshness check")).toBeInTheDocument();
      // The "1 of 3" count in catalog header
      expect(screen.getByText(/2 of 3/)).toBeInTheDocument();
    });
  });

  it("renders filter dropdowns for status and severity", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    expect(screen.getByLabelText("Filter by status")).toBeInTheDocument();
    expect(screen.getByLabelText("Filter by severity")).toBeInTheDocument();
  });

  it("renders Domain Health heading", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    expect(screen.getByText("Domain Health")).toBeInTheDocument();
  });

  // ----- Self-Heal panel tests -----

  it("renders Self-Heal section with scan button", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    expect(screen.getByText("Self-Heal")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Scan for Fixes/i })).toBeInTheDocument();
  });

  it("opens Self-Heal panel and shows fix items after scan", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    // Click scan
    fireEvent.click(screen.getByRole("button", { name: /Scan for Fixes/i }));

    // Fix items should appear
    await waitFor(() => {
      expect(screen.getByText(/Clamp fact_sales\.qty/)).toBeInTheDocument();
      expect(screen.getByText(/Impute dim_item\.brand_name/)).toBeInTheDocument();
      expect(screen.getByText(/Orphan keys/)).toBeInTheDocument();
    });

    // Toolbar should show
    expect(screen.getByText(/0 of 3 selected/)).toBeInTheDocument();
  });

  it("selects and deselects all fixes", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    fireEvent.click(screen.getByRole("button", { name: /Scan for Fixes/i }));

    // Wait for fix items to render (proof that query loaded)
    await waitFor(() => {
      expect(screen.getByText(/Clamp fact_sales\.qty/)).toBeInTheDocument();
    });

    // Select All
    fireEvent.click(screen.getByRole("button", { name: /Select All/i }));
    expect(screen.getByText(/3 of 3 selected/)).toBeInTheDocument();

    // Deselect All
    fireEvent.click(screen.getByRole("button", { name: /Deselect All/i }));
    expect(screen.getByText(/0 of 3 selected/)).toBeInTheDocument();
  });

  it("shows empty state when no fixable issues found", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchDQFixPreview as ReturnType<typeof vi.fn>).mockResolvedValue({ items: [], total: 0 });

    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    fireEvent.click(screen.getByRole("button", { name: /Scan for Fixes/i }));

    await waitFor(() => {
      expect(screen.getByText(/No fixable issues found/)).toBeInTheDocument();
    });
  });

  it("rejects selected fixes", async () => {
    // Re-set mock (previous test overrides it with empty result)
    const queries = await import("@/api/queries");
    (queries.fetchDQFixPreview as ReturnType<typeof vi.fn>).mockResolvedValue({
      items: [
        { id: 0, fix_type: "range", description: "Clamp fact_sales.qty to [0, 10000000]", affected_rows: 500, recommendation: null, status: "pending" },
        { id: 1, fix_type: "completeness", description: "Impute dim_item.brand_name NULLs with median", affected_rows: 120, recommendation: null, status: "pending" },
        { id: 2, fix_type: "orphans", description: "Orphan keys: fact_sales(dmdunit,loc) → dim_dfu", affected_rows: 42, recommendation: "Reload dimension", status: "pending" },
      ],
      total: 3,
    });

    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    fireEvent.click(screen.getByRole("button", { name: /Scan for Fixes/i }));

    // Wait for fix items to load first
    await waitFor(() => {
      expect(screen.getByText(/Clamp fact_sales\.qty/)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText(/0 of 3 selected/)).toBeInTheDocument();
    });

    // Select All then Reject
    fireEvent.click(screen.getByRole("button", { name: /Select All/i }));
    fireEvent.click(screen.getByRole("button", { name: /Reject Selected/i }));

    // All should now show "Rejected" and toolbar should show 0 of 0
    await waitFor(() => {
      expect(screen.getByText(/0 of 0 selected/)).toBeInTheDocument();
      const rejectedLabels = screen.getAllByText("Rejected");
      expect(rejectedLabels.length).toBe(3);
    });
  });
});

/* ========================================================================== */
/*  Medallion sections — Pipeline Lineage, Corrections, Quarantine            */
/* ========================================================================== */

describe("DataQualityTab — Medallion Sections", () => {
  it("renders empty pipeline lineage section", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(<TestQueryWrapper><DataQualityTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Pipeline Lineage")).toBeInTheDocument();
    });
    expect(screen.getByText(/No medallion batches yet/)).toBeInTheDocument();
  });

  it("renders batches when available", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchBatches as ReturnType<typeof vi.fn>).mockResolvedValue({
      batches: [
        { batch_id: 1, domain: "sales", layer: "bronze", source_file: "sales_clean.csv", row_count_in: 1000, row_count_out: 950, row_count_quarantined: 50, status: "completed", started_at: "2026-03-17T12:00:00", completed_at: "2026-03-17T12:01:00", error_message: null },
      ],
      total: 1,
    });
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(<TestQueryWrapper><DataQualityTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Batch #1")).toBeInTheDocument();
    });
    expect(screen.getByText("1,000 in")).toBeInTheDocument();
    expect(screen.getByText("950 out")).toBeInTheDocument();
  });

  it("renders empty corrections section", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(<TestQueryWrapper><DataQualityTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Corrections Audit Log")).toBeInTheDocument();
    });
    expect(screen.getByText(/No DQ corrections recorded/)).toBeInTheDocument();
  });

  it("renders corrections when available", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchCorrections as ReturnType<typeof vi.fn>).mockResolvedValue({
      corrections: [
        { correction_id: 1, domain: "sales", table_name: "silver_sales", row_key: "k1", column_name: "qty", old_value: "100", new_value: "50", fix_type: "clamp", fix_strategy: "range", applied_by: "system", applied_at: "2026-03-17T12:00:00", load_batch_id: 42 },
      ],
      total: 1,
    });
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(<TestQueryWrapper><DataQualityTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("clamp")).toBeInTheDocument();
    });
  });

  it("renders empty quarantine section", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(<TestQueryWrapper><DataQualityTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Quarantine Queue")).toBeInTheDocument();
    });
    expect(screen.getByText(/No quarantined rows/)).toBeInTheDocument();
  });

  it("renders quarantine items with dismiss button", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchQuarantine as ReturnType<typeof vi.fn>).mockResolvedValue({
      quarantine: [
        { quarantine_id: 1, domain: "sales", bronze_id: 100, load_batch_id: 42, rejection_reason: "null_pk", rejection_details: null, raw_row: null, resolved: false, resolved_by: null, created_at: "2026-03-17T12:00:00" },
      ],
      total: 1,
    });
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(<TestQueryWrapper><DataQualityTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("null_pk")).toBeInTheDocument();
    });
    expect(screen.getByText("Dismiss")).toBeInTheDocument();
  });
});
