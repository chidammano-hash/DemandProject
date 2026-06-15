import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

const mockRunDQChecks = vi.fn().mockResolvedValue({ triggered: 5, message: "ok" });
const mockFetchDQFixPreview = vi.fn().mockResolvedValue({
  items: [
    { id: 0, fix_type: "range", description: "Clamp fact_sales.qty to [0, 10000000]", affected_rows: 500, recommendation: null, status: "pending" },
    { id: 1, fix_type: "completeness", description: "Impute dim_item.brand_name NULLs with median", affected_rows: 120, recommendation: null, status: "pending" },
    { id: 2, fix_type: "orphans", description: "Orphan keys: fact_sales(item_id,loc) → dim_sku", affected_rows: 42, recommendation: "Reload dimension: make normalize-all && make load-all", status: "pending" },
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
  { check_id: 3, check_name: "Volume delta check", check_type: "volume_delta", domain: "forecast", table_name: "fact_external_forecast_monthly", severity: "high", enabled: true, last_status: null, last_value: null, last_run: null },
];

const mockHistoryEntries = [
  { check_id: 2, check_name: "range_forecast_basefcst_pref", check_type: "range", domain: "forecast", table_name: "fact_external_forecast_monthly", severity: "warning", status: "fail", metric_value: 12.5, details: { threshold: 10, actual: 12.5, outliers: 500, outlier_pct: 0.12, min: -1000000, max: 100000000 }, run_ts: "2026-03-17T09:00:00" },
  { check_id: 4, check_name: "uniqueness_sales", check_type: "uniqueness", domain: "sales", table_name: "fact_sales_monthly", severity: "critical", status: "error", metric_value: null, details: { error: "Connection timeout during check execution" }, run_ts: "2026-03-16T22:00:00" },
  { check_id: 1, check_name: "completeness_sales_qty", check_type: "completeness", domain: "sales", table_name: "fact_sales_monthly", severity: "critical", status: "pass", metric_value: 0.0, details: null, run_ts: "2026-03-17T10:00:00" },
];

vi.mock("@/api/queries", () => ({
  fetchDQDashboard: vi.fn().mockResolvedValue({ domains: [] }),
  fetchDQChecks: vi.fn().mockResolvedValue({ checks: [] }),
  fetchDQHistory: vi.fn().mockResolvedValue({ entries: [] }),
  runDQChecks: mockRunDQChecks,
  fetchDQFixPreview: mockFetchDQFixPreview,
  applyDQFixes: mockApplyDQFixes,
  dqKeys: {
    dashboard: ["dq", "dashboard"],
    checks: ["dq", "checks"],
    history: (domain?: string) => ["dq", "history", domain ?? ""],
    fixPreview: ["dq", "fix", "preview"],
  },
  STALE_PLATFORM: 300000,
  // Pipeline lineage mocks
  fetchBatches: vi.fn().mockResolvedValue({ batches: [], total: 0 }),
  fetchCorrections: vi.fn().mockResolvedValue({ corrections: [], total: 0 }),
  fetchCorrectionsByItem: vi.fn().mockResolvedValue({ corrections: [], total: 0 }),
  fetchCorrectionsSummary: vi.fn().mockResolvedValue({ skus: [], total: 0 }),
  lineageKeys: {
    batches: ["lineage", "batches"],
    corrections: ["lineage", "corrections"],
  },
  correctionKeys: {
    byItem: (i: string, l: string) => ["dq", "corrections", i, l],
    summary: (d?: string, f?: string) => ["dq", "corrections", "summary", d ?? "", f ?? ""],
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

  it("renders check catalog section", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    expect(screen.getByText(/Check Catalog/)).toBeInTheDocument();
  });

  it("F5.3: the check-count tile reconciles the per-domain run total with the distinct catalog definitions", async () => {
    const { fetchDQDashboard, fetchDQChecks } = await import("@/api/queries");
    // Dashboard rolls up per domain-pair: 20 + 10 = 30 check-RUNS.
    (fetchDQDashboard as ReturnType<typeof vi.fn>).mockResolvedValue({ domains: mockDomains });
    // Catalog lists 3 DISTINCT check definitions (the 83-vs-166 shape, scaled).
    (fetchDQChecks as ReturnType<typeof vi.fn>).mockResolvedValue({ checks: mockChecks });
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    // The tile must no longer be the bare ambiguous "Total Checks" (which read
    // 30 next to a "Check Catalog (3)" header). It is relabeled to "Check Runs"
    // and discloses the distinct-definition denominator so 30 vs 3 self-explains.
    expect(await screen.findByText("Check Runs")).toBeInTheDocument();
    expect(screen.queryByText("Total Checks")).toBeNull();
    // The denominator disclosure is split across text nodes by the {count}
    // interpolation and renders after the catalog query resolves, so await a
    // normalized-text match scoped to the leaf <p> (selector avoids ancestors).
    const sublabel = await screen.findByText(
      (_content, el) =>
        el?.tagName === "P" &&
        (el?.textContent ?? "").replace(/\s+/g, " ").includes("across 3 definitions"),
    );
    expect(sublabel).toBeInTheDocument();
  });

  it("F7.1: Domain Health card surfaces the skipped count and the summary bar adds a Skipped tile", async () => {
    const { fetchDQDashboard } = await import("@/api/queries");
    (fetchDQDashboard as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      domains: [
        // 10 pass / 0 fail / 0 warn / 6 skip / 16 total — the live "item" shape.
        { domain: "item", score: 100, passed: 10, failed: 0, warnings: 0, skipped: 6, total: 16 },
      ],
    });
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    // Domain card must show the skip count so 100% with 0 fail/0 warn reconciles.
    expect(await screen.findByText("6 skip")).toBeInTheDocument();
    // Summary KPI bar must have a "Skipped" tile so total checks reconcile.
    expect(screen.getByText("Skipped")).toBeInTheDocument();
  });

  it("U8.3: an info-only failing domain shows an info count and is not painted critical-red", async () => {
    const { fetchDQDashboard } = await import("@/api/queries");
    (fetchDQDashboard as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      domains: [
        // sku_to_item: only fail is info-severity -> backend scores it 100,
        // surfaces info_fails:2. The card must badge the score green (not red)
        // and disclose the informational fails as "2 info".
        { domain: "sku_to_item", score: 100, passed: 0, failed: 2, warnings: 0, skipped: 0, info_fails: 2, total: 2 },
      ],
    });
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    // The info count is disclosed so the planner sees it is informational.
    const infoChip = await screen.findByText("2 info");
    expect(infoChip).toBeInTheDocument();
    // The domain-card score badge (rounded-full pill) must be green (>80),
    // never the red critical bucket. (The overall ring also shows 100%.)
    const badges = screen.getAllByText("100%");
    const cardBadge = badges.find((b) => b.className.includes("rounded-full"));
    expect(cardBadge).toBeDefined();
    expect(cardBadge!.className).toContain("green");
    expect(cardBadge!.className).not.toContain("red");
  });

  it("F2.1/U2.18: an info-only domain does not show a red non-zero 'fail' next to green 100%", async () => {
    const { fetchDQDashboard } = await import("@/api/queries");
    (fetchDQDashboard as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      domains: [
        // Both 'fails' are info-severity (excluded from the 100% score). The red
        // "fail" count must not double-count them — it should read "0 fail".
        { domain: "sku_to_item", score: 100, passed: 0, failed: 2, warnings: 0, skipped: 0, info_fails: 2, total: 2 },
      ],
    });
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    // The card shows "0 fail" (scored fails), with the detail carried by "2 info".
    expect(await screen.findByText("0 fail")).toBeInTheDocument();
    expect(screen.getByText("2 info")).toBeInTheDocument();
    // No red "2 fail" chip should exist anywhere on the card.
    expect(screen.queryByText("2 fail")).toBeNull();
  });

  it("F3.1/U4.2: a warning-only failing domain (0 pass) badges NEUTRAL (not red 0%, not green 100%) and rolls warning fails into the amber warn chip", async () => {
    const { fetchDQDashboard } = await import("@/api/queries");
    (fetchDQDashboard as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      domains: [
        // forecast_to_sku: only fails are warning-severity, 0 pass -> nothing
        // scoreable, so the backend returns score:null. The card must NOT badge a
        // red 0% (F3.1) and must NOT badge a green 100% (U4.2) — it renders a
        // neutral "—". The red chip still reads "0 fail" and the 2 warning fails
        // roll into "2 warn".
        { domain: "forecast_to_sku", score: null, passed: 0, failed: 2, warnings: 0, skipped: 0, info_fails: 0, warning_fails: 2, total: 2 },
      ],
    });
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    // Red chip reads "0 fail" (warning fails excluded), warn chip absorbs them.
    expect(await screen.findByText("0 fail")).toBeInTheDocument();
    expect(screen.getByText("2 warn")).toBeInTheDocument();
    // The domain-card score badge (rounded-full pill) renders a neutral "—",
    // never a green 100% (would hide the gap) and never a red 0% (over-alarm).
    const cardBadge = screen
      .getAllByText("—")
      .find((b) => b.className.includes("rounded-full"));
    expect(cardBadge).toBeDefined();
    expect(cardBadge!.className).not.toContain("green");
    expect(cardBadge!.className).not.toContain("red");
    // No misleading green 100% / red 0% over the warn-only domain card.
    expect(screen.queryByText("100%")).toBeNull();
    expect(screen.queryByText("0%")).toBeNull();
    expect(screen.queryByText("2 fail")).toBeNull();
  });

  it("F7.2: the summary 'Failed' tile counts only scored/critical fails, matching the '0 fail' cards (warning/info fails are not red)", async () => {
    const { fetchDQDashboard } = await import("@/api/queries");
    // Cycle-7 live shape: summed across domains failed=26 decomposes entirely
    // into warning_fails=20 + info_fails=6, critical=0. Every domain card reads
    // "0 fail". The header "Failed" tile must NOT contradict them with "26".
    (fetchDQDashboard as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      domains: [
        { domain: "inventory", score: 100, passed: 20, failed: 10, warnings: 0, skipped: 0, info_fails: 0, warning_fails: 10, total: 30 },
        { domain: "customer", score: 100, passed: 4, failed: 16, warnings: 6, skipped: 0, info_fails: 6, warning_fails: 10, total: 32 },
      ],
    });
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    // Wait for the mocked dashboard payload to land (domain cards render).
    await screen.findByRole("button", { name: /inventory/i });
    // Locate the "Failed" summary tile by its label, then read its number.
    const failedLabel = screen.getByText("Failed");
    const failedTile = failedLabel.closest("div.rounded-lg")!;
    // Scored/critical fails = Σ max(0, failed - info_fails - warning_fails) = 0.
    expect(failedTile.textContent).toContain("0");
    expect(failedTile.textContent).not.toContain("26");
    // Warning-severity fails (20) roll into the amber Warnings tile, joining the
    // 6 warn-status checks -> 26. They must not masquerade as hard failures.
    const warnLabel = screen.getByText("Warnings");
    const warnTile = warnLabel.closest("div.rounded-lg")!;
    expect(warnTile.textContent).toContain("26");
  });

  it("U3.2: empty state does not instruct the stale /dq/run 404 path", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    const { container } = render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    // The stale curl pointed at /dq/run (a confirmed 404) and a non-existent
    // scripts/dq_run_checks.py. Neither should appear on screen.
    expect(container.textContent).not.toContain("/dq/run");
    expect(container.textContent).not.toContain("dq_run_checks.py");
  });

  it("U3.2: empty state offers an in-app run action wired to runDQChecks", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    // The empty-state CTA triggers the in-app run (POST /data-quality/run via
    // runDQChecks), not a curl to a 404 path.
    const cta = screen.getByRole("button", { name: /Run DQ checks now/i });
    fireEvent.click(cta);
    await waitFor(() => expect(mockRunDQChecks).toHaveBeenCalled());
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

    fireEvent.click(screen.getByRole("button", { name: /Run Checks Now/ }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Run Checks Now/ })).not.toBeDisabled();
    });

    await waitFor(() => {
      expect(fetchDashboardSpy).toHaveBeenCalled();
      expect(fetchChecksSpy).toHaveBeenCalled();
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
      // F5.3 — "Total Checks" relabeled to "Check Runs" (per-domain run total)
      // with a distinct-definition denominator sublabel.
      expect(screen.getByText("Check Runs")).toBeInTheDocument();
      expect(screen.getByText("Passed")).toBeInTheDocument();
      expect(screen.getByText("Failed")).toBeInTheDocument();
      expect(screen.getByText("Warnings")).toBeInTheDocument();
      // "Last Run" appears in both KPI bar and table header
      expect(screen.getAllByText(/Last Run/).length).toBeGreaterThanOrEqual(1);
    });
  });

  it("U3.12: Overall Health is a check-weighted pass rate, not a mean-of-domain-scores", async () => {
    const queries = await import("@/api/queries");
    // A large healthy domain + a tiny 0% domain. Mean-of-means would read ~47%
    // (dominated by the tiny domain); the check-weighted scored pass rate is 88%.
    (queries.fetchDQDashboard as ReturnType<typeof vi.fn>).mockResolvedValue({
      domains: [
        // 28 pass / 2 critical fail -> domain score 93.3, scored total 30.
        { domain: "inventory", score: 93.3, passed: 28, failed: 2, warnings: 0, skipped: 4, info_fails: 0, warning_fails: 0, total: 34 },
        // 0 pass / 2 critical fail -> domain score 0.0, scored total 2.
        { domain: "tiny", score: 0.0, passed: 0, failed: 2, warnings: 0, skipped: 0, info_fails: 0, warning_fails: 0, total: 2 },
      ],
    });

    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );

    // Check-weighted: round(100 * Σpassed / Σ(passed+critical_fails+warnings))
    // = round(100 * 28 / (30 + 2)) = round(87.5) = 88.  NOT the mean-of-means
    // round((93.3 + 0) / 2) = 47.
    const ring = await screen.findByText("88%");
    expect(ring).toBeInTheDocument();
    expect(screen.queryByText("47%")).toBeNull();
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
        { check_id: 1, check_name: "completeness_sales_qty", check_type: "completeness", domain: "sales", table_name: "fact_sales_monthly", severity: "critical", status: "fail", metric_value: 5.0, details: { total: 1000, nulls: 50, null_pct: 5.0 }, run_ts: "2026-03-17T10:00:00" },
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
      expect(screen.getByText("completeness_sales_qty")).toBeInTheDocument();
      expect(screen.getByText("range_sales_qty")).toBeInTheDocument();
    });

    // Filter to critical only
    const select = screen.getByLabelText("Filter issues by severity");
    fireEvent.change(select, { target: { value: "critical" } });

    await waitFor(() => {
      expect(screen.getByText("completeness_sales_qty")).toBeInTheDocument();
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
      expect(screen.getByText("Volume delta check")).toBeInTheDocument();
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
      expect(screen.getByText("Volume delta check")).toBeInTheDocument();
      // The "2 of 3" count in catalog header
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
        { id: 2, fix_type: "orphans", description: "Orphan keys: fact_sales(item_id,loc) → dim_sku", affected_rows: 42, recommendation: "Reload dimension", status: "pending" },
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
/*  Pipeline Lineage & Corrections sections                                   */
/* ========================================================================== */

describe("DataQualityTab — Pipeline Lineage & Corrections", () => {
  it("renders empty pipeline lineage section", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(<TestQueryWrapper><DataQualityTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Pipeline Lineage")).toBeInTheDocument();
    });
    expect(screen.getByText(/No pipeline batches yet/)).toBeInTheDocument();
  });

  it("renders batches when available", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchBatches as ReturnType<typeof vi.fn>).mockResolvedValue({
      batches: [
        { batch_id: 1, domain: "sales", source_file: "sales_clean.csv", row_count_in: 1000, row_count_out: 950, status: "completed", started_at: "2026-03-17T12:00:00", completed_at: "2026-03-17T12:01:00", error_message: null },
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
      expect(screen.getByText(/DQ Corrections — Corrected SKUs/)).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.getByText(/No DQ corrections recorded/)).toBeInTheDocument();
    });
  });

  it("renders corrections summary when available", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchCorrectionsSummary as ReturnType<typeof vi.fn>).mockResolvedValue({
      skus: [
        {
          item_id: "1401-BULK", loc: "101", correction_count: 5,
          domains: ["sales"], tables: ["fact_sales_monthly"],
          columns: ["qty", "qty_shipped"], fix_types: ["outliers"],
          strategies: ["iqr_per_sku"],
          earliest_period: "2024-01-01", latest_period: "2024-06-01",
          latest_at: "2026-03-22T10:00:00",
        },
      ],
      total: 1,
    });
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(<TestQueryWrapper><DataQualityTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("1401-BULK")).toBeInTheDocument();
    });
    expect(screen.getByText("101")).toBeInTheDocument();
    expect(screen.getByText("5")).toBeInTheDocument();
    expect(screen.getByText("outliers")).toBeInTheDocument();
  });

  it("navigates to Item Analysis on SKU row click", async () => {
    const queries = await import("@/api/queries");
    (queries.fetchCorrectionsSummary as ReturnType<typeof vi.fn>).mockResolvedValue({
      skus: [
        {
          item_id: "1401-BULK", loc: "101", correction_count: 1,
          domains: ["sales"], tables: ["fact_sales_monthly"],
          columns: ["qty"], fix_types: ["outliers"], strategies: ["iqr_per_sku"],
          earliest_period: "2024-06-01", latest_period: "2024-06-01",
          latest_at: "2026-03-22T10:00:00",
        },
      ],
      total: 1,
    });
    // Spy on window.location.href setter
    const hrefSetter = vi.fn();
    Object.defineProperty(window, "location", {
      value: { ...window.location, set href(v: string) { hrefSetter(v); }, get href() { return "http://localhost:3000"; } },
      writable: true,
    });
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(<TestQueryWrapper><DataQualityTab /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("1401-BULK")).toBeInTheDocument();
    });
    // Click the SKU row — should navigate
    fireEvent.click(screen.getByText("1401-BULK"));
    expect(hrefSetter).toHaveBeenCalledWith(
      expect.stringContaining("tab=itemAnalysis"),
    );
    expect(hrefSetter).toHaveBeenCalledWith(
      expect.stringContaining("item=1401-BULK"),
    );
  });
});
