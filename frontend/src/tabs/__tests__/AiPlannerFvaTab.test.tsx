/**
 * Smoke + interaction tests for AiPlannerFvaTab — PRD 02-27.
 *
 * Covers the empty / loaded / no-selection paths, the New Run dialog,
 * run-selection flow, status colour coding, summary KPI severity logic,
 * formatting helpers, polling cadence, and key barrel re-exports.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TestQueryWrapper } from "./test-utils";
import {
  fmtPct,
  fmtPp,
  fmtNum,
  severityForLift,
  humanizeRunError,
  runYieldNote,
} from "@/tabs/AiPlannerFvaTab";

// U4.4 — a succeeded run that produced 0 recommendations looked identical to a
// productive run (emerald "succeeded" + a bare "0"). A planner can't tell
// "ran cleanly, nothing actionable" from "did nothing useful". `runYieldNote`
// supplies a distinct sub-note for the zero-yield-success case only.
describe("runYieldNote — zero-yield success affordance (U4.4)", () => {
  it("returns a 'no recommendations' note for succeeded with 0 recs", () => {
    expect(runYieldNote("succeeded", 0)).toMatch(/no recommendations/i);
  });
  it("returns null for a productive succeeded run", () => {
    expect(runYieldNote("succeeded", 12)).toBeNull();
  });
  it("returns null for non-succeeded statuses regardless of recs", () => {
    expect(runYieldNote("failed", 0)).toBeNull();
    expect(runYieldNote("running", 0)).toBeNull();
  });
});

const PYDANTIC_ERR =
  "2 validation errors for Recommendation proposed_qty.1 Input should be a valid number " +
  "[type=float_type, input_value=None, input_type=NoneType] " +
  "For further information visit https://errors.pydantic.dev/2.12/v/float_type " +
  "proposed_qty.4 Input should be a valid number [type=float_type, input_value=None, " +
  "input_type=NoneType] For further information visit https://errors.pydantic.dev/2.12/v/float_type";

// ---------------------------------------------------------------------------
// Mock the API layer — the barrel re-exports from ./api/queries
// ---------------------------------------------------------------------------

vi.mock("@/api/queries", () => ({
  aiFvaBacktestKeys: {
    root: ["ai-planner", "fva-backtest"],
    list: () => ["ai-planner", "fva-backtest", "list", null],
    detail: (id: string) => ["ai-planner", "fva-backtest", "detail", id],
    summary: (id: string) => ["ai-planner", "fva-backtest", "summary", id],
    byRecommendation: (id: string) => ["ai-planner", "fva-backtest", "by-recommendation", id],
    byMonth: (id: string) => ["ai-planner", "fva-backtest", "by-month", id],
    dfus: (id: string) => ["ai-planner", "fva-backtest", "dfus", id],
    dfuDetail: (id: string, item: string, loc: string) =>
      ["ai-planner", "fva-backtest", "dfu-detail", id, item, loc],
  },
  listFvaBacktestRuns: vi.fn().mockResolvedValue({ runs: [], count: 0 }),
  getFvaBacktestSummary: vi.fn().mockResolvedValue(null),
  getFvaBacktestByMonth: vi.fn().mockResolvedValue({ run_id: "", rows: [] }),
  getFvaBacktestByRecommendation: vi.fn().mockResolvedValue({ run_id: "", rows: [] }),
  getFvaBacktestDfus: vi.fn().mockResolvedValue({ run_id: "", rows: [], count: 0 }),
  getFvaBacktestDfuDetail: vi.fn().mockResolvedValue({
    run_id: "", item_id: "", loc: "",
    summary: { n_obs: 0, baseline_wape_pct: null, ai_wape_pct: null, lift_pp: null },
    lags: [], recommendations: [],
  }),
  startFvaBacktestRun: vi.fn().mockResolvedValue({ status: "accepted", message: "ok" }),
}));

// Mock lucide-react icons that the shared UI primitives + KpiCard pull in.
vi.mock("lucide-react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("lucide-react")>();
  return {
    ...actual,
    TrendingUp: ({ className }: { className?: string }) => (
      <span data-testid="icon-trending-up" className={className} />
    ),
    TrendingDown: ({ className }: { className?: string }) => (
      <span data-testid="icon-trending-down" className={className} />
    ),
    Minus: ({ className }: { className?: string }) => (
      <span data-testid="icon-minus" className={className} />
    ),
    HelpCircle: ({ className }: { className?: string }) => (
      <span data-testid="icon-help" className={className} />
    ),
    X: ({ className }: { className?: string }) => (
      <span data-testid="icon-x" className={className} />
    ),
  };
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeRun(overrides: Record<string, unknown> = {}) {
  return {
    run_id: "11111111-1111-1111-1111-111111111111",
    status: "succeeded" as const,
    started_at: null,
    completed_at: null,
    window_months: 10,
    as_of_date: "2026-04-01",
    horizon_months: 3,
    provider: "ollama" as const,
    ai_model: "qwen2.5:32b",
    n_dfus_sampled: 50,
    n_recommendations: 500,
    estimated_cost_usd: 0,
    actual_cost_usd: 0,
    error_message: null,
    ...overrides,
  };
}

describe("AiPlannerFvaTab — smoke", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the header and empty runs state", async () => {
    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    expect(screen.getByText(/AI Planner — FVA Backtest/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /New Backtest Run/i })).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText(/No runs yet/i)).toBeInTheDocument();
    });
    // Empty state on the right
    expect(
      screen.getByText(/Select a run from the list, or start a new backtest/i),
    ).toBeInTheDocument();
  });

  it("renders the PRD reference as non-interactive text, not a dead href='#' anchor (U6.3)", async () => {
    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    const { container } = render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    // The PRD reference text is still shown for context...
    expect(screen.getByText(/PRD 02-27/i)).toBeInTheDocument();
    // ...but it must NOT be a dead underlined anchor that goes nowhere.
    const deadAnchors = container.querySelectorAll('a[href="#"]');
    expect(deadAnchors.length).toBe(0);
  });

  it("does not show the Printable Report link when no run is selected", async () => {
    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText(/No runs yet/i)).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /Printable Report/i })).not.toBeInTheDocument();
  });

  it("renders runs from the API", async () => {
    const { listFvaBacktestRuns } = await import("@/api/queries");
    (listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      runs: [makeRun()],
      count: 1,
    });
    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("succeeded")).toBeInTheDocument();
      expect(screen.getByText("2026-04-01")).toBeInTheDocument();
      expect(screen.getByText("ollama")).toBeInTheDocument();
    });
  });
});

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

describe("AiPlannerFvaTab — formatting helpers", () => {
  it("fmtPct handles null/undefined/NaN with em-dash", () => {
    expect(fmtPct(null)).toBe("—");
    expect(fmtPct(undefined)).toBe("—");
    expect(fmtPct(Number.NaN)).toBe("—");
  });

  it("fmtPct formats with two decimal places by default", () => {
    expect(fmtPct(12.345)).toBe("12.35%");
    expect(fmtPct(0)).toBe("0.00%");
  });

  it("fmtPct respects the places override", () => {
    expect(fmtPct(12.345, 1)).toBe("12.3%");
  });

  it("fmtPp handles null/undefined/NaN with em-dash", () => {
    expect(fmtPp(null)).toBe("—");
    expect(fmtPp(undefined)).toBe("—");
    expect(fmtPp(Number.NaN)).toBe("—");
  });

  it("fmtPp uses a leading + for positive values", () => {
    expect(fmtPp(2.5)).toBe("+2.50pp");
  });

  it("fmtPp keeps the minus sign for negative values", () => {
    expect(fmtPp(-1.5)).toBe("-1.50pp");
  });

  it("fmtPp prints 0 as 0.00pp (no sign)", () => {
    expect(fmtPp(0)).toBe("0.00pp");
  });

  it("fmtNum returns em-dash for null/undefined", () => {
    expect(fmtNum(null)).toBe("—");
    expect(fmtNum(undefined)).toBe("—");
  });

  it("fmtNum uses locale-formatted thousands separators", () => {
    // toLocaleString in Node defaults to en-US-style commas in jsdom.
    expect(fmtNum(1234567)).toBe((1234567).toLocaleString());
    expect(fmtNum(0)).toBe("0");
  });

  it("severityForLift maps the three thresholds", () => {
    expect(severityForLift(null)).toBe("neutral");
    expect(severityForLift(undefined)).toBe("neutral");
    expect(severityForLift(1.0)).toBe("best");
    expect(severityForLift(0.51)).toBe("best");
    expect(severityForLift(0.5)).toBe("neutral");
    expect(severityForLift(0)).toBe("neutral");
    expect(severityForLift(-0.5)).toBe("neutral");
    expect(severityForLift(-0.51)).toBe("warning");
    expect(severityForLift(-2)).toBe("warning");
  });
});

// ---------------------------------------------------------------------------
// NewRunDialog interaction
// ---------------------------------------------------------------------------

describe("AiPlannerFvaTab — NewRunDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("opens the dialog when the trigger is clicked, with default form values", async () => {
    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /New Backtest Run/i }));

    // Dialog title appears
    expect(await screen.findByText("Start AI FVA Backtest")).toBeInTheDocument();

    // Default values
    const windowInput = screen.getByLabelText(/Window \(months\)/i) as HTMLInputElement;
    const horizonInput = screen.getByLabelText(/Horizon \(months\)/i) as HTMLInputElement;
    const dfuInput = screen.getByLabelText(/DFU limit/i) as HTMLInputElement;
    expect(windowInput.value).toBe("10");
    expect(horizonInput.value).toBe("3");
    expect(dfuInput.value).toBe("50");

    // Provider select renders with "ollama" selected (the local SelectValue
    // component echoes the raw value, not the label).
    expect(screen.getByText("ollama")).toBeInTheDocument();
  });

  it("submits the form payload with custom notes + as_of_date", async () => {
    const { startFvaBacktestRun } = await import("@/api/queries");
    const startMock = startFvaBacktestRun as unknown as ReturnType<typeof vi.fn>;
    startMock.mockResolvedValueOnce({ status: "accepted", message: "ok" });

    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /New Backtest Run/i }));

    const notesInput = await screen.findByLabelText(/Notes/i);
    await user.type(notesInput, "smoke");

    const dateInput = screen.getByLabelText(/As-of date/i);
    await user.type(dateInput, "2026-04-01");

    await user.click(screen.getByRole("button", { name: /Start Backtest/i }));

    await waitFor(() => {
      expect(startMock).toHaveBeenCalledTimes(1);
    });
    const payload = startMock.mock.calls[0][0];
    expect(payload).toMatchObject({
      window_months: 10,
      horizon_months: 3,
      provider: "ollama",
      limit_dfus: 50,
      notes: "smoke",
      as_of_date: "2026-04-01",
    });
  });

  it("closes the dialog and invalidates the runs cache on success", async () => {
    const { startFvaBacktestRun } = await import("@/api/queries");
    (startFvaBacktestRun as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      status: "accepted",
      message: "ok",
    });

    // Spy on invalidateQueries through a custom client.
    const { QueryClient, QueryClientProvider } = await import("@tanstack/react-query");
    const { ThemeProvider } = await import("@/context/ThemeContext");
    const { GlobalFilterProvider } = await import("@/context/GlobalFilterContext");

    const client = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 } },
    });
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    const filterValue = {
      filters: {
        brand: [] as string[],
        category: [] as string[],
        market: [] as string[],
        channel: [] as string[],
        item: [] as string[],
        location: [] as string[],
        cluster: [] as string[],
        timeGrain: "month" as const,
      },
      setFilters: vi.fn(),
      resetFilters: vi.fn(),
      hasActiveFilters: false,
      planningDate: null,
    };

    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <ThemeProvider value={{ theme: "light" }}>
        <QueryClientProvider client={client}>
          <GlobalFilterProvider value={filterValue}>
            <AiPlannerFvaTab />
          </GlobalFilterProvider>
        </QueryClientProvider>
      </ThemeProvider>,
    );
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /New Backtest Run/i }));
    expect(await screen.findByText("Start AI FVA Backtest")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Start Backtest/i }));

    // Dialog closes
    await waitFor(() => {
      expect(screen.queryByText("Start AI FVA Backtest")).not.toBeInTheDocument();
    });

    // invalidateQueries called with the root key
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: ["ai-planner", "fva-backtest"],
    });
  });

  it("shows an inline error and keeps the dialog open on mutation failure", async () => {
    const { startFvaBacktestRun } = await import("@/api/queries");
    (startFvaBacktestRun as unknown as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new Error("provider unreachable"),
    );

    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /New Backtest Run/i }));
    expect(await screen.findByText("Start AI FVA Backtest")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Start Backtest/i }));

    // Error message renders
    expect(await screen.findByText(/provider unreachable/i)).toBeInTheDocument();
    // Dialog is still open
    expect(screen.getByText("Start AI FVA Backtest")).toBeInTheDocument();
  });

  it("closes the dialog without firing the mutation when Cancel is clicked", async () => {
    const { startFvaBacktestRun } = await import("@/api/queries");
    const startMock = startFvaBacktestRun as unknown as ReturnType<typeof vi.fn>;

    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /New Backtest Run/i }));
    expect(await screen.findByText("Start AI FVA Backtest")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^Cancel$/i }));

    await waitFor(() => {
      expect(screen.queryByText("Start AI FVA Backtest")).not.toBeInTheDocument();
    });
    expect(startMock).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// Run selection flow + printable report link
// ---------------------------------------------------------------------------

describe("AiPlannerFvaTab — run selection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("selecting a run triggers detail fetches and reveals the Printable Report link", async () => {
    const {
      listFvaBacktestRuns,
      getFvaBacktestSummary,
      getFvaBacktestByMonth,
      getFvaBacktestByRecommendation,
      getFvaBacktestDfus,
    } = await import("@/api/queries");

    (listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      runs: [makeRun()],
      count: 1,
    });
    const summaryMock = getFvaBacktestSummary as unknown as ReturnType<typeof vi.fn>;
    const byMonthMock = getFvaBacktestByMonth as unknown as ReturnType<typeof vi.fn>;
    const byRecMock = getFvaBacktestByRecommendation as unknown as ReturnType<typeof vi.fn>;
    const dfusMock = getFvaBacktestDfus as unknown as ReturnType<typeof vi.fn>;

    summaryMock.mockResolvedValue({
      run_id: "11111111-1111-1111-1111-111111111111",
      baseline_wape_pct: 30.5,
      ai_wape_pct: 28.0,
      lift_pct: 2.5,
      n_dfus: 50,
      n_winners: 30,
      n_losers: 15,
      n_ties: 5,
      win_rate_pct: 60.0,
    });
    byMonthMock.mockResolvedValue({ run_id: "x", rows: [] });
    byRecMock.mockResolvedValue({ run_id: "x", rows: [] });
    dfusMock.mockResolvedValue({ run_id: "x", rows: [], count: 0 });

    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    const user = userEvent.setup();

    // Wait for the run row to appear, then click it.
    const statusCell = await screen.findByText("succeeded");
    await user.click(statusCell);

    // Placeholder is replaced
    await waitFor(() => {
      expect(
        screen.queryByText(/Select a run from the list, or start a new backtest/i),
      ).not.toBeInTheDocument();
    });

    // Detail fetchers fired
    await waitFor(() => {
      expect(summaryMock).toHaveBeenCalledWith("11111111-1111-1111-1111-111111111111");
      expect(byMonthMock).toHaveBeenCalledWith("11111111-1111-1111-1111-111111111111");
      expect(byRecMock).toHaveBeenCalledWith("11111111-1111-1111-1111-111111111111");
      expect(dfusMock).toHaveBeenCalled();
    });

    // Printable Report link
    const reportLink = await screen.findByRole("link", { name: /Printable Report/i });
    expect(reportLink).toHaveAttribute(
      "href",
      "/ai-planner/fva-backtest/runs/11111111-1111-1111-1111-111111111111/report.html",
    );
    expect(reportLink).toHaveAttribute("target", "_blank");
    expect(reportLink).toHaveAttribute("rel", "noopener noreferrer");
  });
});

// ---------------------------------------------------------------------------
// U7.2 — failed / 0-rec runs must not dead-end into blank detail panels
// ---------------------------------------------------------------------------

describe("AiPlannerFvaTab — failed & zero-recommendation runs (U7.2)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("surfaces the error_message inline in the failed run row", async () => {
    const { listFvaBacktestRuns } = await import("@/api/queries");
    (listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      runs: [
        makeRun({
          run_id: "id-failed",
          status: "failed",
          n_recommendations: null,
          n_dfus_sampled: null,
          error_message: "ollama provider unreachable: connection refused",
        }),
      ],
      count: 1,
    });
    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    // The failure reason (already returned by the API) is shown in the row.
    expect(
      await screen.findByText(/ollama provider unreachable: connection refused/i),
    ).toBeInTheDocument();
  });

  it("shows an explicit error state in the detail column when a failed run is selected", async () => {
    const { listFvaBacktestRuns } = await import("@/api/queries");
    (listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      runs: [
        makeRun({
          run_id: "id-failed",
          status: "failed",
          n_recommendations: null,
          error_message: "model timed out after 600s",
        }),
      ],
      count: 1,
    });
    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    const user = userEvent.setup();
    const statusCell = await screen.findByText("failed");
    await user.click(statusCell);

    // The right column explains the failure instead of three blank "No data yet." cards.
    expect(await screen.findByText(/This run failed/i)).toBeInTheDocument();
    // The error message shows in the detail column (and also inline in the row).
    expect(screen.getAllByText(/model timed out after 600s/i).length).toBeGreaterThanOrEqual(1);
    // No misleading KPI/empty panels for a failed run.
    expect(screen.queryByText("FVA by Month (Walk-Forward)")).not.toBeInTheDocument();
  });

  it("shows a single 'no recommendations' empty state for a succeeded 0-rec run", async () => {
    const { listFvaBacktestRuns } = await import("@/api/queries");
    (listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      runs: [
        makeRun({
          run_id: "id-zero",
          status: "succeeded",
          n_recommendations: 0,
          n_dfus_sampled: 50,
        }),
      ],
      count: 1,
    });
    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    const user = userEvent.setup();
    const statusCell = await screen.findByText("succeeded");
    await user.click(statusCell);

    expect(
      await screen.findByText(/No recommendations were generated for this run/i),
    ).toBeInTheDocument();
    // The blank KPI/table panels are suppressed for a 0-rec run.
    expect(screen.queryByText("FVA by Month (Walk-Forward)")).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// humanizeRunError — U2.19: never leak raw pydantic jargon to the planner
// ---------------------------------------------------------------------------
describe("humanizeRunError (U2.19)", () => {
  it("maps the proposed_qty pydantic error to a planner-readable summary", () => {
    const out = humanizeRunError(PYDANTIC_ERR);
    expect(out).toMatch(/no quantity|null quantit|2 recommendation/i);
    expect(out).not.toMatch(/type=float_type/);
    expect(out).not.toMatch(/pydantic\.dev/);
    expect(out.length).toBeLessThanOrEqual(120);
  });

  it("clamps and de-jargons an unknown long error", () => {
    const raw =
      "ValueError: something went very wrong ".repeat(20) +
      "see https://errors.pydantic.dev/2.12/v/float_type";
    const out = humanizeRunError(raw);
    expect(out.length).toBeLessThanOrEqual(120);
    expect(out).not.toMatch(/pydantic\.dev/);
  });

  it("passes a short, clean message through unchanged", () => {
    expect(humanizeRunError("ollama provider unreachable")).toBe(
      "ollama provider unreachable",
    );
  });

  it("returns a neutral fallback for null/empty", () => {
    expect(humanizeRunError(null)).toMatch(/no error/i);
    expect(humanizeRunError("")).toMatch(/no error/i);
  });
});

// ---------------------------------------------------------------------------
// Status colour coding in RunsListPanel
// ---------------------------------------------------------------------------

describe("AiPlannerFvaTab — RunsListPanel status colours", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  async function renderWithStatus(status: string, expectedClassSubstring: string) {
    const { listFvaBacktestRuns } = await import("@/api/queries");
    (listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      runs: [makeRun({ status, run_id: `id-${status}` })],
      count: 1,
    });
    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    const span = await screen.findByText(status);
    expect(span.className).toContain(expectedClassSubstring);
  }

  it("succeeded -> emerald-600", async () => {
    await renderWithStatus("succeeded", "text-emerald-600");
  });

  it("running -> blue-600", async () => {
    await renderWithStatus("running", "text-blue-600");
  });

  it("failed -> rose-600", async () => {
    await renderWithStatus("failed", "text-rose-600");
  });

  it("other (cancelled) -> muted-foreground", async () => {
    await renderWithStatus("cancelled", "text-muted-foreground");
  });
});

// ---------------------------------------------------------------------------
// SummaryKpis rendering
// ---------------------------------------------------------------------------

describe("AiPlannerFvaTab — SummaryKpis", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  async function selectFirstRun() {
    const user = userEvent.setup();
    const cell = await screen.findByText("succeeded");
    await user.click(cell);
  }

  it("renders Lift with best severity when lift > 0.5", async () => {
    const {
      listFvaBacktestRuns,
      getFvaBacktestSummary,
    } = await import("@/api/queries");
    (listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      runs: [makeRun()],
      count: 1,
    });
    (getFvaBacktestSummary as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      run_id: "x",
      baseline_wape_pct: 30,
      ai_wape_pct: 28,
      lift_pct: 2.0,
      n_dfus: 1,
      n_winners: 1,
      n_losers: 0,
      n_ties: 0,
      win_rate_pct: 100,
    });

    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    await selectFirstRun();

    const liftLabel = await screen.findByText("FVA Lift");
    const liftCard = liftLabel.closest("div.rounded-lg");
    expect(liftCard).not.toBeNull();
    // best severity yields text-[var(--kpi-best)] on the value text
    expect(liftCard!.innerHTML).toContain("text-[var(--kpi-best)]");
    // The lift value renders as +2.00pp
    expect(within(liftCard as HTMLElement).getByText("+2.00pp")).toBeInTheDocument();
  });

  it("renders Lift with warning severity when lift < -0.5", async () => {
    const {
      listFvaBacktestRuns,
      getFvaBacktestSummary,
    } = await import("@/api/queries");
    (listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      runs: [makeRun()],
      count: 1,
    });
    (getFvaBacktestSummary as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      run_id: "x",
      baseline_wape_pct: 28,
      ai_wape_pct: 30,
      lift_pct: -2.0,
      n_dfus: 1,
      n_winners: 0,
      n_losers: 1,
      n_ties: 0,
      win_rate_pct: 0,
    });

    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    await selectFirstRun();

    const liftLabel = await screen.findByText("FVA Lift");
    const liftCard = liftLabel.closest("div.rounded-lg");
    expect(liftCard).not.toBeNull();
    expect(liftCard!.innerHTML).toContain("text-[var(--kpi-warning)]");
    expect(within(liftCard as HTMLElement).getByText("-2.00pp")).toBeInTheDocument();
  });

  it("renders Lift with neutral severity when |lift| <= 0.5", async () => {
    const {
      listFvaBacktestRuns,
      getFvaBacktestSummary,
    } = await import("@/api/queries");
    (listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      runs: [makeRun()],
      count: 1,
    });
    (getFvaBacktestSummary as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      run_id: "x",
      baseline_wape_pct: 30,
      ai_wape_pct: 30,
      lift_pct: 0.3,
      n_dfus: 1,
      n_winners: 0,
      n_losers: 0,
      n_ties: 1,
      win_rate_pct: 0,
    });

    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    await selectFirstRun();

    const liftLabel = await screen.findByText("FVA Lift");
    const liftCard = liftLabel.closest("div.rounded-lg");
    expect(liftCard).not.toBeNull();
    // Neutral severity should NOT add best/warning colour to the value
    expect(liftCard!.innerHTML).not.toContain("text-[var(--kpi-best)]");
    expect(liftCard!.innerHTML).not.toContain("text-[var(--kpi-warning)]");
    expect(within(liftCard as HTMLElement).getByText("+0.30pp")).toBeInTheDocument();
  });

  it("renders em-dashes for all KPIs when lift is null", async () => {
    const {
      listFvaBacktestRuns,
      getFvaBacktestSummary,
    } = await import("@/api/queries");
    (listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      runs: [makeRun()],
      count: 1,
    });
    (getFvaBacktestSummary as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      run_id: "x",
      baseline_wape_pct: null,
      ai_wape_pct: null,
      lift_pct: null,
      n_dfus: 0,
      n_winners: 0,
      n_losers: 0,
      n_ties: 0,
      win_rate_pct: null,
    });

    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    await selectFirstRun();

    // Wait for the panel to render
    await screen.findByText("FVA Lift");

    // Each main KPI value should show em-dash. With 4 KPIs (baseline, ai, lift, win rate)
    // and the win-rate sublabel using fmtNum (0 not em-dash), we expect ≥4 dashes.
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(4);
  });

  it("formats the Win Rate sublabel as W / L / T", async () => {
    const {
      listFvaBacktestRuns,
      getFvaBacktestSummary,
    } = await import("@/api/queries");
    (listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      runs: [makeRun()],
      count: 1,
    });
    (getFvaBacktestSummary as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      run_id: "x",
      baseline_wape_pct: 30,
      ai_wape_pct: 28,
      lift_pct: 2,
      n_dfus: 50,
      n_winners: 30,
      n_losers: 15,
      n_ties: 5,
      win_rate_pct: 60,
    });

    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    await selectFirstRun();

    expect(await screen.findByText(/30 W \/ 15 L \/ 5 T/)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Polling cadence
// ---------------------------------------------------------------------------

describe("AiPlannerFvaTab — polling cadence", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("re-fetches the runs list on the 5s poll interval", async () => {
    const { listFvaBacktestRuns } = await import("@/api/queries");
    const listMock = listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>;
    listMock.mockResolvedValue({ runs: [], count: 0 });

    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );

    // First call fires on mount
    await vi.waitFor(() => {
      expect(listMock).toHaveBeenCalledTimes(1);
    });

    // Advance fake time by 10s — should trigger two more polls at 5s and 10s.
    await vi.advanceTimersByTimeAsync(10_000);

    expect(listMock.mock.calls.length).toBeGreaterThanOrEqual(3);
  });
});

// ---------------------------------------------------------------------------
// DfuDetailDialog — clicking a DFU row opens the dialog and renders
// the walk-forward detail returned by getFvaBacktestDfuDetail.
// ---------------------------------------------------------------------------

describe("AiPlannerFvaTab — DfuDetailDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("clicking a DFU row opens the dialog with WAPE + per-lag detail", async () => {
    const {
      listFvaBacktestRuns,
      getFvaBacktestDfus,
      getFvaBacktestDfuDetail,
    } = await import("@/api/queries");

    (listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      runs: [makeRun()],
      count: 1,
    });

    const dfusMock = getFvaBacktestDfus as unknown as ReturnType<typeof vi.fn>;
    dfusMock.mockResolvedValue({
      run_id: "11111111-1111-1111-1111-111111111111",
      rows: [
        {
          item_id: "916045", loc: "1401-BULK",
          sae_baseline: 100, sae_ai: 60,
          abs_error_reduction: 40, n_obs: 9,
        },
      ],
      count: 1,
    });

    const detailMock = getFvaBacktestDfuDetail as unknown as ReturnType<typeof vi.fn>;
    detailMock.mockResolvedValue({
      run_id: "11111111-1111-1111-1111-111111111111",
      item_id: "916045", loc: "1401-BULK",
      summary: { n_obs: 3, baseline_wape_pct: 63.6, ai_wape_pct: 81.8, lift_pp: 18.2 },
      lags: [
        {
          forecast_run_month: "2025-11-01", target_month: "2025-11-01", lag: 1,
          baseline_qty: 100, ai_qty: 80, actual_qty: 90,
        },
      ],
      recommendations: [
        {
          forecast_run_month: "2025-11-01",
          recommendation_code: "SCALE_DOWN",
          pct_change: -20, confidence: 0.85,
          rationale: "downward trend in recent actuals",
          evidence_keys: ["recent_drop"],
        },
      ],
    });

    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    const user = userEvent.setup();

    // Select the run so DfuDrillPanel mounts.
    const statusCell = await screen.findByText("succeeded");
    await user.click(statusCell);

    // Wait for the DFU row, then click it.
    const dfuItemCell = await screen.findByText("916045");
    await user.click(dfuItemCell);

    // Detail fetcher is called with (runId, itemId, loc).
    await waitFor(() => {
      expect(detailMock).toHaveBeenCalledWith(
        "11111111-1111-1111-1111-111111111111", "916045", "1401-BULK",
      );
    });

    // Dialog renders title + recommendation rationale + the per-DFU lift value.
    // (Don't assert on "Baseline WAPE" — that label also appears in the
    // run-level SummaryKpis, so a plain getByText would match twice.)
    await screen.findByText(/916045 @ 1401-BULK/);
    expect(screen.getByText("SCALE_DOWN")).toBeInTheDocument();
    expect(
      screen.getByText("downward trend in recent actuals"),
    ).toBeInTheDocument();
    expect(screen.getByText(/\+18\.20pp/)).toBeInTheDocument();
  });

  it("typing item_id + loc and clicking 'Inspect DFU' opens the dialog", async () => {
    const {
      listFvaBacktestRuns,
      getFvaBacktestDfus,
      getFvaBacktestDfuDetail,
    } = await import("@/api/queries");

    (listFvaBacktestRuns as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      runs: [makeRun()],
      count: 1,
    });
    (getFvaBacktestDfus as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      run_id: "x", rows: [], count: 0,
    });
    const detailMock = getFvaBacktestDfuDetail as unknown as ReturnType<typeof vi.fn>;
    detailMock.mockResolvedValue({
      run_id: "11111111-1111-1111-1111-111111111111",
      item_id: "ABC", loc: "L9",
      summary: { n_obs: 0, baseline_wape_pct: null, ai_wape_pct: null, lift_pp: null },
      lags: [], recommendations: [],
    });

    const { default: AiPlannerFvaTab } = await import("@/tabs/AiPlannerFvaTab");
    render(
      <TestQueryWrapper>
        <AiPlannerFvaTab />
      </TestQueryWrapper>,
    );
    const user = userEvent.setup();

    const statusCell = await screen.findByText("succeeded");
    await user.click(statusCell);

    const itemInput = await screen.findByPlaceholderText(/e\.g\. 916045/);
    const locInput = await screen.findByPlaceholderText(/e\.g\. 1401-BULK/);
    await user.type(itemInput, "ABC");
    await user.type(locInput, "L9");
    await user.click(screen.getByRole("button", { name: /Inspect DFU/i }));

    await waitFor(() => {
      expect(detailMock).toHaveBeenCalledWith(
        "11111111-1111-1111-1111-111111111111", "ABC", "L9",
      );
    });
    await screen.findByText(/ABC @ L9/);
  });
});
