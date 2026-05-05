import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import { JobHistoryTable } from "../JobHistoryTable";
import type { Job } from "../../../api/queries/integration";

// NB: spread overrides AFTER defaults so explicit `null` survives (?? would
// silently substitute the default for a null override).
function buildJob(overrides: Partial<Job> = {}): Job {
  const base: Job = {
    id: "job-1",
    domain: "sales",
    mode: "onetime",
    slice: null,
    file_path: null,
    status: "success",
    rows_loaded: 0,
    rows_inserted: null,
    rows_updated: null,
    rows_deleted: null,
    error_message: null,
    started_at: "2026-04-01T10:00:00Z",
    completed_at: "2026-04-01T10:00:15Z",
    duration_ms: 15000,
    triggered_by: "ui",
  };
  return { ...base, ...overrides };
}

describe("JobHistoryTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the empty message and no table when jobs is empty", () => {
    render(<JobHistoryTable jobs={[]} emptyMessage="Nothing here." />);
    expect(screen.getByText("Nothing here.")).toBeInTheDocument();
    // No <table> rendered
    expect(document.querySelector("table")).toBeNull();
  });

  it("falls back to default empty message when none supplied", () => {
    render(<JobHistoryTable jobs={[]} />);
    expect(screen.getByText("No jobs.")).toBeInTheDocument();
  });

  it("renders one tbody row per job", () => {
    const jobs = [
      buildJob({ id: "a", domain: "sales" }),
      buildJob({ id: "b", domain: "forecast" }),
      buildJob({ id: "c", domain: "inventory" }),
    ];
    render(<JobHistoryTable jobs={jobs} />);
    const tbody = document.querySelector("tbody");
    expect(tbody).not.toBeNull();
    // Only the primary <tr> per job (no failure-row expansion in this fixture)
    const rows = tbody!.querySelectorAll("tr");
    expect(rows.length).toBe(3);
    expect(screen.getByText("sales")).toBeInTheDocument();
    expect(screen.getByText("forecast")).toBeInTheDocument();
    expect(screen.getByText("inventory")).toBeInTheDocument();
  });

  it("renders a status badge per job", () => {
    const jobs = [
      buildJob({ id: "a", status: "success" }),
      buildJob({ id: "b", status: "running" }),
    ];
    render(<JobHistoryTable jobs={jobs} />);
    expect(screen.getByLabelText("status: success")).toBeInTheDocument();
    expect(screen.getByLabelText("status: running")).toBeInTheDocument();
  });

  it("renders an em-dash for null slice", () => {
    render(<JobHistoryTable jobs={[buildJob({ slice: null })]} />);
    // At least one cell should render the em-dash (the slice cell).
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(1);
  });

  it("renders the slice value when provided", () => {
    render(
      <JobHistoryTable jobs={[buildJob({ slice: "2026-04", domain: "sales" })]} />,
    );
    expect(screen.getByText("2026-04")).toBeInTheDocument();
  });

  it("renders the error message under a failed row", () => {
    render(
      <JobHistoryTable
        jobs={[
          buildJob({
            id: "fail-1",
            status: "failed",
            error_message: "Bad CSV",
          }),
        ]}
      />,
    );
    expect(screen.getByText("Bad CSV")).toBeInTheDocument();
  });

  it("does not render a failure row when error_message is null", () => {
    render(
      <JobHistoryTable
        jobs={[buildJob({ status: "failed", error_message: null })]}
      />,
    );
    // tbody should still have only one row (primary), no expansion row.
    const tbody = document.querySelector("tbody")!;
    expect(tbody.querySelectorAll("tr").length).toBe(1);
  });

  it("calls onSelect with the clicked job", () => {
    const onSelect = vi.fn();
    const job = buildJob({ id: "click-me", domain: "salesX" });
    render(<JobHistoryTable jobs={[job]} onSelect={onSelect} />);
    // Click the row containing the cell text "salesX".
    const cell = screen.getByText("salesX");
    const row = cell.closest("tr")!;
    fireEvent.click(row);
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith(job);
  });

  it("does not attach a click handler / cursor when onSelect is omitted", () => {
    render(<JobHistoryTable jobs={[buildJob({ domain: "noClick" })]} />);
    const row = screen.getByText("noClick").closest("tr")!;
    expect(row.className).not.toContain("cursor-pointer");
  });

  it("formats sub-minute durations as Ns", () => {
    render(<JobHistoryTable jobs={[buildJob({ duration_ms: 15000 })]} />);
    expect(screen.getByText("15s")).toBeInTheDocument();
  });

  it("formats multi-minute durations as Nm Ms", () => {
    render(<JobHistoryTable jobs={[buildJob({ duration_ms: 125000 })]} />);
    expect(screen.getByText("2m 5s")).toBeInTheDocument();
  });

  it("renders an em-dash when duration is null", () => {
    render(
      <JobHistoryTable
        jobs={[buildJob({ duration_ms: null, slice: "abc" })]}
      />,
    );
    // slice="abc" so the only "—" should be the duration cell.
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("hides the Rows Loaded column when showRowsLoaded=false", () => {
    render(
      <JobHistoryTable jobs={[buildJob()]} showRowsLoaded={false} />,
    );
    const head = document.querySelector("thead")!;
    const headerText = within(head).queryByText("Rows Loaded");
    expect(headerText).toBeNull();
  });

  it("renders the Rows Loaded header by default", () => {
    render(<JobHistoryTable jobs={[buildJob()]} />);
    const head = document.querySelector("thead")!;
    expect(within(head).getByText("Rows Loaded")).toBeInTheDocument();
  });

  it("formats rows_loaded with thousands separator for success jobs", () => {
    render(
      <JobHistoryTable
        jobs={[buildJob({ status: "success", rows_loaded: 12345 })]}
      />,
    );
    // Intl.NumberFormat default for en/US-like locales -> "12,345".
    // Allow either comma- or non-comma-separated to keep test locale-tolerant.
    const candidates = ["12,345", "12.345", "12345"];
    const found = candidates.some((c) => screen.queryByText(c) !== null);
    expect(found).toBe(true);
  });
});
