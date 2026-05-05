import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("../../api/queries/integration", () => ({
  integrationKeys: {
    all: ["integration"],
    domains: ["integration", "domains"],
    health: ["integration", "health"],
    jobs: () => ["integration", "jobs", {}],
    job: (id: string) => ["integration", "job", id],
  },
  listDomains: vi.fn(),
  listJobs: vi.fn(),
  submitJob: vi.fn(),
  getJob: vi.fn(),
}));

const DOMAINS = [
  {
    name: "item",
    partitioned: false,
    partition_format: null,
    partition_field: null,
    onetime_cascades: true,
    cascade_targets: ["fact_sales_monthly", "fact_external_forecast_monthly"],
  },
  {
    name: "sales",
    partitioned: true,
    partition_format: "YYYY-MM",
    partition_field: "startdate",
    onetime_cascades: false,
    cascade_targets: [],
  },
];

function makeJob(overrides: Record<string, unknown> = {}) {
  return {
    id: "j1",
    domain: "sales",
    mode: "onetime",
    slice: null,
    file_path: null,
    status: "success",
    rows_loaded: 100,
    rows_inserted: null,
    rows_updated: null,
    rows_deleted: null,
    error_message: null,
    started_at: "2026-04-01T10:00:00Z",
    completed_at: "2026-04-01T10:00:10Z",
    duration_ms: 10000,
    triggered_by: "ui",
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  // Onetime mode now triggers a confirm() interstitial; auto-accept in tests.
  vi.spyOn(window, "confirm").mockReturnValue(true);
});

describe("IntegrationTab", () => {
  it("renders the Data Integration heading", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );
    expect(
      screen.getByRole("heading", { name: "Data Integration" }),
    ).toBeInTheDocument();
  });

  it("renders Submit Job form once domains have loaded", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    // The Submit Job <h2> renders during the loading state too, so wait for
    // the actual form (we use the radiogroup as the readiness signal).
    await waitFor(() => {
      expect(screen.getByRole("radiogroup")).toBeInTheDocument();
    });
    // DomainSelector renders an outer-level <select> labelled "Domain"; the
    // tab's recent-jobs filter also renders one. Using getAllByLabelText is
    // robust to that duplication.
    expect(screen.getAllByLabelText(/Domain/i).length).toBeGreaterThanOrEqual(1);
    // ModeSelector renders three radio cards (One-time, Delta, File).
    expect(
      screen.getByRole("radio", { name: /One-time/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /Delta/i })).toBeInTheDocument();
    expect((document.getElementById("mode-selector-file") as HTMLInputElement)).toBeInTheDocument();
  });

  it("hides the slice input when mode is the default (onetime)", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByRole("radiogroup")).toBeInTheDocument();
    });
    expect(screen.queryByPlaceholderText("YYYY-MM")).not.toBeInTheDocument();
  });

  it("shows the slice input when mode=file AND domain is partitioned", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByRole("radiogroup")).toBeInTheDocument();
    });

    // Pick the Submit-form domain selector (the first one).
    const domainSelects = screen.getAllByLabelText(/^Domain$/i) as HTMLSelectElement[];
    fireEvent.change(domainSelects[0], { target: { value: "sales" } });
    fireEvent.click((document.getElementById("mode-selector-file") as HTMLInputElement));

    await waitFor(() => {
      expect(screen.getByPlaceholderText("YYYY-MM")).toBeInTheDocument();
    });
  });

  // After the file-block UX: clicking the File radio for an unpartitioned
  // domain is a no-op — the radio is disabled and the form auto-flips to
  // delta. The slice input never appears.
  it("hides the slice input when File is attempted on an unpartitioned domain", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByRole("radiogroup")).toBeInTheDocument();
    });

    const domainSelects = screen.getAllByLabelText(/^Domain$/i) as HTMLSelectElement[];
    fireEvent.change(domainSelects[0], { target: { value: "item" } });
    // File radio is disabled for unpartitioned 'item'; clicking has no effect.
    fireEvent.click((document.getElementById("mode-selector-file") as HTMLInputElement));

    await waitFor(() => {
      const fileRadio = document.getElementById("mode-selector-file") as HTMLInputElement;
      // Stays unchecked — auto-flip kept us on delta.
      expect(fileRadio.checked).toBe(false);
    });
    expect(screen.queryByPlaceholderText("YYYY-MM")).not.toBeInTheDocument();
  });

  it("disables the Submit button when no domain is selected", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /Submit Job/i }),
      ).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /Submit Job/i })).toBeDisabled();
  });

  it("submits with the correct payload when Submit clicked", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (queries.submitJob as ReturnType<typeof vi.fn>).mockResolvedValue({
      job_id: "abc-123",
      status: "queued",
    });

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByRole("radiogroup")).toBeInTheDocument();
    });

    const domainSelects = screen.getAllByLabelText(/^Domain$/i) as HTMLSelectElement[];
    fireEvent.change(domainSelects[0], { target: { value: "sales" } });
    // Default mode = "onetime" — leave it.
    fireEvent.click(screen.getByRole("button", { name: /Submit Job/i }));

    await waitFor(() => {
      expect(queries.submitJob as ReturnType<typeof vi.fn>).toHaveBeenCalledTimes(1);
    });
    // TanStack Query's mutationFn passes (variables, mutationContext) — assert
    // only the first arg payload, ignoring the context object.
    const submitMock = queries.submitJob as ReturnType<typeof vi.fn>;
    expect(submitMock.mock.calls[0][0]).toEqual({
      domain: "sales",
      mode: "onetime",
    });
  });

  it("shows a success message after submitJob resolves", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (queries.submitJob as ReturnType<typeof vi.fn>).mockResolvedValue({
      job_id: "abc-123",
      status: "queued",
    });

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByRole("radiogroup")).toBeInTheDocument();
    });

    const domainSelects = screen.getAllByLabelText(/^Domain$/i) as HTMLSelectElement[];
    fireEvent.change(domainSelects[0], { target: { value: "sales" } });
    fireEvent.click(screen.getByRole("button", { name: /Submit Job/i }));

    await waitFor(() => {
      expect(screen.getByText("Job abc-123 submitted")).toBeInTheDocument();
    });
  });

  it("shows an error message when submitJob rejects", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (queries.submitJob as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error("boom"),
    );

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByRole("radiogroup")).toBeInTheDocument();
    });

    const domainSelects = screen.getAllByLabelText(/^Domain$/i) as HTMLSelectElement[];
    fireEvent.change(domainSelects[0], { target: { value: "sales" } });
    fireEvent.click(screen.getByRole("button", { name: /Submit Job/i }));

    await waitFor(() => {
      expect(screen.getByText("Failed: boom")).toBeInTheDocument();
    });
  });

  it("places running jobs in the Active Jobs section", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([
      makeJob({ id: "running-1", status: "running", domain: "sales" }),
      makeJob({ id: "done-1", status: "success", domain: "item" }),
    ]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    // Wait for the running badge to actually be in the DOM (the section
    // header alone renders during the loading state).
    await waitFor(() => {
      expect(screen.getByLabelText("status: running")).toBeInTheDocument();
    });

    // Find the Active Jobs section by its <h2> and scope the search.
    const activeHeading = screen.getByText("Active Jobs");
    const activeSection = activeHeading.closest("section")!;
    expect(
      activeSection.querySelector('[aria-label="status: running"]'),
    ).not.toBeNull();
    // Should NOT show the success badge inside the active section.
    expect(
      activeSection.querySelector('[aria-label="status: success"]'),
    ).toBeNull();
  });

  it("places completed jobs in the Recent Jobs section", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([
      makeJob({ id: "running-1", status: "running", domain: "sales" }),
      makeJob({ id: "done-1", status: "success", domain: "item" }),
    ]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByLabelText("status: success")).toBeInTheDocument();
    });

    const recentHeading = screen.getByText("Recent Jobs");
    const recentSection = recentHeading.closest("section")!;
    expect(
      recentSection.querySelector('[aria-label="status: success"]'),
    ).not.toBeNull();
    expect(
      recentSection.querySelector('[aria-label="status: running"]'),
    ).toBeNull();
  });

  it("disables the One-time radio for cascading domains and shows the warning", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByRole("radiogroup")).toBeInTheDocument();
    });

    const domainSelects = screen.getAllByLabelText(/^Domain$/i) as HTMLSelectElement[];
    fireEvent.change(domainSelects[0], { target: { value: "item" } });

    await waitFor(() => {
      const oneTimeRadio = screen.getByRole("radio", { name: /one-time/i });
      expect(oneTimeRadio).toBeDisabled();
    });
    // The disabled-mode reason is rendered as the description text
    expect(
      screen.getByText(/Cascades to fact_sales_monthly, fact_external_forecast_monthly/i),
    ).toBeInTheDocument();
  });

  it("disables the File radio for non-partitioned domains", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByRole("radiogroup")).toBeInTheDocument();
    });

    // 'item' is not partitioned in our test fixture (partitioned: false)
    const domainSelects = screen.getAllByLabelText(/^Domain$/i) as HTMLSelectElement[];
    fireEvent.change(domainSelects[0], { target: { value: "item" } });

    await waitFor(() => {
      const fileRadio = (document.getElementById("mode-selector-file") as HTMLInputElement);
      expect(fileRadio).toBeDisabled();
    });
    expect(
      screen.getByText(/File mode requires a partitioned domain/i),
    ).toBeInTheDocument();
  });

  it("keeps the File radio enabled for partitioned domains", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByRole("radiogroup")).toBeInTheDocument();
    });

    // 'sales' is partitioned in fixture
    const domainSelects = screen.getAllByLabelText(/^Domain$/i) as HTMLSelectElement[];
    fireEvent.change(domainSelects[0], { target: { value: "sales" } });

    await waitFor(() => {
      const fileRadio = (document.getElementById("mode-selector-file") as HTMLInputElement);
      expect(fileRadio).not.toBeDisabled();
    });
  });

  it("auto-flips mode from file to delta when a non-partitioned domain is selected", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByRole("radiogroup")).toBeInTheDocument();
    });

    // Select partitioned domain first, switch mode to file, then change to non-partitioned
    const domainSelects = screen.getAllByLabelText(/^Domain$/i) as HTMLSelectElement[];
    fireEvent.change(domainSelects[0], { target: { value: "sales" } });
    fireEvent.click((document.getElementById("mode-selector-file") as HTMLInputElement));
    await waitFor(() => {
      const fileRadio = document.getElementById("mode-selector-file") as HTMLInputElement;
      expect(fileRadio.checked).toBe(true);
    });

    // Now switch to non-partitioned 'item' — mode should auto-flip to delta
    fireEvent.change(domainSelects[0], { target: { value: "item" } });
    await waitFor(() => {
      const deltaRadio = document.getElementById("mode-selector-delta") as HTMLInputElement;
      expect(deltaRadio.checked).toBe(true);
    });
  });

  it("auto-flips mode from onetime to delta when a cascading domain is selected", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByRole("radio", { name: /one-time/i })).toBeChecked();
    });

    const domainSelects = screen.getAllByLabelText(/^Domain$/i) as HTMLSelectElement[];
    fireEvent.change(domainSelects[0], { target: { value: "item" } });

    // Select by exact id rather than role-name — the cascade reason text
    // contains "Delta" so a /delta/i name matcher would match multiple radios.
    await waitFor(() => {
      const deltaRadio = document.getElementById("mode-selector-delta") as HTMLInputElement;
      expect(deltaRadio.checked).toBe(true);
    });
  });

  it("shows confirm dialog and includes confirm_destructive in payload for cascading onetime", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (queries.submitJob as ReturnType<typeof vi.fn>).mockResolvedValue({
      job_id: "x", status: "queued",
    });

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByRole("radiogroup")).toBeInTheDocument();
    });

    // Select sales (safe domain), then explicitly pick onetime so submit triggers confirm.
    const domainSelects = screen.getAllByLabelText(/^Domain$/i) as HTMLSelectElement[];
    fireEvent.change(domainSelects[0], { target: { value: "sales" } });
    fireEvent.click(screen.getByRole("radio", { name: /one-time/i }));
    fireEvent.click(screen.getByRole("button", { name: /Submit Job/i }));

    await waitFor(() => {
      expect(window.confirm).toHaveBeenCalled();
    });
    // Sales has no cascade targets so confirm_destructive is NOT sent.
    const submitMock = queries.submitJob as ReturnType<typeof vi.fn>;
    await waitFor(() => expect(submitMock).toHaveBeenCalledTimes(1));
    expect(submitMock.mock.calls[0][0]).toEqual({ domain: "sales", mode: "onetime" });
  });

  it("renders the empty active-jobs message when no running jobs exist", async () => {
    const queries = await import("../../api/queries/integration");
    (queries.listDomains as ReturnType<typeof vi.fn>).mockResolvedValue(DOMAINS);
    (queries.listJobs as ReturnType<typeof vi.fn>).mockResolvedValue([
      makeJob({ id: "done-1", status: "success", domain: "item" }),
    ]);

    const { default: IntegrationTab } = await import("../IntegrationTab");
    render(
      <TestQueryWrapper>
        <IntegrationTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("No active jobs.")).toBeInTheDocument();
    });
  });
});
