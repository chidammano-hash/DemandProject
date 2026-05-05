import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ChainProgress } from "../ChainProgress";
import { TestQueryWrapper } from "../../../tabs/__tests__/test-utils";
import {
  getChain,
  type ChainDetail,
  type ChainJob,
} from "../../../api/queries/integration_chain";

vi.mock("../../../api/queries/integration_chain", () => ({
  chainKeys: { chain: (id: string) => ["integration_chain", "chain", id] },
  getChain: vi.fn(),
}));

const mockedGetChain = vi.mocked(getChain);

function buildJob(overrides: Partial<ChainJob> = {}): ChainJob {
  const base: ChainJob = {
    step: 1,
    job_id: "job-1",
    domain: "sales",
    mode: "delta",
    slice: null,
    status: "success",
    rows_loaded: 0,
    rows_inserted: null,
    rows_updated: null,
    rows_deleted: null,
    error_message: null,
    started_at: "2026-04-01T10:00:00Z",
    completed_at: "2026-04-01T10:00:15Z",
    duration_ms: 15000,
  };
  return { ...base, ...overrides };
}

function buildChain(overrides: Partial<ChainDetail> = {}): ChainDetail {
  const base: ChainDetail = {
    id: "abcdef0123456789",
    status: "running",
    total_steps: 3,
    completed_steps: 1,
    failed_step: null,
    started_at: "2026-04-01T10:00:00Z",
    completed_at: null,
    duration_ms: null,
    triggered_by: "ui",
    jobs: [],
  };
  return { ...base, ...overrides };
}

function renderChain(): void {
  render(
    <TestQueryWrapper>
      <ChainProgress chainId="abcdef0123456789" />
    </TestQueryWrapper>,
  );
}

describe("ChainProgress", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state when query has no data yet", () => {
    // never resolves -> stays in loading state
    mockedGetChain.mockReturnValue(new Promise<ChainDetail>(() => {}));
    renderChain();
    expect(screen.getByText("Loading chain…")).toBeInTheDocument();
  });

  it("renders chain id (first 8 chars) and a status badge", async () => {
    mockedGetChain.mockResolvedValue(buildChain({ status: "running" }));
    renderChain();
    expect(await screen.findByText("Chain abcdef01…")).toBeInTheDocument();
    // Header status badge for chain.status (running -> 'running')
    expect(screen.getByLabelText("status: running")).toBeInTheDocument();
  });

  it("renders progress bar with correct percentage and step text", async () => {
    mockedGetChain.mockResolvedValue(
      buildChain({ total_steps: 4, completed_steps: 2 }),
    );
    renderChain();
    const bar = await screen.findByRole("progressbar");
    expect(bar.getAttribute("aria-valuenow")).toBe("50");
    expect(screen.getByText("2 of 4 steps complete")).toBeInTheDocument();
  });

  it("renders one entry per job sorted by step ascending", async () => {
    mockedGetChain.mockResolvedValue(
      buildChain({
        total_steps: 3,
        completed_steps: 3,
        status: "success",
        jobs: [
          buildJob({ step: 3, job_id: "j3", domain: "inventory" }),
          buildJob({ step: 1, job_id: "j1", domain: "item" }),
          buildJob({ step: 2, job_id: "j2", domain: "sales" }),
        ],
      }),
    );
    renderChain();
    const step1 = await screen.findByTestId("chain-step-1");
    const step2 = screen.getByTestId("chain-step-2");
    const step3 = screen.getByTestId("chain-step-3");
    // DOM order matches step order
    const order = [step1, step2, step3];
    for (let i = 1; i < order.length; i++) {
      expect(
        order[i - 1].compareDocumentPosition(order[i]) &
          Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
    }
    expect(step1).toHaveTextContent("item");
    expect(step2).toHaveTextContent("sales");
    expect(step3).toHaveTextContent("inventory");
  });

  it("shows 'X new · Y updated · Z deleted' for completed delta jobs with non-null counts", async () => {
    mockedGetChain.mockResolvedValue(
      buildChain({
        status: "success",
        completed_steps: 1,
        total_steps: 1,
        jobs: [
          buildJob({
            step: 1,
            status: "success",
            rows_inserted: 10,
            rows_updated: 20,
            rows_deleted: 3,
          }),
        ],
      }),
    );
    renderChain();
    expect(await screen.findByText(/10 new/)).toBeInTheDocument();
    expect(screen.getByText(/20 updated/)).toBeInTheDocument();
    expect(screen.getByText(/3 deleted/)).toBeInTheDocument();
  });

  it("shows error_message in red for failed jobs", async () => {
    mockedGetChain.mockResolvedValue(
      buildChain({
        status: "failed",
        completed_steps: 1,
        total_steps: 2,
        failed_step: 2,
        jobs: [
          buildJob({ step: 1, status: "success" }),
          buildJob({
            step: 2,
            job_id: "j2",
            status: "failed",
            error_message: "CSV parse error",
          }),
        ],
      }),
    );
    renderChain();
    const err = await screen.findByText("CSV parse error");
    expect(err).toBeInTheDocument();
    expect(err.className).toMatch(/red/);
  });

  it("shows 'Running…' indicator for in-flight jobs", async () => {
    mockedGetChain.mockResolvedValue(
      buildChain({
        jobs: [buildJob({ step: 1, status: "running" })],
      }),
    );
    renderChain();
    expect(await screen.findByText("Running…")).toBeInTheDocument();
  });

  it("shows '(cancelled)' for queued jobs when chain status is halted", async () => {
    mockedGetChain.mockResolvedValue(
      buildChain({
        status: "halted",
        completed_steps: 1,
        total_steps: 3,
        jobs: [
          buildJob({ step: 1, status: "success" }),
          buildJob({ step: 2, job_id: "j2", status: "queued" }),
          buildJob({ step: 3, job_id: "j3", status: "queued" }),
        ],
      }),
    );
    renderChain();
    const cancelled = await screen.findAllByText("(cancelled)");
    expect(cancelled.length).toBe(2);
  });

  it("renders formatted started_at and duration in the footer", async () => {
    mockedGetChain.mockResolvedValue(
      buildChain({
        status: "success",
        completed_steps: 1,
        total_steps: 1,
        duration_ms: 125000,
      }),
    );
    renderChain();
    expect(await screen.findByText(/Started/)).toBeInTheDocument();
    expect(screen.getByText(/Duration 2m 5s/)).toBeInTheDocument();
  });

  it("renders '—' for null duration in the footer", async () => {
    mockedGetChain.mockResolvedValue(buildChain({ duration_ms: null }));
    renderChain();
    expect(await screen.findByText(/Duration —/)).toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", async () => {
    mockedGetChain.mockResolvedValue(buildChain());
    const onClose = vi.fn();
    render(
      <TestQueryWrapper>
        <ChainProgress chainId="abcdef0123456789" onClose={onClose} />
      </TestQueryWrapper>,
    );
    const btn = await screen.findByLabelText("close chain progress");
    fireEvent.click(btn);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not render a close button when onClose is undefined", async () => {
    mockedGetChain.mockResolvedValue(buildChain());
    renderChain();
    // wait for header before asserting absence
    await screen.findByText("Chain abcdef01…");
    expect(screen.queryByLabelText("close chain progress")).toBeNull();
  });

  it("uses the 'failed' badge palette when chain.status is 'halted'", async () => {
    mockedGetChain.mockResolvedValue(buildChain({ status: "halted" }));
    renderChain();
    // halted -> mapped to 'failed' for the header badge
    expect(await screen.findByLabelText("status: failed")).toBeInTheDocument();
  });
});
