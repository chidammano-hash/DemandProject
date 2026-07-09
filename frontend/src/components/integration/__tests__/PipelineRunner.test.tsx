import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import { PipelineRunner } from "../PipelineRunner";
import { TestQueryWrapper } from "../../../tabs/__tests__/test-utils";
import { runPipeline } from "../../../api/queries/integration";
import { fetchJobDetail } from "../../../api/queries/jobs";

vi.mock("../../../api/queries/integration", () => ({
  runPipeline: vi.fn(),
}));
vi.mock("../../../api/queries/jobs", () => ({
  fetchJobDetail: vi.fn(),
}));

const mockedRun = vi.mocked(runPipeline);
const mockedJob = vi.mocked(fetchJobDetail);

function renderRunner() {
  return render(
    <TestQueryWrapper>
      <PipelineRunner />
    </TestQueryWrapper>,
  );
}

describe("PipelineRunner", () => {
  beforeEach(() => {
    mockedRun.mockReset();
    mockedJob.mockReset();
    mockedRun.mockResolvedValue({ job_id: "etl-1", mode: "refresh", status: "queued" });
    mockedJob.mockResolvedValue({
      job_id: "etl-1",
      job_type: "etl_pipeline",
      job_label: "ETL pipeline (refresh)",
      status: "running",
      params: {},
      result: null,
      error: null,
      submitted_at: "2026-06-14T00:00:00Z",
      started_at: "2026-06-14T00:00:01Z",
      completed_at: null,
      progress_pct: 42,
      progress_msg: "loading sales",
      pid: null,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("submits an incremental refresh by default", async () => {
    renderRunner();
    fireEvent.click(screen.getByRole("button", { name: /run pipeline/i }));
    await waitFor(() => expect(mockedRun).toHaveBeenCalledTimes(1));
    expect(mockedRun.mock.calls[0][0]).toEqual({ mode: "refresh", parallel: false });
  });

  it("requires confirmation for a destructive full reload", async () => {
    renderRunner();
    fireEvent.change(screen.getByLabelText("Pipeline mode"), {
      target: { value: "full" },
    });
    fireEvent.click(screen.getByRole("button", { name: /run pipeline/i }));
    expect(
      screen.getByRole("dialog", { name: /run full pipeline reload/i }),
    ).toBeInTheDocument();
    expect(mockedRun).not.toHaveBeenCalled();
  });

  it("submits a full reload once confirmed, passing parallel", async () => {
    renderRunner();
    fireEvent.change(screen.getByLabelText("Pipeline mode"), {
      target: { value: "full" },
    });
    fireEvent.click(screen.getByLabelText("Parallel"));
    fireEvent.click(screen.getByRole("button", { name: /run pipeline/i }));
    fireEvent.click(screen.getByRole("button", { name: /run full reload/i }));
    await waitFor(() => expect(mockedRun).toHaveBeenCalledTimes(1));
    expect(mockedRun.mock.calls[0][0]).toEqual({ mode: "full", parallel: true });
  });

  it("shows live job status after submitting", async () => {
    renderRunner();
    fireEvent.click(screen.getByRole("button", { name: /run pipeline/i }));
    const status = await screen.findByTestId("pipeline-status");
    expect(status).toHaveTextContent("running");
    expect(status).toHaveTextContent("42%");
    expect(status).toHaveTextContent("loading sales");
  });
});
