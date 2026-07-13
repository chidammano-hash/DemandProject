import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ActiveJobsPanel } from "@/tabs/jobs/ActiveJobsPanel";
import type { Job } from "@/types/jobs";

import { TestQueryWrapper } from "./test-utils";

vi.mock("lucide-react", () => {
  const Stub = () => <span />;
  return {
    Activity: Stub,
    BarChart2: Stub,
    Boxes: Stub,
    CheckCircle2: Stub,
    Clock: Stub,
    Loader2: Stub,
    Network: Stub,
    Package: Stub,
    ScrollText: Stub,
    Sparkles: Stub,
    Square: Stub,
    Timer: Stub,
    TrendingUp: Stub,
    Trophy: Stub,
    XCircle: Stub,
  };
});

function activePipelineJob(overrides: Partial<Job> = {}): Job {
  return {
    job_id: "job-model-refresh",
    job_type: "backtest_lgbm",
    job_label: "[model-refresh 2/7] LightGBM",
    status: "running",
    params: {
      __pipeline_label: "model-refresh",
      __pipeline_step: 2,
      __pipeline_total_steps: 7,
    },
    result: null,
    error: null,
    submitted_at: "2026-07-12T10:00:00Z",
    started_at: "2026-07-12T10:00:01Z",
    completed_at: null,
    progress_pct: 20,
    progress_msg: "Backtesting LightGBM",
    pid: 42,
    ...overrides,
  };
}

describe("ActiveJobsPanel cancellation", () => {
  it("makes clear that cancelling a pipeline stops its remaining steps", () => {
    const onCancel = vi.fn();
    render(
      <TestQueryWrapper>
        <ActiveJobsPanel activeJobs={[activePipelineJob()]} onCancel={onCancel} />
      </TestQueryWrapper>
    );

    const cancel = screen.getByRole("button", { name: "Cancel workflow" });
    expect(cancel.getAttribute("title")).toMatch(/remaining pipeline steps will not start/i);
    fireEvent.click(cancel);
    fireEvent.click(screen.getByRole("button", { name: "Confirm cancel" }));
    expect(onCancel).toHaveBeenCalledWith("job-model-refresh");
  });

  it("shows a cancellation request as pending instead of already cancelled", () => {
    render(
      <TestQueryWrapper>
        <ActiveJobsPanel
          activeJobs={[activePipelineJob()]}
          onCancel={vi.fn()}
          cancellingJobId="job-model-refresh"
        />
      </TestQueryWrapper>
    );

    expect(screen.getByText("Cancellation requested; waiting for the worker to stop…")).toBeDefined();
    expect(
      (screen.getByRole("button", { name: "Cancelling workflow" }) as HTMLButtonElement).disabled
    ).toBe(true);
  });
});
