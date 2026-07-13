import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { TestQueryWrapper } from "../../__tests__/test-utils";
import { PeriodRollPanel } from "../PeriodRollPanel";

const runNamedPipeline = vi.fn();

vi.mock("@/api/queries/jobs", () => ({
  jobKeys: {
    periodRoll: (pipelineId: string) => ["jobs", "period-roll", pipelineId],
  },
  runNamedPipeline: (...args: unknown[]) => runNamedPipeline(...args),
  fetchJobs: vi.fn().mockResolvedValue({ jobs: [], total: 0, limit: 100, offset: 0 }),
}));

describe("PeriodRollPanel", () => {
  it("explains the separate beginning-of-month workflow and its four operations", () => {
    render(
      <TestQueryWrapper>
        <PeriodRollPanel />
      </TestQueryWrapper>
    );

    expect(screen.getByText("Period Roll · Score Prior + Archive Current")).toBeInTheDocument();
    expect(screen.getByText("Ready")).toBeInTheDocument();
    expect(screen.getByText("Calculate Snapshot KPIs")).toBeInTheDocument();
    expect(screen.getByText("Prepare Forecast Snapshot Contenders")).toBeInTheDocument();
    expect(screen.getByText("Archive Forecast Snapshot")).toBeInTheDocument();
    expect(screen.getByText("Clean Forecast Staging")).toBeInTheDocument();
  });

  it("launches the durable period-roll pipeline", async () => {
    runNamedPipeline.mockResolvedValue({
      pipeline_id: "period-roll-1",
      name: "period-roll",
      status: "running",
      steps: 4,
    });
    render(
      <TestQueryWrapper>
        <PeriodRollPanel />
      </TestQueryWrapper>
    );

    fireEvent.click(screen.getByRole("button", { name: "Run Period Roll" }));
    await waitFor(() => expect(runNamedPipeline).toHaveBeenCalledWith("period-roll"));
  });
});
