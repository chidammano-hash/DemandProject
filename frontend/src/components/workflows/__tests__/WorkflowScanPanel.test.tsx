import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";
import { planOperationalWorkflows, runNamedPipeline } from "@/api/queries/jobs";
import { WorkflowScanPanel } from "../WorkflowScanPanel";

vi.mock("@/api/queries/jobs", async () => {
  const actual = await vi.importActual<typeof import("@/api/queries/jobs")>("@/api/queries/jobs");
  return {
    ...actual,
    planOperationalWorkflows: vi.fn(),
    runNamedPipeline: vi.fn(),
  };
});

const mockedPlan = vi.mocked(planOperationalWorkflows);
const mockedRun = vi.mocked(runNamedPipeline);

describe("WorkflowScanPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedPlan.mockResolvedValue({
      plan_id: "plan-1",
      provider: "codex",
      model: "gpt-5.5",
      ai_verified: true,
      status: "planned",
      confidence: 0.94,
      explanation: "Refresh inputs before rebuilding downstream workflows.",
      risk_flags: [],
      questions: [],
      recommendations: [
        {
          pipeline_name: "data-refresh",
          title: "Data Refresh",
          description: "Refresh changed inputs and downstream views.",
          priority: "critical",
          reason: "Sales changed.",
          blockers: [],
          steps: [{ position: 1, job_type: "etl_pipeline", params: {}, label: null }],
        },
      ],
      evidence: {
        planning_month: "2026-07-01",
        changed_domains: ["sales"],
        active_job_count: 0,
        clustered_skus: 100,
        latest_feature_refresh: "2026-06-30T10:00:00Z",
        latest_cluster_promotion: "2026-07-01T10:00:00Z",
        stale_tuning_profiles: 0,
        active_production_month: "2026-07-01",
        planning_month_production_rows: 100,
        planning_month_roster_models: 4,
        planning_month_snapshot_rows: 100,
      },
      scanned_at: "2026-07-10T10:00:00Z",
    });
    mockedRun.mockResolvedValue({
      pipeline_id: "pipeline-1",
      name: "data-refresh",
      status: "running",
      steps: 1,
    });
  });

  it("scans, explains, and runs the first grounded recommendation", async () => {
    render(
      <TestQueryWrapper>
        <WorkflowScanPanel />
      </TestQueryWrapper>
    );

    fireEvent.click(screen.getByRole("button", { name: "Analyze workflows" }));

    expect(await screen.findByText("Data Refresh")).toBeInTheDocument();
    expect(screen.getByText("AI verified")).toBeInTheDocument();
    expect(screen.getByText("Sales changed.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Run Data Refresh" }));

    await waitFor(() => expect(mockedRun.mock.calls[0]?.[0]).toBe("data-refresh"));
    expect(await screen.findByText(/pipeline-1/)).toBeInTheDocument();
  });
});
