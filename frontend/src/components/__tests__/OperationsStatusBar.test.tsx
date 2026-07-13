import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { OperationsStatusBar } from "../OperationsStatusBar";
import { TestQueryWrapper } from "../../tabs/__tests__/test-utils";
import { JobNotificationProvider } from "../../context/JobNotificationContext";
import { fetchActiveJobs } from "../../api/queries/jobs";
import {
  fetchPipelineReadiness,
  fetchPlanningDate,
} from "../../api/queries/dashboard";

vi.mock("../../context/AuthContext", () => ({
  useAuth: () => ({ user: null, loading: false, login: vi.fn(), logout: vi.fn() }),
}));

vi.mock("../../api/queries/dashboard", () => ({
  fetchPlanningDate: vi.fn(),
  fetchPipelineReadiness: vi.fn(),
  pipelineReadinessKeys: {
    readiness: ["dashboard", "pipeline-readiness"],
  },
}));

vi.mock("../../api/queries/jobs", () => ({
  fetchActiveJobs: vi.fn(),
}));

const mockedPlanningDate = vi.mocked(fetchPlanningDate);
const mockedReadiness = vi.mocked(fetchPipelineReadiness);
const mockedActiveJobs = vi.mocked(fetchActiveJobs);

function renderStatusBar(onNavigate = vi.fn()) {
  render(
    <TestQueryWrapper>
      <JobNotificationProvider>
        <OperationsStatusBar activeTab="commandCenter" onNavigate={onNavigate} />
      </JobNotificationProvider>
    </TestQueryWrapper>,
  );
  return onNavigate;
}

describe("OperationsStatusBar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedPlanningDate.mockResolvedValue({
      planning_date: "2026-07-08",
      system_date: "2026-07-08",
      is_frozen: false,
      days_behind: 0,
    });
    mockedReadiness.mockResolvedValue({ ready: true, checks: [] });
    mockedActiveJobs.mockResolvedValue({ jobs: [] });
  });

  it("shows operational context for the current tab", async () => {
    renderStatusBar();

    expect(await screen.findByText("Command Center")).toBeInTheDocument();
    expect(await screen.findByText(/Jul .*2026/)).toBeInTheDocument();
    expect(screen.getByText("No active jobs")).toBeInTheDocument();
    expect(screen.getByText("ML inputs current")).toBeInTheDocument();
  });

  it("opens Workflows when the active job pill is clicked", async () => {
    mockedActiveJobs.mockResolvedValue({
      jobs: [
        {
          job_id: "job-1",
          job_type: "forecast",
          job_label: "Forecast run",
          status: "running",
          params: {},
          result: null,
          error: null,
          submitted_at: "2026-07-08T10:00:00Z",
          started_at: "2026-07-08T10:00:01Z",
          completed_at: null,
          progress_pct: 35,
          progress_msg: "training",
          pid: null,
        },
      ],
    });
    const onNavigate = renderStatusBar();

    fireEvent.click(await screen.findByRole("button", { name: /1 active job/i }));

    expect(onNavigate).toHaveBeenCalledWith("integration");
  });

  it("opens the readiness remediation target for stale stages", async () => {
    mockedReadiness.mockResolvedValue({
      ready: false,
      checks: [
        {
          stage: "clustering",
          status: "stale",
          severity: "high",
          title: "Clusters are stale",
          detail: "SKU features changed after the last cluster run.",
          action: { kind: "navigate", target: "clusters", label: "Open clustering" },
        },
      ],
    });
    const onNavigate = renderStatusBar();

    fireEvent.click(await screen.findByRole("button", { name: /1 stale stage/i }));

    expect(onNavigate).toHaveBeenCalledWith("clusters");
  });
});
