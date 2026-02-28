import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";
import { JobNotificationProvider } from "@/context/JobNotificationContext";

vi.mock("@/api/queries", () => ({
  queryKeys: {
    jobTypes: () => ["job-types"],
    jobs: (p: Record<string, unknown>) => ["jobs", p],
    jobDetail: (id: string) => ["job-detail", id],
    activeJobs: () => ["active-jobs"],
    jobStats: () => ["job-stats"],
    jobSchedules: () => ["job-schedules"],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchJobTypes: vi.fn().mockResolvedValue({
    types: [
      { type_id: "cluster_scenario", label: "Clustering What-If", description: "Run trial clustering", group: "clustering", params_schema: {} },
      { type_id: "backtest_lgbm", label: "LGBM Backtest", description: "Run LightGBM backtest", group: "backtest", params_schema: { cluster_strategy: "global" } },
    ],
  }),
  fetchJobs: vi.fn().mockResolvedValue({ jobs: [], total: 0, limit: 50, offset: 0 }),
  fetchActiveJobs: vi.fn().mockResolvedValue({ jobs: [] }),
  fetchJobStats: vi.fn().mockResolvedValue({
    total: 10, active: 1, completed: 8, failed: 1, cancelled: 0,
    avg_duration_seconds: 120.5,
    last_24h: { submitted: 3, completed: 2, failed: 1 },
  }),
  fetchJobSchedules: vi.fn().mockResolvedValue({ schedules: [] }),
  fetchJobDetail: vi.fn(),
  submitJob: vi.fn(),
  cancelJob: vi.fn(),
  deleteJob: vi.fn(),
  createSchedule: vi.fn(),
  deleteSchedule: vi.fn(),
}));

// Mock lucide-react to avoid rendering issues
vi.mock("lucide-react", () => {
  const Stub = (props: Record<string, unknown>) => <span data-testid={props["data-testid"] as string || "icon"} />;
  return {
    PlayCircle: Stub, Square: Stub, CheckCircle2: Stub, XCircle: Stub,
    Loader2: Stub, Trash2: Stub, ChevronDown: Stub, ChevronRight: Stub,
    Clock: Stub, AlertCircle: Stub, BarChart3: Stub, Zap: Stub,
    Trophy: Stub, Activity: Stub, Network: Stub, TrendingUp: Stub,
    Calendar: Stub, Timer: Stub, Repeat: Stub, X: Stub,
  };
});

const JobsTab = (await import("@/tabs/JobsTab")).default;

describe("JobsTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <JobNotificationProvider>
          <JobsTab theme="light" />
        </JobNotificationProvider>
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Job Scheduler")).toBeDefined();
    });
  });

  it("renders available job type cards", async () => {
    render(
      <TestQueryWrapper>
        <JobNotificationProvider>
          <JobsTab theme="light" />
        </JobNotificationProvider>
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      // Job type names appear in both cards and history filter select
      expect(screen.getAllByText("Clustering What-If").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("LGBM Backtest").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("shows job group headers", async () => {
    render(
      <TestQueryWrapper>
        <JobNotificationProvider>
          <JobsTab theme="light" />
        </JobNotificationProvider>
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Clustering")).toBeDefined();
      expect(screen.getByText("Backtesting")).toBeDefined();
    });
  });

  it("shows empty history message", async () => {
    render(
      <TestQueryWrapper>
        <JobNotificationProvider>
          <JobsTab theme="light" />
        </JobNotificationProvider>
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText(/No jobs in history/)).toBeDefined();
    });
  });

  it("renders header description", async () => {
    render(
      <TestQueryWrapper>
        <JobNotificationProvider>
          <JobsTab theme="light" />
        </JobNotificationProvider>
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText(/Automate, schedule, and monitor/)).toBeDefined();
    });
  });

  it("shows KPI cards with stats", async () => {
    render(
      <TestQueryWrapper>
        <JobNotificationProvider>
          <JobsTab theme="light" />
        </JobNotificationProvider>
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("Total Jobs")).toBeDefined();
      expect(screen.getByText("Active Now")).toBeDefined();
      expect(screen.getByText("Success Rate")).toBeDefined();
      expect(screen.getByText("Avg Duration")).toBeDefined();
    });
  });

  it("shows APScheduler engine badge", async () => {
    render(
      <TestQueryWrapper>
        <JobNotificationProvider>
          <JobsTab theme="light" />
        </JobNotificationProvider>
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByText("APScheduler Engine")).toBeDefined();
    });
  });

  it("renders View Results button for completed cluster_scenario jobs", async () => {
    const { fetchJobs } = await import("@/api/queries");
    (fetchJobs as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      jobs: [{
        job_id: "j_cluster_done",
        job_type: "cluster_scenario",
        job_label: "What-If Scenario A",
        status: "completed",
        submitted_at: "2026-02-27T10:00:00Z",
        completed_at: "2026-02-27T10:01:00Z",
        progress_pct: 100,
        progress_msg: "Done",
        result: { scenario_id: "sc_abc", optimal_k: 5 },
      }],
      total: 1,
      limit: 50,
      offset: 0,
    });

    const mockNavigate = vi.fn();
    render(
      <TestQueryWrapper>
        <JobNotificationProvider>
          <JobsTab theme="light" onNavigateToScenario={mockNavigate} />
        </JobNotificationProvider>
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      const btn = screen.getByTitle("View Results in Clusters Tab");
      expect(btn).toBeDefined();
    });
  });

  it("does NOT render View Results button for non-cluster or non-completed jobs", async () => {
    const { fetchJobs } = await import("@/api/queries");
    (fetchJobs as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      jobs: [
        {
          job_id: "j_lgbm",
          job_type: "backtest_lgbm",
          job_label: "LGBM Run",
          status: "completed",
          submitted_at: "2026-02-27T10:00:00Z",
          completed_at: "2026-02-27T10:02:00Z",
          progress_pct: 100,
          progress_msg: "Done",
          result: {},
        },
        {
          job_id: "j_cluster_running",
          job_type: "cluster_scenario",
          job_label: "What-If Scenario B",
          status: "running",
          submitted_at: "2026-02-27T10:00:00Z",
          progress_pct: 50,
          progress_msg: "Running",
        },
      ],
      total: 2,
      limit: 50,
      offset: 0,
    });

    render(
      <TestQueryWrapper>
        <JobNotificationProvider>
          <JobsTab theme="light" onNavigateToScenario={vi.fn()} />
        </JobNotificationProvider>
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.queryByTitle("View Results in Clusters Tab")).toBeNull();
    });
  });
});
