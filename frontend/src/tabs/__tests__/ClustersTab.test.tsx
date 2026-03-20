import { describe, it, expect, vi } from "vitest";
import { render, waitFor, screen, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";
import { ScenarioNotificationProvider } from "@/context/ScenarioNotificationContext";

vi.mock("@/api/queries", () => ({
  queryKeys: {
    dfuClusters: (s: string) => ["dfu-clusters", s],
    clusterProfiles: () => ["cluster-profiles"],
    clusteringDefaults: () => ["clustering-defaults"],
    clusteringScenario: (id: string) => ["clustering-scenario", id],
    scenarioEstimate: (p: Record<string, unknown>) => ["scenario-estimate", p],
    scenarioStatus: (id: string) => ["scenario-status", id],
    scenarioHistory: () => ["scenario-history"],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchDfuClusters: vi.fn().mockResolvedValue({
    domain: "dfu",
    total_assigned: 10,
    clusters: [
      { cluster_id: "1", label: "high_volume_steady", count: 5, pct_of_total: 50, avg_demand: 1000, cv_demand: 0.3 },
    ],
  }),
  fetchClusterProfiles: vi.fn().mockResolvedValue({
    profiles: [],
    metadata: { optimal_k: 5, silhouette_score: 0.45, inertia: 12345 },
  }),
  fetchClusteringDefaults: vi.fn().mockResolvedValue({
    feature_params: { time_window_months: 24, min_months_history: 1 },
    model_params: { k_range: [3, 12], min_cluster_size_pct: 2.0, use_pca: false, pca_components: null, all_features: false },
    label_params: { volume_high: 0.75, volume_low: 0.25, cv_steady: 0.3, cv_volatile: 0.8, seasonality_threshold: 0.5, zero_demand_threshold: 0.2 },
  }),
  fetchScenarioEstimate: vi.fn().mockResolvedValue({
    estimated_seconds: 45,
    dfu_count: 1200,
    k_range: 10,
  }),
  fetchScenarioStatus: vi.fn().mockResolvedValue({
    scenario_id: "sc_test",
    status: "running",
    elapsed_seconds: 5,
  }),
  fetchScenarioHistory: vi.fn().mockResolvedValue([]),
  fetchJobDetail: vi.fn(),
  runClusteringScenario: vi.fn(),
  promoteScenario: vi.fn(),
  submitJob: vi.fn().mockResolvedValue({ job_id: "job_pipeline_123", status: "queued" }),
}));

vi.mock("@/hooks/useUrlState", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/hooks/useUrlState")>();
  return {
    ...actual,
    getScenarioJobParam: vi.fn().mockReturnValue(null),
    setScenarioJobParam: vi.fn(),
  };
});

// Mock recharts to avoid rendering issues in test env
vi.mock("recharts", () => ({
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => null,
  Cell: () => null,
  ReferenceLine: () => null,
  PieChart: ({ children }: { children: React.ReactNode }) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => null,
  RadarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="radar-chart">{children}</div>,
  Radar: () => null,
  PolarGrid: () => null,
  PolarAngleAxis: () => null,
  PolarRadiusAxis: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

const ClustersTab = (await import("@/tabs/ClustersTab")).default;

function renderTab() {
  return render(
    <TestQueryWrapper>
      <ScenarioNotificationProvider>
        <ClustersTab domain="dfu" onDomainChange={vi.fn()} />
      </ScenarioNotificationProvider>
    </TestQueryWrapper>
  );
}

describe("ClustersTab", () => {
  it("renders without crashing", async () => {
    renderTab();
    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });

  it("renders cluster summary table", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("high_volume_steady")).toBeDefined();
    });
  });

  it("renders What-If Scenarios section", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("What-If Scenarios")).toBeDefined();
    });
  });

  it("auto-loads scenario result from scenario_job URL param", async () => {
    const { getScenarioJobParam } = await import("@/hooks/useUrlState");
    const { fetchJobDetail } = await import("@/api/queries");
    (getScenarioJobParam as ReturnType<typeof vi.fn>).mockReturnValueOnce("job_test123");
    (fetchJobDetail as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      job_id: "job_test123",
      job_type: "cluster_scenario",
      job_label: "What-If Scenario A",
      status: "completed",
      result: {
        scenario_id: "sc_fromurl",
        status: "completed",
        runtime_seconds: 42,
        result: {
          optimal_k: 5,
          silhouette_score: 0.45,
          inertia: 20000,
          total_dfus: 500,
          profiles: [
            { label: "high_volume_steady", count: 200, pct_of_total: 40, mean_demand: 1000, cv_demand: 0.2, seasonality_strength: 0.1, trend_slope: 0.02, growth_rate: 0.05, zero_demand_pct: 0.0 },
          ],
          k_selection_results: {
            k_values: [3, 4, 5],
            inertias: [50000, 30000, 20000],
            silhouette_scores: [0.35, 0.40, 0.45],
          },
          feature_importance: [
            { feature: "mean_demand", importance: 0.8 },
          ],
        },
      },
    });

    renderTab();
    await waitFor(() => {
      expect(fetchJobDetail).toHaveBeenCalledWith("job_test123");
    });
  });

  it("renders Re-run Clustering Pipeline button", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Re-run Clustering Pipeline")).toBeDefined();
    });
  });

  it("shows confirmation dialog when Run Pipeline is clicked", async () => {
    renderTab();
    const btn = await screen.findByText("Re-run Clustering Pipeline");
    fireEvent.click(btn);
    await waitFor(() => {
      expect(screen.getByText("Run Production Clustering Pipeline?")).toBeDefined();
      expect(screen.getByText(/This will overwrite existing cluster assignments/)).toBeDefined();
    });
  });

  it("submits pipeline job on confirm and shows job ID", async () => {
    const { submitJob } = await import("@/api/queries");
    renderTab();
    const btn = await screen.findByText("Re-run Clustering Pipeline");
    fireEvent.click(btn);
    const confirmBtn = await screen.findByText("Run Pipeline");
    fireEvent.click(confirmBtn);
    await waitFor(() => {
      expect(submitJob).toHaveBeenCalledWith("cluster_pipeline", {}, "Production Clustering Pipeline");
    });
    await waitFor(() => {
      expect(screen.getByText("job_pipeline_123")).toBeDefined();
    });
  });

  it("renders Past Scenarios section when What-If is expanded and history exists", async () => {
    const { fetchScenarioHistory } = await import("@/api/queries");
    (fetchScenarioHistory as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        job_id: "j_past1",
        job_type: "cluster_scenario",
        job_label: "What-If Scenario G",
        status: "completed",
        submitted_at: "2026-02-27T14:15:00Z",
        completed_at: "2026-02-27T14:15:45Z",
        result: {
          scenario_id: "sc_past1",
          status: "completed",
          runtime_seconds: 45.2,
          result: {
            optimal_k: 5,
            profiles: [],
            k_selection_results: { k_values: [3, 4, 5], inertias: [50000, 30000, 20000], silhouette_scores: [0.35, 0.40, 0.45] },
            feature_importance: [],
          },
        },
      },
    ]);

    renderTab();
    // The Past Scenarios section is inside the What-If panel — expand it first
    const toggle = await screen.findByText("What-If Scenarios");
    fireEvent.click(toggle);
    await waitFor(() => {
      expect(screen.getByText(/Past Scenarios/)).toBeDefined();
    });
  });
});
