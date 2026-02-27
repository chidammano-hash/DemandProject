import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";
import { ScenarioNotificationProvider } from "@/context/ScenarioNotificationContext";

const mockRunScenario = vi.fn();
const mockPromote = vi.fn();

vi.mock("@/api/queries", () => ({
  queryKeys: {
    dfuClusters: (s: string) => ["dfu-clusters", s],
    clusterProfiles: () => ["cluster-profiles"],
    clusteringDefaults: () => ["clustering-defaults"],
    clusteringScenario: (id: string) => ["clustering-scenario", id],
    seasonalityProfiles: () => ["seasonality-profiles"],
    scenarioEstimate: (p: Record<string, unknown>) => ["scenario-estimate", p],
    scenarioStatus: (id: string) => ["scenario-status", id],
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
    model_params: { k_range: [3, 12], min_cluster_size_pct: 2.0, use_pca: false, pca_components: null, skip_gap: true, all_features: false },
    label_params: { volume_high: 0.75, volume_low: 0.25, cv_steady: 0.3, cv_volatile: 0.8, seasonality_threshold: 0.5, zero_demand_threshold: 0.2 },
  }),
  fetchSeasonalityProfiles: vi.fn().mockResolvedValue({ profiles: [] }),
  fetchScenarioEstimate: vi.fn().mockResolvedValue({ estimated_seconds: 45, dfu_count: 1200, k_range: 10, skip_gap: true }),
  fetchScenarioStatus: vi.fn(),
  runClusteringScenario: mockRunScenario,
  promoteScenario: mockPromote,
}));

const { fetchScenarioStatus: mockFetchStatus } = await import("@/api/queries") as { fetchScenarioStatus: ReturnType<typeof vi.fn> };

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

const MOCK_SCENARIO_RESULT = {
  scenario_id: "sc_20260223_120000_ab12",
  status: "completed" as const,
  runtime_seconds: 12.5,
  params: {
    feature_params: { time_window_months: 24, min_months_history: 1 },
    model_params: { k_range: [3, 12], min_cluster_size_pct: 2.0, use_pca: false, pca_components: null, skip_gap: true, all_features: false },
    label_params: { volume_high: 0.75, volume_low: 0.25, cv_steady: 0.3, cv_volatile: 0.8, seasonality_threshold: 0.5, zero_demand_threshold: 0.2 },
  },
  result: {
    optimal_k: 5,
    silhouette_score: 0.45,
    inertia: 12345,
    total_dfus: 500,
    k_selection_results: {
      k_values: [3, 4, 5, 6, 7],
      inertias: [50000, 30000, 20000, 15000, 12000],
      silhouette_scores: [0.35, 0.40, 0.45, 0.42, 0.38],
    },
    profiles: [
      { label: "high_volume_steady", count: 200, pct_of_total: 40, mean_demand: 1000, cv_demand: 0.2, seasonality_strength: 0.1, trend_slope: 0.02, growth_rate: 0.05, zero_demand_pct: 0.0 },
      { label: "low_volume_volatile", count: 150, pct_of_total: 30, mean_demand: 50, cv_demand: 1.2, seasonality_strength: 0.3, trend_slope: -0.01, growth_rate: -0.02, zero_demand_pct: 0.15 },
      { label: "seasonal_medium", count: 150, pct_of_total: 30, mean_demand: 300, cv_demand: 0.6, seasonality_strength: 0.8, trend_slope: 0.01, growth_rate: 0.03, zero_demand_pct: 0.05 },
    ],
  },
  error: null,
};

describe("WhatIfScenarios", () => {
  beforeEach(() => {
    mockRunScenario.mockReset();
    mockPromote.mockReset();
    (mockFetchStatus as ReturnType<typeof vi.fn>).mockReset();
  });

  it("renders What-If Scenarios toggle button", async () => {
    render(
      <TestQueryWrapper>
        <ScenarioNotificationProvider>
          <ClustersTab domain="dfu" onDomainChange={vi.fn()} theme="light" />
        </ScenarioNotificationProvider>
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText("What-If Scenarios")).toBeDefined();
    });
  });

  it("expands What-If panel on click", async () => {
    render(
      <TestQueryWrapper>
        <ScenarioNotificationProvider>
          <ClustersTab domain="dfu" onDomainChange={vi.fn()} theme="light" />
        </ScenarioNotificationProvider>
      </TestQueryWrapper>
    );
    const toggle = await screen.findByText("What-If Scenarios");
    fireEvent.click(toggle);
    await waitFor(() => {
      expect(screen.getByText("Data Scope")).toBeDefined();
      expect(screen.getByText("Model")).toBeDefined();
      expect(screen.getByText("Labeling Thresholds")).toBeDefined();
    });
  });

  it("shows parameter controls when expanded", async () => {
    render(
      <TestQueryWrapper>
        <ScenarioNotificationProvider>
          <ClustersTab domain="dfu" onDomainChange={vi.fn()} theme="light" />
        </ScenarioNotificationProvider>
      </TestQueryWrapper>
    );
    fireEvent.click(await screen.findByText("What-If Scenarios"));
    await waitFor(() => {
      expect(screen.getByText("Time Window (months)")).toBeDefined();
      expect(screen.getByText("Min History (months)")).toBeDefined();
      expect(screen.getByText("K Range")).toBeDefined();
      expect(screen.getByText("Min Cluster Size (%)")).toBeDefined();
      expect(screen.getByText("Volume High (pctl)")).toBeDefined();
      expect(screen.getByText("CV Steady (<)")).toBeDefined();
    });
  });

  it("shows Schedule Scenario Job and Reset buttons", async () => {
    render(
      <TestQueryWrapper>
        <ScenarioNotificationProvider>
          <ClustersTab domain="dfu" onDomainChange={vi.fn()} theme="light" />
        </ScenarioNotificationProvider>
      </TestQueryWrapper>
    );
    fireEvent.click(await screen.findByText("What-If Scenarios"));
    await waitFor(() => {
      expect(screen.getByText("Schedule Scenario Job")).toBeDefined();
      expect(screen.getByText("Reset to Defaults")).toBeDefined();
    });
  });

  it("calls runClusteringScenario on Run click", async () => {
    mockRunScenario.mockResolvedValue({ scenario_id: "sc_20260223_120000_ab12", status: "running" });
    (mockFetchStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      scenario_id: "sc_20260223_120000_ab12",
      status: "completed",
      runtime_seconds: 12.5,
      result: MOCK_SCENARIO_RESULT,
    });

    render(
      <TestQueryWrapper>
        <ScenarioNotificationProvider>
          <ClustersTab domain="dfu" onDomainChange={vi.fn()} theme="light" />
        </ScenarioNotificationProvider>
      </TestQueryWrapper>
    );
    fireEvent.click(await screen.findByText("What-If Scenarios"));
    await waitFor(() => expect(screen.getByText("Schedule Scenario Job")).toBeDefined());

    fireEvent.click(screen.getByText("Schedule Scenario Job"));

    await waitFor(() => {
      expect(mockRunScenario).toHaveBeenCalledTimes(1);
    });
  });

  it("shows error when scenario POST fails", async () => {
    mockRunScenario.mockRejectedValue(new Error("Pipeline blew up"));

    render(
      <TestQueryWrapper>
        <ScenarioNotificationProvider>
          <ClustersTab domain="dfu" onDomainChange={vi.fn()} theme="light" />
        </ScenarioNotificationProvider>
      </TestQueryWrapper>
    );
    fireEvent.click(await screen.findByText("What-If Scenarios"));
    await waitFor(() => expect(screen.getByText("Schedule Scenario Job")).toBeDefined());

    fireEvent.click(screen.getByText("Schedule Scenario Job"));

    await waitFor(() => {
      expect(screen.getByText("Pipeline blew up")).toBeDefined();
    });
  });

  it("shows promote confirmation dialog", async () => {
    mockRunScenario.mockResolvedValue({ scenario_id: "sc_promote", status: "running" });
    (mockFetchStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      scenario_id: "sc_promote",
      status: "completed",
      runtime_seconds: 12.5,
      result: MOCK_SCENARIO_RESULT,
    });

    render(
      <TestQueryWrapper>
        <ScenarioNotificationProvider>
          <ClustersTab domain="dfu" onDomainChange={vi.fn()} theme="light" />
        </ScenarioNotificationProvider>
      </TestQueryWrapper>
    );
    fireEvent.click(await screen.findByText("What-If Scenarios"));
    fireEvent.click(await screen.findByText("Schedule Scenario Job"));

    await waitFor(() => expect(screen.getByText(/Promote Scenario/)).toBeDefined());
    fireEvent.click(screen.getByText(/Promote Scenario/));

    await waitFor(() => {
      expect(screen.getByText(/Promote Scenario.*to Production\?/)).toBeDefined();
      expect(screen.getByText("Cancel")).toBeDefined();
    });
  });

  it("renders charts when scenario results are available", async () => {
    mockRunScenario.mockResolvedValue({ scenario_id: "sc_charts", status: "running" });
    (mockFetchStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      scenario_id: "sc_charts",
      status: "completed",
      runtime_seconds: 12.5,
      result: MOCK_SCENARIO_RESULT,
    });

    render(
      <TestQueryWrapper>
        <ScenarioNotificationProvider>
          <ClustersTab domain="dfu" onDomainChange={vi.fn()} theme="light" />
        </ScenarioNotificationProvider>
      </TestQueryWrapper>
    );
    fireEvent.click(await screen.findByText("What-If Scenarios"));
    fireEvent.click(await screen.findByText("Schedule Scenario Job"));

    await waitFor(() => {
      expect(screen.getByText("Elbow (WCSS/Inertia)")).toBeDefined();
      expect(screen.getByText("Silhouette Score")).toBeDefined();
      expect(screen.getByText("Cluster Size Distribution")).toBeDefined();
    });
  });
});
