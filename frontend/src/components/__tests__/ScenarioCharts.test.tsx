/**
 * ScenarioCharts — PCA scatter canvas migration tests (P0-4)
 *
 * Verifies that the PCA 2D scatter renders via ModularReactECharts (canvas),
 * not the legacy recharts SVG path, and that the client-side point cap
 * (PCA_RENDER_CAP = 5 000) works correctly.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { ThemeProvider } from "@/context/ThemeContext";
import type { ClusteringScenarioResult, PCAScatterData } from "@/api/queries";
import type { ReactNode } from "react";

// ── Mock ModularReactECharts (canvas renderer; can't run in jsdom) ──────────
vi.mock("@/components/echarts-modular", () => ({
  ModularReactECharts: (props: { style?: React.CSSProperties; option?: unknown }) => (
    <div
      data-testid="modular-echart"
      data-option={JSON.stringify(props.option ?? {})}
      style={props.style}
    />
  ),
  default: (props: { style?: React.CSSProperties }) => (
    <div data-testid="modular-echart-default" style={props.style} />
  ),
}));

// ── recharts is still used for the other charts; mock the whole lib ──────────
vi.mock("recharts");

function Wrapper({ children }: { children: ReactNode }) {
  return <ThemeProvider value={{ theme: "light" }}>{children}</ThemeProvider>;
}

// Minimal result shape that satisfies ScenarioChartsProps
const BASE_RESULT: NonNullable<ClusteringScenarioResult["result"]> = {
  optimal_k: 3,
  silhouette_score: 0.45,
  inertia: 20000,
  total_dfus: 30,
  profiles: [
    {
      label: "high_volume_steady",
      count: 10,
      pct_of_total: 33.3,
      mean_demand: 1000,
      cv_demand: 0.2,
      seasonality_strength: 0.1,
      trend_slope: 0.01,
      growth_rate: 0.05,
      zero_demand_pct: 0.02,
    },
    {
      label: "low_volume_volatile",
      count: 20,
      pct_of_total: 66.7,
      mean_demand: 100,
      cv_demand: 1.2,
      seasonality_strength: 0.3,
      trend_slope: -0.01,
      growth_rate: -0.02,
      zero_demand_pct: 0.4,
    },
  ],
  k_selection_results: {
    k_values: [2, 3, 4],
    inertias: [50000, 20000, 15000],
    silhouette_scores: [0.3, 0.45, 0.4],
  },
  feature_importance: [
    { feature: "mean_demand", variance_ratio: 0.8 },
  ],
};

function makePcaScatter(n: number): PCAScatterData {
  return {
    pc1_variance: 42.1,
    pc2_variance: 18.3,
    points: Array.from({ length: n }, (_, i) => ({
      pc1: Math.sin(i) * 2,
      pc2: Math.cos(i) * 2,
      cluster: i % 3,
    })),
  };
}

// Dynamic import so the mock above is in place before the module loads.
async function importScenarioCharts() {
  const mod = await import("@/components/ScenarioCharts");
  return mod.ScenarioCharts;
}

describe("ScenarioCharts — PCA scatter (P0-4)", () => {
  it("renders the canvas scatter (ModularReactECharts) when pcaScatter is provided", async () => {
    const ScenarioCharts = await importScenarioCharts();
    const pca = makePcaScatter(50);

    render(
      <Wrapper>
        <ScenarioCharts result={BASE_RESULT} pcaScatter={pca} />
      </Wrapper>,
    );

    // ModularReactECharts must be present (canvas path)
    expect(screen.getByTestId("modular-echart")).toBeDefined();
    // The heading text
    expect(screen.getByText("Cluster Visualization (2D PCA)")).toBeDefined();
  });

  it("hides the PCA scatter section when pcaScatter is absent", async () => {
    const ScenarioCharts = await importScenarioCharts();

    render(
      <Wrapper>
        <ScenarioCharts result={BASE_RESULT} />
      </Wrapper>,
    );

    expect(screen.queryByTestId("modular-echart")).toBeNull();
    expect(screen.queryByText("Cluster Visualization (2D PCA)")).toBeNull();
  });

  it("hides the PCA scatter section when pcaScatter has zero points", async () => {
    const ScenarioCharts = await importScenarioCharts();
    const pca = makePcaScatter(0);

    render(
      <Wrapper>
        <ScenarioCharts result={BASE_RESULT} pcaScatter={pca} />
      </Wrapper>,
    );

    expect(screen.queryByTestId("modular-echart")).toBeNull();
  });

  it("does NOT show the point-cap note when total ≤ 5 000", async () => {
    const ScenarioCharts = await importScenarioCharts();
    const pca = makePcaScatter(500);

    render(
      <Wrapper>
        <ScenarioCharts result={BASE_RESULT} pcaScatter={pca} />
      </Wrapper>,
    );

    expect(screen.queryByText(/showing \d+ of \d+ points/)).toBeNull();
  });

  it("shows the point-cap note when total > 5 000", async () => {
    const ScenarioCharts = await importScenarioCharts();
    const pca = makePcaScatter(10_000);

    render(
      <Wrapper>
        <ScenarioCharts result={BASE_RESULT} pcaScatter={pca} />
      </Wrapper>,
    );

    // The capped note: "showing N of 10,000 points"
    const note = screen.getByText(/showing .+ of 10,000 points/);
    expect(note).toBeDefined();
  });

  it("builds ECharts option with one scatter series per cluster", async () => {
    const ScenarioCharts = await importScenarioCharts();
    // 6 points across 3 clusters (2 per cluster)
    const pca: PCAScatterData = {
      pc1_variance: 40,
      pc2_variance: 20,
      points: [
        { pc1: 1, pc2: 2, cluster: 0 },
        { pc1: 3, pc2: 4, cluster: 1 },
        { pc1: 5, pc2: 6, cluster: 2 },
        { pc1: 7, pc2: 8, cluster: 0 },
        { pc1: 9, pc2: 0, cluster: 1 },
        { pc1: 1, pc2: 3, cluster: 2 },
      ],
    };

    render(
      <Wrapper>
        <ScenarioCharts result={BASE_RESULT} pcaScatter={pca} />
      </Wrapper>,
    );

    const chart = screen.getByTestId("modular-echart");
    const option = JSON.parse(chart.getAttribute("data-option") ?? "{}");

    // Exactly 3 series (one per cluster id)
    expect(option.series).toHaveLength(3);
    expect(option.series[0].name).toBe("Cluster 0");
    expect(option.series[1].name).toBe("Cluster 1");
    expect(option.series[2].name).toBe("Cluster 2");
    // Each series must be type "scatter"
    for (const s of option.series) {
      expect(s.type).toBe("scatter");
    }
  });

  it("includes PC1/PC2 variance labels in the axis names", async () => {
    const ScenarioCharts = await importScenarioCharts();
    const pca: PCAScatterData = {
      pc1_variance: 55.5,
      pc2_variance: 22.2,
      points: [{ pc1: 1, pc2: 2, cluster: 0 }],
    };

    render(
      <Wrapper>
        <ScenarioCharts result={BASE_RESULT} pcaScatter={pca} />
      </Wrapper>,
    );

    const chart = screen.getByTestId("modular-echart");
    const option = JSON.parse(chart.getAttribute("data-option") ?? "{}");

    expect(option.xAxis.name).toContain("55.5%");
    expect(option.yAxis.name).toContain("22.2%");
  });
});
