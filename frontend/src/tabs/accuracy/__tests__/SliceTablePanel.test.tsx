import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { ThemeProvider } from "@/context/ThemeContext";
import type { AccuracySliceRow } from "@/types";

// Mock @tanstack/react-virtual — jsdom has no layout engine so useVirtualizer
// would produce 0 virtual items; render all items synchronously instead so the
// virtualized body's rows are assertable.
vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: ({ count, estimateSize }: { count: number; estimateSize: () => number }) => ({
    getVirtualItems: () =>
      Array.from({ length: count }, (_, i) => ({
        index: i,
        start: i * estimateSize(),
        size: estimateSize(),
        key: i,
        lane: 0,
        end: (i + 1) * estimateSize(),
      })),
    getTotalSize: () => count * estimateSize(),
  }),
}));

import { SliceTablePanel, type SliceTablePanelProps } from "../SliceTablePanel";

function makeRow(bucket: string, acc: number): AccuracySliceRow {
  return {
    bucket,
    n_rows: 100,
    by_model: {
      external: { accuracy_pct: acc, wape: 100 - acc, bias: 0.02, sum_forecast: 1000, sum_actual: 950, sku_count: 10 },
    },
  };
}

const NOOP = () => {};

function renderPanel(overrides: Partial<SliceTablePanelProps> = {}) {
  const props: SliceTablePanelProps = {
    sliceGroupBy: "cluster_assignment",
    sliceLag: 0,
    sliceModels: "external",
    sliceKpis: ["accuracy_pct"],
    sliceMonths: 3,
    commonDfus: false,
    seasonalityProfile: "",
    seasonalityProfiles: [],
    loadingSlice: false,
    sliceData: [makeRow("cluster_a", 88), makeRow("cluster_b", 72)],
    allModels: ["external"],
    commonDfuCount: null,
    skuCounts: null,
    truncated: false,
    sliceLimit: 1000,
    onSliceGroupByChange: NOOP,
    onSliceLagChange: NOOP,
    onSliceModelsChange: NOOP,
    onSliceMonthsChange: NOOP,
    onCommonDfusToggle: NOOP,
    onKpiToggle: NOOP,
    onSeasonalityProfileChange: NOOP,
    ...overrides,
  };
  return render(
    <ThemeProvider value={{ theme: "light" }}>
      <SliceTablePanel {...props} />
    </ThemeProvider>,
  );
}

describe("SliceTablePanel (virtualized body)", () => {
  it("suggests only retained forecast models in the model filter", () => {
    renderPanel();

    expect(screen.getByPlaceholderText("e.g. lgbm_cluster,nhits")).toBeDefined();
    expect(screen.queryByPlaceholderText(/lgbm_global/i)).toBeNull();
  });

  it("renders the virtualized bucket rows", () => {
    renderPanel();
    expect(screen.getByText("cluster_a")).toBeDefined();
    expect(screen.getByText("cluster_b")).toBeDefined();
    // Header reflects the bucket count.
    expect(screen.getByText(/2 cluster assignment bucket\(s\)/i)).toBeDefined();
  });

  it("does not show the truncation note when truncated=false", () => {
    renderPanel({ truncated: false });
    expect(screen.queryByText(/truncated/i)).toBeNull();
  });

  it("shows a 'showing top N (truncated)' note when truncated=true", () => {
    renderPanel({ truncated: true, sliceLimit: 1000 });
    const note = screen.getByText(/Showing top 1,000 .*bucket\(s\) by\s*volume \(truncated\)/i);
    expect(note).toBeDefined();
  });

  it("renders an empty-state hint when there is no data", () => {
    renderPanel({ sliceData: [] });
    expect(screen.getByText(/make backtest-load/i)).toBeDefined();
  });
});
