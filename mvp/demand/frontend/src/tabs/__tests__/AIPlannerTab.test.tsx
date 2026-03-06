/**
 * Smoke tests for AIPlannerTab — IPAIfeature1.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// ---------------------------------------------------------------------------
// Mock API layer
// ---------------------------------------------------------------------------
vi.mock("@/api/queries", () => ({
  queryKeys: {
    aiInsights: (p?: object) => ["ai-insights", p ?? {}],
    aiMemos: (p?: object) => ["ai-memos", p ?? {}],
  },
  fetchAiInsights: vi.fn().mockResolvedValue({ insights: [], total: 0, page: 1 }),
  fetchAiMemos: vi.fn().mockResolvedValue({ memos: [] }),
  triggerPortfolioScan: vi.fn().mockResolvedValue({ scan_run_id: "scan-001", status: "accepted" }),
  updateInsightStatus: vi.fn().mockResolvedValue({ insight_id: 1, status: "acknowledged" }),
  STALE: { THIRTY_SEC: 30_000, FIVE_MIN: 300_000 },
}));

// Mock lucide-react icons used in the tab
vi.mock("lucide-react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("lucide-react")>();
  return {
    ...actual,
    AlertTriangle: ({ className }: { className?: string }) => (
      <span data-testid="icon-alert-triangle" className={className} />
    ),
    AlertCircle: ({ className }: { className?: string }) => (
      <span data-testid="icon-alert-circle" className={className} />
    ),
    Info: ({ className }: { className?: string }) => (
      <span data-testid="icon-info" className={className} />
    ),
    CheckCircle2: ({ className }: { className?: string }) => (
      <span data-testid="icon-check" className={className} />
    ),
    Brain: ({ className }: { className?: string }) => (
      <span data-testid="icon-brain" className={className} />
    ),
    RefreshCw: ({ className }: { className?: string }) => (
      <span data-testid="icon-refresh" className={className} />
    ),
    Loader2: ({ className }: { className?: string }) => (
      <span data-testid="icon-loader" className={className} />
    ),
    ChevronDown: ({ className }: { className?: string }) => (
      <span data-testid="icon-chevron-down" className={className} />
    ),
    ChevronUp: ({ className }: { className?: string }) => (
      <span data-testid="icon-chevron-up" className={className} />
    ),
    Package: ({ className }: { className?: string }) => (
      <span data-testid="icon-package" className={className} />
    ),
    TrendingDown: ({ className }: { className?: string }) => (
      <span data-testid="icon-trending-down" className={className} />
    ),
    BarChart3: ({ className }: { className?: string }) => (
      <span data-testid="icon-bar-chart" className={className} />
    ),
    Zap: ({ className }: { className?: string }) => (
      <span data-testid="icon-zap" className={className} />
    ),
    Target: ({ className }: { className?: string }) => (
      <span data-testid="icon-target" className={className} />
    ),
    Monitor: ({ className }: { className?: string }) => (
      <span data-testid="icon-monitor" className={className} />
    ),
    Sparkles: ({ className }: { className?: string }) => (
      <span data-testid="icon-sparkles" className={className} />
    ),
  };
});

// ---------------------------------------------------------------------------
// Helper: sample insight
// ---------------------------------------------------------------------------
const sampleInsight = {
  insight_id: 1,
  insight_type: "stockout_risk" as const,
  severity: "critical" as const,
  item_no: "100320",
  loc: "1401-BULK",
  abc_vol: "A",
  cluster_assignment: "high_volume_steady",
  summary: "Low DOS vs lead time",
  recommendation: "Trigger emergency reorder",
  reasoning: "DOS 18 < LT 21",
  financial_impact_estimate: 8500,
  dos: 18,
  champion_wape: 0.41,
  current_policy_id: "continuous_rop",
  status: "open" as const,
  created_at: "2026-03-01T00:00:00Z",
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe("AIPlannerTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders without crashing with empty data", async () => {
    const { fetchAiInsights, fetchAiMemos } = await import("@/api/queries");
    vi.mocked(fetchAiInsights).mockResolvedValue({ insights: [], total: 0, page: 1 });
    vi.mocked(fetchAiMemos).mockResolvedValue({ memos: [] });

    const { default: AIPlannerTab } = await import("@/tabs/AIPlannerTab");
    render(
      <TestQueryWrapper>
        <AIPlannerTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("AI Planner")).toBeDefined();
    });
  });

  it("shows Generate Now button", async () => {
    const { default: AIPlannerTab } = await import("@/tabs/AIPlannerTab");
    render(
      <TestQueryWrapper>
        <AIPlannerTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("Generate Now")).toBeDefined();
    });
  });

  it("shows portfolio health KPI labels", async () => {
    const { default: AIPlannerTab } = await import("@/tabs/AIPlannerTab");
    render(
      <TestQueryWrapper>
        <AIPlannerTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("Open Insights")).toBeDefined();
      // "Critical" may appear multiple times (KPI card + select item) — use getAllByText
      expect(screen.getAllByText("Critical").length).toBeGreaterThan(0);
    });
  });

  it("renders insight cards when data is present", async () => {
    const { fetchAiInsights } = await import("@/api/queries");
    vi.mocked(fetchAiInsights).mockResolvedValue({
      insights: [sampleInsight],
      total: 1,
      page: 1,
    });

    const { default: AIPlannerTab } = await import("@/tabs/AIPlannerTab");
    render(
      <TestQueryWrapper>
        <AIPlannerTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("100320 @ 1401-BULK")).toBeDefined();
    });
  });

  it("renders insight summary text", async () => {
    const { fetchAiInsights } = await import("@/api/queries");
    vi.mocked(fetchAiInsights).mockResolvedValue({
      insights: [sampleInsight],
      total: 1,
      page: 1,
    });

    const { default: AIPlannerTab } = await import("@/tabs/AIPlannerTab");
    render(
      <TestQueryWrapper>
        <AIPlannerTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("Low DOS vs lead time")).toBeDefined();
    });
  });

  it("renders Acknowledge button for open insights", async () => {
    const { fetchAiInsights } = await import("@/api/queries");
    vi.mocked(fetchAiInsights).mockResolvedValue({
      insights: [sampleInsight],
      total: 1,
      page: 1,
    });

    const { default: AIPlannerTab } = await import("@/tabs/AIPlannerTab");
    render(
      <TestQueryWrapper>
        <AIPlannerTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("Acknowledge")).toBeDefined();
    });
  });

  it("shows severity badge text for critical insight", async () => {
    const { fetchAiInsights } = await import("@/api/queries");
    vi.mocked(fetchAiInsights).mockResolvedValue({
      insights: [sampleInsight],
      total: 1,
      page: 1,
    });

    const { default: AIPlannerTab } = await import("@/tabs/AIPlannerTab");
    render(
      <TestQueryWrapper>
        <AIPlannerTab />
      </TestQueryWrapper>,
    );

    await waitFor(() => {
      expect(screen.getByText("critical")).toBeDefined();
    });
  });
});
