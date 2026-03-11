import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";
import { GlobalFilterProvider } from "@/context/GlobalFilterContext";
import { ScenarioNotificationProvider } from "@/context/ScenarioNotificationContext";
import { JobNotificationProvider } from "@/context/JobNotificationContext";
import type { GlobalFilterContextValue } from "@/context/GlobalFilterContext";
import type { GlobalFilters } from "@/types/theme";

// Provide localStorage mock (useTheme depends on it)
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => { store[key] = value; },
    removeItem: (key: string) => { delete store[key]; },
    clear: () => { store = {}; },
    get length() { return Object.keys(store).length; },
    key: (i: number) => Object.keys(store)[i] ?? null,
  };
})();

Object.defineProperty(window, "localStorage", { value: localStorageMock });

vi.mock("@/api/queries", () => ({
  queryKeys: {
    dashboardKpis: (p: Record<string, unknown>) => ["dashboard-kpis", p],
    dashboardAlerts: (p: Record<string, unknown>) => ["dashboard-alerts", p],
    aiInsights: (p: Record<string, unknown>) => ["ai-insights", p],
    aiMemos: (p: Record<string, unknown>) => ["ai-memos", p],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchDashboardKpis: vi.fn().mockResolvedValue({
    accuracy_pct: 88.5,
    wape_pct: 11.2,
    bias_pct: -3.4,
    total_forecast: 500000,
    total_actual: 480000,
    weeks_of_supply: 4.2,
    window_months: 3,
    deltas: { accuracy_pct: 1.2, wape_pct: -0.5, bias_pct: 0.8, weeks_of_supply: 0.3 },
  }),
  fetchDashboardAlerts: vi.fn().mockResolvedValue({
    alerts: [
      { id: "a1", type: "oos_risk", severity: "critical", title: "OOS Risk", detail: "5 items at risk", count: 5 },
    ],
  }),
  fetchAiInsights: vi.fn().mockResolvedValue({
    insights: [
      {
        insight_id: 1,
        item_no: "587382",
        loc: "1401-BULK",
        severity: "critical",
        insight_type: "stockout_risk",
        summary: "Stockout in ~8 days",
        status: "open",
        champion_wape: 0.63,
        forecast_bias_pct: 0.87,
        dos: 6.4,
        total_lt_days: 14,
        financial_impact_estimate: 42000,
        current_policy_id: null,
        abc_vol: "A",
        ml_cluster: null,
        created_at: new Date().toISOString(),
      },
    ],
    total: 3,
  }),
  fetchAiMemos: vi.fn().mockResolvedValue({
    memos: [
      {
        memo_id: 1,
        period: "2026-03-01",
        narrative_text: "Portfolio is under pressure this week. Three A-class DFUs show critical stockout risk.",
        model_version: "claude-sonnet-4-6",
        created_at: new Date().toISOString(),
      },
    ],
  }),
}));

const DashboardTab = (await import("@/tabs/DashboardTab")).default;

function makeFilterContext(): GlobalFilterContextValue {
  const filters: GlobalFilters = {
    brand: [],
    category: [],
    market: [],
    channel: [],
    item: [],
    location: [],
    timeGrain: "month",
  };
  return {
    filters,
    setFilters: vi.fn(),
    resetFilters: vi.fn(),
    hasActiveFilters: false,
    planningDate: null,
  };
}

function renderDashboard() {
  return render(
    <TestQueryWrapper>
      <GlobalFilterProvider value={makeFilterContext()}>
        <ScenarioNotificationProvider>
          <JobNotificationProvider>
            <DashboardTab />
          </JobNotificationProvider>
        </ScenarioNotificationProvider>
      </GlobalFilterProvider>
    </TestQueryWrapper>
  );
}

describe("DashboardTab", () => {
  beforeEach(() => {
    localStorageMock.clear();
    document.documentElement.classList.remove("light", "dark");
    document.documentElement.removeAttribute("data-transitioning");
    document.documentElement.removeAttribute("data-theme");
  });

  it("renders without crashing when wrapped with providers", async () => {
    renderDashboard();
    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });

  it("shows loading skeletons initially", () => {
    const { container } = renderDashboard();
    const skeletons = container.querySelectorAll(".animate-shimmer");
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it("renders Command Center KPI cards after data loads", async () => {
    const { getByText } = renderDashboard();
    await waitFor(() => {
      expect(getByText("Open Insights")).toBeDefined();
      expect(getByText("Critical")).toBeDefined();
      expect(getByText("Portfolio DOS")).toBeDefined();
      expect(getByText("$ at Risk")).toBeDefined();
    });
  });

  it("renders priority work queue with top insight", async () => {
    const { getByText } = renderDashboard();
    await waitFor(() => {
      expect(getByText("Priority Work Queue")).toBeDefined();
      expect(getByText("587382 @ 1401-BULK")).toBeDefined();
    });
  });

  it("renders AI Planning Digest when memo is available", async () => {
    const { getByText } = renderDashboard();
    await waitFor(() => {
      expect(getByText("AI Planning Digest")).toBeDefined();
    });
  });

  it("renders system alerts section", async () => {
    const { getByText } = renderDashboard();
    await waitFor(() => {
      expect(getByText("System Alerts")).toBeDefined();
    });
  });

  it("renders Command Center heading", async () => {
    const { getByText } = renderDashboard();
    await waitFor(() => {
      expect(getByText("Command Center")).toBeDefined();
    });
  });
});
