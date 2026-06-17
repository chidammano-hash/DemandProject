import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AiChampionPanel } from "../AiChampionPanel";

vi.mock("@/api/queries/ai-champion", () => ({
  aiChampionKeys: {
    latest: () => ["ai-champion", "latest"],
    forecast: (p: unknown) => ["ai-champion", "forecast", p],
  },
  fetchAiChampionLatest: vi.fn(),
  fetchAiChampionForecast: vi.fn(),
  triggerAiChampionGenerate: vi.fn(),
}));

vi.mock("@/api/queries/jobs", () => ({
  fetchActiveJobs: vi.fn().mockResolvedValue({ jobs: [] }),
  fetchJobDetail: vi.fn(),
}));

import { fetchAiChampionLatest, fetchAiChampionForecast } from "@/api/queries/ai-champion";

function wrapper(children: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("AiChampionPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows empty state when no run exists", async () => {
    vi.mocked(fetchAiChampionLatest).mockResolvedValue({ run: null, by_recommendation: [] });
    render(wrapper(<AiChampionPanel />));
    expect(await screen.findByText(/No AI Champion run yet/i)).toBeInTheDocument();
  });

  it("renders run summary and forecast table", async () => {
    vi.mocked(fetchAiChampionLatest).mockResolvedValue({
      run: {
        run_id: "r1",
        plan_version: "2026-04",
        provider: "ollama",
        ai_model: "llama3.1:8b",
        status: "succeeded",
        n_dfus: 100,
        n_adjusted: 20,
        est_cost_usd: 0,
        started_at: "2026-04-01T00:00:00Z",
        completed_at: "2026-04-01T00:05:00Z",
      },
      by_recommendation: [{ recommendation_code: "SCALE_UP", dfus: 20 }],
    });
    vi.mocked(fetchAiChampionForecast).mockResolvedValue({
      total: 1,
      rows: [{
        item_id: "100",
        loc: "L1",
        forecast_month: "2026-05-01",
        horizon_months: 1,
        champion_qty: 100,
        ai_qty: 110,
        recommendation_code: "SCALE_UP",
        pct_change: 10,
        confidence: 0.8,
        rationale: "uptrend",
      }],
    });
    render(wrapper(<AiChampionPanel />));
    expect(await screen.findByText(/AI Champion Forecast/i)).toBeInTheDocument();
    expect(await screen.findByText(/100-L1/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Generate/i })).toBeInTheDocument();
  });
});
