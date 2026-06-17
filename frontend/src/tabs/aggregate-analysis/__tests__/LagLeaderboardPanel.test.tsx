import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LagLeaderboardPanel } from "../LagLeaderboardPanel";

vi.mock("@/api/queries/accuracy", () => ({
  lagLeaderboardKeys: { list: (p: unknown) => ["lag-leaderboard", p] },
  fetchLagLeaderboard: vi.fn().mockResolvedValue({
    lags: [{ lag: 0, rankings: [{ rank: 1, model_id: "lgbm_cluster", accuracy_pct: 68, wape: 32, bias: 0, n_rows: 100 }] }],
    limit: 5,
    source: "agg_accuracy_lag_archive",
  }),
}));

function wrapper(children: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("LagLeaderboardPanel", () => {
  it("renders lag rankings", async () => {
    render(wrapper(<LagLeaderboardPanel />));
    expect(await screen.findByText(/Lag Leaderboard/i)).toBeInTheDocument();
    expect(await screen.findByText(/lgbm_cluster/)).toBeInTheDocument();
  });
});
