/** Lag-decomposed accuracy leaderboard (spec 02-forecasting/28). */
import { useQuery } from "@tanstack/react-query";
import { fetchLagLeaderboard, lagLeaderboardKeys } from "@/api/queries/accuracy";

export function LagLeaderboardPanel() {
  const { data, isLoading } = useQuery({
    queryKey: lagLeaderboardKeys.list({ limit: 5 }),
    queryFn: () => fetchLagLeaderboard({ limit: 5 }),
    staleTime: 120_000,
  });

  if (isLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground">
        Loading lag leaderboard…
      </div>
    );
  }

  if (!data?.lags?.length) {
    return null;
  }

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h3 className="text-sm font-medium text-foreground">Lag Leaderboard</h3>
      <p className="mt-1 text-xs text-muted-foreground">
        Top models by accuracy at each execution lag (0–4 months). Source: {data.source}.
      </p>
      <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {data.lags.map((lagBlock) => (
          <div key={lagBlock.lag} className="rounded-md border border-border/60 p-3">
            <p className="text-xs font-medium text-foreground">Lag {lagBlock.lag}</p>
            {lagBlock.rankings.length === 0 ? (
              <p className="mt-2 text-xs text-muted-foreground">No data</p>
            ) : (
              <ol className="mt-2 space-y-1 text-xs">
                {lagBlock.rankings.map((r) => (
                  <li key={`${lagBlock.lag}-${r.model_id}`} className="flex justify-between gap-2">
                    <span className="truncate font-mono" title={r.model_id}>
                      {r.rank}. {r.model_id}
                    </span>
                    <span className="shrink-0 text-muted-foreground">
                      {r.accuracy_pct != null ? `${r.accuracy_pct.toFixed(1)}%` : "—"}
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
