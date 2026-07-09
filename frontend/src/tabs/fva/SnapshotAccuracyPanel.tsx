import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchFVASnapshotAccuracy,
  fetchFVASnapshotMonths,
  fvaKeys,
  STALE_PLATFORM,
  type FVASnapshotAccuracyRow,
} from "@/api/queries";

function formatPercent(value: number | null): string {
  return value == null ? "Pending" : `${value.toFixed(1)}%`;
}

function rowLabel(row: FVASnapshotAccuracyRow): string {
  return row.snapshot_role === "champion" ? "Champion" : `#${row.contender_rank} ${row.model_id}`;
}

export function SnapshotAccuracyPanel() {
  const [selectedMonth, setSelectedMonth] = useState("");
  const { data: monthsData } = useQuery({
    queryKey: fvaKeys.snapshotMonths,
    queryFn: fetchFVASnapshotMonths,
    staleTime: STALE_PLATFORM,
  });
  const recordMonth = selectedMonth || monthsData?.months[0]?.record_month || "";
  const selectedMonthInfo = monthsData?.months.find((month) => month.record_month === recordMonth);
  const { data: accuracyData } = useQuery({
    queryKey: fvaKeys.snapshotAccuracy(recordMonth),
    queryFn: () => fetchFVASnapshotAccuracy(recordMonth),
    enabled: Boolean(recordMonth),
    staleTime: STALE_PLATFORM,
  });

  const byModel = new Map<string, FVASnapshotAccuracyRow[]>();
  for (const row of accuracyData?.rows ?? []) {
    byModel.set(row.model_id, [...(byModel.get(row.model_id) ?? []), row]);
  }
  const models = [...byModel.values()]
    .map((rows) => rows.sort((a, b) => a.lag - b.lag))
    .sort((left, right) => {
      const leftRank = left[0]?.snapshot_role === "champion" ? 0 : (left[0]?.contender_rank ?? 99);
      const rightRank = right[0]?.snapshot_role === "champion" ? 0 : (right[0]?.contender_rank ?? 99);
      return leftRank - rightRank;
    });

  return (
    <section className="rounded-lg border border-border bg-card p-4" aria-labelledby="snapshot-accuracy-heading">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h3 id="snapshot-accuracy-heading" className="text-sm font-medium text-foreground">Live Forward Snapshot Accuracy</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Champion versus the three frozen, WAPE-ranked alternatives. Deltas use only common DFUs.
          </p>
        </div>
        <label className="text-xs text-muted-foreground">
          Record month
          <select
            aria-label="Snapshot record month"
            className="ml-2 rounded border border-border bg-card px-2 py-1 text-sm text-foreground"
            value={recordMonth}
            onChange={(event) => setSelectedMonth(event.target.value)}
          >
            {monthsData?.months.map((month) => (
              <option key={month.record_month} value={month.record_month}>{month.record_month.slice(0, 7)}</option>
            ))}
          </select>
        </label>
      </div>

      {models.length === 0 ? (
        <p className="py-6 text-center text-sm text-muted-foreground">No archived snapshot accuracy is available yet.</p>
      ) : (
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full text-left text-xs">
            <thead className="text-muted-foreground">
              <tr>
                <th className="px-2 py-2 font-medium">Model</th>
                {[0, 1, 2, 3, 4, 5].map((lag) => <th key={lag} className="px-2 py-2 font-medium">Lag {lag}</th>)}
              </tr>
            </thead>
            <tbody>
              {models.map((rows) => {
                const first = rows[0];
                const byLag = new Map(rows.map((row) => [row.lag, row]));
                return (
                  <tr key={first.model_id} className="border-t border-border/60">
                    <th scope="row" className="whitespace-nowrap px-2 py-2 font-medium text-foreground">{rowLabel(first)}</th>
                    {[0, 1, 2, 3, 4, 5].map((lag) => {
                      const row = byLag.get(lag);
                      return (
                        <td key={lag} className="min-w-28 px-2 py-2 align-top text-muted-foreground">
                          <div className="font-mono text-foreground">{formatPercent(row?.accuracy_pct ?? null)}</div>
                          {row ? <div>{row.n_dfus.toLocaleString()} DFUs</div> : <div>Pending</div>}
                          {row?.snapshot_role === "contender" && row.fva_vs_champion_pts != null ? (
                            <div className={row.fva_vs_champion_pts >= 0 ? "text-emerald-600" : "text-rose-600"}>
                              {row.fva_vs_champion_pts >= 0 ? "+" : ""}{row.fva_vs_champion_pts.toFixed(1)} pts
                            </div>
                          ) : null}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {selectedMonthInfo ? (
        <p className="mt-3 text-xs text-muted-foreground">
          {selectedMonthInfo.closed_lag_count}/6 closed lags
          {selectedMonthInfo.latest_closed_forecast_month ? ` through ${selectedMonthInfo.latest_closed_forecast_month.slice(0, 7)}` : ""}.
          {selectedMonthInfo.last_refresh_at ? ` Refreshed ${selectedMonthInfo.last_refresh_at}.` : " Refresh timestamp unavailable."}
        </p>
      ) : null}
    </section>
  );
}
