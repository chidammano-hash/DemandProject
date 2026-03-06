import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  safetyStockKeys,
  fetchSafetyStockSummary,
  fetchSafetyStockDetail,
  STALE,
  type SafetyStockRow,
} from "@/api/queries";

const PAGE = 50;

function fmt(n: number | null | undefined, decimals = 1): string {
  if (n == null) return "—";
  return Number(n).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtPct(n: number | null | undefined): string {
  if (n == null) return "—";
  return `${Number(n).toFixed(1)}%`;
}

export function SafetyStockPanel() {
  const [belowOnly, setBelowOnly] = useState(false);
  const [ssItemFilter, setSsItemFilter] = useState("");
  const [ssLocFilter, setSsLocFilter] = useState("");
  const [ssOffset, setSsOffset] = useState(0);

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: safetyStockKeys.summary(),
    queryFn: () => fetchSafetyStockSummary(),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: detail, isLoading: detailLoading } = useQuery({
    queryKey: safetyStockKeys.detail({
      is_below_ss: belowOnly ? true : undefined,
      item: ssItemFilter || undefined,
      loc: ssLocFilter || undefined,
      limit: PAGE,
      offset: ssOffset,
    }),
    queryFn: () =>
      fetchSafetyStockDetail({
        is_below_ss: belowOnly ? true : undefined,
        item: ssItemFilter || undefined,
        loc: ssLocFilter || undefined,
        limit: PAGE,
        offset: ssOffset,
      }),
    staleTime: STALE.FIVE_MIN,
  });

  const totalPages = detail ? Math.ceil(detail.total / PAGE) : 0;
  const currentPage = Math.floor(ssOffset / PAGE) + 1;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Items Below SS</p>
          <p className={`text-xl font-bold ${(summary?.below_ss_count ?? 0) > 0 ? "text-red-600" : "text-foreground"}`}>
            {summaryLoading ? "..." : (summary?.below_ss_count ?? 0).toLocaleString()}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Avg SS Coverage</p>
          <p className="text-xl font-bold">
            {summaryLoading ? "..." : fmtPct(summary?.avg_ss_coverage != null ? summary.avg_ss_coverage * 100 : null)}
          </p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Total DFUs</p>
          <p className="text-xl font-bold">{summaryLoading ? "..." : (summary?.total_dfus ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Avg SS Days</p>
          <p className="text-xl font-bold">{summaryLoading ? "..." : fmt(summary?.avg_ss_days, 1)}</p>
        </div>
      </div>

      {summary && summary.by_abc.length > 0 && (
        <div className="overflow-x-auto">
          <p className="text-xs font-medium mb-2">Safety Stock by ABC Class</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-muted-foreground">
                <th className="text-left py-1 pr-3">ABC Class</th>
                <th className="text-right py-1 pr-3">Count</th>
                <th className="text-right py-1 pr-3">Below SS</th>
                <th className="text-right py-1">Avg Coverage</th>
              </tr>
            </thead>
            <tbody>
              {summary.by_abc.map((row) => (
                <tr key={row.abc_vol} className="border-b last:border-0">
                  <td className="py-1 pr-3 font-medium">{row.abc_vol}</td>
                  <td className="py-1 pr-3 text-right">{row.count.toLocaleString()}</td>
                  <td className={`py-1 pr-3 text-right ${row.below_ss_count > 0 ? "text-red-600 font-medium" : ""}`}>
                    {row.below_ss_count.toLocaleString()}
                  </td>
                  <td className="py-1 text-right">
                    {fmtPct(row.avg_coverage != null ? row.avg_coverage * 100 : null)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        <button
          className={`px-3 py-1 text-xs rounded border transition-colors ${belowOnly ? "bg-foreground text-background border-foreground" : "border-border hover:bg-accent"}`}
          onClick={() => { setBelowOnly(!belowOnly); setSsOffset(0); }}
        >
          {belowOnly ? "All Items" : "Below SS Only"}
        </button>
        <input
          className="h-7 rounded border border-input bg-background px-2 text-xs w-32"
          placeholder="Filter by item..."
          value={ssItemFilter}
          onChange={(e) => { setSsItemFilter(e.target.value); setSsOffset(0); }}
        />
        <input
          className="h-7 rounded border border-input bg-background px-2 text-xs w-32"
          placeholder="Filter by location..."
          value={ssLocFilter}
          onChange={(e) => { setSsLocFilter(e.target.value); setSsOffset(0); }}
        />
      </div>

      {detailLoading ? (
        <p className="text-xs text-muted-foreground">Loading...</p>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-muted-foreground">
                  <th className="text-left py-1 pr-2">Item No</th>
                  <th className="text-left py-1 pr-2">Location</th>
                  <th className="text-right py-1 pr-2">SS (qty)</th>
                  <th className="text-right py-1 pr-2">Coverage %</th>
                  <th className="text-center py-1 pr-2">Below SS</th>
                  <th className="text-right py-1 pr-2">Reorder Point</th>
                  <th className="text-center py-1">ABC</th>
                </tr>
              </thead>
              <tbody>
                {(detail?.rows ?? []).length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-4 text-center text-muted-foreground">
                      No data. Run make health-schema health-refresh to populate.
                    </td>
                  </tr>
                ) : (
                  (detail?.rows ?? []).map((r: SafetyStockRow, i: number) => (
                    <tr
                      key={`${r.item_no}-${r.loc}-${i}`}
                      className={`border-b last:border-0 hover:bg-muted/40 ${r.is_below_ss ? "bg-red-50 dark:bg-red-950/20" : ""}`}
                    >
                      <td className="py-1 pr-2 font-mono">{r.item_no}</td>
                      <td className="py-1 pr-2">{r.loc}</td>
                      <td className="py-1 pr-2 text-right">{fmt(r.ss_combined, 1)}</td>
                      <td className="py-1 pr-2 text-right">
                        {fmtPct(r.ss_coverage != null ? r.ss_coverage * 100 : null)}
                      </td>
                      <td className="py-1 pr-2 text-center">
                        {r.is_below_ss ? (
                          <span className="px-1.5 py-0.5 rounded text-xs bg-red-100 text-red-800 font-medium">Yes</span>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </td>
                      <td className="py-1 pr-2 text-right">{fmt(r.reorder_point, 1)}</td>
                      <td className="py-1 text-center">{r.abc_vol ?? "-"}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={ssOffset === 0}
                onClick={() => setSsOffset(Math.max(0, ssOffset - PAGE))}
              >
                Prev
              </button>
              <span>Page {currentPage} / {totalPages}</span>
              <button
                className="px-2 py-1 rounded border disabled:opacity-40"
                disabled={currentPage >= totalPages}
                onClick={() => setSsOffset(ssOffset + PAGE)}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
