import { useQuery } from "@tanstack/react-query";
import {
  intramonthKeys,
  fetchIntramonthSummary,
  fetchIntramonthDetail,
  STALE,
  type IntramonthStockoutRow,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function IntramonthPanel() {
  const { data: summary } = useQuery({
    queryKey: intramonthKeys.summary(),
    queryFn: () => fetchIntramonthSummary(),
    staleTime: STALE.FIVE_MIN,
  });
  const { data: detail } = useQuery({
    queryKey: intramonthKeys.detail({ limit: 10, had_stockout: true }),
    queryFn: () => fetchIntramonthDetail({ limit: 10, had_stockout: "true", sort_by: "stockout_day_rate", sort_dir: "desc" }),
    staleTime: STALE.FIVE_MIN,
  });

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard className={PANEL_KPI} label="Items with Stockout" value={(summary?.items_with_stockout ?? 0).toLocaleString()} colorClass="text-red-600" />
        <KpiCard className={PANEL_KPI} label="Extended Stockouts (7d+)" value={(summary?.items_with_extended_stockout ?? 0).toLocaleString()} colorClass="text-red-700" />
        <KpiCard className={PANEL_KPI} label="Total Stockout Days" value={(summary?.total_stockout_days ?? 0).toLocaleString()} />
        <KpiCard className={PANEL_KPI} label="Est. Lost Sales" value={Number(summary?.total_est_lost_sales ?? 0).toFixed(0)} />
      </div>
      {detail && detail.rows.length > 0 && (
        <div className="overflow-x-auto">
          <p className="text-xs font-medium mb-2">Top Stockout Items (current period)</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="text-left py-1 pr-2">Item</th>
                <th className="text-left py-1 px-2">Loc</th>
                <th className="text-right py-1 px-2">Stockout Days</th>
                <th className="text-right py-1 px-2">Day Rate</th>
                <th className="text-right py-1 px-2">Est. Lost Sales</th>
                <th className="text-center py-1 px-2">Extended?</th>
              </tr>
            </thead>
            <tbody>
              {detail.rows.map((r: IntramonthStockoutRow) => (
                <tr key={`${r.item_no}-${r.loc}-${r.month_start}`} className="border-b hover:bg-muted/30">
                  <td className="py-1 pr-2 font-medium">{r.item_no}</td>
                  <td className="py-1 px-2">{r.loc}</td>
                  <td className="text-right py-1 px-2 text-red-600 font-semibold">{r.stockout_days}</td>
                  <td className="text-right py-1 px-2">{r.stockout_day_rate != null ? `${(r.stockout_day_rate * 100).toFixed(0)}%` : "—"}</td>
                  <td className="text-right py-1 px-2">{r.est_lost_sales?.toFixed(0) ?? "—"}</td>
                  <td className="text-center py-1 px-2">{r.had_extended_stockout ? <span className="text-red-600 font-bold">Yes</span> : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
