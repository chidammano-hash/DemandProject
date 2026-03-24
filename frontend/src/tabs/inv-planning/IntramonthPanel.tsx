import { useQuery } from "@tanstack/react-query";
import {
  intramonthKeys,
  fetchIntramonthSummary,
  fetchIntramonthDetail,
  STALE,
  type IntramonthStockoutRow,
} from "@/api/queries";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { Clock, CheckCircle2 } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function IntramonthPanel() {
  const { filters } = useGlobalFilterContext();
  const gf = {
    brand: filters.brand.length > 0 ? filters.brand.join(",") : undefined,
    category: filters.category.length > 0 ? filters.category.join(",") : undefined,
    market: filters.market.length > 0 ? filters.market.join(",") : undefined,
    item: filters.item.length === 1 ? filters.item[0] : undefined,
    location: filters.location.length === 1 ? filters.location[0] : undefined,
  };

  const { data: summary } = useQuery({
    queryKey: intramonthKeys.summary(gf),
    queryFn: () => fetchIntramonthSummary(gf),
    staleTime: STALE.FIVE_MIN,
  });
  const { data: detail, isLoading } = useQuery({
    queryKey: intramonthKeys.detail({ ...gf, limit: 10, had_stockout: true }),
    queryFn: () => fetchIntramonthDetail({ ...gf, limit: 10, had_stockout: "true", sort_by: "stockout_day_rate", sort_dir: "desc" }),
    staleTime: STALE.FIVE_MIN,
  });

  const totalItems = summary?.total_items ?? 0;
  const itemsWithStockout = summary?.items_with_stockout ?? 0;
  const viewUnpopulated = !isLoading && totalItems === 0 && itemsWithStockout === 0;
  const allClear = !isLoading && totalItems > 0 && itemsWithStockout === 0;

  return (
    <div className="space-y-4">
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2 mb-3">
        <strong className="text-foreground">Intramonth stockouts</strong> scan daily inventory, not just end-of-month snapshots. An item can be out of stock for 20 days, recover on day 28, and still show green in monthly CSL — this panel shows the true picture.
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard className={PANEL_KPI} label="Items with Stockout" value={(summary?.items_with_stockout ?? 0).toLocaleString()} colorClass="text-red-600" />
        <KpiCard className={PANEL_KPI} label="Extended Stockouts (7d+)" value={(summary?.items_with_extended_stockout ?? 0).toLocaleString()} colorClass="text-red-700" tooltip={{ title: "Extended Stockouts (7d+)", description: "Items out of stock 7+ consecutive days. Service impact rises sharply after day 5." }} />
        <KpiCard className={PANEL_KPI} label="Total Stockout Days" value={(summary?.total_stockout_days ?? 0).toLocaleString()} />
        <KpiCard className={PANEL_KPI} label="Est. Lost Sales" value={Number(summary?.total_est_lost_sales ?? 0).toFixed(0)} tooltip={{ title: "Est. Lost Sales", description: "Estimated units lost to stockout, derived from daily sales velocity × stockout days. Conservative estimate." }} />
      </div>
      {viewUnpopulated && (
        <EmptyState
          icon={Clock}
          title="No intra-month stockouts detected"
          description="Intra-month stockout detection scans daily inventory snapshots for zero on-hand events that occur before the end-of-month snapshot, capturing hidden stockouts that would otherwise be masked by EOM recovery."
          steps={[
            { label: "Load daily inventory snapshots", command: "make load-inventory" },
            { label: "Apply schema (first time only)", command: "make intramonth-schema" },
            { label: "Refresh intramonth stockout view", command: "make intramonth-refresh" },
          ]}
        />
      )}
      {allClear && (
        <EmptyState
          icon={CheckCircle2}
          title="All clear — no stockouts this period"
          description="No intra-month stockout events were detected across the portfolio. Daily inventory snapshots are being scanned and all items maintained positive on-hand quantities throughout the month."
        />
      )}
      {(detail?.rows?.length ?? 0) > 0 && detail && (
        <div className="overflow-x-auto">
          <p className="text-xs font-medium mb-2">Top Stockout Items (current period)</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="text-left py-1 pr-2">Item</th>
                <th className="text-left py-1 px-2">Loc</th>
                <th className="text-right py-1 px-2">Stockout Days</th>
                <th className="text-right py-1 px-2" title="Percentage of days in the month where on-hand inventory reached zero">Stockout %</th>
                <th className="text-right py-1 px-2">Est. Lost Sales</th>
                <th className="text-center py-1 px-2">Extended?</th>
              </tr>
            </thead>
            <tbody>
              {detail.rows.map((r: IntramonthStockoutRow) => (
                <tr key={`${r.item_id}-${r.loc}-${r.month_start}`} className="border-b hover:bg-muted/30">
                  <td className="py-1 pr-2 font-medium">{r.item_id}</td>
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
