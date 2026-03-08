import { useQuery } from "@tanstack/react-query";
import {
  supplierKeys,
  fetchSupplierSummary,
  fetchSupplierDetail,
  STALE,
  type SupplierRow,
} from "@/api/queries";

export function SupplierPanel() {
  const { data: summary } = useQuery({
    queryKey: supplierKeys.summary(),
    queryFn: fetchSupplierSummary,
    staleTime: STALE.FIVE_MIN,
  });
  const { data: detail } = useQuery({
    queryKey: supplierKeys.detail({ limit: 10 }),
    queryFn: () => fetchSupplierDetail({ limit: 10, sort_by: "supplier_reliability_score", sort_dir: "asc" }),
    staleTime: STALE.FIVE_MIN,
  });

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Total Suppliers</p>
          <p className="text-xl font-bold">{Number(summary?.total_suppliers ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Avg Reliability Score</p>
          <p className="text-xl font-bold">{Number(summary?.avg_reliability_score ?? 0).toFixed(0)}/100</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Avg Lead Time (days)</p>
          <p className="text-xl font-bold">{Number(summary?.avg_lead_time_days ?? 0).toFixed(1)}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Low Reliability (&lt;40)</p>
          <p className="text-xl font-bold text-red-600">{Number(summary?.low_reliability_count ?? 0).toLocaleString()}</p>
        </div>
      </div>
      {detail && detail.rows.length > 0 && (
        <div className="overflow-x-auto">
          <p className="text-xs font-medium mb-2">Suppliers by Reliability (lowest first)</p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="text-left py-1 pr-2">Supplier</th>
                <th className="text-right py-1 px-2">Score</th>
                <th className="text-right py-1 px-2">SKU-Locs</th>
                <th className="text-right py-1 px-2">Avg LT (d)</th>
                <th className="text-right py-1 px-2">LT CV</th>
                <th className="text-right py-1 px-2">% Stable</th>
              </tr>
            </thead>
            <tbody>
              {detail.rows.map((r: SupplierRow) => (
                <tr key={r.supplier_no} className="border-b hover:bg-muted/30">
                  <td className="py-1 pr-2 font-medium">{r.supplier_name ?? r.supplier_no}</td>
                  <td className={`text-right py-1 px-2 font-semibold ${(r.supplier_reliability_score ?? 100) < 40 ? "text-red-600" : (r.supplier_reliability_score ?? 100) < 70 ? "text-amber-600" : "text-green-600"}`}>
                    {r.supplier_reliability_score ?? "—"}
                  </td>
                  <td className="text-right py-1 px-2">{r.sku_loc_count?.toLocaleString() ?? "—"}</td>
                  <td className="text-right py-1 px-2">{r.avg_lt_mean_days != null ? Number(r.avg_lt_mean_days).toFixed(1) : "—"}</td>
                  <td className="text-right py-1 px-2">{r.avg_lt_cv != null ? Number(r.avg_lt_cv).toFixed(2) : "—"}</td>
                  <td className="text-right py-1 px-2">{r.pct_stable_lt != null ? `${(Number(r.pct_stable_lt) * 100).toFixed(0)}%` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
