import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchSourcingRows,
  fetchSourcingNetwork,
  STALE,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { Package } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function SourcingPanel() {
  const [item, setItem] = useState("");
  const [loc, setLoc] = useState("");
  const [supplier, setSupplier] = useState("");
  const [page, setPage] = useState(0);
  const pageSize = 50;

  const { data: network } = useQuery({
    queryKey: ["sourcing", "network"],
    queryFn: fetchSourcingNetwork,
    staleTime: STALE.FIVE_MIN,
  });

  const { data, isLoading } = useQuery({
    queryKey: ["sourcing", "rows", item, loc, supplier, page],
    queryFn: () =>
      fetchSourcingRows({
        item: item || undefined,
        loc: loc || undefined,
        supplier: supplier || undefined,
        limit: pageSize,
        offset: page * pageSize,
      }),
    staleTime: STALE.ONE_MIN,
  });

  return (
    <div className="space-y-4">
      {/* KPI summary */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <KpiCard className={PANEL_KPI} label="Total Mappings" value={(network?.total_rows ?? 0).toLocaleString()} />
        <KpiCard className={PANEL_KPI} label="Suppliers" value={(network?.supplier_count ?? 0).toLocaleString()} />
        <KpiCard className={PANEL_KPI} label="Item-Locations" value={(network?.item_location_count ?? 0).toLocaleString()} />
        <KpiCard
          className={PANEL_KPI}
          label="Single-Source"
          value={(network?.single_source_count ?? 0).toLocaleString()}
          colorClass="text-amber-600"
          sublabel="supply risk"
        />
        <KpiCard className={PANEL_KPI} label="Multi-Source" value={(network?.multi_source_count ?? 0).toLocaleString()} />
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-2">
        <input
          className="border rounded px-2 py-1 text-sm w-36"
          placeholder="Filter item..."
          value={item}
          onChange={(e) => { setItem(e.target.value); setPage(0); }}
        />
        <input
          className="border rounded px-2 py-1 text-sm w-36"
          placeholder="Filter location..."
          value={loc}
          onChange={(e) => { setLoc(e.target.value); setPage(0); }}
        />
        <input
          className="border rounded px-2 py-1 text-sm w-36"
          placeholder="Filter supplier..."
          value={supplier}
          onChange={(e) => { setSupplier(e.target.value); setPage(0); }}
        />
      </div>

      {/* Table */}
      {!isLoading && (!data || data.rows.length === 0) ? (
        <EmptyState
          icon={Package}
          title="No sourcing data"
          description="Sourcing maps items and locations to their supply sources (supplier-plant combinations)."
          steps={[
            { label: "Normalize sourcing data", command: "make normalize-sourcing" },
            { label: "Load via data pipeline", command: "make load-sourcing" },
          ]}
        />
      ) : (
        <>
          <div className="text-sm text-muted-foreground">
            {(data?.total ?? 0).toLocaleString()} rows
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-muted-foreground">
                  <th className="py-2 px-2">Item</th>
                  <th className="py-2 px-2">Location</th>
                  <th className="py-2 px-2">Source</th>
                  <th className="py-2 px-2">Supplier</th>
                  <th className="py-2 px-2">Plant</th>
                  <th className="py-2 px-2">Transit</th>
                  <th className="py-2 px-2">Site</th>
                </tr>
              </thead>
              <tbody>
                {data?.rows.map((r) => (
                  <tr key={r.sourcing_ck} className="border-b hover:bg-muted/20">
                    <td className="py-1.5 px-2 font-mono">{r.item_id}</td>
                    <td className="py-1.5 px-2">{r.loc}</td>
                    <td className="py-1.5 px-2 font-mono">{r.source_cd}</td>
                    <td className="py-1.5 px-2">{r.supplier_id}</td>
                    <td className="py-1.5 px-2">{r.plant_id}</td>
                    <td className="py-1.5 px-2">{r.transit_mode}</td>
                    <td className="py-1.5 px-2">{r.site_id}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex gap-2 justify-end">
            <button
              className="px-3 py-1 text-sm border rounded disabled:opacity-40"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
            >
              Prev
            </button>
            <button
              className="px-3 py-1 text-sm border rounded disabled:opacity-40"
              disabled={(data?.rows.length ?? 0) < pageSize}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
}
