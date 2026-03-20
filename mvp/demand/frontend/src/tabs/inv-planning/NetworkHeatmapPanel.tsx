import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  insightKeys,
  fetchNetworkHeatmap,
  STALE_INSIGHTS,
  type NetworkHeatmapCell,
} from "@/api/queries";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { formatFixed, formatInt } from "@/lib/formatters";
import { Map as MapIcon } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

function dosTierColor(dos: number | null): string {
  if (dos == null) return "bg-gray-100 dark:bg-gray-800";
  if (dos < 7) return "bg-red-400 dark:bg-red-600 text-white";
  if (dos < 15) return "bg-orange-300 dark:bg-orange-600 text-white";
  if (dos < 31) return "bg-yellow-200 dark:bg-yellow-500";
  if (dos <= 60) return "bg-green-200 dark:bg-green-600";
  return "bg-blue-200 dark:bg-blue-600";
}

function dosTierLabel(dos: number | null): string {
  if (dos == null) return "No data";
  if (dos < 7) return "Critical (<7d)";
  if (dos < 15) return "Low (7-14d)";
  if (dos < 31) return "Normal (15-30d)";
  if (dos <= 60) return "Healthy (30-60d)";
  return "Excess (>60d)";
}

export function NetworkHeatmapPanel() {
  const [hoveredCell, setHoveredCell] = useState<NetworkHeatmapCell | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: insightKeys.networkHeatmap(),
    queryFn: fetchNetworkHeatmap,
    staleTime: STALE_INSIGHTS.FIVE_MIN,
  });

  const cellMap = useMemo(() => {
    const map = new Map<string, NetworkHeatmapCell>();
    for (const cell of data?.cells ?? []) {
      map.set(`${cell.location}|${cell.category}`, cell);
    }
    return map;
  }, [data?.cells]);

  if (error) {
    return (
      <div className="text-xs text-red-600 p-4">
        Failed to load network heatmap: {(error as Error).message}
      </div>
    );
  }

  const summary = data?.summary;
  const locations = data?.locations ?? [];
  const categories = data?.categories ?? [];

  return (
    <div className="space-y-4">
      <div className="text-xs text-muted-foreground bg-muted/20 border rounded px-3 py-2">
        Network balance heatmap showing days of supply (DOS) across locations and product categories.
        Cells are colored by DOS tier to highlight critical shortages and excess inventory positions.
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard
          className={PANEL_KPI}
          label="Total Locations"
          value={isLoading ? "..." : formatInt(summary?.total_locations)}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Critical Cells (<7d)"
          value={isLoading ? "..." : formatInt(summary?.critical_cells)}
          colorClass={(summary?.critical_cells ?? 0) > 0 ? "text-red-600" : undefined}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Excess Cells (>60d)"
          value={isLoading ? "..." : formatInt(summary?.excess_cells)}
          colorClass={(summary?.excess_cells ?? 0) > 0 ? "text-blue-600" : undefined}
        />
        <KpiCard
          className={PANEL_KPI}
          label="Avg Network DOS"
          value={isLoading ? "..." : formatFixed(summary?.avg_network_dos)}
          sublabel="days"
        />
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-2 text-[10px]">
        {[
          { color: "bg-red-400", label: "<7d" },
          { color: "bg-orange-300", label: "7-14d" },
          { color: "bg-yellow-200", label: "15-30d" },
          { color: "bg-green-200", label: "30-60d" },
          { color: "bg-blue-200", label: ">60d" },
        ].map((tier) => (
          <div key={tier.label} className="flex items-center gap-1">
            <div className={`w-3 h-3 rounded ${tier.color}`} />
            <span className="text-muted-foreground">{tier.label}</span>
          </div>
        ))}
      </div>

      {isLoading ? (
        <p className="text-xs text-muted-foreground">Loading heatmap...</p>
      ) : locations.length === 0 || categories.length === 0 ? (
        <EmptyState
          icon={MapIcon}
          title="No network data available"
          description="The network heatmap displays days of supply across locations and categories. Ensure inventory data has been loaded and materialized views refreshed."
          steps={[
            { label: "Load inventory data", command: "make load-all" },
            { label: "Refresh materialized views", command: "make db-apply-sql" },
          ]}
        />
      ) : (
        <>
          {/* Hover tooltip */}
          {hoveredCell && (
            <div className="text-xs bg-card border rounded px-3 py-2 shadow-sm">
              <span className="font-medium">{hoveredCell.location}</span> / <span className="font-medium">{hoveredCell.category}</span>
              {" — "}
              Avg DOS: <span className="font-mono">{formatFixed(hoveredCell.avg_dos)}</span>,
              Items: <span className="font-mono">{hoveredCell.item_count.toLocaleString()}</span>
              {" — "}{dosTierLabel(hoveredCell.avg_dos)}
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="text-xs border-collapse">
              <thead>
                <tr>
                  <th className="text-left py-1 pr-2 text-muted-foreground sticky left-0 bg-background z-10">
                    Location
                  </th>
                  {categories.map((cat) => (
                    <th
                      key={cat}
                      className="text-center py-1 px-1 text-muted-foreground font-normal min-w-[60px]"
                    >
                      {cat}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {locations.map((loc) => (
                  <tr key={loc}>
                    <td className="py-1 pr-2 font-mono whitespace-nowrap sticky left-0 bg-background z-10">
                      {loc}
                    </td>
                    {categories.map((cat) => {
                      const cell = cellMap.get(`${loc}|${cat}`);
                      const dos = cell?.avg_dos ?? null;
                      return (
                        <td
                          key={cat}
                          className={`py-1 px-1 text-center font-mono cursor-default rounded ${dosTierColor(dos)}`}
                          onMouseEnter={() =>
                            cell && setHoveredCell(cell)
                          }
                          onMouseLeave={() => setHoveredCell(null)}
                        >
                          {dos != null ? formatFixed(dos, 0) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
