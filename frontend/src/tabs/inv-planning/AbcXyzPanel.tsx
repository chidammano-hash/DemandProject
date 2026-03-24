import { useQuery } from "@tanstack/react-query";
import {
  abcXyzKeys,
  fetchAbcXyzMatrix,
  fetchAbcXyzSummary,
  STALE,
  type AbcXyzCell,
} from "@/api/queries";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { KpiCard } from "@/components/KpiCard";
import { EmptyState } from "@/components/EmptyState";
import { Grid3x3 } from "lucide-react";

const PANEL_KPI = "rounded-lg bg-muted/30 p-3";

export function AbcXyzPanel() {
  const { filters } = useGlobalFilterContext();
  const gf = {
    brand: filters.brand.length > 0 ? filters.brand.join(",") : undefined,
    category: filters.category.length > 0 ? filters.category.join(",") : undefined,
    market: filters.market.length > 0 ? filters.market.join(",") : undefined,
    item: filters.item.length === 1 ? filters.item[0] : undefined,
    location: filters.location.length === 1 ? filters.location[0] : undefined,
  };

  const { data: matrix, isLoading } = useQuery({
    queryKey: abcXyzKeys.matrix(gf),
    queryFn: () => fetchAbcXyzMatrix(gf),
    staleTime: STALE.FIVE_MIN,
  });
  const { data: summary } = useQuery({
    queryKey: abcXyzKeys.summary(gf),
    queryFn: () => fetchAbcXyzSummary(gf),
    staleTime: STALE.FIVE_MIN,
  });

  const ABC = ["A", "B", "C"];
  const XYZ = ["X", "Y", "Z"];
  const cellMap = new Map((matrix?.cells ?? []).map((c: AbcXyzCell) => [c.segment, c]));

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <KpiCard className={PANEL_KPI} label="Total SKUs" value={(summary?.total_skus ?? 0).toLocaleString()} />
        <KpiCard className={PANEL_KPI} label="Classified" value={(matrix?.total_classified ?? 0).toLocaleString()} />
        <KpiCard className={PANEL_KPI} label="Z-Class (High Variability)" value={(summary?.z_count ?? 0).toLocaleString()} colorClass="text-amber-600" />
      </div>
      {!isLoading && (matrix?.cells ?? []).length === 0 && (
        <EmptyState
          icon={Grid3x3}
          title="ABC-XYZ classification not run"
          description="ABC-XYZ segments each DFU by demand volume (A=top 80%, B=next 15%, C=bottom 5%) cross-classified with demand variability (X=stable, Y=moderate, Z=volatile). The 3×3 matrix guides differentiated service-level policies."
          steps={[
            { label: "Apply schema (first time only)", command: "make abc-xyz-schema" },
            { label: "Classify all SKUs", command: "make abc-xyz-classify" },
          ]}
        />
      )}
      {(matrix?.cells ?? []).length > 0 && (
        <>
          <details className="border rounded p-2 text-xs mb-3 bg-muted/20">
            <summary className="cursor-pointer font-medium text-foreground">What is ABC-XYZ Classification? ▸</summary>
            <div className="mt-2 space-y-1 text-muted-foreground">
              <p><strong className="text-foreground">Volume (ABC):</strong> A = top 80% of demand value · B = next 15% · C = bottom 5%</p>
              <p><strong className="text-foreground">Variability (XYZ):</strong> X = stable (CV &lt; 0.3) · Y = moderate (CV 0.3–0.8) · Z = volatile (CV &gt; 0.8 or lumpy demand)</p>
              <p><strong className="text-foreground">Use:</strong> Each segment gets a differentiated service level, review frequency, and safety stock policy.</p>
            </div>
          </details>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-1 pr-2">ABC╲XYZ</th>
                  {XYZ.map(x => <th key={x} className="text-center py-1 px-2">{x}</th>)}
                </tr>
              </thead>
              <tbody>
                {ABC.map(a => (
                  <tr key={a} className="border-b">
                    <td className="py-1 pr-2 font-medium">{a}</td>
                    {XYZ.map(x => {
                      const seg = a + x;
                      const cell = cellMap.get(seg);
                      const colorMap: Record<string, string> = {
                        AX: "bg-green-50 dark:bg-green-900/20",
                        AY: "bg-blue-50 dark:bg-blue-900/20",
                        BX: "bg-blue-50 dark:bg-blue-900/20",
                        AZ: "bg-amber-50 dark:bg-amber-900/20",
                        BY: "bg-amber-50 dark:bg-amber-900/20",
                        CX: "bg-blue-50 dark:bg-blue-900/20",
                        BZ: "bg-red-50 dark:bg-red-900/20",
                        CY: "bg-amber-50 dark:bg-amber-900/20",
                        CZ: "bg-red-50 dark:bg-red-900/20",
                      };
                      const bgClass = colorMap[seg] ?? "bg-muted/20";
                      return (
                        <td key={x} className="text-center py-1 px-2">
                          {cell ? (
                            <div className={`rounded ${bgClass} px-2 py-1`}>
                              <span className="font-semibold">{cell.sku_count}</span>
                              <br />
                              <span className="text-muted-foreground">SL {((cell.avg_service_level ?? 0) * 100).toFixed(0)}%</span>
                            </div>
                          ) : <span className="text-muted-foreground">—</span>}
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
