import { useQuery } from "@tanstack/react-query";
import {
  abcXyzKeys,
  fetchAbcXyzMatrix,
  fetchAbcXyzSummary,
  STALE,
  type AbcXyzCell,
} from "@/api/queries";

export function AbcXyzPanel() {
  const { data: matrix } = useQuery({
    queryKey: abcXyzKeys.matrix(),
    queryFn: fetchAbcXyzMatrix,
    staleTime: STALE,
  });
  const { data: summary } = useQuery({
    queryKey: abcXyzKeys.summary(),
    queryFn: fetchAbcXyzSummary,
    staleTime: STALE,
  });

  const ABC = ["A", "B", "C"];
  const XYZ = ["X", "Y", "Z"];
  const cellMap = new Map((matrix?.cells ?? []).map((c: AbcXyzCell) => [c.segment, c]));

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Total DFUs</p>
          <p className="text-xl font-bold">{Number(summary?.total_dfus ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Classified</p>
          <p className="text-xl font-bold">{(matrix?.total_classified ?? 0).toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">Z-Class (High Variability)</p>
          <p className="text-xl font-bold text-amber-600">{Number(summary?.z_count ?? 0).toLocaleString()}</p>
        </div>
      </div>
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
                  return (
                    <td key={x} className="text-center py-1 px-2">
                      {cell ? (
                        <div className="rounded bg-blue-50 dark:bg-blue-900/20 px-2 py-1">
                          <span className="font-semibold">{cell.dfu_count}</span>
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
    </div>
  );
}
