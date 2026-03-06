import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { LoadingElement } from "@/components/LoadingElement";
import { formatNumber } from "@/lib/formatters";

interface DetailSnapshot {
  snapshot_date: string;
  qty_on_hand: number;
  qty_on_hand_on_order: number;
  qty_on_order: number;
  lead_time_days: number | null;
  mtd_sales: number;
}

interface ItemDetailPanelProps {
  selectedRow: { item: string; location: string };
  snapshots: DetailSnapshot[] | undefined;
  isLoading: boolean;
}

export function ItemDetailPanel({
  selectedRow,
  snapshots,
  isLoading,
}: ItemDetailPanelProps) {
  return (
    <Card className="animate-fade-in border-primary/20">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">
          Item Detail: {selectedRow.item} @ {selectedRow.location}
        </CardTitle>
        <CardDescription>
          Snapshot history for this item-location pair.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <LoadingElement
            tabKey="inventory"
            message="Loading item detail..."
          />
        ) : snapshots && snapshots.length > 0 ? (
          <div className="max-h-[300px] overflow-auto rounded-md border border-input">
            <Table>
              <TableHeader>
                <TableRow className="border-muted bg-muted/30">
                  <TableHead className="text-xs">Snapshot Date</TableHead>
                  <TableHead className="text-xs text-right">On Hand</TableHead>
                  <TableHead className="text-xs text-right">
                    On Hand+Order
                  </TableHead>
                  <TableHead className="text-xs text-right">On Order</TableHead>
                  <TableHead className="text-xs text-right">Lead Time</TableHead>
                  <TableHead className="text-xs text-right">MTD Sales</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {snapshots.map((snap, idx) => (
                  <TableRow
                    key={`${snap.snapshot_date}-${idx}`}
                    className="hover:bg-muted/30"
                  >
                    <TableCell className="text-sm tabular-nums">
                      {snap.snapshot_date}
                    </TableCell>
                    <TableCell className="text-sm text-right tabular-nums">
                      {formatNumber(snap.qty_on_hand)}
                    </TableCell>
                    <TableCell className="text-sm text-right tabular-nums">
                      {formatNumber(snap.qty_on_hand_on_order)}
                    </TableCell>
                    <TableCell className="text-sm text-right tabular-nums">
                      {formatNumber(snap.qty_on_order)}
                    </TableCell>
                    <TableCell className="text-sm text-right tabular-nums">
                      {snap.lead_time_days != null
                        ? formatNumber(snap.lead_time_days)
                        : "-"}
                    </TableCell>
                    <TableCell className="text-sm text-right tabular-nums">
                      {formatNumber(snap.mtd_sales)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            No snapshot history available for this item-location pair.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
