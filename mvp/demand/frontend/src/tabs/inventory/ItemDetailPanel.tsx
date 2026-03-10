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
        <div className="mb-2 rounded bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
          Daily inventory snapshots for <strong>{selectedRow.item}</strong> at <strong>{selectedRow.location}</strong>.{" "}
          Rows show end-of-day positions — use to trace stockout events or intra-month inventory movements.
        </div>
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
                  <TableHead
                    className="text-xs text-right"
                    title="Inventory units in stock at the time of this daily snapshot"
                  >
                    On Hand
                  </TableHead>
                  <TableHead
                    className="text-xs text-right"
                    title="Combined inventory position: in stock plus already ordered"
                  >
                    On Hand + On Order
                  </TableHead>
                  <TableHead
                    className="text-xs text-right"
                    title="Units on open purchase orders not yet received"
                  >
                    On Order
                  </TableHead>
                  <TableHead
                    className="text-xs text-right"
                    title="Supplier lead time in days for this item at this location"
                  >
                    Lead Time (days)
                  </TableHead>
                  <TableHead
                    className="text-xs text-right"
                    title="Cumulative sales units month-to-date at snapshot date"
                  >
                    MTD Sales
                  </TableHead>
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
                      {snap.qty_on_hand.toLocaleString()}
                    </TableCell>
                    <TableCell className="text-sm text-right tabular-nums">
                      {snap.qty_on_hand_on_order.toLocaleString()}
                    </TableCell>
                    <TableCell className="text-sm text-right tabular-nums">
                      {snap.qty_on_order.toLocaleString()}
                    </TableCell>
                    <TableCell className="text-sm text-right tabular-nums">
                      {snap.lead_time_days != null
                        ? snap.lead_time_days.toLocaleString()
                        : "-"}
                    </TableCell>
                    <TableCell className="text-sm text-right tabular-nums">
                      {snap.mtd_sales.toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        ) : snapshots && snapshots.length === 0 ? (
          <p className="py-4 text-center text-sm text-muted-foreground">
            No snapshot history found for this item-location.
          </p>
        ) : (
          <p className="text-sm text-muted-foreground">
            No snapshot history available for this item-location pair.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
