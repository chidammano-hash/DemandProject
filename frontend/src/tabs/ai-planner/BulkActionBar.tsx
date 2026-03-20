/**
 * BulkActionBar — sticky floating bar shown when 1+ insights are selected.
 */
import { Button } from "@/components/ui/button";
import { Loader2, X } from "lucide-react";

export function BulkActionBar({
  count,
  onAcknowledgeAll,
  onClear,
  isPending,
}: {
  count: number;
  onAcknowledgeAll: () => void;
  onClear: () => void;
  isPending: boolean;
}) {
  return (
    <div className="fixed bottom-6 left-1/2 z-40 -translate-x-1/2 flex items-center gap-3 rounded-full border bg-card px-5 py-2.5 shadow-lg">
      <span className="text-sm font-medium">{count} selected</span>
      <div className="h-4 w-px bg-border" />
      <Button size="sm" onClick={onAcknowledgeAll} disabled={isPending} className="rounded-full gap-1.5">
        {isPending && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
        Accept all
      </Button>
      <Button size="sm" variant="ghost" onClick={onClear} className="rounded-full text-muted-foreground">
        <X className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
