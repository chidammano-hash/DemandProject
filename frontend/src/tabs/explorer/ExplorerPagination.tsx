/**
 * Pagination footer with row range, page indicator, and prev/next buttons.
 */
import { Button } from "@/components/ui/button";
import { formatNumber } from "@/lib/formatters";

export interface ExplorerPaginationProps {
  offset: number;
  limit: number;
  total: number;
  totalApproximate: boolean;
  onPrev: () => void;
  onNext: () => void;
}

export function ExplorerPagination({
  offset,
  limit,
  total,
  totalApproximate,
  onPrev,
  onNext,
}: ExplorerPaginationProps) {
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(offset + limit, total);

  return (
    <div className="mt-3 flex items-center justify-between gap-2 text-sm">
      <span className="text-muted-foreground">
        Showing {start}-{end} of{" "}
        {totalApproximate ? `${formatNumber(total - 1)}+` : formatNumber(total)}
        {total > 0 && (
          <span className="ml-2 tabular-nums">
            (Page {Math.floor(offset / limit) + 1} of{" "}
            {totalApproximate
              ? `${Math.ceil((total - 1) / limit)}+`
              : Math.ceil(total / limit)}
            )
          </span>
        )}
      </span>
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={offset === 0}
          onClick={onPrev}
        >
          Previous
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={offset + limit >= total}
          onClick={onNext}
        >
          Next
        </Button>
      </div>
    </div>
  );
}
