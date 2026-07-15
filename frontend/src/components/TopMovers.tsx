import { TrendingUp, TrendingDown } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Mover } from "@/types/theme";

function formatCompact(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(0);
}

interface TopMoversProps {
  movers: Mover[];
  className?: string;
}

export function TopMovers({ movers, className }: TopMoversProps) {
  if (movers.length === 0) {
    return (
      <div className={cn("flex items-center justify-center py-8 text-sm text-muted-foreground", className)}>
        No movers data
      </div>
    );
  }

  return (
    <div className={cn("space-y-1.5", className)}>
      {movers.map((mover, idx) => {
        const isUp = mover.direction === "up";
        return (
          <div
            key={`${mover.item_description}-${idx}`}
            className="flex items-center gap-2 rounded-md px-2 py-1.5 hover:bg-muted/40 transition-colors duration-100 cursor-pointer"
          >
            <span className={cn("flex-shrink-0", isUp ? "text-kpi-best" : "text-kpi-warning")}>
              {isUp ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
            </span>
            <span className="min-w-0 flex-1 truncate text-sm text-card-foreground">
              {mover.item_description}
            </span>
            <span className={cn(
              "flex-shrink-0 text-sm font-semibold tabular-nums",
              isUp ? "text-kpi-best" : "text-kpi-warning",
            )}>
              {isUp ? "+" : ""}{formatCompact(mover.delta)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
